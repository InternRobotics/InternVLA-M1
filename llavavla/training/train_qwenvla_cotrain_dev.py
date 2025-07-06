"""
train.py
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist

import yaml
import wandb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import AutoProcessor
from transformers import get_scheduler

from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import argparse
from omegaconf import OmegaConf
from hydra import initialize

from llavavla.training.metrics import normalize_dotlist_args

from prismatic.overwatch import initialize_overwatch
# from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


from llavavla.dataloader.vlm_datasets import make_vlm_dataloader

from llavavla.dataloader.rlds_datasets import get_vla_dataset, collate_fn# TODO 要移动到dataloader 下面
from accelerate import Accelerator, DeepSpeedPlugin

deepspeed_plugin = DeepSpeedPlugin()# 这个插件是否能使用到 config 的参数呢？ 其实这里应该是可以飞显示用的， 感觉有版本问题 #zero_stage=2, gradient_accumulation_steps=1 ：v2: hf_ds_config="scripts/run_scripts/ds_config.yaml"
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state) # TODO 之后要移动到trainer 内部， --> 直接搬LLaVA trainer

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__) # 后期移除， 不要基于 prismatic 来玩输出
logger = get_logger(__name__)


from llavavla.model.framework.qwenpi_dev import build_model_framework

def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )
    return fast_tokenizer


class VLAMTrainer:
    def __init__(self, cfg, model, vla_train_dataloader, vlm_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        
        # 训练状态跟踪
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        # 初始化训练组件
        self._init_wandb()
        self._init_checkpointing()
    
    def _calculate_total_batch_size(self):
        """计算全局批量大小"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )
    
    def _init_wandb(self):
        """初始化Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )
    
    def _init_checkpointing(self):
        """初始化检查点目录"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # 恢复训练状态
        # 要判断是否有self.config.trainer.pretrained_checkpoint
        if pretrained_checkpoint and is_resume: # TODO 这里还没能够保存state, 思考是否必要
            self._load_checkpoint(self.config.resume_from_checkpoint)
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # TODO: 恢复训练步数和其他状态
    
    def _save_checkpoint(self):
        """保存当前训练状态"""

        if accelerator.is_main_process:
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # 保存模型状态
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            
            # 保存训练元数据
            summary_data = {
                "steps": self.completed_steps,
                # TODO: 添加其他需要保存的训练状态
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            
        self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """记录训练指标"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0: # 有些参数应该是需要intial 给 class 的了
            if self.accelerator.is_main_process:
                # 计算梯度范数
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                metrics["grad_norm"] = total_norm ** 0.5
                
                # 添加学习率
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                
                # 添加epoch信息
                metrics["epoch"] = self.completed_steps // len(self.vla_train_dataloader)
                
                # 记录到W&B
                wandb.log(metrics, step=self.completed_steps)
                
                # 调试输出
                if self.config.is_debug:
                    print(f"Step {self.completed_steps}: {metrics}")
    
    def _create_data_iterators(self):
        """创建数据迭代器"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.vlm_train_dataloader)
    
    def _get_next_batch(self):
        """获取下一批数据（自动处理数据循环）"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)
        
        try:
            batch_vlm = next(self.vlm_iter) # TODO 首尾循环应该是dataset 自己的功能， 这里是考虑到很多人的dataset 是没有这个功能的
        except StopIteration:
            self.vlm_iter = iter(self.vlm_train_dataloader)
            batch_vlm = next(self.vlm_iter)
        
        return batch_vla, batch_vlm
    
    def train(self):
        """执行训练循环"""
        # 打印训练配置
        self._log_training_config()
        
        # 准备数据迭代器
        self._create_data_iterators()
        
        # 创建进度条
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        
        # 主训练循环
        while self.completed_steps < self.config.trainer.max_train_steps:
            # 获取数据批次
            batch_vla, batch_vlm = self._get_next_batch()
            
            # 执行训练步骤
            step_metrics = self._train_step(batch_vla, batch_vlm)
            
            # 更新进度
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            # 记录指标
            self._log_metrics(step_metrics)
            
            # 保存检查点
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
            
            # 检查终止条件
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break
        
        # 训练结束处理
        self._finalize_training()
    
    def _log_training_config(self):
        """记录训练配置"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f" Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    
    def _train_step(self, batch_vla, batch_vlm):
        """执行单个训练步骤"""
        # TODO: 实现梯度累积
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            
            # VLA任务前向传播
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                action_loss, action_vlm_loss = self.model.forward(batch_vla)
                total_loss = action_loss + action_vlm_loss
            
            # VLA反向传播
            self.accelerator.backward(total_loss)
            
            # VLM任务前向传播
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                vlm_output = self.model.qwen_vl_interface(**batch_vlm)
                vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
            
            # VLM反向传播
            self.accelerator.backward(vlm_loss)
            
            # 梯度裁剪
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.trainer.gradient_clipping
                )
            
            # 优化器步骤
            self.optimizer.step()
            self.lr_scheduler.step()
        
        return {
            "action_dit_loss": action_loss.item(),
            "action_vlm_loss": action_vlm_loss.item(),
            "vlm_loss": vlm_loss.item(),
        }
    
    def _finalize_training(self):
        """训练结束处理"""
        # 保存最终模型
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        
        # 关闭W&B
        if self.accelerator.is_main_process:
            wandb.finish()
        
        self.accelerator.wait_for_everyone()

from llavavla.training.metrics import build_param_lr_groups
def train(cfg) -> None:
    overwatch.info("VLA Training :: Warming Up")
    # accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # dist.barrier()  # Ensure all processes are synchronized before starting training
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # Save as YAML using OmegaConf
        OmegaConf.save(cfg, output_dir / "config.yaml")
        # Additionally save as JSON TODO 之后要将 .model 的参数单独save json
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    dist.barrier()
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!

    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    vla = build_model_framework(cfg)
    # fast_tokenizer = load_fast_tokenizer() # TODO 考虑架构时候的事情
    # processor = vla.vlm.processor # @Jinhui TODO 不应该在这个地方 赋值， 数据准备应该和 封装类绑定为函数
    # [Validate] Model should be in Full Precision! @Jinhui TODO Why?
    # for param in vla.parameters():
    #     if param.dtype != torch.float32: #@Jinhui TODO Check, why?
    #         param.data = param.data.float()
    #     assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"


    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    
    vla.freeze_backbones() # TODO 应该是trainer 要做的事情

    # Print number of total/trainable model parameters # TODO 应该集成到trainer 中
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )


    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.datasets.keys()}`")
    #   text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vla_dataset = get_vla_dataset( # 拒绝任何内部转换
        cfg.datasets.vla_data.data_root_dir, # 太多参数了， 应该config 穿越过去， 或者是 ** 的方式
        cfg.datasets.vla_data.data_mix,
        default_image_resolution=tuple(cfg.datasets.vla_data.default_image_resolution),
        shuffle_buffer_size=cfg.datasets.vla_data.shuffle_buffer_size,
        image_aug=cfg.datasets.vla_data.image_aug,
        future_action_window_size=cfg.framework.action_model.future_action_window_size,
        past_action_window_size=cfg.framework.action_model.past_action_window_size,
        load_all_data_for_training=cfg.datasets.vla_data.load_all_data_for_training,
    )

    # Create DataLoader
    
    vla_train_dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size, # @Jinhui TODO 感觉即使有个空的 collate_fn 也会让代码 扩展性 更好
        collate_fn=collate_fn
    )

    vlm_data_mudule = make_vlm_dataloader(cfg) # TODO 👆构建dataloader 的逻辑也不能放到这里。 思考一下，为什么 SFTTrainer 需要这样写
    vlm_train_dataloader = vlm_data_mudule["train_dataloader"]
    # sample = next(iter(vla_dataset)) #for debug

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, output_dir)
    
    # Create Train Strategy
    dist.barrier()
    accelerator.dataloader_config.dispatch_batches =  False # TODO 是不是可以写到 config 内部？
    # Initialize optimizer

    param_groups = build_param_lr_groups(vla=vla, cfg=cfg) # TODO 这里的参数应该是从 config 中获取的， 而不是直接写死
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas), # 这是用于 一阶和二阶动量估计 的两个超参数：
        weight_decay=1e-8, # 这是用于 L2 正则化 的项（惩罚参数值太大的趋势）：
        eps=1e-8,
    )
    pass
    dist.barrier()
    if overwatch.is_rank_zero(): # 想办法写成一个修饰函数
        for i, group in enumerate(optimizer.param_groups):
            print(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    # Initialize learning rate scheduler
    
    num_warmup_steps = min(int(cfg.trainer.max_train_steps*cfg.trainer.warmup_ratio), cfg.trainer.max_warmup_steps)
    cfg.trainer.num_warmup_steps = num_warmup_steps

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps
    )

    # Prepare everything with Accelerator, setup
    vla, optimizer, vla_train_dataloader, vlm_train_dataloader = accelerator.prepare( # @JinhuiYE 第三方工具 or DDP？
        vla, optimizer, vla_train_dataloader, vlm_train_dataloader
    )
    

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")

    # Run VLA Training # TODO move them to class tainer 
    # 创建Trainer实例
    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator
    )
    
    # 执行训练
    trainer.train()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="llavavla/conf/qwenact.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and overwatch.is_rank_zero():
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    train(cfg)
