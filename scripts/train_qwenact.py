"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import json
import os
import re
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
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
from transformers import SchedulerType, get_scheduler
from qwen_vl_utils import process_vision_info
import math
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Callable, Optional

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
# from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# from llavavla.training import VLAMetrics, get_train_strategy
from llavavla.training.materialize_qwen import get_train_strategy
from llavavla.training import VLAMetrics

from llavavla.conf import VLAConfig, VLARegistry
from llavavla.model.vla import load_qwenvl, load_qwenvla
from llavavla.model.vla import CogACT_Qwen
from llavavla.training.materialize_qwen import get_vla_dataset_and_collator
from llavavla.model.tools import * #TODO just for fast debug, remove later
from accelerate import Accelerator, DeepSpeedPlugin

# 设置 DeepSpeed 插件
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=4)
accelerator = Accelerator(mixed_precision='bf16', deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__) # 后期移除， 不要基于 prismatic 来玩输出
logger = get_logger(__name__)

@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Union[str, Path]] = None                  # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    # (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 42                                                  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    #trackers: Tuple[str, ...] = ("jsonl",)                         # Trackers to initialize (if W&B, add config!)
    wandb_project: str = ""                                         # Name of W&B project to log to (use default!)
    wandb_entity: str = ""                                          # Name of entity to log under
    repeated_diffusion_steps: int = 8                               # Repeated steps for training action model (a diffusion model)
    load_all_data_for_training: bool = True                         # Load all training data 
    future_action_window_size: int = 15                             # Action chunking, predicting future actions + current action
    past_action_window_size: int = 0                                # Action history window size, not used now, set to be 0 
    action_model_type: str = 'DiT-B'                                # Action model type, chose from ['DiT-S', 'DiT-B', 'DiT-L']
    use_ema: bool = False                                           # EMA version of action model
    action_dim: int = 7                                             # Dimension of action space

    #@Jinhui overwrite 
    is_debug: Optional[bool] = False                              # Epoch to Resume (should match checkpoint)

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on

def load_model(cfg, hf_token):
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None: #这里还没有检查过
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            pretrained_checkpoint = Path(cfg.pretrained_checkpoint) #@Jinhui TODO Check cfg.pretrained_checkpoint 的参数类型为什么一直对不上
            # cfg.pretrained_checkpoint = pretrained_checkpoint #TO MV 看一下为什么要检查这个
            assert int(re.search("step-(.+?)-", pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", pretrained_checkpoint.name).group(1)) == cfg.resume_epoch
        overwatch.info("Loading VLA Checkpoint")
        if cfg.use_ema:
            overwatch.info("Loading EMA of Diffusion")
        vla = load_qwenvla(cfg.pretrained_checkpoint,
                        hf_token=hf_token,
                        load_for_training=True,  #jinhui False
                        action_model_type=cfg.action_model_type, 
                        action_dim=cfg.action_dim,
                        future_action_window_size=cfg.future_action_window_size,
                        past_action_window_size=cfg.past_action_window_size,
                        use_ema=cfg.use_ema,
                        )

    else:
        # TODO 这里不应该对读取什么 vlm 有假设，应该用接口方法
        # cfg.vla.base_vlm = "playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
        # print(cfg.vla.base_vlm)
        vlm = load_qwenvl(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True) # @Jinhui True
        overwatch.info("Creating VLA from Base VLM")
        if cfg.use_ema:
            overwatch.info("Creating EMA for Diffusion") 
        vla = CogACT_Qwen(vlm, 
                    action_model_type=cfg.action_model_type,
                    action_dim=cfg.action_dim,
                    future_action_window_size=cfg.future_action_window_size,
                    past_action_window_size=cfg.past_action_window_size,
                    use_ema=cfg.use_ema,
                    )
        # del this variable to avoid bugs. The vlm shouldn't be used anymore
        # del vlm
    

    return vla

def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )
    return fast_tokenizer

def trainer(model, train_dataloader, optimizer, lr_scheduler, accelerator, cfg): # @TODO make it as trainer

    cfg.logging_frequency = 10
    cfg.checkpoint_save_frequency = 5000
    cfg.gradient_accumulation_steps = 1
    cfg.gradient_clipping = 1.0
    cfg.vla.max_steps = 135000
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)


    # Initialize Weights and Biases
    if accelerator.is_main_process: # @Jinhui TODO 这里可以查看Openvla 之类的，把它坐着tools
        # wandb.init(project=cfg.wandb_project_name)

        wandb.init(
            name=cfg.run_id,
            dir=os.path.join(cfg.output_dir, "wandb"),
            # config=self.hparams,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group="vla-train",
        )


    # Resume from checkpoint if provided
    if cfg.pretrained_checkpoint and cfg.is_resume:
        accelerator.load_state(cfg.resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {cfg.resume_from_checkpoint}")

    
    # Training loop
    # Right now we assume single node training. I did not test on multi node training.
    total_batch_size = cfg.vla.per_device_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    max_train_steps = 100000
    logger.info(f"  Num steps = {cfg.vla.max_steps}") # cfg.vla.max_train_steps 
    logger.info(f"  Instantaneous batch size per device = {cfg.vla.per_device_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.vla.max_steps}")

    completed_steps = 0

    progress_bar = tqdm(range(cfg.vla.max_steps), disable=not accelerator.is_local_main_process)
    total_loss = 0.0

    
    while completed_steps < max_train_steps:
        for batch in train_dataloader:
            # with accelerator.accumulate(model): # zero2 不允许gred 累计, 先保留， 看看zero3 是否允许
            optimizer.zero_grad() # @Jinhui TODO 之后 put data_processing here 
            # dist.barrier()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                action_loss, output = model.forward(**batch) # TODO make vlm and action loss
                # dist.barrier()
                # vlm_loss = output.vlm_loss
                # dist.barrier()
                total_loss += action_loss.detach().float()

            
            accelerator.backward(action_loss)

            if cfg.gradient_clipping is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg.gradient_clipping)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            optimizer.step()
            lr_scheduler.step()

            # Logging
            if completed_steps % cfg.logging_frequency == 0:
                if accelerator.is_main_process:
                    
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm**0.5
                    lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {completed_steps}, Loss: {action_loss.item()}, Grad Norm: {total_norm}")
                    lr = lr_scheduler.get_last_lr()[0]
                    result = {
                        "train_loss": action_loss.item(),
                        "grad_norm": total_norm,
                        "learning_rate": lr,
                    }
                    wandb.log({"train_loss": action_loss.item(), "learning_rate": lr}, step=completed_steps)
               
            # Checkpointing
            if completed_steps% cfg.save_interval == 0 and completed_steps > 0:
                if accelerator.is_main_process:
                    accelerator.save_state(os.path.join(cfg.output_dir, "checkpoints", f"steps_{completed_steps}"))
                    summary_data = {"steps": completed_steps, "train_loss": total_loss/cfg.save_interval}
                    with open(os.path.join(cfg.output_dir, "summary.jsonl"), "a") as f:
                        f.write(json.dumps(summary_data) + "\n")
                    logger.info(f"Checkpoint saved at step {completed_steps}")
                    total_loss = 0.0
                
            # dist.barrier()  # Ensure all processes log at the same time
                    
            if completed_steps >= max_train_steps:
                break



    # Save final checkpoint
    if accelerator.is_main_process:
        accelerator.save_state(os.path.join(cfg.output_dir, f"steps_{completed_steps}"))
        checkpoint_path = os.path.join(cfg.output_dir, f"steps_{completed_steps}")
        logger.info(f"Training finished. Final checkpoint saved at {checkpoint_path}")
        wandb.finish()

@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("CogACT-VLA Training :: Warming Up")
    gradient_accumulation_steps = 1
    # accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    if cfg.is_debug:
        if int(os.environ.get("RANK", -1)) == 0:
            import debugpy
            debugpy.listen(("0.0.0.0", 5878))
            print("🔍 Rank 0 waiting for debugger attach on port 5678...")
            debugpy.wait_for_client()
    # 显式传递 DeepSpeed 配置 # @Jinhui accelerator 没有找到 合理是使用方式， 像torhrun 之类的可以自动管理这些，而不会有错误出现
    # deepspeed_plugin = DeepSpeedPlugin(
    #     zero_stage=2,
    #     offload_optimizer_device="none",
    #     offload_param_device="none",
    #     logging_level="info",  # 可选：设置为 "debug" 以获取更多日志
    # )

    # accelerator = Accelerator()  # plugins auto-injected based on your YAML
    


    # accelerator.dataloader_config.dispatch_batches =  False
    # Configure Unique Run Name & Save Directory
    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    # torch.cuda.set_device(device_id := overwatch.local_rank())
    # torch.cuda.empty_cache() # 全权交给 Accelerator 管理多机多卡
    

    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.vla.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = Path(cfg.hf_token).read_text().strip() if "/" in cfg.hf_token else os.environ[cfg.hf_token]
    
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    dist.barrier()
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!

    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    vla = load_model(cfg, hf_token)
    fast_tokenizer = load_fast_tokenizer()
    processor = vla.vlm.processor # @Jinhui TODO 不应该在这个地方 赋值， 数据准备应该和 封装类绑定为函数
    # [Validate] Model should be in Full Precision! @Jinhui TODO Why?
    for param in vla.parameters():
        if param.dtype != torch.float32: #@Jinhui TODO Check, why?
            param.data = param.data.float()
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"


    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    stage = "full-finetune"  # Full fine-tuning
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )


    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    #   text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vla_dataset, _, collate_fn = get_vla_dataset_and_collator( # 拒绝任何内部转换
        cfg.data_root_dir,
        cfg.vla.data_mix,
        vlp_processor=vla.vlm.processor, #这个实现很不好，其实现在已经开始和模型绑定了
        tokenizer=vla.vlm.processor.tokenizer,
        prompt_builder_fn=vla.vlm.prompt_builder_fn, #add_turn
        default_image_resolution=(3, 224, 224),
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=cfg.future_action_window_size,
        past_action_window_size=cfg.past_action_window_size,
    )

    # Create DataLoader
    
    train_dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.vla.per_device_batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor,fast_tokenizer)
    )

    # sample = next(iter(vla_dataset)) #for debug

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    
    dist.barrier()
    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    # Prepare everything with Accelerator
    gradient_accumulation_steps = 1
    
    accelerator.dataloader_config.dispatch_batches =  False

    # Initialize optimizer
    learning_rate = 1e-4

    optimizer = torch.optim.AdamW(
        vla.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-8,
        eps=1e-8,
    )
    # Initialize learning rate scheduler
    
    max_train_steps = 135000
    cfg.vla.max_steps = max_train_steps
    num_warmup_steps = 1000
    cfg.num_warmup_steps = num_warmup_steps

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )

    # Prepare everything with Accelerator, setup
    vla, optimizer, train_dataloader = accelerator.prepare( # @JinhuiYE 第三方工具 or DDP？
        vla, optimizer, train_dataloader
    )
    # @Jinhui 推荐用 accelerator， 这里用DDP是因为之前的脚本是torch run


    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    # metrics = VLAMetrics(
    #     cfg.trackers,
    #     cfg.run_id,
    #     run_dir,
    #     draccus.encode(cfg),
    #     wandb_project=cfg.wandb_project,
    #     wandb_entity=cfg.wandb_entity,
    #     resume_step=cfg.resume_step,
    #     resume_epoch=cfg.resume_epoch,
    # )

    # Run VLA Training # TODO move them to class tainer 
    trainer(
        model=vla,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        cfg=cfg
    )

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    
    train()
