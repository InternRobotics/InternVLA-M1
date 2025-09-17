"""
train.py
"""
# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from InternVLA.dataloader.rlds_datasets import collate_fn, get_vla_dataset
from InternVLA.dataloader.qwenvl_llavajson.vlm_datasets import make_vlm_dataloader

from InternVLA.training.trainer_utils.metrics import normalize_dotlist_args
from InternVLA.model.framework import build_framework
from InternVLA.training.trainer_utils.metrics import only_main_process
from InternVLA.training.trainer_utils.metrics import TrainerUtils
from InternVLA.dataloader import save_dataset_statistics


deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)

def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )
    return fast_tokenizer



def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        
        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, \
                open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
        
    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)
    
    return model


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA dataset
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_dataset = get_vla_dataset(
        cfg.datasets.vla_data.data_root_dir,
        cfg.datasets.vla_data.data_mix,
        default_image_resolution=tuple(cfg.datasets.vla_data.default_image_resolution),
        shuffle_buffer_size=cfg.datasets.vla_data.shuffle_buffer_size,
        image_aug=cfg.datasets.vla_data.image_aug,
        future_action_window_size=cfg.framework.action_model.future_action_window_size,
        past_action_window_size=cfg.framework.action_model.past_action_window_size,
        load_all_data_for_training=cfg.datasets.vla_data.load_all_data_for_training,
    )
    
    # VLA dataloader
    vla_train_dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn
    )
    
    # VLM data loader
    vlm_data_module = make_vlm_dataloader(cfg)
    vlm_train_dataloader = vlm_data_module["train_dataloader"]
    
    # save dataset statistics
    if accelerator.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, output_dir)
    
    # reject automatic dispatch # TODO should write to accelerator config
    accelerator.dataloader_config.dispatch_batches =  False
    dist.barrier()

    return vla_train_dataloader, vlm_train_dataloader

def setup_optimizer_and_scheduler(
    model, cfg
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and learning rate scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    
    
    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    
    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps
    )
    
    # TODO mv to trainer
    # # prepare all components
    # (model, optimizer, vla_train_dataloader, vlm_train_dataloader) = accelerator.prepare(
    #     model, optimizer, vla_train_dataloader, vlm_train_dataloader
    # )
    
    return optimizer, lr_scheduler

class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vlm_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        
        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        
    def prepare_training(self):
        

        # load pretrained weights
        if (hasattr(self.config.trainer, 'pretrained_checkpoint') and self.config.trainer.pretrained_checkpoint):
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = self.config.trainer.reload_modules if hasattr(self.config.trainer, 'reload_modules') else None
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
        
        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules) # TODO think about self.config is global parameter or relative parameter?

        #  print trainable parameters of model --> TODO he should be the last one to summarize the check, consider centralized management
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader, self.vlm_train_dataloader = self.setup_distributed_training(
            self.accelerator, # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.vlm_train_dataloader
        )


        self._init_wandb()
        self._init_checkpointing()
    

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )
    
    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )
    
    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # resume training state
        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)
    
    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # TODO: resume training steps and other states
    
    def _save_checkpoint(self):
        """save current training state"""

        if accelerator.is_main_process:
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            
            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
                # TODO: add other training states to save
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if self.accelerator.is_main_process:
                # calculate gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                metrics["grad_norm"] = total_norm ** 0.5
                
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                
                # add epoch info
                metrics["epoch"] = self.completed_steps // len(self.vla_train_dataloader)
                
                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")
    
    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vlm_iter = iter(self.vlm_train_dataloader)
    
    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)
        
        try:
            batch_vlm = next(self.vlm_iter) # TODO first and last loop should be the function of dataset, here is considering that many datasets don't have this function
        except StopIteration:
            self.vlm_iter = iter(self.vlm_train_dataloader)
            batch_vlm = next(self.vlm_iter)
        
        return batch_vla, batch_vlm
    
    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()
        
        # prepare data iterators
        self._create_data_iterators()
        
        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        
        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            batch_vla, batch_vlm = self._get_next_batch()
            
            # execute training step
            step_metrics = self._train_step(batch_vla, batch_vlm)
            
            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            # record metrics
            self._log_metrics(step_metrics)
            
            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

                # TODO add eval logic @MichaelYu781
                dist.barrier()  # ensure all processes have saved
            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break
        
        # training end processing
        self._finalize_training()
    
    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f" Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

        # TODO here should print all key information in training: model size, freeze, lr group and so on.
    
    def _train_step(self, batch_vla, batch_vlm):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                action_loss, action_vlm_loss = self.model.forward(batch_vla)
                total_loss = action_loss + action_vlm_loss
            
            # VLA backward propagation
            self.accelerator.backward(total_loss)
            
            # VLM task forward propagation
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vlm_output = self.model.qwen_vl_interface(**batch_vlm)
                vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
            
            # VLM backward propagation
            self.accelerator.backward(vlm_loss)
            
            # gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.trainer.gradient_clipping
                )
            
            # optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
        
        return {
            "action_dit_loss": action_loss.item(),
            "action_vlm_loss": action_vlm_loss.item(),
            "vlm_loss": vlm_loss.item(),
        }
    
    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        
        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()
        
        self.accelerator.wait_for_everyone()

from InternVLA.training.trainer_utils.metrics import build_param_lr_groups
def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader, vlm_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    
    # create trainer
    # Run VLA Training
    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator
    )
    
    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
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
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    # TODO consider whether to merge trainer? --> users initially thought it was better to integrate more
    main(cfg)
