

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


from accelerate.utils import (
    MODEL_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_PATTERN_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_PATTERN_NAME,
    AORecipeKwargs,
    AutocastKwargs,
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    DynamoBackend,
    FP8RecipeKwargs,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    MSAMPRecipeKwargs,
    PrecisionType,
    ProfileKwargs,
    ProjectConfiguration,
    RNGType,
    TERecipeKwargs,
    TorchDynamoPlugin,
    TorchTensorParallelPlugin,
    apply_fp8_autowrap,
    check_os_kernel,
    clean_state_dict_for_safetensors,
    compare_versions,
    convert_model,
    convert_model_to_fp8_ao,
    convert_outputs_to_fp32,
    ensure_weights_retied,
    extract_model_from_parallel,
    gather,
    gather_object,
    get_grad_scaler,
    get_mixed_precision_context_manager,
    get_pretty_name,
    has_offloaded_params,
    is_bf16_available,
    is_bitsandbytes_multi_backend_available,
    is_deepspeed_available,
    is_ipex_available,
    is_lomo_available,
    is_megatron_lm_available,
    is_mlu_available,
    is_msamp_available,
    is_musa_available,
    is_npu_available,
    is_torch_version,
    is_torch_xla_available,
    is_transformer_engine_available,
    is_xpu_available,
    load_fsdp_model,
    load_fsdp_optimizer,
    pad_across_processes,
    parse_choice_from_env,
    recursively_apply,
    reduce,
    release_memory,
    save,
    save_fsdp_model,
    save_fsdp_optimizer,
    wait_for_everyone,
)
import glob

def safe_save_checkpoint(accelerator, output_dir, step, total_loss, save_interval):
    """安全保存检查点的主进程异步方法"""
    # 0. 确保所有进程完成当前训练步骤
    accelerator.wait_for_everyone()
    
    # 1. 只在主进程执行保存操作
    if accelerator.is_main_process:
        # 创建临时本地目录避免共享存储竞争
        temp_dir = f"/tmp/checkpoint_{step}_{os.getpid()}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 2. 异步保存到本地临时目录
        def async_save():
            # 保存加速器状态
            accelerator.save_state(
                output_dir=temp_dir,
                safe_serialization=True
            )
            
            # 保存模型特定状态（DeepSpeed需要单独处理）
            model = accelerator.unwrap_model(accelerator._models[0])
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                model.save_checkpoint(
                    save_dir=temp_dir,
                    tag=f"step_{step}",
                    client_state={"step": step}
                )
            
            # 移动到最终共享存储位置
            final_dir = os.path.join(output_dir, "checkpoints", f"steps_{step}")
            shutil.copytree(temp_dir, final_dir, dirs_exist_ok=True)
            
            # 记录元数据
            summary_data = {
                "steps": step,
                "train_loss": total_loss / save_interval
            }
            with open(os.path.join(output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            
            # 清理临时文件
            shutil.rmtree(temp_dir)

        
        # 在独立线程中执行保存
        import threading
        save_thread = threading.Thread(target=async_save)
        save_thread.start()
    
    # 3. 非主进程立即返回
    accelerator.wait_for_everyone()
    return True

# 在训练循环中使用
 

def load_safe_checkpoint(accelerator, output_dir, resume_step=None):
    """从安全保存的检查点恢复训练"""
    # 1. 确定恢复点
    if resume_step is None:
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoints", "steps_*"))
        if not checkpoint_dirs:
            return 0
        resume_step = max([int(d.split("_")[-1]) for d in checkpoint_dirs])
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"steps_{resume_step}")
    
    # 2. 加载模型权重
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(accelerator._models[0])
        
        # 加载16位模型权重
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            # 创建临时加载器
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            state_dict = load_state_dict_from_zero_checkpoint(model, checkpoint_dir)
            model.load_state_dict(state_dict)
        
        # 加载优化器状态
        optimizer_path = os.path.join(checkpoint_dir, "optimizer_state.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path)
            model.optimizer.load_state_dict(optimizer_state)
    
    # 3. 加载元数据
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
            global_step = metadata.get("global_step", resume_step)
    else:
        global_step = resume_step
    
    # 4. 广播恢复步数到所有进程
    global_step_tensor = torch.tensor([global_step], device=accelerator.device)
    torch.distributed.broadcast(global_step_tensor, src=0)
    global_step = global_step_tensor.item()
    
    accelerator.print(f"Resumed training from step {global_step}")
    return global_step