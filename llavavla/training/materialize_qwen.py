"""
materialize.py

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
"""

from typing import Callable, Optional, Union

import torch

from prismatic.models.vlms import PrismaticVLM
from llavavla.model.vla import CogACT, CogACT_Qwen
from llavavla.training.strategies import FSDPStrategy_QWen, TrainingStrategy_Qwen
# Registry =>> Maps ID --> {cls(), kwargs} :: supports FSDP for now, but DDP handler is also implemented!
# 这个逻辑太绕了，不应该构建这个复杂逻辑，看看auto 能够自己解决
TRAIN_STRATEGIES = {
    "fsdp-shard-grad-op": {"cls": FSDPStrategy_QWen, "kwargs": {"sharding_strategy": "shard-grad-op"}},
    "fsdp-full-shard": {"cls": FSDPStrategy_QWen, "kwargs": {"sharding_strategy": "full-shard"}},
}



from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from llavavla.dataloader import EpisodicRLDSDataset, RLDSBatchQwenTransform, RLDSDataset

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers import Qwen2_5_VLProcessor

def get_train_strategy(
    train_strategy: str,
    vlm: Union[PrismaticVLM, CogACT_Qwen],
    device_id: int,
    stage: str,
    epochs: int,
    max_steps: Optional[int],
    global_batch_size: int,
    per_device_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    enable_gradient_checkpointing: bool = True,
    enable_mixed_precision_training: bool = True,
    reduce_in_full_precision: bool = False,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> TrainingStrategy_Qwen:
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]
        strategy = strategy_cfg["cls"](
            vlm=vlm,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            **strategy_cfg["kwargs"],
        )
        return strategy
    else:
        raise ValueError(f"Train Strategy {train_strategy} is not supported!")


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    vlp_processor: Qwen2_5_VLProcessor,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,         # Concatenated `past_action_window_size-1' actions and the current action for the input
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    base_action_tokenizer: PreTrainedTokenizerBase = None
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    if base_action_tokenizer is None:
        action_tokenizer = None
    else:
        action_tokenizer = ActionTokenizer(base_action_tokenizer)
    # action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchQwenTransform(
        action_tokenizer, vlp_processor, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        future_action_window_size=future_action_window_size,
        past_action_window_size=past_action_window_size,
        image_aug=image_aug,
        load_all_data_for_training=load_all_data_for_training,
    )

    return dataset, action_tokenizer, collator