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

# @Jinhui TODO 这个文件夹不应该存在， 应该是每个文件直接链接， 不要高层级结构

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from llavavla.dataloader import EpisodicRLDSDataset, RLDSBatchQwenTransform, RLDSDataset,RLDSBatchTransform

from pathlib import Path
from typing import Tuple, Type
from typing import List, Dict, Any, Callable, Optional
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

def get_train_align_strategy(
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


# NORA中有很多 fask tokenizor 的内容可以借鉴


def map_fast_token_to_vlm_action(tokens: List[str]) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])

from typing import List, Dict, Any, Callable, Optional
from transformers import AutoProcessor, PreTrainedTokenizerBase, Qwen2_5_VLForConditionalGeneration
def process_example(example: Dict[str, Any], fast_tokenizer: AutoProcessor) -> Dict[str, Any]:
    """Processes a single example from the dataset."""
    pixel_values = example['image']
    action = example['action']
    lang = example['lang']
    if "action" in example:
        fast_tokens = fast_tokenizer(action)
        vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])
    # @Jinhui TODO 这个应该是要和 main model 封装在一起的
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pixel_values},
                {"type": "text", "text": lang},
            ],
        },
        # { #@Jinhui TODO 这里是值得study 的点， 是否加入 robot message
        #     "role": "assistant",
        #     "content": [
        #         {"type": "text", "text": vlm_action},
        #     ],
        # },
    ]
    return messages

from qwen_vl_utils import process_vision_info

def collate_fn(examples,processor,fast_tokenizer): # @Jinhui TODO 要想清楚这个应该放到那个位置，他应该是和模型绑定的内容, 包括各种mask 策略， 其实是和每个模型相关的， 而且都是逻辑代码，没必要复用
        messages = [process_example(example,fast_tokenizer) for example in examples]
        actions = [example["action"] for example in examples]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        action_token_min = 151665
        action_token_max = 153712
        labels = batch_input['input_ids'].clone()
        # For each sequence in the batch, find the first occurrence of an action token.
        
        # @Jinhui TODO 看是否要处理这个，这个的意思是这处理 answer 上的loss
        # for i in range(labels.size(0)): # TODO 这里是先要mask vlm message， 不要参与 VLA training
        #     seq = labels[i]
        #     # Create a mask for tokens within the action token range.
        #     mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
        #     nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
        #     if nonzero_indices.numel() > 0:
        #         first_action_index = nonzero_indices[0].item()
        #         # Mask out all tokens before the first action token.
        #         seq[:first_action_index] = -100

        #     else:
        #         # If no action token is found, mask the entire sequence.
        #         seq[:] = -100
        
        labels[labels == processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
        batch_input['labels'] = labels # 确认确实是又对齐的
        batch_input["actions"] = torch.stack([torch.tensor(a) for a in actions], dim=0)  # numpy -> tensor shape: (B, 16, 7)
        return batch_input

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    vlp_processor: Qwen2_5_VLProcessor,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "left", # 很有可能是这里导致了 bug
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
    batch_transform = RLDSBatchTransform( # TODO 不能和数据集耦合，应该实现高内聚
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

    return dataset, action_tokenizer, collate_fn

if __name__ == "__main__":
    pass
    #@Jinhui TODO 全部 模块文件必须能够独立 执行测试单元
