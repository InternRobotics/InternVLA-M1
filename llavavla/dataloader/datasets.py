"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
from transformers.models. qwen2_5_vl import Qwen2_5_VLProcessor
from llavavla.dataloader.promt_builder import QwenVLPromptHelper

@dataclass
class RLDSBatchTransform: #
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img.shape in rlds_batch = 224,224, 3 = h,w,c
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]) # B 要被去掉？
        lang = rlds_batch["task"]["language_instruction"].decode().lower() + "🔍" #cognition token
        # <PIL.Image.Image image mode=RGB size=224x224 at 0x7EFFCBD42530>
        # Construct Chat-based Prompt #@Jinhui 其实挺好的， 但是不用它来维护 system prompt, 因为Qwen 有他自己的 system prompt
        # prompt_builder = self.prompt_builder_fn("openvla") # 这个应该内聚到 Main model 里面
        # 这里应该用单例的，因为要保持全文统一 TODO @Jinhui

        # Add future actions to batch  # 好像action achunk 不在这？
        # if rlds_batch["action"].shape[0] > 1:
        #     action = torch.tensor(action, dtype=torch.float32)
        #     action_mask = None
        #     # if "action_mask" in rlds_batch: # 好像action achunk 不在这？
        #         action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        #@Jinhui TODO add new keys for Qwen # Jinhui 你应该涉及为一个 inputs 的参数，保持灵活传参数
        # 不要在这里做任何数据处理
  
        return dict(action=action,image=img,lang=lang, dataset_name=dataset_name)



@dataclass
class RLDSBatchQwenTransform: # @Jinhui TODO 这里要实现一个和模型无关的 Transform
    action_tokenizer: ActionTokenizer
    # base_tokenizer: PreTrainedTokenizerBase
    # image_transform: ImageTransform #qwen 是合并到了 @Jinhui TODO mv them
    qwen_VLProcessor: Qwen2_5_VLProcessor
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        
        # For future action predictions
        if rlds_batch["action"].shape[0] > 1:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        else:
            dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # img.shape in rlds_batch = 224,224, 3 = h,w,c
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]) # B 要被去掉？
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # <PIL.Image.Image image mode=RGB size=224x224 at 0x7EFFCBD42530>
        # Construct Chat-based Prompt #@Jinhui 其实挺好的， 但是不用它来维护 system prompt, 因为Qwen 有他自己的 system prompt
        # prompt_builder = self.prompt_builder_fn("openvla") # 这个应该内聚到 Main model 里面
        # 这里应该用单例的，因为要保持全文统一 TODO @Jinhui
        self.promptHelper = QwenVLPromptHelper(processor=self.qwen_VLProcessor, system_prompt="You are a helpful assistant")
        # If action tokenizer is not used, we don't add the action to the chat answer
        if self.action_tokenizer is None: # 之后考虑是否有更好的方式，其实这个是和模型强绑定的
            conversation = self.promptHelper.build_conversation(instruction=lang, image = [img], answer=None)
            
        else:
            # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
            conversation = self.promptHelper.build_conversation(instruction=lang, image = [img], answer= self.action_tokenizer(action))
            
        
        # TODO emergency check for speedup
        # minin version of QwenPromptBuilder --> @Jinhui TODO 后续可以实现到 QwenPromptBuilder 中进行对话管理
        # 拿到 对话的 text 文本 
        # prompt_text = self.qwen_VLProcessor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # # Tokenize (w/ `base_tokenizer`)
        # inputs = self.qwen_VLProcessor(text=[prompt_text], images=[img], padding=True, return_tensors="pt")
        inputs, prompt_text = self.promptHelper.build_multimodal_inputs(
        conversation, img, return_prompt_text=True)
        # dict_keys(['pixel_values', 'image_grid_thw']) # (256, 1176) # (1, 3) --> 符合 Qwen 的要求 N_patch, C*patch_w*patch_h
        input_ids = inputs.input_ids[0]
        labels = inputs.input_ids.clone()[0]
        attention_mask = inputs.attention_mask[0]

        image_grid_thw = inputs.image_grid_thw
        pixel_values = inputs.pixel_values # value in patch size

        # Add future actions to batch
        if rlds_batch["action"].shape[0] > 1:
            action = torch.tensor(action, dtype=torch.float32)
            action_mask = None
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        if self.action_tokenizer is None:
            labels[: -1] = IGNORE_INDEX
        else:
            # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            labels[: -(len(action) + 1)] = IGNORE_INDEX

        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        #@Jinhui TODO add new keys for Qwen # Jinhui 你应该涉及为一个 inputs 的参数，保持灵活传参数

        qwen_inputs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            
        }
        return dict(pixel_values=qwen_inputs, input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels, dataset_name=dataset_name, actions=action, action_masks=action_mask)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchQwenTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        future_action_window_size: int = 0,
        past_action_window_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
        load_all_data_for_training: bool = True,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=past_action_window_size + 1,                                    # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                                        # Skip trajectories without language labels
                #goal_relabeling_strategy="uniform",                                        # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            load_all_data_for_training=load_all_data_for_training,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch) # 这个感觉上是个很不好的实现

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            load_all_data_for_training=rlds_config["load_all_data_for_training"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
