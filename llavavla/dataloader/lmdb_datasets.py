"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import debugpy 
debugpy.listen(("0.0.0.0", 10092))  # 监听端口 
print("Waiting for debugger to attach...")
debugpy.wait_for_client()  # 等待 VS Code 附加

import os, json, pickle, bisect, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import lmdb
from itertools import accumulate
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


@dataclass
class LMDBBatchTransform:
    def __call__(self, lmdb_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a LMDB batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = lmdb_batch["dataset_name"], lmdb_batch["action"][0]

        # For future action predictions
        if lmdb_batch["action"].shape[0] > 1:
            dataset_name, action = lmdb_batch["dataset_name"], lmdb_batch["action"]
        else:
            dataset_name, action = lmdb_batch["dataset_name"], lmdb_batch["action"][0]

        # img.shape in lmdb_batch = 480,640, 3 = h,w,c, RGB
        img = Image.fromarray(lmdb_batch["observation"]["image_primary"][0]) # B 要被去掉？
        
        # img = torch.tensor(img, dtype=torch.float32)  # TODO Check 这里要看是否执行了数据增强 h,w,c
        lang = lmdb_batch["task"]["language_instruction"].decode().lower() #+ "🔍" #cognition token

        return dict(action=action,image=[img],lang=lang, dataset_name=dataset_name)


class LMDBDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        dataset_info_name: str = None,
        obs_type: str = "obs_camera",
        action_type: str = "abs_qpos",
        window_size: int = 16,
        image_aug: bool = False,
        batch_transform: LMDBBatchTransform = None,
        **kwargs: Any,
    ) -> None:
        """Dataset wrapper for LMDB format data."""
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset_info_name = dataset_info_name if dataset_info_name is not None else dataset_name
        self.dataset_path = f'{data_dir}/{dataset_name}/render'
        self.obs_type = obs_type
        self.action_type = action_type
        self.window_size = window_size
        self.image_aug = image_aug

        # Load dataset info
        logger.info(f"loading dataset at {data_dir}/{dataset_name}")
        assert os.path.exists(f"{data_dir}/data_info/{self.dataset_info_name}.json")
        with open(f"{data_dir}/data_info/{self.dataset_info_name}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        # Load action statistics
        meta_info = pickle.load(open(f"{data_dir}/data_info/{self.dataset_info_name}.pkl", "rb"))
        try:
            if self.action_type == "abs_qpos":
                self.arm_action_mean = np.array(meta_info["abs_qpos_action_mean"])
                self.arm_action_std = np.array(meta_info["abs_qpos_action_std"])
                self.arm_action_min = np.array(meta_info["abs_qpos_action_min"])
                self.arm_action_max = np.array(meta_info["abs_qpos_action_max"])
            elif self.action_type == "delta_qpos":
                self.arm_action_mean = np.array(meta_info["delta_qpos_action_mean"])
                self.arm_action_std = np.array(meta_info["delta_qpos_action_std"])
                self.arm_action_min = np.array(meta_info["delta_qpos_action_min"])
                self.arm_action_max = np.array(meta_info["delta_qpos_action_max"])
            elif self.action_type == "abs_ee_pose":
                self.arm_action_mean = np.array(meta_info["abs_eepose_action_mean"])
                self.arm_action_std = np.array(meta_info["abs_eepose_action_std"])
                self.arm_action_min = np.array(meta_info["abs_eepose_action_min"])
                self.arm_action_max = np.array(meta_info["abs_eepose_action_max"])
            elif self.action_type == "delta_ee_pose":
                self.arm_action_mean = np.array(meta_info["delta_eepose_action_mean"])
                self.arm_action_std = np.array(meta_info["delta_eepose_action_std"])
                self.arm_action_min = np.array(meta_info["delta_eepose_action_min"])
                self.arm_action_max = np.array(meta_info["delta_eepose_action_max"])
            else:
                raise NotImplementedError(f"Action type {self.action_type} not supported")
        except Exception as e:
            logger.error(f"Error loading action statistics: {e}")
            raise e

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sequence from dataset at given index."""
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
            
        episode_name = self.episode_list[episode_id]

        # Open LMDB environment
        lmdb_env = lmdb.open(
            f"{self.dataset_path}/{episode_name}/lmdb",
            readonly=True,
            lock=False,
            readahead=True,
            meminit=True
        )

        # Load meta info
        meta_info = pickle.load(open(f"{self.dataset_path}/{episode_name}/meta_info.pkl", "rb"))

        # Get data keys
        if self.action_type == "abs_qpos":
            arm_index = meta_info["keys"]["scalar_data"].index(b'arm_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
        elif self.action_type == "delta_qpos":
            arm_index = meta_info["keys"]["scalar_data"].index(b'delta_arm_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
        elif self.action_type == "abs_ee_pose":
            arm_index = meta_info["keys"]["scalar_data"].index(b'ee_pose_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
        elif self.action_type == "delta_ee_pose":
            arm_index = meta_info["keys"]["scalar_data"].index(b'delta_ee_pose')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
        else:
            raise NotImplementedError(f"Action type {self.action_type} not supported")
        
        gripper_index = meta_info["keys"]["scalar_data"].index(b'gripper_close') 
        gripper_key = meta_info["keys"]["scalar_data"][gripper_index]
        qpos_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
        qpos_key = meta_info["keys"]["scalar_data"][qpos_index]
        
        primary_index = meta_info["keys"][f"observation/{self.obs_type}/color_image"]
        wrist_index = meta_info["keys"]["observation/realsense/color_image"]

        language_instruction = meta_info["language_instruction"]

        # Load sequence data
        sequence = []
        with lmdb_env.begin(write=False) as txn:

            arm_action = pickle.loads(txn.get(arm_key))
            gripper_action = pickle.loads(txn.get(gripper_key))

            primary_data = pickle.loads(txn.get(primary_index[start_id]))
            primary_data = cv2.imdecode(np.frombuffer(primary_data, np.uint8), cv2.IMREAD_COLOR)
            # convert to PIL Image
            primary_data = Image.fromarray(primary_data)

        lmdb_env.close()
        # action chuck: window_size
        action_length = len(arm_action)
        if action_length >= self.window_size + start_id:
            action = arm_action[start_id:start_id + self.window_size]
            gripper = gripper_action[start_id:start_id + self.window_size]
        else:
            action = arm_action[start_id:action_length] + np.zeros(self.window_size - (action_length - start_id))
            gripper = gripper_action[start_id:action_length] + np.ones(self.window_size - (action_length - start_id))

        collected_action = []
        for a, g in zip(action, gripper):
            collected_action.append(self.load_robot_action(np.concatenate([a[0], a[1]]), g))

        
        return dict(action=collected_action,image=[primary_data],lang=language_instruction, dataset_name=self.dataset_name)

    def __iter__(self):
        """Iterate through the dataset sequentially."""
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    # TODO 这个函数 需要重构， 不能和数据集耦合
    def load_robot_action(self, arm_action, gripper_action):
        if self.action_type in ["abs_qpos", "delta_qpos", "abs_ee_pose", "delta_ee_pose"]:
            actions = np.zeros(8)
            actions[:7] = 2 * (arm_action[:7] - self.arm_action_min[:7]) / (self.arm_action_max[:7] - self.arm_action_min[:7] + 1e-8) - 1
            actions[-1] = gripper_action
            assert np.all(actions <= 1) and np.all(actions >= -1)
            return actions
        else:
            raise NotImplementedError(f"Action type {self.action_type} not supported")


class EpisodicLMDBDataset(LMDBDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    # TODO 实现
    pass



# class DummyDataset(Dataset):
#     def __init__(
#         self,
#         action_tokenizer: ActionTokenizer,
#         base_tokenizer: PreTrainedTokenizerBase,
#         image_transform: ImageTransform,
#         prompt_builder_fn: Type[PromptBuilder],
#     ) -> None:
#         self.action_tokenizer = action_tokenizer
#         self.base_tokenizer = base_tokenizer
#         self.image_transform = image_transform
#         self.prompt_builder_fn = prompt_builder_fn

#         # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
#         # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
#         self.dataset_statistics = {
#             "dummy_dataset": {
#                 "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
#             }
#         }

#     def __len__(self):
#         # TODO =>> Replace with number of elements in your dataset!
#         return 10000

#     def __getitem__(self, idx):
#         # TODO =>> Load image, action and instruction from disk -- we use dummy values
#         image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
#         action = np.asarray(np.random.rand(7), dtype=np.float32)
#         instruction = "do something spectacular"

#         # Add instruction to VLA prompt
#         prompt_builder = self.prompt_builder_fn("openvla")
#         conversation = [
#             {"from": "human", "value": f"What action should the robot take to {instruction}?"},
#             {"from": "gpt", "value": self.action_tokenizer(action)},
#         ]
#         for turn in conversation:
#             prompt_builder.add_turn(turn["from"], turn["value"])

#         # Tokenize (w/ `base_tokenizer`)
#         input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
#         labels = list(input_ids)

#         # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
#         #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
#         input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
#         pixel_values = self.image_transform(image)

#         # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
#         labels[: -(len(action) + 1)] = IGNORE_INDEX

#         return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


def get_dummy_dataset(dataconfig: dict):

    pass

from typing import List, Dict, Any, Callable, Optional
from torch.utils.data import DataLoader


def get_vla_dataset(
    data_root_dir: Path,
    dataset_name: str,
    obs_type: str = "obs_camera",
    action_type: str = "abs_qpos",
    window_size: int = 16,
    image_aug: bool = False,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,         # Concatenated `past_action_window_size-1' actions and the current action for the input
    load_all_data_for_training: bool = True,  # Load all data for training, or only a subset
    **kwargs: Any,  # Additional arguments for RLDSBatchTransform
) -> Tuple[Dataset]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    batch_transform = LMDBBatchTransform( # TODO 不能和数据集耦合，应该实现高内聚
    )
    
    # Build RLDS Iterable Dataset
    cls = LMDBDataset if not episodic else EpisodicLMDBDataset
    dataset = cls(
        dataset_name,
        data_root_dir,
        dataset_name,
        obs_type=obs_type,
        action_type=action_type,
        window_size=window_size,
        image_aug=image_aug,
        batch_transform=batch_transform,
    )

    return dataset

from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor

import torch.distributed as dist
from types import SimpleNamespace

def collate_fn(batch):
    # batch: list of items, 假设每个 item 是 (PIL.Image, other_info)

    pass # TODO 如果要动态 input， 就不能用 default_collate
    # dist.barrier()  # 确保所有进程都在同一时间点

    return batch # 我们宁愿返回一个 list_of_dict for 动态的 inputs

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

if __name__ == "__main__":
    pass
    #@Jinhui TODO 全部 模块文件必须能够独立 执行测试单元

    # test  get_vla_dataset
    cfg = {
        'data_root_dir': '/mnt/petrelfs/share/efm_p/wangfangjing/datasets',
        'dataset_name': 'Banana/banana_plate_4035_none-wot-wow-woh-0529',
        'obs_type': 'obs_camera',
        'action_type': 'abs_ee_pose',
        'window_size': 16,
        'vla': {
            'per_device_batch_size': 1,
        }
    }
    cfg = dict_to_namespace(cfg)

    vla_dataset = get_vla_dataset(
        data_root_dir=cfg.data_root_dir,
        dataset_name=cfg.dataset_name,
        obs_type=cfg.obs_type,
        action_type=cfg.action_type,
        window_size=cfg.window_size,
    )
    
    # 方法2: 使用迭代器
    dataset_iter = iter(vla_dataset)
    while True:
        try:
            batch_samples = next(dataset_iter)
            print(batch_samples['action'])
        except StopIteration:
            break
