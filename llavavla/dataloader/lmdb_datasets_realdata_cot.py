"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import re
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
from llavavla.dataloader.lmdb.data_utils import get_lmdb_dataset_statistics, save_dataset_statistics

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

# temp map dict:
pick_object_map = {
    "奶牛玩具": "cow toy",
    "河马玩具": "hippo toy",
    "大象玩具": "elephant toy",
    "奥利奥": "Oreo",
    "茄子": "eggplant",
    "葡萄": "grape",
    "白色运动饮料瓶子": "white sports drink bottle",
    "橙色瓶子": "orange bottle",
    "橙子": "orange",
    "小绿色薯片": "small green chips",
    "桃子": "peach",
    "东方树叶饮料": "Dongfangshuye tea drink",
    "菠萝": "pineapple",
    "照相机": "camera",
    "香蕉": "banana",
    "牛奶": "milk",
    "蓝白手套": "blue-white glove",
    "黄瓜": "cucumber",
    "胶水瓶子": "glue bottle",
    "苹果": "apple",
}

place_objects_map = {
    "棕色台子": "brown table",
    "白色果篮": "white fruit basket",
    "棕色方形果篮": "brown square fruit basket",
    "紫色果盘": "purple fruit plate",
    "咖啡色箱子": "brown box"
}

def get_episode_cot(episode_name, frame_index, obs= "primary_images", dir="/mnt/petrelfs/share/efm_p/yujunqiu/grounding/filtered_results"):

    json_path  = os.path.join(dir, episode_name, "filtered_bboxes.json")
    
    # 这里数据构建到的问题太多
    pick_name = None
    place_name = None
    pick_bbox = None
    place_bbox = None
    #load json
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            CoT_episodic = json.load(f)
            primary_view_annotaions = CoT_episodic.get(obs, {})


        object_list = list(primary_view_annotaions.keys())
        for obj in object_list:
            if obj in pick_object_map.values():
                pick_name = obj
            if obj in place_objects_map.values():
                place_name = obj
        if pick_name is not None:
            pick_bbox = primary_view_annotaions[pick_name].get(str(frame_index), None)
        if place_name is not None:
            place_bbox = primary_view_annotaions[place_name].get(str(frame_index), None)
    
    return pick_name, pick_bbox, place_name, place_bbox # 返回的东西可能是 none

def crop_bbox(bbox, crop_coords):
    """
    Adjust the bounding box coordinates for cropping.

    :param bbox: Original bounding box [x_min, y_min, x_max, y_max].
    :param crop_coords: Crop coordinates [crop_y_start, crop_y_end, crop_x_start, crop_x_end].
    :return: Adjusted bounding box [x_min, y_min, x_max, y_max].
    """
    if bbox is None:
        return None

    crop_y_start, crop_y_end, crop_x_start, crop_x_end = crop_coords

    # Adjust bbox for cropping
    x_min, y_min, x_max, y_max = bbox
    x_min = max(x_min - crop_x_start, 0)
    y_min = max(y_min - crop_y_start, 0)
    x_max = min(x_max - crop_x_start, crop_x_end - crop_x_start)
    y_max = min(y_max - crop_y_start, crop_y_end - crop_y_start)

    return [x_min, y_min, x_max, y_max]


def resize_bbox(bbox, original_size, target_size):
    """
    Adjust the bounding box coordinates for resizing.

    :param bbox: Bounding box adjusted for cropping [x_min, y_min, x_max, y_max].
    :param original_size: Original size of the cropped image (height, width).
    :param target_size: Target size for resizing (height, width).
    :return: Adjusted bounding box [x_min, y_min, x_max, y_max].
    """
    if bbox is None:
        return None

    orig_h, orig_w = original_size
    target_h, target_w = target_size

    # Scale bbox for resizing
    x_min, y_min, x_max, y_max = bbox
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    x_min = int(x_min * scale_x)
    y_min = int(y_min * scale_y)
    x_max = int(x_max * scale_x)
    y_max = int(y_max * scale_y)

    return [x_min, y_min, x_max, y_max]

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
        normalization_type: str = "q01_q99",
        save_statistics_dir: str = None,
        shuffle: bool = True,
        seed: int = 42,
        crop_obs_camera: bool = False,
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
        self.normalization_type = normalization_type
        self.crop_obs_camera = crop_obs_camera

        # Load dataset info
        logger.info(f"loading dataset at {data_dir}/{dataset_name}")
        assert os.path.exists(f"{data_dir}/data_info/{self.dataset_info_name}.json")
        with open(f"{data_dir}/data_info/{self.dataset_info_name}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)

        # Get dataset statistics (with caching)
        self.dataset_statistics = get_lmdb_dataset_statistics(
            dataset_name=self.dataset_name,
            data_dir=self.data_dir,
            action_type=self.action_type,
            dataset_info_name=self.dataset_info_name,
            save_dir=save_statistics_dir,
        )

        # Load action statistics
        meta_info = pickle.load(open(f"{data_dir}/data_info/{self.dataset_info_name}.pkl", "rb"))
        try:
            if self.action_type == "abs_qpos":
                self.arm_action_mean = np.array(meta_info["abs_arm_action_mean"])
                self.arm_action_std = np.array(meta_info["abs_arm_action_std"])
                self.arm_action_min = np.array(meta_info["abs_arm_action_min"])
                self.arm_action_max = np.array(meta_info["abs_arm_action_max"])
                # Load q01 and q99 if available
                if "abs_arm_action_q01" in meta_info:
                    self.arm_action_q01 = np.array(meta_info["abs_arm_action_q01"])
                    self.arm_action_q99 = np.array(meta_info["abs_arm_action_q99"])
            elif self.action_type == "delta_qpos":
                self.arm_action_mean = np.array(meta_info["delta_arm_action_mean"])
                self.arm_action_std = np.array(meta_info["delta_arm_action_std"])
                self.arm_action_min = np.array(meta_info["delta_arm_action_min"])
                self.arm_action_max = np.array(meta_info["delta_arm_action_max"])
                # Load q01 and q99 if available
                if "delta_arm_action_q01" in meta_info:
                    self.arm_action_q01 = np.array(meta_info["delta_arm_action_q01"])
                    self.arm_action_q99 = np.array(meta_info["delta_arm_action_q99"])
            elif self.action_type == "abs_ee_pose":
                self.arm_action_mean = np.array(meta_info["abs_eepose_action_mean"])
                self.arm_action_std = np.array(meta_info["abs_eepose_action_std"])
                self.arm_action_min = np.array(meta_info["abs_eepose_action_min"])
                self.arm_action_max = np.array(meta_info["abs_eepose_action_max"])
                # Load q01 and q99 if available
                if "abs_eepose_action_q01" in meta_info:
                    self.arm_action_q01 = np.array(meta_info["abs_eepose_action_q01"])
                    self.arm_action_q99 = np.array(meta_info["abs_eepose_action_q99"])
            elif self.action_type == "delta_ee_pose":
                self.arm_action_mean = np.array(meta_info["delta_eepose_action_mean"])
                self.arm_action_std = np.array(meta_info["delta_eepose_action_std"])
                self.arm_action_min = np.array(meta_info["delta_eepose_action_min"])
                self.arm_action_max = np.array(meta_info["delta_eepose_action_max"])
                # Load q01 and q99 if available
                if "delta_eepose_action_q01" in meta_info:
                    self.arm_action_q01 = np.array(meta_info["delta_eepose_action_q01"])
                    self.arm_action_q99 = np.array(meta_info["delta_eepose_action_q99"])
            else:
                raise NotImplementedError(f"Action type {self.action_type} not supported")
        except Exception as e:
            logger.error(f"Error loading action statistics: {e}")
            raise e

        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        
        # Create shuffle mapping if needed
        self.shuffle = shuffle
        if shuffle:
            # Create a list of all valid indices
            all_indices = list(range(self.length))
            # Shuffle with fixed seed for reproducibility
            import random
            random.seed(seed)
            random.shuffle(all_indices)
            
            # Create mapping from shuffled index to original index
            self.shuffle_mapping = {shuffled_idx: original_idx for shuffled_idx, original_idx in enumerate(all_indices)}
        else:
            self.shuffle_mapping = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sequence from dataset at given index."""
        # Apply shuffle mapping if enabled
        if self.shuffle_mapping is not None:
            original_idx = self.shuffle_mapping[idx]
        else:
            original_idx = idx
            
        # Use original logic with the mapped index
        episode_id = bisect.bisect_right(self.accumulated_num_step, original_idx)
        if episode_id - 1 >= 0:
            start_id = original_idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = original_idx
            
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

        # Get data keys based on real data format
        if self.action_type == "abs_qpos":
            # Real data doesn't have abs_qpos action, fallback to qpos state
            state_key = b'observation/robot/qpos'
            arm_key = state_key  # Use state as action for abs_qpos
        elif self.action_type == "delta_qpos":
            # Real data doesn't have delta_qpos action, skip this type
            raise NotImplementedError(f"Action type {self.action_type} not supported in real data")
        elif self.action_type == "abs_ee_pose":
            arm_key = b'ee_pose_action'
            state_key = b'observation/robot/ee_pose_state'
        elif self.action_type == "delta_ee_pose":
            arm_key = b'delta_ee_pose_action'
            state_key = b'observation/robot/ee_pose_state'
        else:
            raise NotImplementedError(f"Action type {self.action_type} not supported")
        
        gripper_key = b'gripper_close'

        # Try to get image keys from meta_info, fallback to direct keys if not available
        try:
            primary_index = meta_info["keys"][f"observation/{self.obs_type}/color_image"]
            wrist_index = meta_info["keys"]["observation/realsense/color_image"]
        except KeyError:
            # If keys not in meta_info, use direct LMDB keys
            primary_index = None
            wrist_index = None

        language_instruction = meta_info.get("language_instruction", "Complete the task")

        # get pick and place obj TODO 应该在数据annotation的时候就提供
        match = re.match(r"Move the (.+?) to the top of the (.+?)\.", language_instruction)
        if match:
            pick_obj = match.group(1).strip()
            place_obj = match.group(2).strip()

        # Load sequence data
        sequence = []
        with lmdb_env.begin(write=False) as txn:
            arm_action = pickle.loads(txn.get(arm_key))
            gripper_action = pickle.loads(txn.get(gripper_key))

            # Handle image loading
            if primary_index is not None:
                primary_data = pickle.loads(txn.get(primary_index[start_id]))
                primary_data = cv2.imdecode(np.frombuffer(primary_data, np.uint8), cv2.IMREAD_COLOR)
            else:
                # Try to find image data with alternative keys
                try:
                    # Try common image key patterns
                    img_key = f"observation/{self.obs_type}/color_image_{start_id}".encode()
                    primary_data = pickle.loads(txn.get(img_key))
                    primary_data = cv2.imdecode(np.frombuffer(primary_data, np.uint8), cv2.IMREAD_COLOR)
                except:
                    # Create dummy image if no image data found
                    primary_data = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # get image grounding CoT
            pick_name, pick_bbox, place_name, place_bbox = get_episode_cot(episode_name=episode_name, frame_index=0) # @DEBUG 是按照frames 来？
            # @temp 因为annotation 中说数据不完善， 需要额外再处理
            pick_name  = pick_obj
            place_name = place_obj

            # Crop the image according to the red box region (adjust coordinates as needed)
            # Original size is 640x480, crop to the region of interest
            original_size = (480, 640)
            target_size = (224, 224)

            if primary_data.shape[:2] == (480, 640) and self.crop_obs_camera:  # height, width
                # Define crop coordinates based on the red box (adjust these values as needed)
                # Format: [y_start:y_end, x_start:x_end]
                crop_y_start = 50   # Top of red box
                crop_y_end = 480    # Bottom of red box  
                crop_x_start = 170   # Left of red box
                crop_x_end = 635    # Right of red box
                
                primary_data = primary_data[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

                # Adjust bbox for cropping and resizing
                crop_coords = [crop_y_start, crop_y_end, crop_x_start, crop_x_end]
                
                # Adjust bbox for cropping
                pick_bbox = crop_bbox(pick_bbox, crop_coords)
                place_bbox = crop_bbox(place_bbox, crop_coords)

                # Adjust bbox for resizing
                original_size = (crop_y_end - crop_y_start, crop_x_end - crop_x_start)

            # Convert to PIL Image
            primary_data = Image.fromarray(primary_data)
            primary_data = primary_data.resize(target_size)
            pick_bbox = resize_bbox(pick_bbox, original_size, target_size)
            place_bbox = resize_bbox(place_bbox, original_size, target_size)
            # @TODO pick_bbox can be none
            # @TODO 因为 bbox 缺失的情况太严重了， --> 是否只返第一 frame

            # 
            if pick_bbox is not None and place_bbox is not None: # 16%
                think_prompt = "Please identify where to pick the object and where to place it."
                language_instruction = f"{language_instruction} {think_prompt}"
                solution = f"Pick {pick_name} at {pick_bbox}, then place it on {place_name} at {place_bbox}."
            elif pick_bbox is not None: # 20%
                think_prompt = "Please identify where to pick the object."
                language_instruction = f"{language_instruction} {think_prompt}"
                solution = f"Pick {pick_name} at {pick_bbox}, then place it on {place_name}."

            # elif place_bbox is not None: # 28%
            #     think_prompt = "Please identify where to place the object."
            #     language_instruction = f"{language_instruction} {think_prompt}"
            #     solution = f"Pick {pick_name}, then place it on {place_name} at {place_bbox}."
            else: # TODO 这里其实应该是有prompt 但是不用输出的模式 --> 下一个版本再做高阶操作
                think_prompt = "Give the action directly."
                language_instruction = f"{language_instruction} {think_prompt}"
                solution = None # 36%
            #  
            
            # Handle wrist camera (optional)
            try:
                if wrist_index is not None:
                    wrist_data = pickle.loads(txn.get(wrist_index[start_id]))
                    wrist_data = cv2.imdecode(np.frombuffer(wrist_data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    # Try alternative wrist key
                    wrist_key = f"observation/realsense/color_image_{start_id}".encode()
                    wrist_data = pickle.loads(txn.get(wrist_key))
                    wrist_data = cv2.imdecode(np.frombuffer(wrist_data, np.uint8), cv2.IMREAD_COLOR)
                
                wrist_data = Image.fromarray(wrist_data)
                wrist_data = wrist_data.resize((224, 224))
                has_wrist = True
            except:
                # No wrist camera data available
                wrist_data = None
                has_wrist = False

        lmdb_env.close()
        
        # action chunk: window_size
        action_length = len(arm_action)
        if action_length >= self.window_size + start_id:
            action = arm_action[start_id:start_id + self.window_size]
            gripper = gripper_action[start_id:start_id + self.window_size]
        else:
            # Handle padding based on action type
            available_actions = arm_action[start_id:action_length]
            available_grippers = gripper_action[start_id:action_length]
            padding_length = self.window_size - (action_length - start_id)
            
            if self.action_type.startswith("delta"):
                # For delta actions, pad with zeros
                last_gripper = gripper_action[-1]
                if isinstance(arm_action, np.ndarray):
                    # Handle numpy array
                    zero_action = np.zeros((padding_length,) + arm_action.shape[1:], dtype=arm_action.dtype)
                    repeated_gripper = np.full((padding_length,), last_gripper, dtype=gripper_action.dtype)
                     
                    action = np.concatenate([available_actions, zero_action], axis=0)
                    gripper = np.concatenate([available_grippers, repeated_gripper], axis=0)
                else:
                    # Handle list of arrays/tuples
                    if isinstance(arm_action[0], (list, tuple)):
                        # Handle nested structure like [(pos, quat), ...]
                        zero_action = [type(arm_action[0])([np.zeros_like(arm_action[0][0]), np.zeros_like(arm_action[0][1])]) for _ in range(padding_length)]
                    else:
                        # Handle flat array structure
                        zero_action = [np.zeros_like(arm_action[0]) for _ in range(padding_length)]
                    repeated_gripper = [last_gripper] * padding_length
                    
                    action = available_actions + zero_action
                    gripper = available_grippers + repeated_gripper
            else:
                # For absolute actions, repeat the last action
                last_action = arm_action[-1]
                last_gripper = gripper_action[-1]
                
                if isinstance(arm_action, np.ndarray):
                    # Handle numpy array
                    repeated_action = np.tile(last_action[None, :], (padding_length, 1))
                    repeated_gripper = np.full((padding_length,), last_gripper, dtype=gripper_action.dtype)
                    
                    action = np.concatenate([available_actions, repeated_action], axis=0)
                    gripper = np.concatenate([available_grippers, repeated_gripper], axis=0)
                else:
                    # Handle list
                    action = available_actions + [last_action] * padding_length
                    gripper = available_grippers + [last_gripper] * padding_length

        collected_action = []
        for a, g in zip(action, gripper):
            collected_action.append(self.load_robot_action(a, g).astype(np.float16))

        # Return data with or without wrist camera
        if has_wrist: # @Fangjin不要定义内在转移逻辑
            return dict(action=collected_action, image=[primary_data, wrist_data], lang=language_instruction, 
                        solution=solution, dataset_name=self.dataset_name)
        # else:
        #     return dict(action=collected_action, image=[primary_data], lang=language_instruction, 
        #                 solution=solution, dataset_name=self.dataset_name)

    def __iter__(self):
        """Iterate through the dataset sequentially."""
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    # TODO 这个函数 需要重构， 不能和数据集耦合
    def load_robot_action(self, arm_action, gripper_action):
        if self.action_type in ["abs_ee_pose", "delta_ee_pose"]:
            # Handle ee_pose format: [(pos, quat), (pos, quat), ...]
            if isinstance(arm_action[0], (list, tuple)) and len(arm_action[0]) == 2:
                # arm_action is in format [(pos, quat), ...]
                positions = np.array([pos for pos, quat in arm_action])
                quaternions = np.array([quat for pos, quat in arm_action])
                arm_action_flat = np.concatenate([positions, quaternions], axis=1)
            else:
                # arm_action is already flattened
                arm_action_flat = np.array(arm_action)
            
            actions = np.zeros(7)
            
            # Apply normalization based on normalization_type
            if self.normalization_type == "q01_q99":
                # Use q01 and q99 for normalization
                arm_q01 = np.array(getattr(self, f"arm_action_q01", self.arm_action_min))
                arm_q99 = np.array(getattr(self, f"arm_action_q99", self.arm_action_max))
                actions[:6] = 2 * (arm_action_flat[:6] - arm_q01[:6]) / (arm_q99[:6] - arm_q01[:6] + 1e-8) - 1
                # 添加 clipping 确保在 [-1, 1] 范围内
                actions[:6] = np.clip(actions[:6], -1, 1)
            elif self.normalization_type == "min_max":
                # Use min/max for normalization
                actions[:6] = 2 * (arm_action_flat[:6] - self.arm_action_min[:6]) / (self.arm_action_max[:6] - self.arm_action_min[:6] + 1e-8) - 1
            elif self.normalization_type == "mean_std":
                # Use mean/std for normalization (z-score normalization)
                actions[:6] = (arm_action_flat[:6] - self.arm_action_mean[:6]) / (self.arm_action_std[:6] + 1e-8)
                # mean_std 归一化本身不保证 [-1,1] 范围，如果需要可以添加 clipping
                actions[:6] = np.clip(actions[:6], -1, 1)  # 限制在 ±3σ 范围内
            else:
                raise ValueError(f"Unknown normalization_type: {self.normalization_type}")
            
            # normalize gripper_action to 0 or 1
            actions[-1] = (gripper_action + 1) / 2
            assert np.all(actions <= 1) and np.all(actions >= -1)
            return actions
        elif self.action_type == "abs_qpos":
            # For abs_qpos, use state as action
            actions = np.zeros(8)
            
            # Apply normalization based on normalization_type
            if self.normalization_type == "q01_q99":
                # Use q01 and q99 for normalization
                arm_q01 = np.array(getattr(self, f"arm_action_q01", self.arm_action_min))
                arm_q99 = np.array(getattr(self, f"arm_action_q99", self.arm_action_max))
                actions[:7] = 2 * (arm_action[:7] - arm_q01[:7]) / (arm_q99[:7] - arm_q01[:7] + 1e-8) - 1
            elif self.normalization_type == "min_max":
                # Use min/max for normalization
                actions[:7] = 2 * (arm_action[:7] - self.arm_action_min[:7]) / (self.arm_action_max[:7] - self.arm_action_min[:7] + 1e-8) - 1
            elif self.normalization_type == "mean_std":
                # Use mean/std for normalization (z-score normalization)
                actions[:7] = (arm_action[:7] - self.arm_action_mean[:7]) / (self.arm_action_std[:7] + 1e-8)
            else:
                raise ValueError(f"Unknown normalization_type: {self.normalization_type}")
            
            actions[-1] = (gripper_action + 1) / 2
            assert np.all(actions <= 1) and np.all(actions >= -1)
            return actions
        else:
            raise NotImplementedError(f"Action type {self.action_type} not supported")

    def get_dataset_statistics(self) -> Dict:
        """Return dataset statistics in the same format as RLDS datasets."""
        return self.dataset_statistics

    def save_dataset_statistics(self, run_dir: Path) -> None:
        """Save dataset statistics to the specified directory."""
        save_dataset_statistics(self.dataset_statistics, run_dir)


class EpisodicLMDBDataset(LMDBDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    # TODO 实现
    pass

def get_dummy_dataset(dataconfig: dict):

    pass

from typing import List, Dict, Any, Callable, Optional
from torch.utils.data import DataLoader


def get_vla_dataset(
    data_root_dir: Path,
    data_mix: str,
    data_mix_info: str,
    obs_type: str = "obs_camera",
    action_type: str = "abs_qpos",
    window_size: int = 16,
    image_aug: bool = False,
    shuffle_buffer_size: int = 100_000,
    default_image_resolution: Tuple[int, int, int] = (3, 224, 224),
    train: bool = True,
    episodic: bool = False,
    future_action_window_size: int = 0,
    past_action_window_size: int = 1,
    load_all_data_for_training: bool = True,
    normalization_type: str = "q01_q99",
    save_statistics_dir: str = None,
    shuffle: bool = True,
    seed: int = 42,
    crop_obs_camera: bool = False,
    **kwargs: Any,
) -> Tuple[Dataset]:
    """Initialize LMDB Dataset and optionally save statistics."""

    batch_transform = LMDBBatchTransform()
    
    # Build LMDB Dataset
    cls = LMDBDataset if not episodic else EpisodicLMDBDataset
    dataset = cls(
        data_mix,
        data_root_dir,
        data_mix_info,
        obs_type=obs_type,
        action_type=action_type,
        window_size=window_size,
        image_aug=image_aug,
        batch_transform=batch_transform,
        normalization_type=normalization_type,
        save_statistics_dir=save_statistics_dir,
        shuffle=shuffle,
        seed=seed,
        crop_obs_camera=crop_obs_camera,
    )

    # Optionally save statistics to run directory
    if save_statistics_dir is not None:
        dataset.save_statistics(Path(save_statistics_dir))

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


    import debugpy 
    debugpy.listen(("0.0.0.0", 10092))  # 监听端口 
    print("Waiting for debugger to attach 10092...")
    debugpy.wait_for_client()  # 等待 VS Code 附加

    # test  get_vla_dataset
    from omegaconf import OmegaConf
    config_yaml = "./llavavla/conf/qwenvla_lmdb_real.yaml"
    cfg = OmegaConf.load(config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_model_cfg = cfg.framework.action_model
    vla_dataset = get_vla_dataset(
        data_root_dir=vla_dataset_cfg.data_root_dir,
        data_mix=vla_dataset_cfg.data_mix,
        data_mix_info=vla_dataset_cfg.data_mix_info,
        obs_type=vla_dataset_cfg.obs_type,
        action_type=vla_dataset_cfg.action_type,
        window_size=vla_model_cfg.future_action_window_size + 1,
        image_aug=vla_dataset_cfg.image_aug,
        default_image_resolution=tuple(vla_dataset_cfg.default_image_resolution),
        shuffle=vla_dataset_cfg.shuffle,
        crop_obs_camera=vla_dataset_cfg.crop_obs_camera,
        normalization_type=vla_dataset_cfg.normalization_type,
    )

    import time
    # 方法2: 使用迭代器
    dataset_iter = iter(vla_dataset)
    while True:
    # for _ in range(10):
        try:
            start_time = time.time()
            batch_samples = next(dataset_iter)
            print(batch_samples['action'])
            end_time = time.time()
            print(f"Each batch Time taken: {end_time - start_time} seconds")
        except StopIteration:
            break
