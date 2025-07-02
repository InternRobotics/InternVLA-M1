"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""


import os, json, pickle, bisect, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, List, Optional

import lmdb
from itertools import accumulate
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from llavavla.dataloader.lmdb.data_utils import get_lmdb_dataset_statistics, save_dataset_statistics, NormalizationType

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
        normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99,
        save_statistics_dir: str = None,
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

        # Load dataset info
        logger.info(f"loading dataset at {data_dir}/{dataset_name}")
        assert os.path.exists(f"{data_dir}/data_info/{self.dataset_info_name}.json")
        with open(f"{data_dir}/data_info/{self.dataset_info_name}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.window_size for f in self.episode_info_list]
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
            elif self.action_type == "delta_qpos":
                self.arm_action_mean = np.array(meta_info["delta_arm_action_mean"])
                self.arm_action_std = np.array(meta_info["delta_arm_action_std"])
                self.arm_action_min = np.array(meta_info["delta_arm_action_min"])
                self.arm_action_max = np.array(meta_info["delta_arm_action_max"])
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
            state_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
            state_key = meta_info["keys"]["scalar_data"][state_index]
        elif self.action_type == "delta_qpos":
            arm_index = meta_info["keys"]["scalar_data"].index(b'delta_arm_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
            state_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
            state_key = meta_info["keys"]["scalar_data"][state_index]
        elif self.action_type == "abs_ee_pose":
            arm_index = meta_info["keys"]["scalar_data"].index(b'ee_pose_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
            state_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/ee_pose_state')
            state_key = meta_info["keys"]["scalar_data"][state_index]
        elif self.action_type == "delta_ee_pose":
            arm_index = meta_info["keys"]["scalar_data"].index(b'delta_ee_pose_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
            state_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/ee_pose_state')
            state_key = meta_info["keys"]["scalar_data"][state_index]
        else:
            raise NotImplementedError(f"Action type {self.action_type} not supported")
        
        gripper_index = meta_info["keys"]["scalar_data"].index(b'gripper_close') 
        gripper_key = meta_info["keys"]["scalar_data"][gripper_index]
        # qpos_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
        # qpos_key = meta_info["keys"]["scalar_data"][qpos_index]

        primary_index = meta_info["keys"][f"observation/{self.obs_type}/color_image"]
        wrist_index = meta_info["keys"]["observation/realsense/color_image"]

        language_instruction = meta_info["language_instruction"]

        # get grounding info
        pick_uid = meta_info['task_data']['goal'][0][0]['obj1_uid']
        place_uid = meta_info['task_data']['goal'][0][0]['obj2_uid']
        target_position = meta_info['task_data']['goal'][0][0]['position']
        
        
        # 一次性缩放轨迹
        # scale_x = image_size[0] / 224
        # scale_y = image_size[1] / 224
        

        # Load sequence data
        sequence = []
        with lmdb_env.begin(write=False) as txn:

            arm_action = pickle.loads(txn.get(arm_key))
            if self.action_type == "abs_ee_pose" or self.action_type == "delta_ee_pose":
                positions = np.array([pos for pos, quat in arm_action])
                quaternions = np.array([quat for pos, quat in arm_action])
                arm_action = np.concatenate([positions, quaternions], axis=1).tolist()
            gripper_action = pickle.loads(txn.get(gripper_key))

            primary_data = pickle.loads(txn.get(primary_index[start_id]))
            primary_data = cv2.imdecode(np.frombuffer(primary_data, np.uint8), cv2.IMREAD_COLOR)
            # convert to PIL Image
            primary_data = Image.fromarray(primary_data)
            
            # TODO mv to func
            all_trace_2d = pickle.loads(txn.get(b'observation/obs_camera/tcp_2d_trace'))
            all_gripper_close = pickle.loads(txn.get(b'gripper_close'))

            # 2) current / target bbox
            all_raw_boxes = pickle.loads(txn.get( b'observation/obs_camera/bbox2d_loose' ))
            all_labels_list = pickle.loads(txn.get(b'observation/obs_camera/bbox2d_loose_id2labels'))
            curr_boxes, curr_labels = get_bbox_and_label_arrays(all_raw_boxes, all_labels_list, start_id)

            # resize to 224*224
            image_size = primary_data.size
            primary_data = primary_data.resize((224, 224), Image.BILINEAR)
            scale_x = 224 / image_size[0]
            scale_y = 224 / image_size[1]
            all_trace_2d_scaled = scale_trace(all_trace_2d, scale_x, scale_y)

             # 3) affordance point
             # 并不是全部都有 这个？
            future_griper_change = detect_first_change_point(start_id, all_gripper_close,
                                                    all_trace_2d_scaled,
                                                    point_index=1)


            # 4) 轨迹
            j = future_griper_change["frame_idx"] #future_griper_change_idx

            future_traj = get_trajectory_plan(all_trace_2d_scaled, start_id, j,
                                                horizon=10)
            
            sol = {
                "future_traj": future_traj
            }


        lmdb_env.close()
        # action chuck: window_size
        action_length = len(arm_action)
        if action_length >= self.window_size + start_id:
            action = arm_action[start_id:start_id + self.window_size]
            gripper = gripper_action[start_id:start_id + self.window_size]
        else:
            # last action repeat
            # action = arm_action[start_id:action_length] + np.zeros(self.window_size - (action_length - start_id))
            # gripper = gripper_action[start_id:action_length] + np.ones(self.window_size - (action_length - start_id))

            action = arm_action[start_id:action_length] + np.repeat(arm_action[-1], self.window_size - (action_length - start_id), axis=0)
            gripper = gripper_action[start_id:action_length] + np.repeat(gripper_action[-1], self.window_size - (action_length - start_id), axis=0)

        collected_action = []
        for a, g in zip(action, gripper):
            collected_action.append(self.load_robot_action(a, g).astype(np.float16))

        
        return dict(action=collected_action,image=[primary_data],lang=language_instruction, 
                    solution=sol,
                    dataset_name=self.dataset_name)

    def __iter__(self):
        """Iterate through the dataset sequentially."""
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    # TODO 这个函数 需要重构， 不能和数据集耦合
    def load_robot_action(self, arm_action, gripper_action):
        if self.action_type in ["abs_qpos", "delta_qpos", "abs_ee_pose", "delta_ee_pose"]:
            actions = np.zeros(8)
            actions[:7] = 2 * (arm_action[:7] - self.arm_action_min[:7]) / (self.arm_action_max[:7] - self.arm_action_min[:7] + 1e-8) - 1
            # normalize gripper_action to 0 or 1
            actions[-1] = (gripper_action + 1) / 2
            assert np.all(actions <= 1) and np.all(actions >= -1)
            return actions
        else:
            raise NotImplementedError(f"Action type {self.action_type} not supported")

    def get_dataset_statistics(self) -> Dict:
        """Return dataset statistics in the same format as RLDS datasets."""
        return self.dataset_statistics

    def save_statistics(self, run_dir: Path) -> None:
        """Save dataset statistics to the specified directory."""
        save_dataset_statistics(self.dataset_statistics, run_dir)


class EpisodicLMDBDataset(LMDBDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    # TODO 实现
    pass


def scale_point(p, scale_x, scale_y):
    return [int(p[0] * scale_x), int(p[1] * scale_y)]

def scale_trace(trace_2d, scale_x, scale_y):
    return [np.array([scale_point(p,scale_x, scale_y) for p in frame]) for frame in trace_2d]

def get_bbox_and_label_arrays(key_bbox, key_label, frame_idx: int) -> tuple:
    """
    从 txn 中读取指定帧的 loose bbox 数组与 id2label dict。
    返回： (boxes_np, labels_dict)
    boxes_np: numpy array shape (N,5) 每行 [label, xmin,ymin,xmax,ymax]
    labels_dict: dict[label_id->{class: uid}]
    """

    frame_boxes = np.stack([np.array(x).item() for x in key_bbox[frame_idx]])
    labels_list = key_label[frame_idx]
    return frame_boxes, labels_list

def get_dummy_dataset(dataconfig: dict):

    pass



def get_trajectory_plan(
    all_trace_2d: list,
    start_idx: int = 0,
    end_idx: int = -1,
    horizon: int=10,
    scale_x: float = 1.0,
    scale_y: float = 1.0
) -> list:
    """
    从 start_idx 到 end_idx 之间，均匀采样 horizon 个第二投影点，并缩放到图像坐标。
    如果 horizon 大于帧数差（end_idx - start_idx + 1），会对相邻帧做线性插值以补齐采样点。

    参数:
      all_trace_2d: list of shape [(n_points, 2), ...]，每帧若有多点，则选第 2 个点
      start_idx: 起始帧索引
      end_idx: 结束帧索引（包含）
      horizon: 采样点总数
      scale_x, scale_y: 缩放因子

    返回:
      traj: list of [x, y]，长度为 horizon
    """
    T = len(all_trace_2d)
    if end_idx == -1:
        end_idx = T - 1
    if not (0 <= start_idx < T):
        raise IndexError(f"start_idx {start_idx} out of range [0, {T})")
    if not (0 <= end_idx < T):
        raise IndexError(f"end_idx {end_idx} out of range [0, {T})")
    if end_idx <= start_idx:
        raise ValueError(f"end_idx ({end_idx}) must be > start_idx ({start_idx})")
    if horizon < 2:
        raise ValueError("horizon must be at least 2 to interpolate")

    # 在 [start_idx, end_idx] 区间生成均匀浮点索引
    positions = np.linspace(start_idx, end_idx, num=horizon)

    traj = []
    for pos in positions:
        # 若 pos 正好是整数索引，直接取该帧
        if pos.is_integer():
            idx = int(pos)
            pts = all_trace_2d[idx]
            x, y = pts[1]
        else:
            # 否则对 floor 和 ceil 帧坐标做线性插值
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            alpha = pos - lo
            pts_lo = all_trace_2d[lo][1]
            pts_hi = all_trace_2d[hi][1]
            x = (1 - alpha) * pts_lo[0] + alpha * pts_hi[0]
            y = (1 - alpha) * pts_lo[1] + alpha * pts_hi[1]

        # 缩放并取整
        traj.append([int(x * scale_x), int(y * scale_y)])

    return traj


def detect_first_change_point(
    current_index: int,
    gripper_close: List[int],
    trace_2d: List[List[List[float]]],
    point_index: int = 1
) -> Optional[Dict]:
    """
    从 current_index 开始，找到 gripper_close 列表中第一次发生值变化的时刻，
    并返回该帧的指定关键点坐标。

    参数:
        current_index: 开始检测的帧索引
        gripper_close: 手爪状态列表（例如，-1 表示张开，1 表示闭合）
        trace_2d: 每帧的 2D 关键点列表，形状为 [帧数][关键点数][2]
        point_index: 要返回的关键点在每帧列表中的索引

    返回:
        {'frame_idx': i, 'point': [x, y]} 或 None（如果到末尾都没有变化）
    """
    assert len(gripper_close) == len(trace_2d), "长度不一致，无法对应每一帧"

    prev = gripper_close[current_index]
    # 从下一个帧开始检测
    for i in range(current_index + 1, len(gripper_close)):
        curr = gripper_close[i]
        x, y = trace_2d[i][point_index]
        change_point =  {
                'frame_idx': i,
                'point': [int(x), int(y)]
            }
        if curr != prev:
            return change_point
        prev = curr

    # 没有检测到任何变化, 返回最后一个点
    return change_point

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
    episodic: bool = False,
    normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99,
    save_statistics_dir: str = None,
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
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()  # 等待 VS Code 附加

    # test  get_vla_dataset
    cfg = {
        'data_root_dir': '/mnt/petrelfs/share/yejinhui/Datasets',
        'obs_type': 'obs_camera',
        'action_type': 'delta_ee_pose',
        'window_size': 16,
        'vla': {
            'per_device_batch_size': 1,
            'data_mix': 'banana_plate_4035_none-wot-wow-woh-0529',
            'data_mix_info': 'Banana/banana_plate_4035_none-wot-wow-woh-0529_200',
        }
    }
    cfg = dict_to_namespace(cfg)

    vla_dataset = get_vla_dataset( # 拒绝任何内部转换
        cfg.data_root_dir, # 太多参数了， 应该config 穿越过去， 或者是 ** 的方式
        cfg.vla.data_mix,
        cfg.vla.data_mix_info,
        action_type=cfg.action_type,
        default_image_resolution=(3, 224, 224),
    )
    
    # 方法2: 使用迭代器
    dataset_iter = iter(vla_dataset)
    while True:
        try:
            batch_samples = next(dataset_iter)
            print(batch_samples['action'])
        except StopIteration:
            break
