import numpy as np
import torch
import os
import h5py
import json
import pickle
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

import logging
logger = logging.getLogger(__name__)

import IPython
e = IPython.embed

from itertools import accumulate
import lmdb

import cv2
import math

from torchvision import transforms
from PIL import Image

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

class EpisodicLmdbDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_name: str,
                 root_dir: str,
                 dataset_info_name: str,
                 camera_names,
                 args,
                 transform=None):
        super(EpisodicLmdbDataset).__init__()
        self.dataset_name = dataset_name
        self.dataset_info_name = dataset_info_name
        self.root_dir = root_dir
        self.dataset_path = f'{root_dir}/{dataset_name}'
        self.camera_names = camera_names
        self.args = args
        self.transform = transform
        logging.info(f"loading dataset at {root_dir}/{dataset_name}")
        assert os.path.exists(f"./data_info/{self.dataset_info_name}.json")
        with open(f"./data_info/{self.dataset_info_name}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)
        meta_info = pickle.load(open(f"./data_info/{self.dataset_info_name}.pkl", "rb"))
        self.norm_stats = {}
        self.norm_stats["action_mean"] = np.array(meta_info["arm_action_mean"])
        self.norm_stats["action_std"] = np.array(meta_info["arm_action_std"])
        self.norm_stats["arm_action_min"] = np.array(meta_info["arm_action_min"])
        self.norm_stats["arm_action_max"] = np.array(meta_info["arm_action_max"])
        self.norm_stats["qpos_mean"] = np.array(meta_info["qpos_mean"])
        self.norm_stats["qpos_std"] = np.array(meta_info["qpos_std"])
        self.norm_stats["qpos_min"] = np.array(meta_info["qpos_min"])
        self.norm_stats["qpos_max"] = np.array(meta_info["qpos_max"])
        self.norm_stats["eepose_action_min"] = np.array(meta_info["eepose_action_min"])
        self.norm_stats["eepose_action_max"] = np.array(meta_info["eepose_action_max"])
        self.norm_stats["eepose_state_min"] = np.array(meta_info["eepose_state_min"])
        self.norm_stats["eepose_state_max"] = np.array(meta_info["eepose_state_max"])
        self.action_length = 600


    def __len__(self):
        return self.num_episode

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_list[index]
        lmdb_env = lmdb.open(
            f"{self.dataset_path}/{episode_id}/lmdb",
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        meta_info = pickle.load(
            open(
                f"{self.dataset_path}/{episode_id}/meta_info.pkl",
                "rb"
            )
        )
        arm_index = meta_info["keys"]["scalar_data"].index(b'arm_action')
        arm_key = meta_info["keys"]["scalar_data"][arm_index]
        gripper_index = meta_info["keys"]["scalar_data"].index(b'gripper_close')
        gripper_key = meta_info["keys"]["scalar_data"][gripper_index]
        qpos_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
        qpos_key = meta_info["keys"]["scalar_data"][qpos_index]
        pre_grasp_index = meta_info['task_data']['frame_status']['0/pre_grasp']
        post_grasp_index = meta_info['task_data']['frame_status']['0/post_grasp']
        post_place_index = meta_info['task_data']['frame_status']['0/post_place']
        eepose_action_index = meta_info['keys']['scalar_data'].index(b'ee_pose_action')
        eepose_action_key = meta_info['keys']['scalar_data'][eepose_action_index]
        eepose_state_index = meta_info['keys']['scalar_data'].index(b'observation/robot/ee_pose_state')
        eepose_state_key = meta_info['keys']['scalar_data'][eepose_state_index]

        if self.args.use_image:
                    # obs_index = meta_info["keys"]["observation/obs_camera/color_image"]
            # realsense_index = meta_info["keys"]["observation/realsense/color_image"]
            camera_keys = []
            for cam_name in self.camera_names:
                camera_keys.append(meta_info["keys"][f"observation/{cam_name}/color_image"])

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(self.num_step_per_episode[index])

        with lmdb_env.begin(write=False) as txn:
            original_action_shape = (self.action_length, 7)
            original_gripper_close_shape = (self.action_length, 2)
            arm_action = pickle.loads(txn.get(arm_key))[start_ts:]
            gripper_close = pickle.loads(txn.get(gripper_key))[start_ts:]
            action_len = self.num_step_per_episode[index] - start_ts
            qpos = pickle.loads(txn.get(qpos_key))[start_ts]

            if self.args.use_image:
                image_dict = dict()
                for cam_name, cam_key in zip(self.camera_names, camera_keys):
                    cam_data = pickle.loads(txn.get(cam_key[start_ts]))
                    image_dict[cam_name] = cv2.imdecode(np.frombuffer(cam_data, np.uint8), cv2.IMREAD_COLOR)

                    # image_dict[cam_name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    # image_dict[cam_name] = cv2.imread('/ssd/home/wangfangjing/saved/visualized_results/vis/grasp_pose_sys2_log/2025-05-09/2025-05-09_16-34-57.png')
            # obs_data = pickle.loads(txn.get(obs_index[start_ts]))
            # obs_data = cv2.imdecode(np.frombuffer(obs_data, np.uint8), cv2.IMREAD_COLOR)
            # realsense_data = pickle.loads(txn.get(realsense_index[start_ts]))
            # realsense_data = cv2.imdecode(np.frombuffer(realsense_data, np.uint8), cv2.IMREAD_COLOR)

            eepose_action = pickle.loads(txn.get(eepose_action_key))
            eepose_state = pickle.loads(txn.get(eepose_state_key))


        pre_grasp_eepose_t, pre_grasp_eepose_q = eepose_action[pre_grasp_index]
        post_grasp_eepose_t, post_grasp_eepose_q = eepose_action[post_grasp_index]
        post_place_eepose_t, post_place_eepose_q = eepose_action[post_place_index]

        pre_grasp_eepose = np.concatenate([pre_grasp_eepose_t, pre_grasp_eepose_q])
        post_grasp_eepose = np.concatenate([post_grasp_eepose_t, post_grasp_eepose_q])
        post_place_eepose = np.concatenate([post_place_eepose_t, post_place_eepose_q])

        eepose_action = [np.concatenate([t, q]) for t, q in eepose_action][start_ts:]
        eepose_state = [np.concatenate([t, q]) for t, q in eepose_state][start_ts]

        grasp_eepose = np.stack([pre_grasp_eepose, post_grasp_eepose, post_place_eepose], axis=0)

        # arm_action = np.array(arm_action)
        # arm_action = 2 * (arm_action - self.norm_stats["arm_action_min"]) / (self.norm_stats["arm_action_max"] - self.norm_stats["arm_action_min"] + 1e-8) - 1

        # padded_action = np.zeros(original_action_shape, dtype=np.float32)
        # padded_action[:action_len] = arm_action
        is_pad = np.zeros(self.action_length)
        is_pad[action_len:] = 1

        eepose_action = np.array(eepose_action)
        eepose_action = 2 * (eepose_action - self.norm_stats["eepose_action_min"]) / (self.norm_stats["eepose_action_max"] - self.norm_stats["eepose_action_min"] + 1e-8) - 1 
        
        padded_eepose_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_eepose_action[:action_len] = eepose_action
        is_pad_eepose_action = np.zeros(self.action_length)
        is_pad_eepose_action[action_len:] = 1

        # Convert gripper_close to numpy array and repeat along new axis
        # gripper action is normalized to （0(开), 1（关））
        gripper_close = np.array(gripper_close)[:, None].repeat(2, axis=1)
        gripper_close = (gripper_close + 1) // 2
        padded_gripper_close = np.zeros(original_gripper_close_shape, dtype=np.float32)
        padded_gripper_close[:action_len] = gripper_close
        # is_pad_gripper = np.zeros(self.action_length)
        # is_pad_gripper[action_len:] = 1

        # all_cam_images = [obs_data, realsense_data]
        # all_cam_images = np.stack(all_cam_images, axis=0)

        # padded_action = (padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        # qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # gripper state is normalized to [-1(关), 1（开）]
        qpos = 2 * (qpos - self.norm_stats["qpos_min"]) / (self.norm_stats["qpos_max"] - self.norm_stats["qpos_min"] + 1e-8) - 1

        eepose_state = np.array(eepose_state)
        eepose_state = 2 * (eepose_state - self.norm_stats["eepose_state_min"]) / (self.norm_stats["eepose_state_max"] - self.norm_stats["eepose_state_min"] + 1e-8) - 1
        eepose_state = np.concatenate([eepose_state, qpos[7:9]], axis=-1)

        # image_data = torch.from_numpy(all_cam_images)
        eepose_action_data = torch.from_numpy(padded_eepose_action).float()
        eepose_state = torch.from_numpy(eepose_state).float()
        grasp_eepose = torch.from_numpy(grasp_eepose).float()
        # qpos_data = torch.from_numpy(qpos).float()
        # action_data = torch.from_numpy(padded_action).float()
        gripper_close_data = torch.from_numpy(padded_gripper_close).float()
        is_pad = torch.from_numpy(is_pad).bool()
        # is_pad_gripper = torch.from_numpy(is_pad_gripper).bool()
        is_pad_eepose_action = torch.from_numpy(is_pad_eepose_action).bool()
        # image_data = torch.einsum('k h w c -> k c h w', image_data)

        # image_data = image_data / 255.0 

        # action_data = torch.cat([action_data, gripper_close_data], dim=-1)
        # is_pad = torch.cat([is_pad, is_pad_gripper], dim=-1)
        eepose_action_data = torch.cat([eepose_action_data, gripper_close_data], dim=-1)
        # is_pad_eepose_action = torch.cat([is_pad_eepose_action, is_pad_gripper], dim=-1)

        # return image_data, qpos_data, action_data, is_pad, 
        # return grasp_eepose, qpos_data, action_data, is_pad

        if self.args.use_image:
            all_cam_images = []
            for cam_name in self.camera_names:
                # augmentation
                if self.transform is not None:
                    tmp_img = image_dict[cam_name]
                    # First convert OpenCV BGR format to RGB
                    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image first
                    tmp_img = Image.fromarray(tmp_img)
                    # Now apply the transform sequence
                    tmp_img = self.transform(tmp_img)
                    all_cam_images.append(tmp_img)
                else:
                    all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            image_data = torch.from_numpy(all_cam_images)
            # image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0

            return image_data, grasp_eepose, eepose_state, eepose_action_data, is_pad_eepose_action
        else:
            return grasp_eepose, eepose_state, eepose_action_data, is_pad_eepose_action


        # episode_id = self.episode_ids[index]
        # dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        # with h5py.File(dataset_path, 'r') as root:
        #     is_sim = root.attrs['sim']
        #     original_action_shape = root['/action'].shape
        #     episode_len = original_action_shape[0]
        #     if sample_full_episode:
        #         start_ts = 0
        #     else:
        #         start_ts = np.random.choice(episode_len)
        #     # get observation at start_ts only
        #     qpos = root['/observations/qpos'][start_ts]
        #     qvel = root['/observations/qvel'][start_ts]
        #     image_dict = dict()
        #     for cam_name in self.camera_names:
        #         image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
        #     # get all actions after and including start_ts
        #     if is_sim:
        #         action = root['/action'][start_ts:]
        #         action_len = episode_len - start_ts
        #     else:
        #         action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
        #         action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        # self.is_sim = is_sim
        # padded_action = np.zeros(original_action_shape, dtype=np.float32)
        # padded_action[:action_len] = action
        # is_pad = np.zeros(episode_len)
        # is_pad[action_len:] = 1

        # # new axis for different cameras
        # all_cam_images = []
        # for cam_name in self.camera_names:
        #     all_cam_images.append(image_dict[cam_name])
        # all_cam_images = np.stack(all_cam_images, axis=0)

        # # construct observations
        # image_data = torch.from_numpy(all_cam_images)
        # qpos_data = torch.from_numpy(qpos).float()
        # action_data = torch.from_numpy(padded_action).float()
        # is_pad = torch.from_numpy(is_pad).bool()

        # # channel last
        # image_data = torch.einsum('k h w c -> k c h w', image_data)

        # # normalize image and change dtype to float
        # image_data = image_data / 255.0
        # action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        # qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # return image_data, qpos_data, action_data, is_pad

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def get_norm_stats_lmdb(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats



def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_lmdb_data(args, dataset_name, dataset_info_name, root_dir, camera_names, batch_size_train, floor=False):

    # patch_h = 16
    # patch_w = 22
    # transform = transforms.Compose([
    #     transforms.ColorJitter(
    #         brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
    #     ),
    #     transforms.RandomPerspective(distortion_scale=0.5),
    #     transforms.RandomAffine(
    #         degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
    #     ),
    #     transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
    #     transforms.Resize((patch_h * 14, patch_w * 14)),
    #     transforms.ToTensor(),  # ensure correct dtype and shape
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                         std=(0.229, 0.224, 0.225)),
    # ])
    transform = None
    # norm_stats = get_norm_stats_lmdb(root_dir, dataset_name, dataset_info_name)
    train_dataset = EpisodicLmdbDataset(dataset_name, root_dir, dataset_info_name, camera_names, args, transform)

    round_fn = math.floor if floor else math.ceil
    num_samples = len(train_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        drop_last=True
    )
    train_dataloader.num_batches = num_batches
    train_dataloader.num_samples = num_samples


    return train_dataloader
    


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)




# ===============================================
# 
# ===============================================

 
import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
import lmdb
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import time 
import cv2
import re
import pickle
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 16,
        "keep_indices": [[0, 16]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}

def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}

def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class BaseBananaDataset(Dataset):
    def __init__(
        self,
        dataset_name: str, # bun_all_none
        root_dir: str, # /ssd/home/wangfangjing/data
        dataset_info_name: str, # bun_all_none_100
        obs_type: str = "obs_camera",
        action_type: str = "abs_qpos",
        obs_space: DictConfig = obs_config,
        proprio_state: DictConfig = prop_state,
        transforms: Dict = {},
        window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        text_aug=False,
        dif_ws=False,
        act_step: int = 1,
        key: str = "lang",
        gripper_width: bool = False,
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        magic_scaling_factor_pos: float = 1.0,
        magic_scaling_factor_orn: float = 1.0,
        use_aug_data: bool = False,
        **kwargs: Any,
    ):
        self.dataset_name = dataset_name
        self.root_dir = root_dir 
        self.action_type = action_type
        self.dataset_path = f'{root_dir}/{dataset_name}'
        self.dataset_info_name = dataset_info_name
        self.obs_type = obs_type
        self.image_preprocess = None
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.pad = pad
        self.window_size = window_size
        self.use_aug_data = use_aug_data
        self.aux_lang_loss_window = aux_lang_loss_window
        self.text_aug = text_aug
        self.act_step = act_step
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.magic_scaling_factor_pos = magic_scaling_factor_pos
        self.magic_scaling_factor_orn = magic_scaling_factor_orn
        logger.info(f"loading dataset at {root_dir}/{dataset_name}")
        logger.info("finished loading dataset")
        print(f"./data_info/{self.dataset_info_name}.json")
        assert os.path.exists(f"./data_info/{self.dataset_info_name}.json")
        with open(f"./data_info/{self.dataset_info_name}.json", 'r') as f:
            self.episode_info_list = json.load(f)
            self.episode_list = [f[0] for f in self.episode_info_list]
            self.num_step_per_episode = [f[1] - self.window_size for f in self.episode_info_list]
            self.num_episode = len(self.episode_list)
        meta_info = pickle.load(open(f"./data_info/{self.dataset_info_name}.pkl", "rb"))
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
        self.accumulated_num_step = list(accumulate(self.num_step_per_episode))
        self.length = self.accumulated_num_step[-1]
        self.gripper_width = gripper_width
    
    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}
        
    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.window_size - len(sequence["actions"])

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))

        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))

        return padded

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )
        #  todo: find better way of distinguishing rk and play action spaces
        seq_acts = torch.cat(
            [
                self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
            ],
            dim=-1,
        )
        seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )

        return seq

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict
    ):
        return {"lang": episode["language"]}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        window_size = self.window_size
        head = False
        sequence = self._get_sequences(idx, window_size, head=head) # TODO
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list

        return sequence
    
    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        episode_id = bisect.bisect_right(self.accumulated_num_step, idx)
        if episode_id - 1 >= 0:
            start_id = idx - self.accumulated_num_step[episode_id - 1]
        else:
            start_id = idx
        num_step_per_episode = self.num_step_per_episode[episode_id]
        end_id = start_id + window_size 
        if self.use_aug_data:
            demo_list = self.episode_info_list[episode_id][2:]
            start_id, end_id = demo_list[start_id]
        episode_id = self.episode_list[episode_id] 
        episodes = []

        # lmdb env
        lmdb_env = lmdb.open(
            f"{self.dataset_path}/{episode_id}/lmdb", 
            # f"/home/tianyang/Data/{episode_id}/lmdb",
            readonly=True, 
            lock=False, 
            readahead=True, 
            meminit=True
        )
        meta_info = pickle.load(
            open(
            f"{self.dataset_path}/{episode_id}/meta_info.pkl", 
            # f"/home/tianyang/Data/{episode_id}/meta_info.pkl", 
            "rb"
            )
        )
        t0 = time.time()
        # arm action and gripper action
        if self.action_type == "abs_qpos":
            arm_index = meta_info["keys"]["scalar_data"].index(b'arm_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]
        elif self.action_type == "delta_qpos":
            arm_index = meta_info["keys"]["scalar_data"].index(b'delta_arm_action')
            arm_key = meta_info["keys"]["scalar_data"][arm_index]      
        elif self.action_type == "delta_ee":
            arm_index = meta_info["keys"]["scalar_data"].index(b'delta_arm_ee_action')  
            arm_key = meta_info["keys"]["scalar_data"][arm_index]      
        gripper_index = meta_info["keys"]["scalar_data"].index(b'gripper_close')
        gripper_key = meta_info["keys"]["scalar_data"][gripper_index]
        # proprio: qpos
        qpos_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/qpos')
        qpos_key = meta_info["keys"]["scalar_data"][qpos_index]
        # forlanpose_index = meta_info["keys"]["scalar_data"].index(b'observation/robot/forlan2robot_pose')
        # forlanpose_key = meta_info["keys"]["scalar_data"][forlanpose_index]
        # primary color image and wrist color image
        primary_index =  meta_info["keys"][f"observation/{self.obs_type}/color_image"]
        wrist_index = meta_info["keys"]["observation/realsense/color_image"]
        t00 = time.time()

        for step_id in range(start_id, end_id):
            data_dict = {}
            str_step_id = str(step_id).zfill(4)
            with lmdb_env.begin(write=False) as txn:
                # torch.cuda.synchronize()
                t1 = time.time()
                arm_action = pickle.loads(txn.get(arm_key))[step_id]
                gripper_action = pickle.loads(txn.get(gripper_key))[step_id]
                qpos = pickle.loads(txn.get(qpos_key))[step_id]
                # forlanpose = pickle.loads(txn.get(forlanpose_key))[step_id]
                # torch.cuda.synchronize()
                t2 = time.time()
                primary_data = pickle.loads(txn.get(primary_index[step_id]))
                primary_data = cv2.imdecode(np.frombuffer(primary_data, np.uint8), cv2.IMREAD_COLOR)
                wrist_data = pickle.loads(txn.get(wrist_index[step_id]))
                wrist_data = cv2.imdecode(np.frombuffer(wrist_data, np.uint8), cv2.IMREAD_COLOR)
                # torch.cuda.synchronize()
                t3 = time.time()
            # images
            data_dict["rgb_static"] = primary_data
            data_dict["rgb_gripper"] = wrist_data
            # actions
            data_dict["actions"] = self.load_robot_action(arm_action, gripper_action)
            # robot_obs
            data_dict["robot_obs"] = self.load_robot_obs(qpos, None)
            # scene obs
            data_dict["scene_obs"] = np.zeros(self.proprio_state.n_scene_obs)
            episodes.append(data_dict)
        lmdb_env.close()

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = meta_info["language_instruction"]
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        info["use_for_aux_lang_loss"] = False
        seq_lang = self.process_language(episode, self.transforms)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  
        seq_dict["episode_id"] = episode_id

        return seq_dict

    def load_robot_action(self, arm_action, gripper_action):
        
        # actions[:7] = (arm_action[:7] - self.arm_action_mean[:7]) / self.arm_action_std[:7]
        if self.action_type == "abs_qpos" or self.action_type == "delta_qpos":
            actions = np.zeros(8)
            actions[:7] = 2 * (arm_action[:7] - self.arm_action_min[:7]) / (self.arm_action_max[:7] - self.arm_action_min[:7] + 1e-8) - 1
            actions[-1] = gripper_action
            assert np.all(actions <= 1) and np.all(actions >= -1)
            # actions = np.clip(actions, -1, 1)
        elif self.action_type == "delta_ee":
            actions = np.zeros(7)
            actions[-1] = gripper_action
            actions[:3] = arm_action[:3, 3] / 0.02
            actions[3:6] = R.from_matrix(arm_action[:3,:3]).as_euler("xyz", degrees=False) / 0.05
        return actions

    def load_robot_obs(self, qpos, forlanpose, open_thresh=0.10):
        robot_obs = np.zeros(self.proprio_state.n_state_obs)
        # robot_obs[:6] = pose_to_6d(forlanpose) # forlan pose 6d
        robot_obs[7:14] = qpos[:7]   
        robot_obs[-2:] = qpos[7:9]

        return robot_obs

    def __len__(self):
        return self.length

class DiskBananaDataset(Dataset):
    def __init__(
        self, 
        image_fn: Callable,
        text_fn: Callable,
        dataset_names: List[str],
        dataset_info_names: List[str],
        obs_type: str = "obs_camera",
        *args: Any,
        rgb_pad: int = -1,
        gripper_pad: int = -1,
        traj_cons: bool = False,
        act_step : int = 1,
        gripper_width: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.dataset_names = dataset_names
        # set_trace()
        self.datasets = [
                BaseBananaDataset(
                    *args, 
                    dataset_name=dataset_name,
                    obs_type=obs_type,
                    act_step=act_step,
                    gripper_width=gripper_width,
                    **kwargs,
                    dataset_info_name=dataset_info_name,
                ) for dataset_name, dataset_info_name in zip(dataset_names, dataset_info_names)
            ]
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        self.act_step = act_step
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        self.length_each_dataset = [len(dataset) for dataset in self.datasets]
        self.accumulated_length_each_dataset = list(accumulate(self.length_each_dataset))

    def register_image_preprocess_hook(self, func):
        self.image_preprocess = func

    def __len__(self):
        return self.accumulated_length_each_dataset[-1]

    def __getitem__(self, idx):
        dataset_id = bisect.bisect_right(self.accumulated_length_each_dataset, idx)
        if dataset_id - 1 >= 0:
            local_idx = idx - self.accumulated_length_each_dataset[dataset_id - 1]
        else:
            local_idx = idx

        return self.datasets[dataset_id].__getitem__(local_idx)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        episode_id = [s["episode_id"] for s in sample]
        text_tensors = self.text_fn(stacked_language)

        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        robot_obs = torch.zeros(1)
        if self.act_step != 1:
            # set_trace()
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]

        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs 

def get_banana_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_names = args.banana_dataset_names
    dataset_info_names = args.dataset_info_names
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    banana_dataset = DiskBananaDataset(
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        dataset_names=dataset_names,
        dataset_info_names=dataset_info_names,
        obs_type=args.obs_type,
        action_type=args.action_type,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        act_step=args.multi_step_action,
        root_dir=args.root_dir,
        window_size=args.window_size,
        dif_ws=args.dif_ws,
        gripper_width=args.gripper_width,
        use_aug_data=args.use_aug_data,
        max_rel_pos=args.max_rel_pos,
        max_rel_orn=args.max_rel_orn,
        magic_scaling_factor_pos=args.magic_scaling_factor_pos,
        magic_scaling_factor_orn=args.magic_scaling_factor_orn,
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(banana_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        banana_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        banana_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=banana_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=banana_dataset)




if __name__ == "__main__":
    dataset = BaseBananaDataset(
        dataset_name="pp_1225" , # pp_1225
        root_dir="/home/tianyang/Data", # /home/tianyang/Data
    )

    dataset._get_sequences(0, 16)
















