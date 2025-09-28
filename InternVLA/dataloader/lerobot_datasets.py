from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from InternVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from InternVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from InternVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from InternVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,  # 新增参数
    delete_pause_frame: bool = False,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="torchvision_av",
        delete_pause_frame=delete_pause_frame,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool = True,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        **kwargs,
    )

if __name__ == "__main__":
    config_yaml = "InternVLA/config/training/internvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=16, # For Debug
        collate_fn=collate_fn,
    )

    from tqdm import tqdm
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        pass