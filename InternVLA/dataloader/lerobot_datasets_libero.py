from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from InternVLA.dataloader.gr00t_lerobot.data_config import DATA_CONFIG_MAP
from InternVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag, DATASET_NAME_TO_EMBODIMENT_TAG
from InternVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from InternVLA.dataloader.gr00t_lerobot.libero.mixtures import LIBERO_NAMED_MIXTURES

def collate_fn(batch):
    return batch


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :return: A LeRobotSingleDataset object.
    """
    data_config = DATA_CONFIG_MAP[data_name]
    modality_config = data_config.modality_config()
    # TODO: wait for test transforms
    transforms = data_config.transform()
    # transforms = None
    dataset_path = data_root_dir / data_name
    embodiment_tag = DATASET_NAME_TO_EMBODIMENT_TAG[data_name]
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=EmbodimentTag[embodiment_tag.name],
        video_backend="torchvision_av",
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_mix: The name of the dataset mixture.
    """
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    mixture_spec = LIBERO_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    dataset_mixture = []  # Changed from Sequence type annotation to actual list
    
    for d_name, d_weight in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(data_root_dir, d_name), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=42,
        **kwargs
    )


if __name__ == "__main__":
    data_root_dir = Path("/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Datasets/OXE_LEROBOT_DATASET")
    data_mix = "bridge" # bridge_rt_1
    import debugpy
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for client to attach 10092...")
    debugpy.wait_for_client()
    
    # Load YAML config & Convert CLI overrides to dotlist config
    config_yaml = "llavavla/config/lerobot_data/qwenvla_cotrain_oxe.yaml"
    config_yaml = "llavavla/config/lerobot_data/qwenvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data


    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    # for i in range(len(dataset)):
    #     print(dataset[i])

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )
    
    from tqdm import tqdm

    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # print(batch)
        pass