
# @Jinhui TODO 不要写这样的方式， 请直接 import from datasets.py

import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from InternVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

# TODO 工具类，注意后续的 重构, 应该写到dataloader class 内部
def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets_oxe":
        from InternVLA.dataloader.lerobot_datasets_oxe import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        data_root_dir = vla_dataset_cfg.data_root_dir
        data_mix = vla_dataset_cfg.data_mix

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=8,
            # shuffle=True # RLSD 不能做这个事情, leberot 也不需要， 但是在考虑中
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    
    elif dataset_py == "lerobot_datasets_libero":
        from InternVLA.dataloader.lerobot_datasets_libero import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(vla_dataset_cfg) # TODO 要将config 传输融合进去
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=8,
            # shuffle=True # RLSD 不能做这个事情, leberot 也不需要， 但是在考虑中
        )        
        if dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader

    elif dataset_py == "vlm_datasets":
        # VLM 数据加载器
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
        

