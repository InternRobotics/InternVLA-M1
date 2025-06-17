"""
data_utils.py

Additional RLDS-specific data utilities.
"""

import hashlib
import json
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: Dict) -> Dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file for LMDB datasets."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for dataset_name, stats in dataset_statistics.items():
            # Convert numpy arrays to lists for JSON serialization
            for action_type in ["action"]:
                if action_type in stats:
                    for k in stats[action_type].keys():
                        if isinstance(stats[action_type][k], np.ndarray):
                            stats[action_type][k] = stats[action_type][k].tolist()
            
            # Handle proprioceptive data if present
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            
            # Handle scalar statistics
            for scalar_key in ["num_trajectories", "num_transitions"]:
                if scalar_key in stats:
                    if isinstance(stats[scalar_key], np.ndarray):
                        stats[scalar_key] = stats[scalar_key].item()
        
        json.dump(dataset_statistics, f_json, indent=2)
    overwatch.info(f"Saved LMDB dataset statistics file at path {out_path}")


def get_lmdb_dataset_statistics(
    dataset_name: str,
    data_dir: str,
    action_type: str = "abs_qpos",
    dataset_info_name: str = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Computes and caches statistics for LMDB datasets similar to RLDS datasets.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Root directory containing the dataset
        action_type: Type of action data to analyze
        dataset_info_name: Optional different name for dataset info files
        save_dir: Directory to save statistics cache
    
    Returns:
        Dictionary containing dataset statistics
    """
    dataset_info_name = dataset_info_name if dataset_info_name is not None else dataset_name
    
    # Create unique hash for caching
    hash_dependencies = (dataset_name, data_dir, action_type, dataset_info_name)
    unique_hash = hashlib.sha256("".join(map(str, hash_dependencies)).encode("utf-8"), usedforsecurity=False).hexdigest()
    
    # Setup cache paths
    local_path = os.path.expanduser(os.path.join("~", ".cache", "llavavla", f"lmdb_dataset_statistics_{unique_hash}.json"))
    if save_dir is not None:
        cache_path = os.path.join(save_dir, f"lmdb_dataset_statistics_{unique_hash}.json")
    else:
        cache_path = local_path
    
    # Check if cache exists
    if os.path.exists(cache_path):
        overwatch.info(f"Loading existing LMDB dataset statistics from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)
    
    if os.path.exists(local_path):
        overwatch.info(f"Loading existing LMDB dataset statistics from {local_path}")
        with open(local_path, "r") as f:
            return json.load(f)
    
    # Compute statistics if not cached
    overwatch.info("Computing LMDB dataset statistics. This may take a bit, but should only need to happen once.")
    
    # Load dataset info
    dataset_info_path = os.path.join(data_dir, "data_info", f"{dataset_info_name}.json")
    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError(f"Dataset info file not found: {dataset_info_path}")
    
    with open(dataset_info_path, 'r') as f:
        episode_info_list = json.load(f)
    
    # Load action statistics from pickle file
    meta_info_path = os.path.join(data_dir, "data_info", f"{dataset_info_name}.pkl")
    if not os.path.exists(meta_info_path):
        raise FileNotFoundError(f"Meta info file not found: {meta_info_path}")
    
    import pickle
    with open(meta_info_path, "rb") as f:
        meta_info = pickle.load(f)
    
    # Extract action statistics based on action_type
    try:
        if action_type == "abs_qpos":
            action_mean = np.array(meta_info["abs_arm_action_mean"])
            action_std = np.array(meta_info["abs_arm_action_std"])
            action_min = np.array(meta_info["abs_arm_action_min"])
            action_max = np.array(meta_info["abs_arm_action_max"])
        elif action_type == "delta_qpos":
            action_mean = np.array(meta_info["delta_arm_action_mean"])
            action_std = np.array(meta_info["delta_arm_action_std"])
            action_min = np.array(meta_info["delta_arm_action_min"])
            action_max = np.array(meta_info["delta_arm_action_max"])
        elif action_type == "abs_ee_pose":
            action_mean = np.array(meta_info["abs_eepose_action_mean"])
            action_std = np.array(meta_info["abs_eepose_action_std"])
            action_min = np.array(meta_info["abs_eepose_action_min"])
            action_max = np.array(meta_info["abs_eepose_action_max"])
        elif action_type == "delta_ee_pose":
            action_mean = np.array(meta_info["delta_eepose_action_mean"])
            action_std = np.array(meta_info["delta_eepose_action_std"])
            action_min = np.array(meta_info["delta_eepose_action_min"])
            action_max = np.array(meta_info["delta_eepose_action_max"])
        else:
            raise NotImplementedError(f"Action type {action_type} not supported")
    except KeyError as e:
        raise KeyError(f"Required action statistics not found in meta_info for action_type {action_type}: {e}")
    
    # Calculate quantiles (q01, q99) - these might not be pre-computed, so we'll use min/max as approximation
    # In a real implementation, you might want to compute these from actual data
    action_q01 = action_min
    action_q99 = action_max
    
    # Count episodes and transitions
    num_trajectories = len(episode_info_list)
    num_transitions = sum(info[1] for info in episode_info_list)
    
    # Create statistics dictionary
    statistics = {
        dataset_name: {
            "action": {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "max": action_max.tolist(),
                "min": action_min.tolist(),
                "q01": action_q01.tolist(),
                "q99": action_q99.tolist(),
            },
            "num_trajectories": num_trajectories,
            "num_transitions": num_transitions,
            "action_type": action_type,
            "dataset_info_name": dataset_info_name,
        }
    }
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(statistics, f, indent=2)
        overwatch.info(f"Saved LMDB dataset statistics to {cache_path}")
    except Exception as e:
        overwatch.warning(f"Could not write dataset statistics to {cache_path}. Error: {e}")
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(statistics, f, indent=2)
            overwatch.info(f"Saved LMDB dataset statistics to {local_path}")
        except Exception as e2:
            overwatch.error(f"Failed to save statistics to backup location: {e2}")
    
    return statistics


def normalize_lmdb_action_and_proprio(traj: Dict, metadata: Dict, normalization_type: NormalizationType):
    """Normalizes the action fields of an LMDB trajectory using the given metadata."""
    if normalization_type == NormalizationType.NORMAL:
        # Normalize to mean=0, std=1
        action = (traj["action"] - metadata["action"]["mean"]) / (metadata["action"]["std"] + 1e-8)
        return {"action": action, **{k: v for k, v in traj.items() if k != "action"}}
    
    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        # Normalize to [-1, 1] bounds
        if normalization_type == NormalizationType.BOUNDS:
            low = np.array(metadata["action"]["min"])
            high = np.array(metadata["action"]["max"])
        else:  # BOUNDS_Q99
            low = np.array(metadata["action"]["q01"])
            high = np.array(metadata["action"]["q99"])
        
        # Normalize to [-1, 1]
        action = np.clip(2 * (traj["action"] - low) / (high - low + 1e-8) - 1, -1, 1)
        
        # Set unused dimensions (where min == max) to 0
        zeros_mask = low == high
        action = np.where(zeros_mask, 0.0, action)
        
        return {"action": action, **{k: v for k, v in traj.items() if k != "action"}}
    
    else:
        raise ValueError(f"Unknown Normalization Type {normalization_type}")

