"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

# TODO 未来不要是一个文件夹
from typing import Dict, List, Tuple

# fmt: off
LIBERO_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {

    # ===  Lebero Datasets ===
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0),
        ("libero_90_no_noops_lerobot", 1.0),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0),
    ],

    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0)
    ],
    
    # === Custom Finetuning Datasets ===
    "custom_finetuning": [
        ("gen_manip_tiny", 1.0),
    ],
}
# fmt: on
