# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The OxE Bridge dataset.
    """

    OXE_RT1 = "oxe_rt1"
    """
    The OxE RT-1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = "franka"
    """
    The Franka Emika Panda robot.
    """


# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_BRIDGE.value: 18,
    EmbodimentTag.OXE_RT1.value: 19,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.FRANKA.value: 25,
}

# dataset name to embodiment tag
DATASET_NAME_TO_EMBODIMENT_TAG = {
    "bridge_orig_1.0.0_lerobot": EmbodimentTag.OXE_BRIDGE,
    "fractal20220817_data_0.1.0_lerobot": EmbodimentTag.OXE_RT1,
    "bench_v6_all_longrange_split0_h264": EmbodimentTag.FRANKA,
    "bench_v6_all_longrange_split1_h264": EmbodimentTag.FRANKA,
    "bench_v6_all_longrange_split2_h264": EmbodimentTag.FRANKA,
    "bench_v6_all_longrange_split3_h264": EmbodimentTag.FRANKA,
    "bench_v6_all_longrange_split4_h264": EmbodimentTag.FRANKA,
    "libero_10_no_noops_1.0.0_lerobot": EmbodimentTag.FRANKA,
    "libero_goal_no_noops_1.0.0_lerobot": EmbodimentTag.FRANKA,
    "libero_spatial_no_noops_1.0.0_lerobot": EmbodimentTag.FRANKA,
    "libero_90_no_noops_lerobot": EmbodimentTag.FRANKA,
    "libero_object_no_noops_1.0.0_lerobot": EmbodimentTag.FRANKA,
}
