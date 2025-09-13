
# 这里定义一个基本的 framework class 以及share 的func

import torch.nn as nn
from typing import Union, List

from pathlib import Path

import torch, json
import torch.nn as nn
import numpy as np

from typing import List

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch, json
import torch.nn as nn
import numpy as np
from PIL import Image
import re
from InternVLA.model.tools import auto_get_trainable_modules # 后续应该是trainer 的职责范围

from InternVLA.model.framework.share_tools import read_mode_config
from InternVLA.training.trainer_utils import initialize_overwatch
from InternVLA.model.framework.share_tools import dict_to_namespace



logger = initialize_overwatch(__name__)


class baseframework(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
    
    @classmethod
    def from_pretrained( # @Jinhui TODO 这里要写如何resume checkpoints
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint) # 读取 config 和 norm_stats
        # TODO 
        config = dict_to_namespace(model_config)
        model_config = config # TODO 不要使用相对变量 model_config， 需要换名字
        model_config.trainer.pretrained_checkpoint = None # 为了加快加载速度，避免重复加载， TODO 其实不应该在initial的位置设置 load_pretrained_backbones
        FrameworkModel = cls(config=model_config, **kwargs) # 初始化模型
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu") #["model"]
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            FrameworkModel = FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e

        # **确保模型在 GPU 上**
        FrameworkModel = FrameworkModel #.to("cuda") # TODO 其实不应该是这里管理to GPU的
        return FrameworkModel

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):# 这个要来了任何和 lerobot 格式的对齐
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"] 

    @property # TODO 这个应该是trainer 的duty
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        keys = auto_get_trainable_modules(self, max_depth=max_depth)# auto 去判断哪些module是trainable的
        return keys
    
    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        将归一化的动作转换为原始动作空间。
        
        :param normalized_actions: 归一化的动作数组，形状为 [action chunk, D]。
        :param action_norm_stats: 包含动作归一化统计信息的字典，必须包含以下键：
            - "q01": 动作的第 1 百分位值。
            - "q99": 动作的第 99 百分位值。
            - "mask": 可选，布尔数组，用于标记哪些动作需要反归一化。
        :return: 反归一化后的动作数组，形状与输入 `normalized_actions` 相同。
        """

        # @BUG 这个是 simpler 的
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):# 这个要来了任何和 lerobot 格式的对齐
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"] 
