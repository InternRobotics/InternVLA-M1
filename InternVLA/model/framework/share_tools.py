"""
TODO  waiting to move # 全部tools 应该是要在一个位置的

"""

import os
from pathlib import Path
from types import SimpleNamespace
import json

from typing import Union, List
import torchvision
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from types import SimpleNamespace
import torch, json
import torch.nn as nn
import numpy as np
from PIL import Image
import re

from InternVLA.training.trainer_utils import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)



from types import SimpleNamespace

class NamespaceWithGet(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        # 返回键值对生成器
        return ((key, getattr(self, key)) for key in self.__dict__)

    def __iter__(self):
        # 使对象支持字典解包
        return iter(self.__dict__)

    def to_dict(self):
        # 递归将 NamespaceWithGet 转换为字典
        return {key: (value.to_dict() if isinstance(value, NamespaceWithGet) else value)
                for key, value in self.items()}


# def dict_to_namespace(d): 
#     if isinstance(d, dict):
#         return NamespaceWithGet(**{k: dict_to_namespace(v) for k, v in d.items()})
#     elif isinstance(d, list):
#         return [dict_to_namespace(i) for i in d]
#     else:
#         return d
    
from omegaconf import OmegaConf

def dict_to_namespace(d):
    """
    将字典转换为 OmegaConf 对象。
    """
    return OmegaConf.create(d)




from types import SimpleNamespace
from omegaconf import OmegaConf
import inspect
import functools
from typing import Any

# TODO use this for much good config handling in model code
def _to_omegaconf(x: Any):
    """
    Convert x to an OmegaConf object.
    Accepts: None, str(path), dict, OmegaConf, NamespaceWithGet, SimpleNamespace.
    """
    if x is None:
        return OmegaConf.create({})
    if isinstance(x, OmegaConf.__class__):  # fallback, typically not hit
        return x
    try:
        # OmegaConf node detection
        from omegaconf import DictConfig, ListConfig

        if isinstance(x, (DictConfig, ListConfig)):
            return x
    except Exception:
        pass

    if isinstance(x, str):
        # treat as path
        return OmegaConf.load(x)
    if isinstance(x, dict):
        return OmegaConf.create(x)
    if isinstance(x, NamespaceWithGet) or isinstance(x, SimpleNamespace):
        # convert to plain dict
        try:
            d = x.to_dict() if hasattr(x, "to_dict") else vars(x)
        except Exception:
            d = vars(x)
        return OmegaConf.create(d)
    # fallback: try to create
    return OmegaConf.create(x)


def merge_pram_config(init):
    """
    Decorator for __init__ that:
    - extracts the 'config' argument (if any) and converts it to OmegaConf;
    - converts other explicitly passed init args to an OmegaConf and merges: merged = OmegaConf.merge(loaded_cfg, params_cfg)
      (explicit params override loaded cfg);
    - sets self.config = merged and calls original __init__ with merged config and original other kwargs.
    """
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # Map positional args to parameter names (excluding self)
        sig = inspect.signature(init)
        param_names = [name for i, (name, p) in enumerate(sig.parameters.items()) if i > 0]

        init_kwargs = {}
        for name, val in zip(param_names, args):
            init_kwargs[name] = val
        # override with explicit kwargs
        init_kwargs.update(kwargs)

        # get provided config (if any)
        provided_config = init_kwargs.get("config", None)

        loaded_cfg = _to_omegaconf(provided_config)

        # build params cfg from explicit init args (other than config)
        params = {k: v for k, v in init_kwargs.items() if k != "config"}
        params_cfg = OmegaConf.create(params) if params else OmegaConf.create({})

        # merge: loaded_cfg <- params_cfg (params override file)
        merged = OmegaConf.merge(loaded_cfg, params_cfg)

        # set on instance
        try:
            # prefer attaching OmegaConf directly
            self.config = merged
        except Exception:
            # fallback to dict
            self.config = OmegaConf.to_container(merged, resolve=True)

        # prepare kwargs for original init: ensure config is the merged OmegaConf
        call_kwargs = dict(init_kwargs)
        call_kwargs["config"] = merged

        # call original __init__ using keyword args only (safer)
        return init(self, **call_kwargs)

    return wrapper




def read_model_config(pretrained_checkpoint):
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt")
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)

        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            vla_cfg = json.load(f) #["vla"]
            # model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])() #@TODO check 我觉得其实不重要，

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return vla_cfg, norm_stats


def read_mode_config(pretrained_checkpoint):
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt")
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)

        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            vla_cfg = json.load(f) #["vla"]
            # model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])() #@TODO check 我觉得其实不重要，

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return vla_cfg, norm_stats
