"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from llavavla.model.vlm import _QWen_VL_Interface
from prismatic.overwatch import initialize_overwatch

from llavavla.model.vla import CogACT_Qwen
from llavavla.model.vla.align_qwen_act import QwenACT
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-AWQ"

# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys()) #用这样的注册变量是增加了耦合性的，不推荐 TODO @Jinhui --> 改为注册类


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load_qwenvl(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None, #TODO mv, 这些不应该由代码控制，而是变为系统变量
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> _QWen_VL_Interface:
    
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
        # Load Vision Backbone

    qwen_vl = _QWen_VL_Interface.from_pretrained(model_id_or_path, enable_mixed_precision_training=True)
    # del qwen_vl.model.lm_head  # Remove LM Head for Inference
    # Load Model Config from `config.json`
    model_cfg = qwen_vl.model.config.to_dict()
    # processing_cfg = qwen_vl.processor
    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['_name_or_path']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['model_type']}[/]\n"
        # f"             LLM Backbone    =>> [bold]{model_cfg['model_type']}[/]\n"
        # f"             Arch Specifier  =>> [bold]{model_cfg['model_type']}[/]\n"
        # f"             Checkpoint Path =>> [underline]`{_name_or_path}`[/]"
    )  

    return qwen_vl
 
# === Load Pretrained VLA Model ===
def load_qwenvla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    model_type: str = "pretrained",
    **kwargs,
) -> CogACT_Qwen:
    """Loads a pretrained CogACT from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)

        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        with open(config_json, "r") as f:
            vla_cfg = json.load(f)["vla"]
            # model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])() #@TODO check 我觉得其实不重要，

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_id_or_path      }[/] with:\n"
        f"             LVM Backbone =>> [bold]{vla_cfg['base_vlm']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    vla = CogACT_Qwen.from_pretrained( # 这个位置就会报错
        checkpoint_pt,
        base_vlm = vla_cfg["base_vlm"],
        freeze_weights=not load_for_training, 
        norm_stats=norm_stats,
        **kwargs,
    )

    return vla


# === Load Pretrained VLA Model ===
def load_qwenact( #@Jinhui TODO 后期要简化 模型的加载了 逻辑，建议是直接 initial初始化，统一入口覆盖参数
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    model_type: str = "pretrained",
    **kwargs,
) -> QwenACT:
    """Loads a pretrained CogACT from either local disk or the HuggingFace Hub."""

    # TODO (siddk, moojink) :: Unify semantics with `load()` above; right now, `load_vla()` assumes path points to
    #   checkpoint `.pt` file, rather than the top-level run directory!
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints"), "Invalid checkpoint!"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        # model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])() #@TODO check 我觉得其实不重要，

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_id_or_path      }[/] with:\n"
        f"             LVM Backbone =>> [bold]{vla_cfg['base_vlm']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    vla = QwenACT.from_pretrained(
        checkpoint_pt,
        base_vlm = vla_cfg["base_vlm"],
        freeze_weights=not load_for_training, 
        norm_stats=norm_stats,
        **kwargs,
    )

    return vla
