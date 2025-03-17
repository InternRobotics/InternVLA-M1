import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from prismatic.models.backbones.llm.prompting import PromptBuilder
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder


from typing import Optional, Sequence, Type

import torch
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    wrap,
)

# Registry =>> Support Qwen-2.5 Models (from HF Transformers)
# fmt: off
QWEN25_MODELS = {
    # === Pure Qwen2.5 (non-instruct/chat-tuned) Models ===
    "qwen25-0_5b-extra": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-0.5B"
    },
    "qwen25-0_5b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-0.5B"
    },
    "qwen25-1_5b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-1.5B"
    },
    "qwen25-3b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-3B"
    },
    "qwen25-7b-pure": {
        "llm_family": "qwen2.5", "llm_cls": AutoModelForCausalLM, "hf_hub_path": "Qwen/Qwen2.5-7B"
    },

}
# fmt: on


# add by jinhui
from llavavla.model.vlm.qwen_prompter import QwenPromptBuilder
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from torch.nn import Linear, Embedding

# 从 transformers 导入 Qwen2.5 相关模块
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,           # LLM Decoder Layer
    Qwen2MLP,                    # MLP Module in LLM
    Qwen2Attention,               # LLM Attention Layer
    Qwen2RMSNorm,                 # Normalization Layer
    Qwen2RotaryEmbedding          # Rotary Position Embedding
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLSdpaAttention,
    Qwen2_5_VLMLP,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2RMSNorm,
    Qwen2_5_VLModel,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLSdpaAttention,
    Qwen2MLP,
    Qwen2_5_VLRotaryEmbedding,
)

class _QWen_VL_Interface(VLM): #TODO @Jinhui 后期不能再向 PrismaticVLM 对齐， 思考更加flexible做法， --》 接口class的实现
    """
    这是对 Qwen2_5_VLForConditionalGeneration 的简单封装，使其在接口层面上更接近 PrismaticVLM，
    例如能够返回类似 CausalLMOutputWithPast 的结构，并拥有类似 vision_backbone、llm_backbone 等属性。
    """

    def __init__(
        self,
        model_id: str,
        vision_backbone=None,
        llm_backbone=None,
        enable_mixed_precision_training: bool = True, #@Jinhui Check
        **kwargs
    ):  
        # QWen 原生模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,  torch_dtype="auto", device_map="auto")
        # 伪造子模块引用，以便 CogACT 里还能访问 想办法拿到
        
        vision_backbone = model.visual
        # 为了对齐 self.llm_backbone # 需要这样干的原因是 VLM_base 写的不好，做了强制假设
        llm_backbone = model.model #
        processor = AutoProcessor.from_pretrained(model_id)
        llm_backbone.llm = llm_backbone.config
        llm_backbone.llm.generation_config  =  llm_backbone.generation_config

        super().__init__(
            "prismatic", #这个其实可以rm
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )
        # QWen 原生模型
        self.model = model
        # 将整个模型转换为所需的精度类型。
        self.model.to(torch.float32)
        # 伪造子模块引用，以便 CogACT 里还能访问 想办法拿到
        # self.projector = self.model.lm_head #
        self.vision_backbone = self.model.visual
        # 如果需要在 forward 过程中做自动混合精度
        self.enable_mixed_precision_training = enable_mixed_precision_training
        
        # 处理图文输入
        self.processor = processor
        # 仅做示例：给出与 PrismaticVLM 接口对应的一些占位属性
        self.trainable_module_keys = ["visual", "model", "lm_head"]
        self.all_module_keys = ["visual", "model", "lm_head"]
        
        # 对齐 Keys
        self.arch_specifier = None #其实是在  self.vision_backbone 内部

        self.llm_backbone.transformer_layer_cls = Qwen2DecoderLayer
  


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,  # 需要 hidden_states
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        调用 QWen2.5 的 forward，输出类似 CausalLMOutputWithPast 的结构，供 CogACT 使用。
        """
        with torch.autocast("cuda", enabled=self.enable_mixed_precision_training, dtype=torch.float16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        
        # QWen2.5 默认返回的可能是 QWenXXXModelOutput；这里示例将它包装成一个 CausalLMOutputWithPast
        # 仅做示例：如果 QWen2.5 返回的字段名不同，你需要做对应处理
        dummy_output = CausalLMOutputWithPast(
            loss=outputs.loss if hasattr(outputs, "loss") else None,
            logits=outputs.logits if hasattr(outputs, "logits") else None,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )
        return dummy_output

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        max_new_tokens: int = 128,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs
    ):
        """
        让 Qwen2.5 和 GPT 类似地进行 generate 生成。
        某些参数可能在 Qwen2.5 中用法不同，需要结合官方文档调整。
        """
        with torch.autocast("cuda", enabled=self.enable_mixed_precision_training, dtype=torch.float16):
            generation_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs
            )
        return generation_output

    def freeze_backbones(self, stage: str):
        """
        原本在 PrismaticVLM 里可能会冻住 vision_backbone 或 llm_backbone 的权重。
        如果 Qwen2.5 也需要分阶段冻结，可以在这里自定义逻辑。
        """
        # 如果不需要就留空
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        enable_mixed_precision_training: bool = True,
        **kwargs
    ):
        """
        类似 PrismaticVLM 的 from_pretrained，用于直接加载 Qwen2.5。
        """
        return cls(model_id, enable_mixed_precision_training, **kwargs)

    ## Padding Methods
    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" | "vla-train" | "vla-full-train" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            # self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"finetune", "vla-train"}:
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            # self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"full-finetune", "vla-full-train"}:
            # self.vision_backbone.dtype = torch.float32 #直接修改dtype属性可能会导致错误
            for param in self.vision_backbone.parameters():
                if param.dtype != torch.float32:
                    param.data = param.data.float()
            
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
        

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.__class__.__name__}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.__class__.__name__}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage in {"last-layer-finetune", "vla-last-layer-train"}:
            self.vision_backbone.requires_grad_(False)
            # self.projector.requires_grad_(False)         
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[Frozen]                    🥶   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen]                    🥶   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        elif stage in {"vla-sandwich-train"}:
            # self.vision_backbone.dtype = torch.float32
            self.vision_backbone.to(torch.float32)
            self.vision_backbone.requires_grad_(True)
            # self.projector.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)

            # Unfreeze final LLM layer
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            # fmt: off
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[Frozen, except last layer] 🥶🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)  # noqa: E501
            overwatch.info(f"[TRAINABLE]                 🔥   =>> Projector `{self.arch_specifier}`", ctx_level=1)
            # fmt: on

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

        overwatch.debug("##################################################")
        overwatch.debug("#####      Trainable Network Parameters:     #####")
        overwatch.debug("##################################################")
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            # self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            # self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Return an FSDP wrapping policy that combines size-based and transformer-based auto-wrapping policies.
        """

        # 1️⃣ Transformer-based Auto-Wrap Policy
        transformer_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2_5_VLDecoderLayer,  # LLM 解码层
                Qwen2_5_VLVisionBlock,   # 视觉 Transformer Block
            }
        )

        # 2️⃣ Size-based Auto-Wrap Policy (用于超大层)
        size_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=1e8  # 1 亿参数以上的层进行 FSDP 包装
        )

        # 3️⃣ 组合策略：优先匹配 Transformer 层，否则基于参数数量包装
        def combined_policy(module, recurse, nonwrapped_numel):
            return transformer_policy(module, recurse, nonwrapped_numel) or size_policy(module, recurse, nonwrapped_numel)

        return combined_policy

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return QwenPromptBuilder
    
    @property
    def transformer_layer_cls(self) -> Type[torch.nn.Module]:
        return Qwen2DecoderLayer
    
    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[torch.nn.Module]:
        # TODO not sure that this works
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)

    
def get_qwen2_5_vl(model_id="playground/Pretrained_models/Qwen2.5-VL-7B-Instruct"):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained( # 里面有奇怪的bug, 来自cookbooks
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


if __name__ == "__main__":
    model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    qwen_vl = _QWen_VL_Interface(model_id)
    pass