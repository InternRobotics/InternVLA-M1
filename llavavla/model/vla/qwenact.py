"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path, os
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch, json
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast
import re
from prismatic.overwatch import initialize_overwatch

from llavavla.model.action_model.action_model import ActionModel
from llavavla.model.action_model.models import DiT
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


# get QWen2.5
from llavavla.model.vlm import _QWen_VL_Interface #不应该强依赖于这个，应该是一个接口类，而不是一个具体的类, TODO 不要实现 hard 接口类， 使用 **kwargs
from llavavla.model.tools import auto_get_module_keys, auto_get_trainable_modules
from llavavla.model.vlm.QWen2_5 import get_qwen2_5_interface
from llavavla.model.projector.QDormer import get_layerwise_qformer

class QwenQFormerDiT(nn.Module):
    def __init__(
        self,
        qwen_model_name:str = './playground/Pretrained_models/Qwen2.5-VL-3B-Instruct', # 这是不好的实现， 一定不能是互相依赖
        action_model_type: str = 'DiT-B', 
        vl_token_dim: int = 2048,
        action_hidden_dim: int = 768,  # @Jinhui # 这个 应该是和DiT-B
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # TODO 全部转 全局config, 要面向对象编程
        self.qwen_vl_interface = get_qwen2_5_interface(qwen_model_name) 
        self.layer_qformer = get_layerwise_qformer(input_hidden_dim=vl_token_dim, output_hidden_dim=action_hidden_dim) # @Jinhui 需要逻辑从QWen 中对齐 hidden
        self.action_model = ActionModel(model_type = action_model_type,  # TODO @Jinhui 应该写到 get_action_model()
                                            action_hidden_dim = action_hidden_dim, # 这些参数关系要 TODO集中 设置到config
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size) # 也应该用 函数封装
        # TODO ActionModel 需要和qformer 一起设计

        # self.qwen_processor = vlm.processor # 要面向对象编程， 不要 属性外泄
        # 这些是 action chunck 的参数
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size

        self.all_module_keys = auto_get_module_keys(self) #  TODO 这个是trainer的 funx
        self.norm_stats = norm_stats # 这个是 inference 时候用到的， 不应该是放到这个位置？


    @property
    def trainable_module_keys(self) -> List[str]:

        # TODO check, 原版返回的死 vlm.model, 新的实现是vlm --> 看一下保存逻辑是否发上变化
        keys = auto_get_trainable_modules(self, max_depth=1)# auto 去判断哪些module是trainable的
        return keys
    

    def forward( # TODO 需要将 loss 计算分离出来
        self, # 只面对最原始的 data exmaples, 为了可读性，这里还是要写成显示的参数
        examples: List[dict] = None,  # 这里的 examples 是指原始的输入数据
        repeated_diffusion_steps: int = 4,
        **kwargs,  # 👈 敏捷代码的灵活性， 允许任何形式的传参数
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # @Jinhui TBD TODO 
        # pixel_values = pixel_values["pixel_values"] # labeles = pixel_values["labels"]
        # dist.barrier()

        # images: Optional[torch.FloatTensor] = None,
        # instructions: Optional[List] = None,
        # actions: Optional[torch.FloatTensor] = None,
        images = [example["image"] for example in examples]  # [B, H, W, C]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]
        
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions = instructions) # @Jinhui TODO add instruction to qwenvl inputs
        with torch.autocast("cuda", dtype=torch.float16):

            qwenvl_outputs = self.qwen_vl_interface( # 都是local的参数变化， 不要写到config, 但是为了保持可复现，应该有个默认的 yaml
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values, # [512, 1176] 石斛没有 B,  
                image_grid_thw =qwen_inputs.image_grid_thw, # 2* [1,16,16] --> 512 = 16*16*2, 1176 = (224/16)^2 * 3 * 2 @JinhuiYE TODO 这个需要找Qwen 的官方文档验证
                labels= qwen_inputs.input_ids.clone(),
                # use_cache=use_cache,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                # past_key_values=past_key_values,
                # **kwargs
                )
        
        vlm_loss = qwenvl_outputs.loss # @Jinhui TODO 这里是可以study 的地方， 是否 training lang
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[-6:]) # [B, 64, D_action]
            
        actions = torch.stack([torch.tensor(a) for a in actions], dim=0).to(action_latent_feature.device)  # [B, chunk, 7] @Jinhui TODO to tensor 的逻辑可以放到 transform 里面
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        action_latent_feature = action_latent_feature.repeat(repeated_diffusion_steps, 1, 1)  # [repeated_diffusion_steps*B, T, D_action]
        # Action model forward and compute loss # 这里功能有点 越俎代庖 TODO 将loss 集中到 main module中统一处理
        action_loss = self.action_model.loss(actions_repeated, action_latent_feature) # TODO loss 应该放到另一个函数
        return action_loss, qwenvl_outputs

    # @torch.inference_mode() # @Jinhui DEBUG 临时取消
    def predict_action(
        self, image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """

        # @之后写入模型内部， 变成私有化方法
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]) # B 要被去掉？
        # image resize to 224*224
        img = image.resize((224, 224))  # Resize to Qwen-VL default input size
        lang = instruction.lower()
        # messages = [
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "image", "image": img}, # 224*224 rgb
        #         {"type": "text", "text": lang},
        #     ],
        # },]
        # text = self.qwen_vl_interface.processor.apply_chat_template( # TODO check if align with training
        #     messages, tokenize=False, add_generation_prompt=False
        # )
        # image_inputs, video_inputs = process_vision_info(messages) # image_inputs = list of PIL
        # inputs = self.qwen_vl_interface.processor(
        #     text=text,
        #     images=image_inputs, # inputs["pixel_values"].shape [256, 1176]
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )

        inferface_inputs =  self.qwen_vl_interface.build_qwenvl_inputs(images=[[img]], instructions = [lang]) # @Jinhui TODO add instruction to qwenvl inputs
        qwen_inputs = inferface_inputs
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`

        
        # add by Jinhui

        # end add by Jinhui
        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fmt: off
            qwenvl_outputs = self.qwen_vl_interface( # TODO 这里之后要用generation func
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values, # [512, 1176] 石斛没有 B,  
                image_grid_thw =qwen_inputs.image_grid_thw, # 2* [1,16,16] --> 512 = 16*16*2, 1176 = (224/16)^2 * 3 * 2 @JinhuiYE TODO 这个需要找Qwen 的官方文档验证
                labels= qwen_inputs.input_ids.clone(),
                output_hidden_states=True, 
                return_dict=True,
            ) # generation 拿不到前面token 的信息，考虑使用 forward?

        with torch.autocast("cuda", dtype=torch.bfloat16):
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[-6:]) # [B, 64, D_action]
            
            # Jinhui see text # outputs.sequences.shape: B, len with prefix
            # outputs.input_ids = outputs.sequences # 为了和 input dict 保持一致， 方便调用 self._get_cognition_features# 还真不太一样，因为generation的逻辑和 forward不一样
            # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)]
            # output_text = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print("output:\n",output_text[0])
            # fmt: on
            # 我们training的时候是 image 不固定在最前面没，是没办法只max_new = 1 的

        
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype
        B = action_latent_feature.shape[0]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=action_latent_feature.device).to(model_dtype)  #[B, T, D]

        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition # [64, 768]
            uncondition_shape = uncondition.shape
            uncondition = uncondition.unsqueeze(0)  #[1, 64, D]
            uncondition = uncondition.expand(B, uncondition_shape[0], uncondition_shape[1]) #[B, n_qformer_token, D] # 
            z = torch.cat([action_latent_feature, uncondition], 0) # [2, 64, 768] TODO check 看看 training的时候是剁手
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=action_latent_feature)
            sample_fn = self.action_model.net.forward
        
        if os.environ.get("DEBUG"):
            print(z .shape)
        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=action_latent_feature.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=action_latent_feature.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions


    def freeze_backbones(self, stage): # TODO freeze 是架构的一部分， 得全局控制， 要写个提示 用户在这里 做声明
        # self.vlm.freeze_backbones(stage)

        """
        根据给定的正则模式列表冻结模块。
        如果某个模块的名称匹配（例如公共前缀匹配），则冻结该模块下所有参数（不再递归冻结子模块），
        并返回冻结模块名称的有序浅层列表。
        
        参数：
            patterns: 正则表达式字符串列表，模块名称只要匹配其中一个模式，就会被冻结。
            
        返回：
            一个冻结模块名称的列表（按递归顺序）。
        """
        # r"^vlm\.model\.visual", r"^action_model"
        patterns = ["qwen_vl_interface"] #TODO 时候要参数化
        def freeze_module(module: nn.Module, prefix: str) -> List[str]:
            # 如果当前模块名称匹配任一模式，则冻结当前模块，不再递归子模块
            if any(re.match(pattern, prefix) for pattern in patterns if prefix):
                for param in module.parameters(recurse=False):
                    param.requires_grad = False
                return [prefix]
            # 否则，递归遍历子模块
            frozen_keys = []
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                frozen_keys.extend(freeze_module(child, full_name))
            
            self.qwen_vl_interface.eval()
            # 在模型初始化或加载之后调用
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad = False
            # freeze the qwen_vl_interface
            
            return frozen_keys
            

        # 对整个模块（self）递归检查。注意，根目录通常为空字符串，这里不冻结根模块
        frozen = []
        for name, child in self.named_children():
            full_name = name  # 顶层模块名称
            frozen.extend(freeze_module(child, full_name))
        dist.barrier()  # 确保所有进程都完成冻结操作
        return frozen


    @classmethod
    def from_pretrained( # @Jinhui TODO 这里要写如何resume checkpoints
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> None:
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint) # 读取 config 和 norm_stats
        # Initialize CogACT
        qwenQFormerACT = build_model_framework(model_config) 
        # set for action un-norm
        qwenQFormerACT.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu") #["model"]
        
        model_keys = set(qwenQFormerACT.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())

        # ✅ 1. 加载匹配的权重
        for key in checkpoint_keys:
            if key in model_keys:
                try:
                    qwenQFormerACT.state_dict()[key].copy_(model_state_dict[key])
                    # overwatch.info(f"✅ Loaded: {key}")
                except Exception as e:
                    overwatch.warning(f"⚠️ Failed to copy weight for key '{key}': {e}")
            else:
                overwatch.warning(f"⚠️ Checkpoint has unknown key '{key}' (not in model). Ignoring.")

        # ✅ 2. 反向检查：模型中有但 checkpoint 中缺失的
        missing_keys = model_keys - checkpoint_keys # TODO 这里之后要考虑 nontrainable params --> 我觉得没必要省存储空间
        for key in sorted(missing_keys):
                overwatch.warning(f"⚠️ Model expects key '{key}' but it's missing in checkpoint.")
                
        return qwenQFormerACT
    
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

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

# TODO 写一个build model 函数

def build_model_framework(model_config: dict = {}) -> QwenQFormerDiT:
    # TODO  实现和config 对应的 load 逻辑

    model = QwenQFormerDiT(
    qwen_model_name='/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct',
    action_model_type='DiT-B',
    vl_token_dim=2048,
    action_dim=7,
    future_action_window_size=15,
    past_action_window_size=0,
    # use_ema=False,
    )
        
    return model


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
            vla_cfg = json.load(f)["vla"]
            # model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])() #@TODO check 我觉得其实不重要，

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    return vla_cfg, norm_stats

def load_from_pretrained(pretrained_checkpoint):
    """Load a pretrained QwenQFormerDiT model from a checkpoint."""



    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =


    # TODO 这里应该是从config中加载
    model = QwenQFormerDiT.from_pretrained(
        pretrained_checkpoint=pretrained_checkpoint)
    return model

if __name__ == "__main__":

    # 模型参数
    import debugpy
    debugpy.listen(("0.0.0.0", 5878))
    print("🔍 Rank 0 waiting for debugger attach on port 5878...")
    debugpy.wait_for_client()

    model_frameword = build_model_framework()
    pass