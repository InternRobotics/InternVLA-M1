"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast
import re
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

from llavavla.model.action_model.action_model import ActionModel
from llavavla.model.action_model.models import DiT
from llavavla.dataloader.promt_builder import QwenVLPromptHelper
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


class CogACT_Qwen(nn.Module):
    def __init__(
        self,
        vlm:_QWen_VL_Interface, # 这是不好的实现， 一定不能是互相依赖
        action_model_type: str = 'DiT-B',
        token_size: int = 2048,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.action_model = ActionModel(model_type = action_model_type,  # TODO @Jinhui 应该写到 get_action_model()
                                            action_hidden_dim = token_size, 
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size)
        self.vlm = vlm

        # print("Freezing QWEN-VL model parameters") # @Jinhui Bug, 应该统一位置来 设置这个
        # for param in self.vlm.parameters():
        #     param.requires_grad = False
        
        self.qwen_processor = vlm.processor # 
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
        #     self.all_module_keys = ['action_model', 'ema_diffusion']
        # else:
        #     self.all_module_keys = ['action_model']
        
        # TODO check 为什么改文件model 名字么？ 
        # for module_keys in self.vlm.all_module_keys: #@Jinhui checking
        #     self.all_module_keys.append("vlm." + module_keys)
        self.all_module_keys = auto_get_module_keys(self)
        # Diffusion head is always trainable
        # self._trainable_module_keys = ['action_model'] # 应该放到一个集中的地方
        self.norm_stats = norm_stats

        # 这里应该和 data loader tranfomation 对齐的
        self.promptHelper = QwenVLPromptHelper(processor=self.vlm.processor, system_prompt="You are a helpful assistant")

    @property
    def trainable_module_keys(self) -> List[str]:
        # keys = []
        # for module_keys in self.vlm.trainable_module_keys:
        #     keys.append("vlm." + module_keys)
        # keys += self._trainable_module_keys
        # TODO check, 原版返回的死 vlm.model, 新的实现是vlm --> 看一下保存逻辑是否发上变化
        keys = auto_get_trainable_modules(self, max_depth=1)# auto 去判断哪些module是trainable的
        return keys
    
    
    @staticmethod
    def align_module_names(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aligns module names in the state dict to match the current model's module names."""
        # TODO 是一个临时方法，后期要考虑align save 的逻辑，让他能够直接load 进来
        """
        Align the keys in the state_dict to match the expected model structure.

        Args:
            state_dict (dict): Original model state_dict with misaligned keys.

        Returns:
            dict: Aligned state_dict.
        """
        aligned_dict = {}

        # Step 1: 处理 `model`，重命名为 `vlm.model`
        if "model" in state_dict:
            for key, value in state_dict["model"].items():
                aligned_dict[f"vlm.model.{key}"] = value  # 添加前缀

        # Step 2: 删除 `visual` 和 `lm_head`（假设它们为空）
        if "visual" in state_dict:
            if state_dict["visual"]:  # 如果 visual 不是空的，可能需要处理
                print("Warning: 'visual' is expected to be empty but contains data.")
            del state_dict["visual"]

        if "lm_head" in state_dict:
            if state_dict["lm_head"]:  # 如果 lm_head 不是空的，可能需要处理
                print("Warning: 'lm_head' is expected to be empty but contains data.")
            del state_dict["lm_head"]

        return aligned_dict

    def freeze_backbones(self, stage):
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
        patterns = [] #TODO 时候要参数化
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
                
            return frozen_keys
        
        # 对整个模块（self）递归检查。注意，根目录通常为空字符串，这里不冻结根模块
        frozen = []
        for name, child in self.named_children():
            full_name = name  # 顶层模块名称
            frozen.extend(freeze_module(child, full_name))
        return frozen


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks = None,
        **kwargs,  # 👈 敏捷代码的灵活性， 允许任何形式的传参数
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # @Jinhui TBD TODO 
        # pixel_values = pixel_values["pixel_values"] # labeles = pixel_values["labels"]
        # dist.barrier()
        output: CausalLMOutputWithPast = self.vlm( #system 
            input_ids=input_ids,
            image_grid_thw=kwargs.get("image_grid_thw", None),  # 可能是一个图像网格
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels, # label 全是 -100 @Jinhui TODO Bug here, input 也全是一样的
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        vlm_loss = output.loss # @Jinhui TODO 这里是可以study 的地方， 是否 training lang
        # extract the last hidden state and the learnable EOS token feature
        last_hidden = output.hidden_states[-1] # B,len,D
        cognition_features = self._get_cognition_features(last_hidden, input_ids, attention_mask=attention_mask)


        actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]

        # Action model forward and compute loss
        loss = self.action_model.loss(actions_repeated, cognition_features_repeated) # TODO loss 应该放到另一个函数
        return loss, output

    def _get_cognition_features_old(self, last_hidden, input_ids) -> torch.Tensor:
        # last_hidden = outputs.hidden_states[-1] # B,len,D

        # extract the visual token number #@Jinhui 他要拿一个token 去做 下游 TODO 展示不要用关
        # @Discussion 这个位置需要讨论 --> 其实可以通过检查 visual token 的 indexs 来
        image_token_id, vido_token_id, pad_token_id = 151655, 151654, 151643
        # 1️⃣ 找到第一个匹配的 image_token_id 的索引
        match_indices = (input_ids == image_token_id).float().argmax(dim=1)  # [B]

        # 2️⃣ 确保索引不会越界（前一个 token 不能是 -1）# 不对的，因为没有causal attention 还没看到 image token， 应该是next token
        prev_indices = torch.clamp(match_indices - 1, min=0)  # [B] 

        # 3️⃣ 扩展索引，使其匹配 last_hidden 维度
        expanded_indices = prev_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  # [B, D]

        # 4️⃣ 提取 cognition_features (前一个 token 的隐藏状态)
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        cognition_features = cognition_features.to(torch.bfloat16)
        
        return cognition_features
    def _get_cognition_features(self, last_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        提取每个样本中 cognition_token（🔍）位置的 hidden state，作为 cognition feature。

        Args:
            last_hidden: Tensor, shape [B, T, D]
            input_ids: Tensor, shape [B, T]
            attention_mask: Tensor, shape [B, T]

        Returns:
            cognition_features: Tensor, shape [B, 1, D]
        """
        cognition_token_id = self.promptHelper.cognition_token_ids

        B, T, D = last_hidden.shape

        # 找到每个样本中 cognition_token_id 出现的位置
        cognition_indices = (input_ids == cognition_token_id).int()  # [B, T]，为1的位置是 cognition
        has_cognition_token = cognition_indices.any(dim=1)

        if not has_cognition_token.all():
            raise ValueError("Not all samples contain the cognition token 🔍")

        # 获取每行 cognition_token 的位置（只取第一个匹配的 token index）
        cognition_pos = cognition_indices.argmax(dim=1)  # [B]
        
        # 构建用于 gather 的索引
        gather_index = cognition_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, D)  # [B, 1, D]
        cognition_features = last_hidden.gather(dim=1, index=gather_index)  # [B, 1, D]

        return cognition_features

    
    def get_fsdp_wrapping_policy(self) -> Callable: 
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        # vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        # llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={DiT},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vlm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy, # @Jinhui TODO Checking这个应该保留么？
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        base_vlm: str,
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        **kwargs,
    ) -> CogACT_Qwen:

        # Load VLM backbone, borrowed from PrismaticVLM

        # 仅加载模型配置，而不加载权重
        base_vlm = "/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct" # Jinhui Can be a bug TODO 需要调整training 和测试的工作目录
        # 只初始化模型结构，不加载参数
        vlm = _QWen_VL_Interface(model_id=base_vlm, load_for_training=False)
   
        

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu") #["model"]
        # qwen_state_dict = QwenACT_state_dict["model"]
        # Initialize CogACT
        cogact = CogACT_Qwen(vlm,
                        token_size = vlm.model.config.hidden_size, # 这里的 model 的分成很奇怪
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        )
        cogact.qwen_processor = vlm.processor # @Jinhui TODO 为什么不放到 inital
        # Load VLM from Checkpoint # TODO 后期要对齐 save 的逻辑
        # qwen_state_dict = CogACT_Qwen.align_module_names(model_state_dict)
        # assert CogACT_Qwen.check_unexpected_keys(qwen_state_dict,cogact),  "check_point 中有参数没有被 load"
        # cogact.vlm.load_state_dict(model_state_dict["vlm"])  # @Jinhui 任务整个model一起，逻辑写到里面里面
 
        # 自动加载 checkpoint 中的权重到对应模块 #@Jinhui TODO 怎么保证全部trainable参数被save 了？
        # 遍历 checkpoint 中的每个键，若 cogact 有相应属性且该属性支持 load_state_dict，则加载权重
        model_keys = cogact.state_dict().keys()

        for key, state in model_state_dict.items():
            if key in model_keys:
                try:
                    cogact.state_dict()[key].copy_(state)
                    # overwatch.info(f"✅ Successfully loaded weights for key '{key}'")
                except Exception as e:
                    overwatch.warning(f"⚠️ Failed to copy weight for key '{key}': {e}")
            else:
                overwatch.warning(f"⚠️ Unknown key '{key}' in checkpoint. Ignoring.")
        
        # TODO 需要一个逻辑检查是否全部参数就load 好了？ --> 不直接 cogact.load 的原因
        # TODO Jinhui 很必要有个检查流程，保证 all tranable 参数被 save 了, all tranable  被load 了
        return cogact

    @torch.inference_mode()
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
        # qwen_processor = self.qwen_processor
        #@Jinhui 想办法怎么和dataloader 对齐？
        # Build VLA Prompt
        # prompt_builder = self.vlm.get_prompt_builder()
        # prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")

        # TODO 为了保证测试一致性心里应该是 用func, 但是如果预期这里是 template-free, 就应该是这样的
        # minin version of QwenPromptBuilder --> @Jinhui TODO 后续可以实现到 QwenPromptBuilder 中进行对话管理
        # 拿到 对话的 text 文本 
        # conversation = [
        #     {"role": "user", "content": [{"type": "text", "text":f"What action should the robot take to {instruction.lower()}?"}, {"image": None}]},
        #     ]
                
        # @之后写入模型内部， 变成私有化方法
        # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0]) # B 要被去掉？
        # image resize to 224*224
        img = image.resize((224, 224))  # Resize to Qwen-VL default input size
        lang = instruction.lower() + "🔍" #cognition token
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img}, # rgb
                {"type": "text", "text": lang},
            ],
        },]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )



        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.half_precision_dtype # 为什么用半精度推理
        
        # add by Jinhui
        device = self.vlm.device  # 确保所有输入和模型都在同一设备上
        dtype = next(self.vlm.parameters()).dtype  # 获取模型的默认数据类型（float16 或 float32）

        
        # 确保所有张量都移动到 GPU，并转换为正确的数据类型
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                if key in ["input_ids", "attention_mask"]:  # 保证 input_ids 和 attention_mask 仍然是 long 类型
                    inputs[key] = inputs[key].to(device, dtype=torch.long)
                elif key in ["image_grid_thw"]:
                    continue
                else:  # 其他 Tensor（如 pixel_values）转换成模型 dtype（float16 或 float32）
                    inputs[key] = inputs[key].to(device, dtype=dtype)


        # end add by Jinhui
        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True, 
                return_dict=True,
            ) # generation 拿不到前面token 的信息，考虑使用 forward?

            # Jinhui see text # outputs.sequences.shape: B, len with prefix
            # outputs.input_ids = outputs.sequences # 为了和 input dict 保持一致， 方便调用 self._get_cognition_features# 还真不太一样，因为generation的逻辑和 forward不一样
            # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)]
            # output_text = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print("output:\n",output_text[0])
            # fmt: on
            # 我们training的时候是 image 不固定在最前面没，是没办法只max_new = 1 的
        # Extract cognition feature
        # outputs.hidden_states = list = next tokens 
        # be careful about the where the cognition_features comes from would align with training
        # cognition_features = output.hidden_states[0][-1][:,-1,:]  # nexx tokens, layers, B, len, D #@Jinhui to Think 这里为什么每一个 next token 都记录了 全部都 hidden? 不是的，只有第一个会
        last_hidden_states = outputs.hidden_states[-1] #torch.Size([1, 428, 2048]) # last hidden_states for next token generation
        cognition_features = self._get_cognition_features(last_hidden_states, inputs.input_ids, attention_mask=inputs.attention_mask) # [B,1,D] TODO carefully checking with align training

        assert (cognition_features.shape[0], cognition_features.shape[1], cognition_features.shape[-1]) == (1, 1,2048), "Batch size must be 1 for action prediction"
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

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
                                                                device=cognition_features.device,
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
                                                                    device=cognition_features.device
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

    @torch.inference_mode()
    def predict_action_batch(
        self, image: List[Image], 
        instruction: List[str], 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference in batch; maps input image and task instruction to continuous action.
        This function is used for batch inference in the simulators.
        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        
        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction[id].lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)), dim=0
            ) # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch :]

        cumulative_sum = attention_mask.cumsum(dim=1)  
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1) #[B, D]

        assert (cognition_features.shape[0], cognition_features.shape[1]) == (B, 4096), "Batch size must be B for action prediction"
        using_cfg = cfg_scale > 1.0


        model_dtype = next(self.action_model.net.parameters()).dtype

        B = cognition_features.shape[0]
        
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,#False, try to set True 
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0)
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,#False, try to set True 
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

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
    
    @staticmethod
    def check_unexpected_keys(state_dict, model):
        """
        检查 state_dict 是否包含 unexpected_keys。
        
        Args:
            state_dict (dict): 需要加载的模型权重。
            model (torch.nn.Module): 目标模型（如 cogact）。
        
        Returns:
            bool: 如果没有 unexpected_keys，则返回 True；否则返回 False 并报错。
        """
        # 获取模型已有的参数
        model_keys = set(model.state_dict().keys())

        # 获取 state_dict 里的参数
        state_dict_keys = set(state_dict.keys())

        # 计算 unexpected_keys
        unexpected_keys = state_dict_keys - model_keys

        # 如果发现 unexpected_keys，报错或者警告
        if unexpected_keys:
            print(f"❌ Unexpected keys found in state_dict: {unexpected_keys}")
            return False  # 发现不匹配的 key，返回 False
        
        print("✅ No unexpected keys found.")
        return True  # 所有 key 都匹配，返回 True
