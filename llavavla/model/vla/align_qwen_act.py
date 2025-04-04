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

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

from llavavla.model.action_model.action_model import ActionModel
from llavavla.model.action_model.models import DiT
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


# get QWen2.5
from llavavla.model.vlm import _QWen_VL_Interface #不应该强依赖于这个，应该是一个接口类，而不是一个具体的类

class QwenACT(nn.Module):
    def __init__(
        self,
        vlm:_QWen_VL_Interface,
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

        # self.projection = nn.Linear(token_size, 4096)# token_size --> token_size*2
        self.image2cognition_projector = CNNModelWithMaxPool(hidden_dim=token_size, fc_out_features=4096)
        # 4096 in cogact
        self.action_model = ActionModel(model_type = action_model_type,  # TODO @Jinhui 应该写到 get_action_model()
                                            token_size = 4096,  #TODO 应该设置为 config 管理
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size)
        
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion']
        else:
            self.all_module_keys = ['action_model']
        self.all_module_keys.append("image2cognition_projector")
        # TODO check 为什么改文件model 名字么？ 
        for module_keys in self.vlm.all_module_keys: #@Jinhui checking
            self.all_module_keys.append("vlm." + module_keys)

        # Diffusion head is always trainable
        # self._trainable_module_keys = ['action_model']
        self.norm_stats = norm_stats


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
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        # @Jinhui TBD TODO 
        # pixel_values = pixel_values["pixel_values"] # labeles = pixel_values["labels"]
        output: CausalLMOutputWithPast = self.vlm( #system 2
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        vlm_loss = output.loss # TODO 需要修改dataloader 这里应该增加 预测 grounding的 loss
        # extract the last hidden state and the learnable EOS token feature
        last_hidden = output.hidden_states[-1] # B,len,D
        cognition_features = self._get_cognition_features(last_hidden, input_ids, attention_mask=attention_mask)


        actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]
        # align qwen feature and DiT features
        # cognition_features_repeated = self.projection(cognition_features_repeated)

        # Action model forward and compute loss
        loss = self.action_model.loss(actions_repeated, cognition_features_repeated)
        return loss, output

    def _get_cognition_features(self, last_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        提取每个样本中最后一个有效 token 的 hidden state，作为 cognition feature。

        Args:
            last_hidden: Tensor, shape [B, T, D]
            input_ids: Tensor, shape [B, T]
            attention_mask: Tensor, shape [B, T]

        Returns:
            cognition_features: Tensor, shape [B, 1, D]
        """

        return self.image2cognition_projector(last_hidden, input_ids, attention_mask=attention_mask)
        # extract the visual token number #@Jinhui 他要拿一个token 去做 下游 TODO 展示不要用关
        # @Discussion 这个位置需要讨论 --> 其实可以通过检查 visual token 的 indexs 来, 
        image_token_id, vido_token_id, pad_token_id = 151655, 151654, 151643
        # assert TODO 需要假设 文本中不带 action 的， 不然需要新的 id 来标记位置
        # 获取 batch size 和 hidden dim
        B, T, D = last_hidden.shape

        # 计算每个样本最后一个有效 token 的索引位置
        # 方法：cumsum + == max
        cumsum = attention_mask.cumsum(dim=1)  # shape: [B, T]
        max_indices = (cumsum == cumsum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  # shape: [B]
        # train --> 198, test --> 198 ✅， 但是这里我想可能不应该是要同一个
        # 构建 gather 索引：形状 [B, 1, D]
        expanded_indices = max_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, D)  # [B, 1, D]
        # index = ... 198, 151644, 77091, 198, 76478, 114218, 112578]
        # gather 取出每个样本的最后有效 token 的 hidden state
        cognition_features = last_hidden.gather(dim=1, index=expanded_indices)  # [B, 1, D]

        return cognition_features

    def get_fsdp_wrapping_policy(self) -> Callable: 
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        # vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        # llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={DiT, CNNModelWithMaxPool},
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
    def from_pretrained( # TODO 看看怎么和 initial from QWen 合并？
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
    ) -> QwenACT:

        # Load VLM backbone, borrowed from PrismaticVLM

        # 仅加载模型配置，而不加载权重
        base_vlm = "/fs-computility/efm/yejinhui/Projects/CogACT/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct" #TODO 需要调整training 和测试的工作目录
        # config = AutoConfig.from_pretrained(base_vlm)
        # vlm = Qwen2_5_VLForConditionalGeneration(config)  # 只初始化模型结构，不加载参数
        vlm = _QWen_VL_Interface(model_id=base_vlm, load_for_training=False)
        # 加载 Processor（它没有权重，不受影响）
        #TODO 需要更好的逻辑
        qwen_processor = AutoProcessor.from_pretrained(base_vlm)

        # put it to interfance:

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        # qwen_state_dict = QwenACT_state_dict["model"]
        # Initialize CogACT
        qwenact = QwenACT(vlm, #@Jinhui 是传入逻辑还是好？
                        token_size = vlm.model.config.hidden_size, # 这里的 model 的分成很奇怪
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        )
        qwenact.qwen_processor = qwen_processor
        # Load VLM from Checkpoint 
        # qwen_state_dict = CogACT_Qwen.align_module_names(model_state_dict)
        # assert CogACT_Qwen.check_unexpected_keys(qwen_state_dict,cogact),  "check_point 中有参数没有被 load"
        # === Load known modules ===
        standard_modules = { #TODO 后续考虑 auto 判断Key 和 load # TODO 后期要对齐 save 的逻辑
            "model": qwenact.vlm.model,
            "image2cognition_projector": qwenact.image2cognition_projector,
            "action_model": qwenact.action_model,
            "ema_diffusion": qwenact.ema_diffusion if use_ema else None
        }

        # === Attempt to load state_dicts ===
        for key, module in standard_modules.items():
            if key in model_state_dict and module is not None:
                try:
                    module.load_state_dict(model_state_dict[key])
                    overwatch.info(f"✅ Loaded weights for `{key}`")
                except Exception as e:
                    overwatch.warning(f"⚠️ Failed to load `{key}`: {e}")

        # === Auto-load additional submodules if possible ===
        loaded_keys = set(standard_modules.keys())
        remaining_keys = set(model_state_dict.keys()) - loaded_keys
        for key in remaining_keys:
            if hasattr(qwenact, key):
                submodule = getattr(qwenact, key)
                if hasattr(submodule, 'load_state_dict'):
                    try:
                        submodule.load_state_dict(model_state_dict[key])
                        overwatch.info(f"✅ Auto-loaded extra module `{key}`")
                    except Exception as e:
                        overwatch.warning(f"⚠️ Failed to auto-load `{key}`: {e}")
                else:
                    overwatch.warning(f"⚠️ Attribute `{key}` exists but is not a loadable module.")
            else:
                overwatch.warning(f"⚠️ Unknown key `{key}` in checkpoint. Ignoring.")
        

        # Freeze Weights
        if freeze_weights: # TODO 不应该把 freeze 逻辑分散
            vlm.requires_grad_(False)
            vlm.eval()

        return qwenact

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
        conversation = [
            {"role": "user", "content": [{"type": "text", "text":f"What action should the robot take to {instruction.lower()}?"}, {"image": None}]},
            ]
        
        prompt_text = self.qwen_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # Tokenize (w/ `base_tokenizer`)
        inputs = self.qwen_processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")

        # dict_keys(['pixel_values', 'image_grid_thw']) # (256, 1176) # (1, 3) --> 符合 Qwen 的要求 N_patch, C*patch_w*patch_h
        input_ids = inputs.input_ids[0]
        pixel_values = inputs.pixel_values # value in patch size

        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
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
            outputs = self.vlm.generate(
                **inputs,
                max_new_tokens=3, #@Jinhui Checking TODO: 这里很不科学
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs
            ) # generation 拿不到前面token 的信息，考虑使用 forward?

            # Jinhui see text # outputs.sequences.shape: B, len with prefix
            outputs.input_ids = outputs.sequences # 为了和 input dict 保持一致， 方便调用 self._get_cognition_features# 还真不太一样，因为generation的逻辑和 forward不一样
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)]
            output_text = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print("output:\n",output_text[0])
            # fmt: on
            # 我们training的时候是 image 不固定在最前面没，是没办法只max_new = 1 的
        # Extract cognition feature
        # outputs.hidden_states = list = next tokens 
        # be careful about the where the cognition_features comes from would align with training
        # cognition_features = output.hidden_states[0][-1][:,-1,:]  # nexx tokens, layers, B, len, D #@Jinhui to Think 这里为什么每一个 next token 都记录了 全部都 hidden? 不是的，只有第一个会
        last_hidden_states = outputs.hidden_states[0][-1] #torch.Size([1, 428, 2048]) # last hidden_states for next token generation
        cognition_features = self._get_cognition_features(last_hidden_states, outputs.input_ids, attention_mask=inputs.attention_mask) # [B,1,D] TODO carefully checking with align training

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
        #TODO 之后这里看看这吗写出 策略话。但是本质上是要内聚到这里来的，只是说可以读取一个配置
        for param in self.image2cognition_projector.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.float()
        
        # only align peojection between Qwen and DiT
        self.vlm.model.requires_grad_(False)
        self.action_model.requires_grad_(False)
        self.image2cognition_projector.requires_grad_(True)
        # Add to `self.trainable_module_keys`
        self._trainable_module_keys = ["image2cognition_projector"]
        overwatch.info(f"[TRAINABLE] 🔥 =>> Backbone `{self.image2cognition_projector.__class__.__name__}`", ctx_level=1)
        


    @property
    def trainable_module_keys(self) -> List[str]:
        # TODO 这里应该实现的事直接去 扫描 trainable_module_keys
        # keys = []
        # for module_keys in self.vlm.trainable_module_keys:
        #     keys.append("vlm." + module_keys)
        # keys += self._trainable_module_keys
        self._trainable_module_keys = ["image2cognition_projector"]
        return self._trainable_module_keys
    

    def reset_model_pramaeters(self):
        """Reset the model parameters."""

        #TODO 只有使用config来管理
        # 先直接local 变量来敏捷开发
        # local cogact release checkpoint
        model_path = "playground/Pretrained_models/CogACT-Base/checkpoints/CogACT-Base.pt"
        cogact_pramaeters = torch.load(model_path, map_location="cpu")["model"]
        # local action model
        state_dict = cogact_pramaeters["action_model"]
        self.action_model.load_state_dict(state_dict)

        pass


# TODO move to other.py
import torch
import torch.nn as nn

class CNNModelWithMaxPool(nn.Module):
    def __init__(self, 
                 hidden_dim: int = 2048,  # 每个token的隐藏维度
                 conv_out_channels: int = 64,  # 每个卷积层输出的通道数
                 kernel_size: int = 3,  # 卷积核大小
                 stride: int = 1,  # 卷积步幅
                 pool_kernel_size: int = 2,  # 池化层的卷积核大小
                 pool_stride: int = 2,  # 池化步幅
                 fc_out_features: int = 4096):  # 全连接层输出特征数
        """
        CNN模型，使用多个Conv2d层，最后通过MaxPool2d压缩成B, D形状，再通过一个线性层输出。

        Args:
            hidden_dim (int): 每个token的隐藏维度，默认为2048。
            conv_out_channels (int): 卷积层输出的通道数，默认为64。
            kernel_size (int): 卷积核大小，默认为3。
            stride (int): 卷积步幅，默认为1。
            pool_kernel_size (int): 池化层的卷积核大小，默认为2。
            pool_stride (int): 池化步幅，默认为2。
            fc_out_features (int): 全连接层输出的特征数，默认为4096。
        """
        super(CNNModelWithMaxPool, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=hidden_dim, 
                               out_channels=conv_out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride)
        
        self.conv2 = nn.Conv2d(in_channels=conv_out_channels, 
                               out_channels=conv_out_channels * 2,  # 增加通道数
                               kernel_size=kernel_size, 
                               stride=stride)
        
        self.conv3 = nn.Conv2d(in_channels=conv_out_channels * 2, 
                               out_channels=conv_out_channels * 4,  # 增加通道数
                               kernel_size=kernel_size, 
                               stride=stride)
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # 定义全局池化层
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))  # 自适应最大池化，将每个通道池化为一个值
        
        # 全连接层
        self.fc1 = nn.Linear(conv_out_channels * 4, fc_out_features)  # 全连接层1

    def forward(self, last_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，提取图像token的隐藏状态，并通过CNN和全连接层生成特征。

        Args:
            last_hidden (torch.Tensor): 输入的隐藏状态，形状为 [B, T, D]。
            input_ids (torch.Tensor): 输入的token id，形状为 [B, T]。
            attention_mask (torch.Tensor): 注意力掩码，形状为 [B, T]。

        Returns:
            torch.Tensor: 模型输出特征，形状为 [B, 1]。
        """

        # 提取图像token的隐藏状态
        image_token_id = 151655  # 图像token的ID（根据具体任务调整）
        image_token_positions = (input_ids == image_token_id)  # 获取图像token的位置信息
        image_token_hidden_states = last_hidden[image_token_positions].view(last_hidden.size(0), -1, last_hidden.size(-1))  # [B, num_image_tokens, D]

        # 计算 W 和 H（假设图像token的数量是完美的平方数）
        num_image_tokens = image_token_hidden_states.shape[1]
        W = H = int(num_image_tokens ** 0.5)  # W = H = sqrt(num_image_tokens)

        # 重塑隐藏状态为 [B, W, H, D]
        image_token_hidden_states = image_token_hidden_states.view(last_hidden.size(0), W, H, last_hidden.size(-1))

        # 卷积层1
        x = self.conv1(image_token_hidden_states)  # [B, conv_out_channels, W', H']
        x = torch.relu(x)

        # 卷积层2
        x = self.conv2(x)  # [B, conv_out_channels*2, W'', H'']
        x = torch.relu(x)

        # 卷积层3
        x = self.conv3(x)  # [B, conv_out_channels*4, W''', H''']
        x = torch.relu(x)

        # 使用全局最大池化
        x = self.global_pool(x)  # [B, conv_out_channels*4, 1, 1]
        
        # 展平张量
        x = x.view(x.size(0), -1)  # [B, conv_out_channels*4]

        # 全连接层1
        x = self.fc1(x)  # [B, fc_out_features]
        x = torch.relu(x)
        return x

    def __init__(self, 
                 hidden_dim: int = 2048,  # 每个token的隐藏维度
                 conv_out_channels: int = 64,  # 每个卷积层输出的通道数
                 kernel_size: int = 3,  # 卷积核大小
                 stride: int = 1,  # 卷积步幅
                 pool_kernel_size: int = 2,  # 池化层的卷积核大小
                 pool_stride: int = 2,  # 池化步幅
                 fc_out_features: int = 4096):  # 全连接层输出特征数
        """
        CNN模型，使用多个Conv2d层，最后通过MaxPool2d压缩成B, D形状，再通过一个线性层输出。

        Args:
            hidden_dim (int): 每个token的隐藏维度，默认为2048。
            conv_out_channels (int): 卷积层输出的通道数，默认为64。
            kernel_size (int): 卷积核大小，默认为3。
            stride (int): 卷积步幅，默认为1。
            pool_kernel_size (int): 池化层的卷积核大小，默认为2。
            pool_stride (int): 池化步幅，默认为2。
            fc_out_features (int): 全连接层输出的特征数，默认为4096。
        """
        super(CNNModelWithMaxPool, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=hidden_dim, 
                               out_channels=conv_out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride)
        
        self.conv2 = nn.Conv2d(in_channels=conv_out_channels, 
                               out_channels=conv_out_channels * 2,  # 增加通道数
                               kernel_size=kernel_size, 
                               stride=stride)
        
        self.conv3 = nn.Conv2d(in_channels=conv_out_channels * 2, 
                               out_channels=conv_out_channels * 4,  # 增加通道数
                               kernel_size=kernel_size, 
                               stride=stride)
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # 定义全局池化层
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))  # 自适应最大池化，将每个通道池化为一个值
        
        # 全连接层
        self.fc1 = nn.Linear(conv_out_channels * 4, fc_out_features)  # 全连接层1

    def forward(self, last_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，提取图像token的隐藏状态，并通过CNN和全连接层生成特征。

        Args:
            last_hidden (torch.Tensor): 输入的隐藏状态，形状为 [B, T, D]。
            input_ids (torch.Tensor): 输入的token id，形状为 [B, T]。
            attention_mask (torch.Tensor): 注意力掩码，形状为 [B, T]。

        Returns:
            torch.Tensor: 模型输出特征，形状为 [B, 1]。
        """

        # 提取图像token的隐藏状态
        image_token_id = 151655  # 图像token的ID（根据具体任务调整）
        image_token_positions = (input_ids == image_token_id)  # 获取图像token的位置信息
        image_token_hidden_states = last_hidden[image_token_positions].view(last_hidden.size(0), -1, last_hidden.size(-1))  # [B, num_image_tokens, D]

        # 计算 W 和 H（假设图像token的数量是完美的平方数）
        num_image_tokens = image_token_hidden_states.shape[1]
        W = H = int(num_image_tokens ** 0.5)  # W = H = sqrt(num_image_tokens)

        # 重塑隐藏状态为 [B, W, H, D] = 16,8,8,2048
        image_token_hidden_states = image_token_hidden_states.view(last_hidden.size(0), W, H, last_hidden.size(-1))

        # [B, W, H, D] -> [B, D, H, W]
        image_token_hidden_states = image_token_hidden_states.permute(0, 3, 1, 2)
        # 卷积层1
        x = self.conv1(image_token_hidden_states)  # [B, conv_out_channels, W', H']
        x = torch.relu(x)

        # 卷积层2
        x = self.conv2(x)  # [B, conv_out_channels*2, W'', H'']
        x = torch.relu(x)

        # 卷积层3
        x = self.conv3(x)  # [B, conv_out_channels*4, W''', H''']
        x = torch.relu(x)

        # 使用全局最大池化
        x = self.global_pool(x)  # [B, conv_out_channels*4, 1, 1]
        
        # 展平张量
        x = x.view(x.size(0), -1)  # [B, conv_out_channels*4]

        # 全连接层1
        x = self.fc1(x)  # [B, fc_out_features]
        x = torch.relu(x)

        # 输出层
        return x.unsqueeze(1)  # [B, 1, D]

