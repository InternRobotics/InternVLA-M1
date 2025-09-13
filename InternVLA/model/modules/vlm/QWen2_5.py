import torch
import transformers
from typing import Optional, List
import copy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger
logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# add by jinhui

import torch.nn as nn
#@TODO emergency fix @Jinhui more readable and flexible way for VLM interface
# @Jinhui 这里需要讨论是否需要一个 强制要求的 模版类？ TODO @Jinhui：不需要， 不对架构做任何假设
class _QWen_VL_Interface(nn.Module): #TODO @Jinhui 后期不能再向 PrismaticVLM 对齐， 思考更加flexible做法， --》 接口class的实现， TODO 要直接在 model 中扁平的 初始化全部 modules, 不能递归
    """
    这是对 Qwen2_5_VLForConditionalGeneration 的简单封装，使其在接口层面上更接近 PrismaticVLM，
    例如能够返回类似 CausalLMOutputWithPast 的结构，需要一个 class 来包装是因为 不同的VLM 有不一样的api, 但是要保证对外的功能是一致的
    """
    # 这个的存在其实是因为VLM的多样性比较大， 这里封住一下变化

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs
    ):  
        super().__init__()
        # QWen 原生模型
        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen2.5-VL-3B-Instruct")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_id) 
        processor.tokenizer.padding_side  = 'left' #TODO Check  Flash Attention version of Qwen2_5_VL. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. 
        
        self.model = model
        self.processor = processor
        self.config = config

    def forward( # 后期这里应该是总结和qwen forward 对齐， 但是这里 TODO 移除这个逻辑， 直接调用qwen的逻辑
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,  # 需要 hidden_states
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        调用 QWen2.5 的 forward，输出类似 CausalLMOutputWithPast 的结构
        """
        #  TODO 这里需要更加简洁的接口
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw =image_grid_thw,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        
        return outputs

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
    
    def build_qwenvl_inputs(self, images, instructions, **kwargs): # 这个能够加速推理，但是会有几个字符差异，对libero 这种强overfit 的bench 影响很大
        """
        Build Qwen2-VL compatible inputs for a batch of multi-camera images.

        Args:
            images: list B*list of PIL, image format: RGB, value in [0, 255
            processor: Qwen2.5 VL processor (AutoProcessor.from_pretrained)
            instructions: Text prompt to use for each instruction
            device: Target device (default: "cuda")
        # 改变                
        Returns:
            inputs: dict with input_ids, attention_mask, pixel_values, etc., ready for model.generate or model(...)
        """
        # TODO 这里要和 QWen 官方对齐 --> infer 这样更快， 但是 我们可以写成
        # TODO 保留的原因是相比  v2 似乎更快， 更容易出结果
        pass
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions): # 思考多图应该怎么处理？
            content = [{"type": "image", "image": img} for img in imgs] # 其实是支持多图的
            
            if "CoT_prompt" in self.config.datasets.vla_data: # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            messages.append(msg)

        # Prepare text prompts using processor
        # default 流程是 json --> message --> texts --> input_ids
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        return inputs.to(self.model.device)
    

def get_qwen2_5_interface(config=None, **kwargs):

    model = _QWen_VL_Interface(config=config) # 要时刻记住面向对象编程

    return model

if __name__ == "__main__":
    model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    qwen_vl = get_qwen2_5_interface(model_id)
    pass