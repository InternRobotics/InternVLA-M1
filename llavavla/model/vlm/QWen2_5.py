import torch

from typing import Optional, List

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
# from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from qwen_vl_utils import process_vision_info

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Registry =>> Support Qwen-2.5 Models (from HF Transformers)

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
        model_id: str,
        load_for_training: bool = True,
        vl_config: Optional[dict] = None,  # TODO 
        **kwargs
    ):  
        super().__init__()
        # QWen 原生模型
        if load_for_training or True: #TODO model -> vlm 
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,  
                # attn_implementation="flash_attention_2", #"sdpa" TODO 要确认是否和 train 有关， 直觉上是无关的
                torch_dtype="auto", 
                device_map="cuda",
            ) # 只能到 cpu 先 , device_map="cpu" # 试试auto --> FSDP 还是报错了
            # torch.distributed.barrier() # 确保所有进程都加载完毕
        else:
            config = AutoConfig.from_pretrained(model_id)
            model = Qwen2_5_VLForConditionalGeneration(config)  # 只初始化模型结构，不加载参数, @Jinhui 发现load 空模型需要更多的时间
            # 这里会卡住
        processor = AutoProcessor.from_pretrained(model_id) #TODO check 
        processor.tokenizer.padding_side  = 'left' #TODO Check  Flash Attention version of Qwen2_5_VL. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. 


        self.model = model
        self.processor = processor

    def forward( # 后期这里应该是总结和qwen forward 对齐， 但是这里
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
        调用 QWen2.5 的 forward，输出类似 CausalLMOutputWithPast 的结构，供 CogACT 使用。
        """
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # @Jinhui TBD TODO 
            # pixel_values = pixel_values["pixel_values"]
            # print(kwargs["image_grid_thw"])
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
    
    def build_qwenvl_inputs(self, images, instructions, prompt="What is in this image?"):
        """
        Build Qwen2-VL compatible inputs for a batch of multi-camera images.

        Args:
            images: list B*list of PIL, image format: RGB, value in [0, 255
            processor: Qwen2.5 VL processor (AutoProcessor.from_pretrained)
            instructions: Text prompt to use for each instruction
            device: Target device (default: "cuda")

        Returns:
            inputs: dict with input_ids, attention_mask, pixel_values, etc., ready for model.generate or model(...)
        """
        # TODO 这里要和 QWen 官方对齐
        pass
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images,instructions):
            content = [{"type": "image", "image": img} for img in imgs] # 其实是支持多图的
            prompt = f"what is the key object to finish the task: {instruction}. Output the bbox to local the object"
            prompt = f"{instruction}."
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            messages.append(msg)

        # Prepare text prompts using processor
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        # Prepare visual inputs
        # Tokenize all together
        inputs = self.processor( # @JinhuiYE TODO 这里需要检查是否图片是否放到的指定地方， 要去对比 官方dataloader
            text=texts,
            images=image_inputs, # list of PIL, can not to tensor by ourself? yes, will be a bug
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # 检查
        # inputs.keys()
        # inputs["pixel_values"].shape
        # torch.Size([512, 1176]) 这个很奇怪, 它将全部 pixel_values 压缩在一起了？

        return inputs.to(self.model.device)


    
def get_qwen2_5_interface(model_id, config=None):
    if config is None:
        model = _QWen_VL_Interface(model_id= model_id) # 要时刻记住面向对象编程
    else:
        vl_config = config.vla # TODO 后期要统一 config 
        model = _QWen_VL_Interface(model_id, vl_config)

    
    return model

if __name__ == "__main__":
    model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    qwen_vl = get_qwen2_5_interface(model_id)
    pass