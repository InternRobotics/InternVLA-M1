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
        model_id: str,
        config: Optional[dict] = None,
        **kwargs
    ):  
        super().__init__()
        # QWen 原生模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2", #"sdpa" TODO 要确认是否和 train 有关， 直觉上是无关的
            torch_dtype="auto",
            device_map="cuda",
        )
        
        processor = AutoProcessor.from_pretrained(model_id) #TODO check 
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
        # TODO position_ids 是否生效了？
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        调用 QWen2.5 的 forward，输出类似 CausalLMOutputWithPast 的结构，供 CogACT 使用。
        """
        #  TODO 这里需要更加简洁的接口
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model( # TODO 验证   1. 验证position id 是干什么的， 这里要定义**input 就是唯一传惨， 不能够中途调换
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
        dummy_output = CausalLMOutputWithPast( # TODO 后期移除？ 需要讨论返回的格式
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
    
    def build_qwenvl_inputs_v1(self, images, instructions, **kwargs): # 这个能够加速推理，但是会有几个字符差异，对libero 这种强overfit 的bench 影响很大
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
            prompt = f"what is the key object to finish the task: {instruction}. Output the bbox to local the object"
            # prompt = f"{instruction}."
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
        inputs = self.processor( # @JinhuiYE TODO 这里需要检查是否图片是否放到的指定地方， 要去对比 官方dataloader
            text=texts,
            images=image_inputs, # list of PIL
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        # torch.distributed.barrier()
        # inputs.keys() # dict_keys(['input_ids', , 'attention_mask', 'pixel_values', 'image_grid_thw']) # 验证 position_ids 不提的话， 内部会自己算
        return inputs.to(self.model.device)
    
    def build_qwenvl_inputs(self, images, instructions, solutions=None):
        """
        Build Qwen2-VL compatible inputs for a batch of multi-camera images.

        Args:
            images: B*list of PIL (muilti-view), image format: RGB, value in [0, 255]
            processor: Qwen2.5 VL processor (AutoProcessor.from_pretrained)
            instructions: Text prompt to use for each instruction
            device: Target device (default: "cuda")

        Returns:
            inputs: dict with input_ids, attention_mask, pixel_values, etc., ready for model.generate or model(...)
        """
        # default 流程是 json, source --> message --> texts --> input_ids
        # 这里我们做 Lerobot --> message -->json source --> text --> inputs with label
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        
        # build conversation messages
        for imgs, instruction in zip(images, instructions): # 思考多图应该怎么处理？

            content = [{"type": "image", "image": img} for img in imgs] # 其实是支持多图的
            CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", None)
            if CoT_prompt:
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = f"Your task is {instruction} where is the pick object and where is the place object. locate the bbox of pick and place in json" # old prompt for onging ckpt

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                solution = solutions[len(messages)]
                solution_content = [{"type": "text", "text": f": {solution}"}]
                msg.append({"role": "assistant", "content": solution_content})
            
            messages.append(msg)
    
        images, videos = process_vision_info(messages) # 这样可以处理不同 图片的复杂情况
        # TODO v1 暂时不支持video 的处理. # 目前还不能image, video 交错， 如果实现需要而外的统一
        # copy from .../transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py
        # copy from llavavla/dataloader/vlm_datasets.py, ⚠️， 为了 能够适用 批处理，做了修改
        # TODO 要修改 官方的 preprocess_qwen_2_visual 中的函数， 同时确保多模态那边不会出现bug
        # TODO 其实可以直接用 processor， 弊端是处理了两次 tokenizer, 但是我觉得开销 不大， 值得做的更加美观
        
        image_inputs = {}
        image_grid_thw = None
        video_inputs = {}
        video_grid_thw = None

        if images is not None: # TODO 看一下这里是否要并行处理，或者直接
            image_inputs = self.processor.image_processor(images=images, return_tensors="pt") # 这里是直接处理成 tensor 的
            image_grid_thw = copy.deepcopy(image_inputs["image_grid_thw"]) 
            image_grid_thw_merged = [
                merged_thw.prod() // self.processor.image_processor.merge_size**2
                for merged_thw in image_grid_thw
            ]
            grid_thw_merged = image_grid_thw_merged # 目前还不能image, video 交错
            text_inputs = preprocess_qwen_2_visual( # 对 官方代码进行了修改，sources --> massage， 支持 batch padding
                messages, self.processor.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            ) # 拿到 input_ids and SFT labels

        elif videos is not None:
            # need more alignment with official code
            RuntimeWarning("Video inputs are not yet supported in this interface. 还不确定这个框架是否支持这样的混合输出.")
            pass
        else:
            ResourceWarning("No visual inputs provided. 还不确定这个框架是否支持这样的混合输出.")
            pass

        # torch.distributed.barrier()
        inputs = BatchFeature(data={**text_inputs, **image_inputs, **video_inputs}, tensor_type="pt")
        # qwen 官方： dict_keys(['input_ids', 'labels', 'position_ids', 'pixel_values', 'image_grid_thw'])
        # 我们 的： dict_keys(['input_ids', 'labels', 'attention_mask','position_ids', 'pixel_values', 'image_grid_thw']) # 
        # 验证了 position_ids 可以不提供， 内部自己会算 ../transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
        return inputs.to(self.model.device)



def messages_to_sources(batch_messages):
    """
    将 batch 格式的 messages 转换为 sources 格式，支持多模态（image/text）。
    
    Args:
        batch_messages: List[List[Dict]]，每个样本是一组 message 对话

    Returns:
        List[List[Dict]]，每个样本的 source 对话
    """
    batch_sources = []

    for messages in batch_messages:
        source = []
        for msg in messages:
            role = msg["role"]
            segments = msg["content"]

            parts = []
            for seg in segments:
                if seg["type"] == "text":
                    parts.append(seg["text"])
                elif seg["type"] == "image":
                    parts.append(DEFAULT_IMAGE_TOKEN) # VIDEO 还不支持
                else:
                    raise ValueError(f"Unsupported content type: {seg['type']}")

            content = "\n".join(parts)
            source.append({"from": "human" if role == "user" else "gpt", "value": content})

        batch_sources.append(source)

    return batch_sources

def preprocess_qwen_2_visual(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    # message --> sources json
    pass
    

    sources = messages_to_sources(messages)  # 转换为 source 格式
    # torch.distributed.barrier()
    # 复用 QWenvl 代码
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")
    
    # tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []
    action_obs_mask = [] # 记录那些token 是obs 可以看到的 --> 传递给 Q-Former # TODO 看一下是否有更好的实现方式
    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target) # TODO 看一下是如何处理结束符号的 @JinhuiYE


    # TODO Batch padding 
    # TODO 不建议在这里执行padding
    
    # Padding input_ids 和 targets
    input_ids = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side
    )
    targets = pad_sequence(
        [torch.tensor(tgt, dtype=torch.long) for tgt in targets],
        batch_first=True,
        padding_value=IGNORE_INDEX,
        padding_side=tokenizer.padding_side
    )

    # 构建 attention_mask：非 pad 的位置为 1，pad 的为 0
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )

    
def get_qwen2_5_interface(model_id, config=None):

    model = _QWen_VL_Interface(model_id, config=config) # 要时刻记住面向对象编程

    return model

if __name__ == "__main__":
    model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    qwen_vl = get_qwen2_5_interface(model_id)
    pass