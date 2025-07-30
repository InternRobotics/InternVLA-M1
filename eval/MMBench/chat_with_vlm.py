from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from llavavla.model.framework.qwenpi import QwenQFormerDiT
from llavavla.model.framework.DinoQFormerACT import QwenQFormerDiT
import os, torch


import debugpy, torch
debugpy.listen(("0.0.0.0", 10092))
print("🔍 Rank 0 waiting for debugger attach on port 10092...")
debugpy.wait_for_client()

saved_model_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0712_vla_v4_fixvit_vlma/checkpoints/steps_30000_pytorch_model.pt"
saved_model_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0726_v6_vla_dinol_32_cotrain_freezedino/checkpoints/steps_40000_pytorch_model.pt"

qwenact = QwenQFormerDiT.from_pretrained( # a lot of Missing key(s) in state_dict:
          saved_model_path,                       # choose from ['CogACT/CogACT-Small', 'CogACT/CogACT-Base', 'CogACT/CogACT-Large'] or the local path
        )


# default: Load the model on the available device(s)
model = qwenact.qwen_vl_interface.model
# default processer
processor = qwenact.qwen_vl_interface.processor



# model_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"  # or your local path
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     # attn_implementation="flash_attention_2",
# )
# model.forward
# processor = AutoProcessor.from_pretrained(
#     model_path, use_fast=True
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# tokenizer.padding_side = "left"  # BATCH INFER
# processor.tokenizer = tokenizer


model.to("cuda")  # 将模型移动到 GPU
model.eval()


messages0 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/petrelfs/yejinhui/Projects/llavavla/eval/MMBench/COCO_train2014_000000561947.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


messages_text = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "中国的首都是哪里？"}, # 过拟合了
        ],
    }
]

# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "1000+1 * 10 等于多少?"},
        ],
    }
]
messages2 = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "讲个笑话"},
]
# Combine messages for batch processing
messages = [messages0, messages1, messages2, messages3, messages_text]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)


pass
# WindowX p

# 5k 20 --> 1.3 
# 5k 20 --> 18.5 
# ['1+1 等于 2.', 
# 'I am a large language model created by Alibaba Cloud. I am called Qwen.', 
# '北京市, 中国']


#10k WindowX 50 --> 13.3  
# ['2', 
# "I am a large, green octopus plushie. Here are some details:\n\n- I have **eight** tentacles, each one is a vibrant green color. The tentacles are detailed and have a playful appearance.\n- The plushie is lying on its back on a **red blanket**. Its head is in the air and it appears to be floating in the blanket.\n- The background of the image is a **white wall**. To the right of the image, there's a **brown pillow** resting against the wall.\n\nThis description includes the types and colors of the objects, their actions, and their locations. It also", 
# 'The capital city of China is Beijing.']


# 15k: 21.6

# 20k: 59 --> 28.1

# 30k 60 --> 40

# 40k 60 --> 54


# 0 =
# 'The image presents a simple yet intriguing scene. At the center of the frame, a **silver metal lock** stands upright, its metallic surface gleaming against the backdrop. The lock is unattended, standing alone on a **black surface**. The background of the image is blurred, suggesting a depth of field effect from the camera and focusing our attention on the lock.\n\nThe lock is positioned in the center of the frame, drawing our eye immediately. Its solitary presence in the otherwise empty frame creates a sense of mystery and anticipation. Despite its simplicity, the image evokes a sense of security and intrigue, as if inviting us to ponder over'
# 1 =
# "I am a large, yellow-green sea turtle with a pattern of black spots on its body. The turtle is resting on a concrete surface, its long, powerful legs folded neatly beneath it. Its head is tilted slightly to the right, as if it's observing something with a calm and serene expression. The turtle's large eyes are open, looking directly at the camera, giving us a glimpse into its tranquil world. The background is blurred and indistinct, allowing us to focus solely on the turtle and appreciate its beauty in this moment."
# 2 =
# "The People's Republic of China, commonly known as China, is a vast and diverse country. Its territory is composed of various geographical elements, each contributing to the unique landscape that unfolds before your eyes.\n\nIn the north, a majestic mountain range stretches across the horizon, its peaks reaching for the sky. The mountains are blanketed with a lush layer of green trees, creating a beautiful contrast against the blue sky. \n\nTo the left, a large body of water glistens under the sunlight, its surface undulating with gentle ripples. The water's presence adds a sense of tranquility to the scene.\n\nOn the right, there's"
# len() =
# 3
