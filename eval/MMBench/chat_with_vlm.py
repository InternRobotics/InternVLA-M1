from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from llavavla.model.framework.qwenact import QwenQFormerDiT
import os


import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("🔍 Rank 0 waiting for debugger attach on port 5678...")
debugpy.wait_for_client()

saved_model_path = "/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0608_ftqwen_vlm_bridge_rt_1_64gpus_lr_5e-5_qformer_36_37_rp/checkpoints/steps_5000_pytorch_model.pt"
qwenact = QwenQFormerDiT.from_pretrained( # a lot of Missing key(s) in state_dict:
          saved_model_path,                       # choose from ['CogACT/CogACT-Small', 'CogACT/CogACT-Base', 'CogACT/CogACT-Large'] or the local path
        )


# default: Load the model on the available device(s)
model = qwenact.qwen_vl_interface.model
# default processer
processor = qwenact.qwen_vl_interface.processor


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
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


messages_text = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "中国的首都是哪里？"},
        ],
    }
]

# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "1+1 等于多少?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1,messages2, messages_text ]

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



# WindowX 

# 5k 20 --> 1.3 
# ['1+1 等于 2.', 
# 'I am a large language model created by Alibaba Cloud. I am called Qwen.', 
# '北京市, 中国']


#10k WindowX 50 --> 13.3  
# ['2', 
# "I am a large, green octopus plushie. Here are some details:\n\n- I have **eight** tentacles, each one is a vibrant green color. The tentacles are detailed and have a playful appearance.\n- The plushie is lying on its back on a **red blanket**. Its head is in the air and it appears to be floating in the blanket.\n- The background of the image is a **white wall**. To the right of the image, there's a **brown pillow** resting against the wall.\n\nThis description includes the types and colors of the objects, their actions, and their locations. It also", 
# 'The capital city of China is Beijing.']