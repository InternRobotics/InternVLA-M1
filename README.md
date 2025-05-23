# LLaVA-VLA

LLaVA-VLA 是一个开源项目，旨在解耦并统一视觉语言模型（VLM）和视觉语言动作（VLA）模型，使其在同一框架下兼容运行。

## 特性

- **模块化设计**，包含 **视觉编码**、**语言处理** 和 **动作建模** 组件。
- **同时支持 VLM 和 VLA 任务**，可灵活适配不同应用场景。
- **开源可扩展**，适用于进一步研究与开发。

## 文件结构 (预计)

```
LLaVA-VLA
├── model                # 模型相关代码
│   ├── visual_encoder   # 处理图像和视频特征提取
│   ├── language_encoder # 处理文本输入与嵌入
│   ├── action_model     # 执行视觉语言动作
│   ├── vla              # 各种vla 框架 @TODO 这里的模块怎么划分还需要商量
│
├── dataloader           # 收据构建和预处理
│
├── training             # 训练相关代码
│
├── conf                 # 配置文件
│
├── README.md            # 项目说明文件
├── requirements.txt     # 依赖包列表
```

#
愿景: 开发一个可以同时支持 VLM traning (System2) 和 VLA training 的框架

## 希望的feature 和 理想的脚本
1. Pretraining VLM
2. Pretraining DiT
3. align VLM with DiT (希望在 5 epcoh 内完成 alignment)


## 开发规划
1. 支持 QwenACT 的training (done)
2. 支持同时training VLA 和 VLM (done)
3. 支持同时 align openVLA DiT and Qwen (done) 

4. 支持 单独 training VLM with own vision encode (pending) #直接用QWen
5. 支持 单独 training ACT with own vision encode (pending) # 直接用openVLA



### setup envs
'''bash

cd llavavla/model/openvla 
pip install -e .
<!-- 他们的 pyproject.toml 里面已经有很多包的版本很难install， 比如python 版本绑定为 3.10 -->
<!-- 移除 presmiatic 之后将不需要 -->

cd /mnt/petrelfs/yejinhui/Projects/llavavla/llavavla

pip install -e .


<!-- hard to pip flash_attn-->
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


'''
## 许可证

MIT License



## cmd

### 测试QWen 是否还是 ok 的



from qwen_vl_utils import process_vision_info
model = cogact.vlm.model
model.to("cuda")
processor = cogact.vlm.processor

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

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
