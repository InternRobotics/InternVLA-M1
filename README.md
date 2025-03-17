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
├── dataloader           # 数据加载与预处理
│
├── training             # 训练相关代码
│
├── conf                 # 配置文件
│
├── README.md            # 项目说明文件
├── requirements.txt     # 依赖包列表
```

## 安装与使用

```bash
git clone https://github.com/your-repo/LLaVA-VLA.git
cd LLaVA-VLA
pip install -r requirements.txt
```

## 许可证

MIT License

