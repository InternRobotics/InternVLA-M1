# InternVLA-M1

> 一体化视觉-语言-动作 (Vision-Language-Action, VLA) 开源框架  
> End-to-end, modular, research-friendly.

<!-- TODO: 在此处插入项目 Logo / 架构图 / 动图 -->
<!-- TODO: Demo Video 占位: 将 demo.mp4 放到 ./assets 并在此引用 -->

# Introduction
InternVLA-M1 is a open-source, end-to-end vision–language–action (VLA) framework. 

## 🔥 核心特性 (Key Features)

1. Modular & Extensible  
   Core components (VLM, Action Model, Projector, DINO, Trainer) are fully decoupled. You can plug in custom vision-language backbones, action policies, or feature projectors without touching the rest. A unified data interface (e.g., LeRobot + custom robotics datasets) lowers integration and research iteration cost.

2. Dual-System and Dual-Supervision
InternVLA-M1 integrates a unified architecture with both a language head and an action head, enabling collaborative training under dual supervision, combining both language and action signals. This design supports learning from multimodal data, especially robotic perception data, significantly improving instruction-following capability.

3. Efficient Training & Fast Convergence  
   Learns spatial / visual priors from large-scale multimodal pretraining, then transfers them via spatial prompt fine-tuning. Achieves strong performance (e.g., OXE SOTA-level convergence in ~2.5 epochs without separate action pretraining). Built‑in optimizations: FlashAttention2, BF16, gradient accumulation, distributed (torchrun / DeepSpeed‑ready).



---

## 📂 目录结构 (Repo Structure)

```text
InternVLA
├── model
│   ├── framework            # 主框架 (数据流 / loss / forward)
│   ├── modules
│   │   ├── vlm              # 各类多模态 / 语言模型
│   │   ├── action_model     # 动作策略 / 控制模型
│   │   ├── projector        # 特征对齐 / 空间映射
│   │   ├── dino_model       # 视觉细节特征
├── dataloader
│   ├── groot_lerobot        # LeRobot / Groot 数据适配
├── training
│   ├── train_vlm
│   ├── train_vla
│   ├── train_vla_withCotrain
├── config                   # 全局统一实验配置 (YAML)
├── real_deployment          # 部署与推理
│   ├── deploy/server_policy.py
├── scripts                  # 训练与评估脚本
├── playground               # 建议将符号链接放在此
│   ├── Datasets             
│   ├── Pretrain_models
```

---

## 🛠 环境准备 (Environment Setup)

快速安装：

```bash
conda create -n internVLA python=3.10 -y
conda activate internVLA

# 基础依赖
pip install -r requirements.txt

# FlashAttention2 (确保 Torch/CUDA 版本匹配)
pip install flash-attn --no-build-isolation

# 可编辑安装
pip install -e .


```

---


## 🚀 快速上手 (Quick Start)
### Jinhui 在写这部分， 就是高数其他人如果我们要 follow 我们的工作， 大概的路线
复现 --> 准备数据 --> 准备模型 --> 训练 --> 测试

## 1 复现我们的结果 on SimplerENV
我们会在examples/ 
这里 提供 如果 reproduce internVLA-M1 


## 🧩 扩展InternVLA to your work (How to Extend)

### 1. Data Format & Loading：
我们数据数据格式借鉴了开源的最佳实践。例如action数据采用 LeRobot provide by GR00T : https://github.com/NVIDIA/Isaac-GR00T. 多模态数据 follow Qwen2.5-VL : https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune

我可以参考他们的规范准备数据。and in 我们codebase, 
对于 action 数据 应该是
python ./InternVLA/dataloader/lerobot_datasets_oxe.py 
查看是否能否成功返回数据
每个数据的分会规范 是
****

同理， 利用 python ./InternVLA/dataloader/vlm_datasets.py debug 查看 你的dataloader 是否好。

一旦 dataloader 返回的内容符合你的预期， 注册 your own dataloader InternVLA/dataloader/__init__.py


### 2. 模型开发：
你或许会开发你自己的模型
我们约定 你只能有一个 framework.py (例如 InternVLA/model/framework/M1.py ) align with ther framework fig in your papar. 
你可以在 InternVLA/model/modules 中定义你framework 需要的模块。 but 你想要保证 you can python framework.py. then build your model and forward your more with batch sample. 
 
 then 
register your framework in InternVLA/model/framework/__init__.py.


### 3. 模型部署

所有的framework 能够通过
from InternVLA.model.framework.yourmodel import Yourframeork
your_model_ckpt="playground/Checkpoints/debug_0910_internM1_cotrain/checkpoints/steps_2000_pytorch_model.pt"

your_model = Yourframeork.from_pretrained(your_model_ckpt)

then your kan
your_model.predict_action()

你能够使用 deployment/model_server 的服务来 server 评测


### 参数配置：
InternVLA-M1 采用 yaml 文件来管理

只有一个 global 参数, 他们被统一 管理在 InternVLA/config/training/qwenvla_cotrain_oxe.yaml。 这是一个 灵活参数对象（例如dict）， 全局都对完整的参数对象可见，但是只有要使用的时候才访问对应的value， which mean 参数对象 可以冗余，but 不能缺失 if 在你的工程中明确要使用。

InternVLA-M1 已经对参数进行了初步的分组。 例如 datasets, framework, trainer
参数的优先级是 你可以通 CMD 覆盖或者增加 参数。

最后生效的参数会被统一 save 在ckpt 文件夹， which 方便后续的去读


---

## 📈 Model Zoo (占位)

| 模型 | 参数规模 | 预训练数据 | 下游 (LIBERO) 成绩 | 下载 |
|------|----------|------------|--------------------|------|
| InternVLA-M1 Base | ~ | ~ | ~ | TODO |
| InternVLA-M1 Large | ~ | ~ | ~ | TODO |

（TODO: 后续补充权重与日志）

---


## 🧩 扩展指南 (How to Extend)
<!-- as toDO -->
<!-- 新增 your own VLA：
我们约定 你只能有一个 framework.py (例如 InternVLA/model/framework/M1.py ) align with ther framework fig in your papar. 
你可以在 InternVLA/model/modules 中定义你framework 需要的模块。 but 你想要保证 you can python framework.py. then build your model and forward your more with batch sample. 

then  register your framework in InternVLA/model/framework/__init__.py. 我们避免 使用 REGISTRY 方法以保留更好的可读性and 方便用户 review code.

新增 your own 训练参数：
InternVLA-M1 只有一个 global 参数, 他们被统一 管理在 InternVLA/config/training/qwenvla_cotrain_oxe.yaml。 这是一个 灵活参数对象（例如dict），你可以通 CMD 覆盖或者增加 参数。 全局都对完整的参数对象可见，但是只有要使用的时候才访问对应的value， which mean 参数对象 可以冗余，but 不能缺失 if 在你的工程中明确要使用。
InternVLA-M1 已经对参数进行了初步的分组。 例如 datasets, framework, trainer

新增 训练策略：
InternVLA-M1 的trainer 是 自建的base 最基础的 torch 函数完成， 例如对于 freeze modular,  通过 trainer.freeze_modules 直接声明 用户自己framework moduels name， and 我们通过RE 去查看模型模块， 并使用 采用最基础的 torch 函数完成 参数的冻结（看TrainerUtils）


yep，InternVLA-M1 might not 上手就来，因为他使用了很多 最基础的 pytorch 工具来完成 codebase 实现更好的解偶和 保持更好的扩展性。 but 如果try， your will find it 优势。 -->

---

## 📜 Citation (引用)

如果本项目对你的研究有帮助，请引用（占位）：

```bibtex
@misc{internvla2024,
  title  = {InternVLA-M1: An Open Vision-Language-Action Framework},
  author = {InternVLA Contributors},
  year   = {2024},
  url    = {https://github.com/...}
}
```

---

## ✅ TODO Roadmap

- [ ] 发布模型权重
- [ ] 增加多任务混合训练示例
- [ ] 集成 Deepspeed / FSDP
- [ ] 发布真实机器人 Demo
- [ ] 添加日志可视化 (TensorBoard / WandB)
- [ ] 统一评估脚本指标输出

---

## 🤝 Contributing

欢迎 PR / Issue：

---

## 🔐 License

MIT License

---

## 📬 联系

- Issue：提交详细日志 + 复现步骤
- 邮件：TODO (如需添加)
- 交流群：TODO (可放飞书/钉钉/微信群二维码)

---

感谢使用 InternVLA-M1！🎯 如果觉得有用，欢迎 Star 支持。

