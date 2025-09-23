# InternVLA-M1

**InternVLA-M1** is a Open-source Spatially Grounded Foundation Model for Generalist Robot Policy

https://github.com/user-attachments/assets/e83ae046-a503-46a8-95e4-ef381919b7f8

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://github.com/InternRobotics/InternVLA-M1/blob/InternVLA-M1/assets/InternVLA_M1.pdf) [![Website](https://img.shields.io/badge/Website-GitHub%20Pages-blue.svg)](https://internrobotics.github.io/internvla-m1.github.io) [![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://youtu.be/n129VDqJCk4) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![](assets/teaser.png)

## 🔥 Key Features

1. **Modular & Extensible**  
   All core components (model architecture, training data, training strategies, evaluation pipeline) are fully decoupled, enabling independent development, debugging, and extension of each module.


2. **Dual-System and Dual-Supervision**
   InternVLA-M1 integrates both a language head and an action head under a unified framework, enabling collaborative training with dual supervision. 

3. **Efficient Training & Fast Convergence**
   Learns spatial and visual priors from large-scale multimodal pretraining and transfers them via spatial prompt fine-tuning. Achieves strong performance (e.g., SOTA-level convergence on  in \~2.5 epochs without separate action pretraining). 

## 🎯 Target Audience

1. Users who want to leverage open-source VLMs (e.g., Qwen2.5-VL) for robot control.
2. Teams co-training action datasets jointly with multimodal (vision–language) data.
3. Researchers exploring alternative VLA architectures and training strategies.

## 📊 Experimental Results
|             | WindowX | Google Robot(VA) | Google Robot(VM) | LIBERO |
|-------------|---------|------------------|------------------|--------|
| $\pi_0$         | 27.1    | 54.8             | 58.8             | 94.2   |
| GR00t       | 61.9    | 44.5             | 35.2             | 93.9   |
| InternVLA-M1 |**71.7** |**76.0**          |**80.7**          |**95.9**|






# 🚀 Quick Start

## 🛠 Environment Setup

```bash
# Clone the repo
git clone https://github.com/InternRobotics/InternVLA-M1

# Create conda environment
conda create -n internvla-m1 python=3.10 -y
conda activate internvla-m1

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install InternVLA-M1
pip install -e .
```


## ⚡ Quick Interactive M1 Demo

Below are two collapsible examples: InternVLA-M1 chat and action prediction.

<details open>
<summary><b>InternVLA-M1 Chat Demo (image Q&A / Spatial Grounding)</b></summary>

```python
from InternVLA.model.framework.M1 import InternVLA_M1
from PIL import Image
import requests
from io import BytesIO
import torch

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img

saved_model_path = "/PATH/checkpoints/steps_50000_pytorch_model.pt"
internVLA_M1 = InternVLA_M1.from_pretrained(saved_model_path)

# Use the raw image link for direct download
image_url = "https://raw.githubusercontent.com/InternRobotics/InternVLA-M1/InternVLA-M1/assets/table.jpeg"
image = load_image_from_url(image_url)
question = "Give the bounding box for the apple."
response = internVLA_M1.chat_with_M1(image, question)
print(response)
```
</details>

<details>
<summary><b>InternVLA-M1 Action Prediction Demo (two views)</b></summary>

```python
from InternVLA.model.framework.M1 import InternVLA_M1
from PIL import Image
import requests
from io import BytesIO
import torch

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img

saved_model_path = "/PATH/checkpoints/steps_50000_pytorch_model.pt"
internVLA_M1 = InternVLA_M1.from_pretrained(saved_model_path)

image_url = "https://raw.githubusercontent.com/InternRobotics/InternVLA-M1/InternVLA-M1/assets/table.jpeg"
view1 = load_image_from_url(image_url)
view2 = view1.copy()

# Construct input: batch size = 1, two views
batch_images = [[view1, view2]]  # List[List[PIL.Image]]
instructions = ["Pick up the apple and place it on the plate."]

if torch.cuda.is_available():
    internVLA_M1 = internVLA_M1.to("cuda")

pred = internVLA_M1.predict_action(
    batch_images=batch_images,
    instructions=instructions,
    cfg_scale=1.5,
    use_ddim=True,
    num_ddim_steps=10,
)
normalized_actions = pred["normalized_actions"]  # [B, T, action_dim]
print(normalized_actions.shape, type(normalized_actions))
```
</details>


## 📘 Examples

We provide several end-to-end examples for reference:

* **Reproduce InternVLA-M1 in SimplerEnv**
  [Example](/examples/SimplerEnv)

* **Reproduce InternVLA-M1 in LIBERO**
  [Example](/examples/LIBERO)

* **Training/Deployment on real robots**
  [Example](/examples/real_robot)

## 📈 Model Zoo
We release a series of pretrained models and checkpoints to facilitate reproduction and downstream use.

### ✅ Available Checkpoints

| Model | Description | Link |
|-------|-------------|------|
| **InternVLA-M1** | Main pretrained model | [🤗 Hugging Face](https://huggingface.co/InternRobotics/InternVLA-M1) |
| **InternVLA-M1-Pretrain-RT-1-Bridge** | Pretraining on RT-1 Bridge data | [🤗 Hugging Face](https://huggingface.co/InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge) |
| **InternVLA-M1-LIBERO-Long** | Fine-tuned on LIBERO Long-horizon tasks | [🤗 Hugging Face](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Long) |
| **InternVLA-M1-LIBERO-Goal** | Fine-tuned on LIBERO Goal-conditioned tasks | [🤗 Hugging Face](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Goal) |
| **InternVLA-M1-LIBERO-Spatial** | Fine-tuned on LIBERO Spatial reasoning tasks | [🤗 Hugging Face](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Spatial) |
| **InternVLA-M1-LIBERO-Object** | Fine-tuned on LIBERO Object-centric tasks | [🤗 Hugging Face](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Object) |



# 🗺️ Roadmap

* [✅] Release model weights
* [ ] Add multi-task mixed training examples
* [ ] Unify evaluation scripts and metrics

# 🤝 Contributing

We welcome contributions via Pull Requests or Issues.
Please include detailed logs and reproduction steps when reporting bugs.

# 📜 Citation

If you find this useful in your research, please consider citing:

```bibtex
@misc{internvla2024,
  title  = {InternVLA-M1: Latent Spatial Grounding for Instruction-Following Robotic Manipulation},
  author = {InternVLA-M1 Contributors},
  year   = {2025},
  booktitle={arXiv},
}
```

# 📬 Contact

* Issues: Submit via GitHub Issues with detailed logs and steps

# 🙏 Acknowledgements

We thank the open-source community for their inspiring work. This project builds upon and is inspired by the following projects (alphabetical order):
- [IPEC-COMMUNITY](https://huggingface.co/IPEC-COMMUNITY): Curated OXE / LIBERO style multi-task datasets and formatting examples.
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): Standardized action data loader (GR00T-LeRobot).
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md): Multimodal input/output format, data loader, and pretrained VLM backbone.
- [CogACT](https://github.com/microsoft/CogACT/tree/main/action_model): Reference for a DiT-style action head design.
- [llavavla](https://github.com/JinhuiYE/llavavla): Baseline code structure and engineering design references.
- [GenManip Simulation Platform](https://github.com/InternRobotics/GenManip): Simulation platform for generalizable pick-and-place based on Isaac Sim.


Notes:
- If any required attribution or license header is missing, please open an issue and we will correct it promptly.
- All third-party resources remain under their original licenses; users should comply with respective terms.


---

Thanks for using **InternVLA-M1**! 🌟
If you find it useful, please consider giving us a ⭐ on GitHub.
