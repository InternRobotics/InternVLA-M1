# InternVLA-M1

**InternVLA-M1** is an open-source, end-to-end **vision–language–action (VLA) framework** for building and researching generalist robot policies.

<!-- Demo Video -->

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/) [![Website](https://img.shields.io/badge/Website-GitHub%20Pages-blue.svg)](https://internrobotics.github.io/internvla-m1.github.io) [![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://www.youtube.com/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![](assets/teaser.png)

---

## 🔥 Key Features

1. **Modular & Extensible**
   Core components (VLM, Action Model, Projector, DINO, Trainer) are fully decoupled. You can plug in custom vision-language backbones, action policies, or feature projectors without modifying the rest of the codebase. A unified data interface (e.g.,  + custom robotics datasets) reduces integration effort and accelerates research iteration.

2. **Dual-System and Dual-Supervision**
   InternVLA-M1 integrates both a language head and an action head under a unified framework, enabling collaborative training with dual supervision. This design leverages both language and action signals to learn from multimodal data—especially robotic perception data—significantly improving instruction-following capability.

3. **Efficient Training & Fast Convergence**
   Learns spatial and visual priors from large-scale multimodal pretraining and transfers them via spatial prompt fine-tuning. Achieves strong performance (e.g., SOTA-level convergence on  in \~2.5 epochs without separate action pretraining). Built-in optimizations include , BF16, gradient accumulation, and distributed training ( /  ready).

---

## 🚀 Quick Start

### 🛠 Environment Setup

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

---

## 📘 Examples

We provide several end-to-end examples for reference:

* **Training/Evaluation in simpler env**
  [Example](/examples/simplerEnv/setup.md)

* **Training/Deployment on real robots**
  [Example](/examples/real_robot/setup.md)

* **Extending InternVLA-M1**
  [Example](examples/extending-m1/README.md)

---

## 📈 Model Zoo

| Model              | Size | Pretraining Data | Downstream () | Download |
| ------------------ | ---- | ---------------- | ------------- | -------- |
| InternVLA-M1 Base  | \~   | \~               | \~            | TODO     |
| InternVLA-M1 Large | \~   | \~               | \~            | TODO     |

---

## 🗺️ Roadmap

* [ ] Release model weights
* [ ] Add multi-task mixed training examples
* [ ] Integrate  /  (FSDP)
* [ ] Release real-robot demo
* [ ] Add training log visualization ( / )
* [ ] Unify evaluation scripts and metrics

---

## 🤝 Contributing

We welcome contributions via Pull Requests or Issues.
Please include detailed logs and reproduction steps when reporting bugs.

---

## 📜 Citation

If you find this useful in your research, please consider citing:

```bibtex
@misc{internvla2024,
  title  = {InternVLA-M1: An Open Vision-Language-Action Framework},
  author = {InternVLA Contributors},
  year   = {2025},
  url    = {https://github.com/InternRobotics/InternVLA-M1}
}
```

---

## 📬 Contact

* Issues: Submit via GitHub Issues with detailed logs and steps

---

## 🙏 Acknowledgements

We would like to thank the open-source community for their inspiring contributions.
This project builds upon and is inspired by the following works and toolchains:

- PyTorch – The deep learning framework used for model implementation and training.
- FlashAttention2 – Enables efficient attention computation for large-scale training.
- LeRobot – Provides standardized action datasets.
- Qwen2.5-VL – Provides multimodal pretraining data and codebase inspiration for vision-language modeling.

---

Thanks for using **InternVLA-M1**! 🌟
If you find it useful, please consider giving us a ⭐ on GitHub.
