# OpenVLA: An Open-Source Vision-Language-Action Model

[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Getting Started**](#getting-started) | [**Pretrained VLAs**](#pretrained-vlas) | [**Installation**](#installation) | [**Fine-Tuning OpenVLA via LoRA**](#fine-tuning-openvla-via-lora) | [**Fully Fine-Tuning OpenVLA**](#fully-fine-tuning-openvla) |
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Project Website**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>

## Latest Updates
- [2024-07-08] Added new sections: [Fine-Tuning OpenVLA via LoRA](#fine-tuning-openvla-via-lora), [Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla)
- [2024-06-13] Initial release

<hr style="border: 2px solid gray;"></hr>

A simple and scalable codebase for training and fine-tuning vision-language-action models (VLAs) for generalist robotic 
manipulation:

- **Different Dataset Mixtures**: We natively support arbitrary datasets in RLDS format, including arbitrary mixtures of
  data from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/).
- **Easy Scaling**: Powered by PyTorch FSDP and Flash-Attention, we can quickly and efficiently train models from 1B - 
  34B parameters, with easily adaptable model architectures.
- **Native Fine-Tuning Support**: Built-in support (with examples) for various forms of fine-tuning (full, 
  partial, LoRA).

Built on top of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).

## Getting Started

To get started with loading and running OpenVLA models for inference, we provide a lightweight interface that leverages
HuggingFace `transformers` AutoClasses, with minimal dependencies.

For example, to load `openvla-7b` for zero-shot instruction following in the
[BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) with a WidowX robot:

```python
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
robot.act(action, ...)
```

We also provide an [example script for fine-tuning OpenVLA models for new tasks and 
embodiments](./vla-scripts/finetune.py); this script supports different fine-tuning modes -- including (quantized) 
low-rank adaptation (LoRA) supported by [HuggingFace's PEFT library](https://huggingface.co/docs/peft/en/index). 

For deployment, we provide a lightweight script for [serving OpenVLA models over a REST API](./vla-scripts/deploy.py), 
providing an easy way to integrate OpenVLA models into existing robot control stacks, 
removing any requirement for powerful on-device compute.

## Pretrained VLAs

We release two OpenVLA models trained as part of our work, with checkpoints, configs, and model cards available [on our
HuggingFace page](https://huggingface.co/openvla):
- [`openvla-7b`](https://huggingface.co/openvla/openvla-7b): The flagship model from our paper, trained from 
  the Prismatic `prism-dinosiglip-224px` VLM (based on a fused DINOv2 and SigLIP vision backbone, and Llama-2 LLM). 
  Trained on a large mixture of datasets from Open X-Embodiment spanning 970K trajectories 
  ([mixture details - see "Open-X Magic Soup++"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).
- [`openvla-v01-7b`](https://huggingface.co/openvla/openvla-7b-v01): An early model used during development, trained from
  the Prismatic `siglip-224px` VLM (singular SigLIP vision backbone, and a Vicuña v1.5 LLM). Trained on the same mixture
  of datasets as [Octo](https://github.com/octo-models/octo), but for significantly fewer GPU hours than our final model 
  ([mixture details - see "Open-X Magic Soup"](./prismatic/vla/datasets/rlds/oxe/mixtures.py)).

**Explicit Notes on Model Licensing & Commercial Use**: While all code in this repository is released under an MIT 
License, our pretrained models may inherit restrictions from the underlying base models we use. Specifically, both the
above models are derived from Llama-2, and as such are subject to the 
[Llama Community License](https://ai.meta.com/llama/license/).

---

## Installation

> **Note**: These installation instructions are for full-scale pretraining (and distributed fine-tuning); if looking to
  just run inference with OpenVLA models (or perform lightweight fine-tuning), see instructions above!

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require
PyTorch 2.2.* -- installation instructions [can be found here](https://pytorch.org/get-started/locally/). The latest 
version of this repository was developed and thoroughly tested with:
  - PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, and flash-attn 2.5.5

**[5/21/24] Note**: Following reported regressions and breaking changes in later versions of `transformers`, `timm`, and
`tokenizers` we explicitly pin the above versions of the dependencies. We are working on implementing thorough tests, 
and plan on relaxing these constraints as soon as we can.

Once PyTorch has been properly installed, you can install this package locally via an editable installation (or via 
`pip install git+https://github.com/openvla/openvla`):

```bash
cd openvla
pip install -e .

# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

**Note:** See `vla-scripts/` for full training and verification scripts for OpenVLA models. Note that `scripts/` is
mostly a holdover from the original (base) `prismatic-vlms` repository, with support for training and evaluating
visually-conditioned language models; while you can use this repo to train VLMs AND VLAs, note that trying to generate
language (via `scripts/generate.py`) with existing OpenVLA models will not work (as we only train current OpenVLA models
to generate actions, and actions alone).

## Fine-Tuning OpenVLA via LoRA

In this section, we discuss fine-tuning OpenVLA using Low-Rank Adaptation (LoRA) via the Hugging Face `transformers` library,
which is recommended if you do not have sufficient compute to fully fine-tune a 7B-parameter model. The main script for LoRA
fine-tuning is `vla-scripts/finetune.py`. (If you instead wish to do full fine-tuning, please see the
[Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla) section.)

Below we show an example of how you can fine-tune the main OpenVLA checkpoint ([`openvla-7b`](https://huggingface.co/openvla/openvla-7b))
via LoRA. Here we fine-tune on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single A100
GPU with 80 GB VRAM. (You can also fine-tune with a smaller GPU, as long as it has at least ~27 GB of memory,
by modifying the batch size.)

First, download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Now, launch the LoRA fine-tuning script, as shown below. Note that `--batch_size==16` with `--grad_accumulation_steps==1`
requires ~72 GB GPU memory. If you have a smaller GPU, you should reduce `--batch_size` and increase `--grad_accumulation_steps`
to maintain an effective batch size that is large enough for stable training. If you have multiple GPUs and wish to train via
PyTorch Distributed Data Parallel (DDP), simply set `--nproc-per-node` in the `torchrun` command below to the number of available GPUs.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

Note: If you set `--image_aug==False` in the command above, you will observe nearly 100% `action_accuracy` in the training logs,
since the [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained (without augmentations) on a
superset of datasets that includes BridgeData V2.

To LoRA fine-tune on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets
from OXE). Alternatively, if you have a custom dataset that is not part of OXE, you can either (a) convert the dataset to the RLDS format which is
compatible with our fine-tuning script (see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this), or (b) use your own
custom PyTorch Dataset wrapper (see comments in `vla-scripts/finetune.py` for instructions). We recommend option (a) for most users; the RLDS dataset and
dataloader are tested more extensively since we used these for all of our pretraining and fine-tuning experiments.

For option (a), after you converted your dataset to RLDS, you need to register it with our data loader, by registering a dataset
config [here](prismatic/vla/datasets/rlds/oxe/configs.py#L54) and a dataset transform function [here](prismatic/vla/datasets/rlds/oxe/transforms.py#L828).

Once you have integrated your new dataset, you can launch LoRA fine-tuning with the same `vla-scripts/finetune.py` script above. If you run into any issues,
please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the [OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=)
(including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

## Fully Fine-Tuning OpenVLA

In this section, we discuss <ins>fully fine-tuning</ins> OpenVLA (all 7.5 billion parameters) via native PyTorch Fully Sharded Data Parallel (FSDP)
using the [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) training script. Full fine-tuning is more advanced/involved and is only recommended
if you have sufficient compute (e.g., a full node of 8 A100 GPUs) and if LoRA fine-tuning is insufficient for your use case (e.g., if the fine-tuning distribution
varies drastically from the pretraining distribution). Otherwise, we recommend that you try parameter-efficient fine-tuning via LoRA, which is described in the 
[Fine-Tuning OpenVLA via LoRA](#fine-tuning-openvla-via-lora) section.

For full fine-tuning, you will need to download [a different version of the OpenVLA model checkpoint](https://huggingface.co/openvla/openvla-7b-prismatic) that is compatible
with the Prismatic VLMs codebase, which we built on top of to develop the OpenVLA model. You can download this Prismatic-compatible OpenVLA checkpoint using the git commands below
(alternatively, you can download via the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)):

```bash
# Change directory to your base model checkpoints folder
cd <PATH TO BASE MODEL CHECKPOINTS DIR>

# Download checkpoint (30 GB) -- may take a few minutes
git clone git@hf.co:openvla/openvla-7b-prismatic

# If the command above did not download the full checkpoint,
# manually fetch it via git Large File Storage (LFS)
# Note: You may have to configure an SSH key for this to work
cd openvla-7b-prismatic
git lfs fetch --all
```

We show how you can fully fine-tune OpenVLA on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single node with 8 GPUs. If you wish to
use a different number of GPUs (or nodes), you can modify the VLA training configuration in [`prismatic/conf/vla.py`](prismatic/conf/vla.py).

Download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Next, create a [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) and copy the token value (a string that starts with
`hf_...`) into a file named `.hf_token` at the root directory of this repo (`openvla/.hf_token`).

```bash
# Go to openvla root directory
cd openvla

# Copy HF token value into token file. Replace "hf_..." with your own token value!
# See: https://huggingface.co/docs/hub/en/security-tokens
echo hf_... >>> .hf_token
```

Now, launch the training script. If you wish to use a different number of nodes or GPUs, modify the VLA training configuration in
[`prismatic/conf/vla.py`](prismatic/conf/vla.py) and then change the `--nnodes` and `--nproc-per-node` arguments below accordingly.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <PATH TO openvla/openvla-7b-prismatic CHECKPOINT FILE: step-295000-epoch-40-loss=0.2200.pt> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --run_id <OPTIONAL RUN ID FOR WANDB LOGGING> \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_interval <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> \
  --is_resume False
```

Note that the `--is_resume` argument is set to `False` above since we are fine-tuning a pretrained checkpoint rather than resuming a paused training run.

If your training run gets paused and you wish to resume from the latest checkpoint, change `--pretrained_checkpoint` to the latest checkpoint path,
and then set `--is_resume==True` and specify `--resume_step` and `--resume_epoch` as the step and epoch number, respectively. For example, if you wish to
resume training from a checkpoint named `step-010000-epoch-20-loss=0.0160.pt`, you would set `is_resume==True`, `resume_step==10000`, and `resume_epoch==20`.

Note: If you run the BridgeData V2 fine-tuning command above, you should observe nearly 100% Action Token Accuracy in the training logs, since the
[`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained on a superset of datasets that includes BridgeData V2.

To fully fine-tune OpenVLA on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets from OXE).
Alternatively, if you have a custom dataset that is not part of OXE, you can convert the dataset to the RLDS format, which is compatible with our fine-tuning script
(see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this). After downloading/converting the dataset, you will need to modify the following files:

* [`prismatic/conf/vla.py`](prismatic/conf/vla.py): Add a new training configuration by creating an experiment class, and then register it in the `VLARegistry` at the bottom of the file.
  * Make sure to create a new unique `vla_id` for your fine-tuning run, and adjust some configuration variables as needed – e.g., `expected_world_size` (number of GPUs),
  `per_device_batch_size` (batch size per GPU), `global_batch_size` (total batch size), `shuffle_buffer_size` (number of samples in shuffle buffer per GPU), etc. See comments
  under the `VLAConfig` class at the top of the file to understand the purpose of each variable.
* [`prismatic/vla/datasets/rlds/oxe/mixtures.py`](prismatic/vla/datasets/rlds/oxe/mixtures.py): Define a new mixture for your fine-tuning mixture in the `OXE_NAMED_MIXTURES` dictionary.
* [`prismatic/vla/datasets/rlds/oxe/transforms.py`](prismatic/vla/datasets/rlds/oxe/transforms.py): Define a new dataset transform function for your fine-tuning dataset, and add it to the
`OXE_STANDARDIZATION_TRANSFORMS` registry at the bottom of the file.
* [`prismatic/vla/datasets/rlds/oxe/configs.py`](prismatic/vla/datasets/rlds/oxe/configs.py): Add a new configuration specifying your fine-tuning dataset's observation and action spaces
to the `OXE_DATASET_CONFIGS` dictionary.

After completing the steps above, you can start full fine-tuning using the `vla-scripts/train.py` script. Make sure to set the `--vla.type` argument to the new `vla_id` that you added in `prismatic/conf/vla.py`.

When you are finished with fine-tuning, you will need to convert the final model checkpoint to a version that is
compatible with the Hugging Face `transformers` library. See the [Converting Prismatic Models to Hugging Face](#converting-prismatic-models-to-hugging-face) section for instructions.

If you run into any issues, please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the
[OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=) (including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

### Converting Prismatic Models to Hugging Face

If you have used the Prismatic VLMs codebase to train your model (e.g., if you did full fine-tuning of OpenVLA on a
new dataset), you will need to convert the final checkpoint to a version that is compatible with Hugging Face
`transformers` AutoClasses. We discuss how to do so in this section.

Let's say your training run directory is `PRISMATIC_RUN_DIR` (e.g., `prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7`).
Inside this directory, there should be a directory called `checkpoints` which contains saved model checkpoints (e.g.,
`step-295000-epoch-40-loss=0.2200.pt`). The Prismatic-to-Hugging-Face conversion script
([convert_openvla_weights_to_hf.py](vla-scripts/extern/convert_openvla_weights_to_hf.py)) expects a checkpoint file
named `latest-checkpoint.pt`. Therefore, you should first create a symbolic link called `latest-checkpoint.pt` that
points to the checkpoint file that you wish to convert:

```bash
# Go to your Prismatic training run's `checkpoints` directory
cd PRISMATIC_RUN_DIR/checkpoints

# Create symbolic link pointing to your checkpoint file
ln -s <YOUR CHECKPOINT FILENAME> latest-checkpoint.pt
```

Then, launch the conversion script to convert the checkpoint from the Prismatic VLMs format to the Hugging Face format:

```bash
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
    --output_hf_model_local_path <OUTPUT DIR FOR CONVERTED CHECKPOINT>
```

The command above will save the HF-compatible checkpoint in `output_hf_model_local_path`. Now you can load the checkpoint
with HF AutoClasses as normal, as shown below. Note that there is an additional necessary step to register the OpenVLA model
to HF AutoClasses before loading it because you are loading a locally saved checkpoint rather than one that is pushed to the
HF Hub (see [here](https://huggingface.co/docs/transformers/en/custom_models#registering-a-model-with-custom-code-to-the-auto-classes)
for details).

```python
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("<PATH TO CONVERTED CHECKPOINT DIR>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<PATH TO CONVERTED CHECKPOINT DIR>",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

...
```

## Training VLAs from Scratch

We provide full instructions and configurations for training VLA models on (arbitrary subsets of) the
[Open X-Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/). If you run in to any issues with 
the following, see [VLA Troubleshooting](#vla-troubleshooting) below (or file a GitHub Issue).

### VLA Pretraining Datasets

We download and preprocess individual datasets from Open X-Embodiment in [RLDS format](https://github.com/google-research/rlds) following 
[this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh). See 
[mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py) for the full list of component datasets (and mixture 
weights) we use to train `openvla-7b`. 
- **Important**: For the BridgeData V2 component, the version in OXE is out of date (as of 12/20/2023). Instead,
  you should download the dataset from the [official website](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) and place it under the subdirectory `bridge_orig/`. 
  Replace any reference to `bridge` in the OXE code with `bridge_orig`.

### VLA Configuration & Training Script

The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We use 
[`draccus`](https://pypi.org/project/draccus) to provide a modular, dataclass-based interface for specifying VLA 
training configurations; existing VLA configurations are in [`prismatic/conf/vla.py`](prismatic/conf/vla.py). You can 
add your own training configuration and refer to it using the `--vla.type` command line argument.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs. Launch training via `torchrun`:

```bash
# Train VLA on BridgeData V2 with the Prismatic DINO-SigLIP 224px Backbone on a Single Node (w/ 8 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

### VLA Troubleshooting

The following are a list of known problems and corresponding fixes:

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **Fix**: Downgrade `tensorflow-datasets` via `pip install tensorflow-datasets==4.9.3`.


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **Fix**: Upgrade `dlimp` to the newest version. You may have to `--force-reinstall` like so:
`pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `vla-scripts/` - Core scripts for training, fine-tuning, and deploying VLAs.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---

#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
