This document provides instructions for reproducing our **experimental results** with SimplerEnv.  

# 🚀 Eval SimplerEnv


The evaluation process consists of two main parts:  

1. Setting up the `simpler_env` environment and dependencies.  
2. Running the evaluation by launching services in both `internvla_m1` and `simpler_env` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---

## 📊 Experimental Results

### 1. WidowX Robot

| Task                              | Success Rate (%) |
| --------------------------------- | ---------------- |
| Put Spoon on Towel                | 87.5             |
| Put Carrot on Plate               | 67.9             |
| Stack Green Block on Yellow Block | 31.3             |
| Put Eggplant in Yellow Basket     | 100.0            |
| **Average**                       | **71.7**         |

---

### 2. Google Robot (Visual Matching)

| Task                                     | Success Rate (%) |
| ---------------------------------------- | ---------------- |
| Pick Coke Can                            | 95.3             |
| Move Near                                | 90.0             |
| Open/Close Drawer                        | 75.5             |
| Open Top Drawer and Place Apple          | 62.0             |
| **Average**                              | **80.7**         |

---

### 3. Google Robot (Variant Aggregation)

| Task                                     | Success Rate (%) |
| ---------------------------------------- | ---------------- |
| Pick Coke Can                            | 86.1             |
| Move Near                                | 82.0             |
| Open/Close Drawer                        | 72.0             |
| Open Top Drawer and Place Apple          | 64.0             |
| **Average**                              | **76.0**         |

---

## ⬇️ 0. Download Checkpoints
First, download the checkpoints from [[InternVLA-M1-Pretrain-RT-1-Bridge](https://huggingface.co/InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge)]



## 📦 1. Environment Setup

To set up the environment, please first follow the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) to install the base `simpler_env` environment.  Note: 你不需要 将SimplerEnv install 到M1 training 环境上， 因为他们通过socket通讯

Afterwards, inside the `simpler_env` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

⚠️ **Common Issues**
When testing SimplerEnv on NVIDIA A100, you may encounter the following error:
`libvulkan.so.1: cannot open shared object file: No such file or directory`
You can refer to this link to fix: [Installation Guide – Vulkan Section](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **internvla_m1 environment**: runs the policy inference server.  
- **simpler_env environment**: runs the simulation eval code.  

### Step 1. Start the server (internvla_m1 environment)

In the first terminal, activate the `internvla_m1` conda environment and run:  

```bash
bash examples/SimplerEnv/start_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in  
`examples/SimplerEnv/start_server.sh`  

---

### Step 2. Start the simulation (simpler_env environment)

In the second terminal, activate the `simpler_env` conda environment and run:  

```bash
export MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1_need/0906_bestvla_retrain_sota2/checkpoints/steps_50000_pytorch_model.pt
bash examples/SimplerEnv/start_simpler_env.sh ${MODEL_PATH} 
```
This script will automatically launch the WidowX Robot evaluation tasks, reproducing the benchmark results reported above.

⚠️ **Note:** Please ensure that you specify the correct `SimplerEnv_PATH`
`examples/SimplerEnv/start_simpler_env.sh`  


# 🚀 Training on OXE

## Data Preparation
1. Prepare the OXE data following the GR00T / Open-X Embodiment procedure (download + convert). Keep only the subsets you need (e.g. bridge / rt1, etc.).
2. YAML parameter snippet (InternVLA/config/training/internvla_cotrain_oxe.yaml):
```
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: playground/Datasets/OXE_LEROBOT_DATASET
    data_mix: bridge_rt_1   # change or extend if you add more mixture
```
3. Make sure you can load batched data:
```bash
python InternVLA/dataloader/lerobot_datasets.py --config_yaml InternVLA/config/training/internvla_cotrain_oxe.yaml
```

## Training
Run:
```bash
bash /mnt/petrelfs/yejinhui/Projects/llavavla/scripts/run_scripts/run_lerobot_datasets.sh
```
Make sure the script explicitly uses the validated config path in `run_lerobot_datasets.sh` (add --config_yaml if not already passed).



