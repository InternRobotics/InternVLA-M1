# 🚀 Eval SimplerEnv

This document provides instructions for reproducing our **experimental results** with SimplerEnv.  
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

To set up the environment, please first follow the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) to install the base `simpler_env` environment.  

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

- **internvla_m1 environment**: runs the inference server.  
- **simpler_env environment**: runs the simulation.  

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
export PYTHONPATH=$(pwd):$PYTHONPATH
bash examples/SimplerEnv/start_simpler_env.sh
```
This script will automatically launch the WidowX Robot evaluation tasks, reproducing the benchmark results reported above.

⚠️ **Note:** Please ensure that you specify the correct rgb_overlay_path in  
`examples/SimplerEnv/start_simpler_env.sh`  



🚀 Training on OXE

We are finalizing the OXE training readme. A straightforward guide and scripts will be released soon (this week). If you already have OXE-lerobot-formatted data and need early access.

