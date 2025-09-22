# 🚀 Eval Libero

This document provides instructions for reproducing our **experimental results** with libero.  
The evaluation process consists of two main parts:  

1. Setting up the `libero` environment and dependencies.  
2. Running the evaluation by launching services in both `internvla_m1` and `libero` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

---

## 📊 Experimental Results
|                 | Spatial | Objects | Goal | Long | Avg  |
|-----------------|---------|---------|------|------|------|
| GR00T N1        | 94.4    | 97.6    | 93.0 | 90.6 | 93.9 |
| $\pi_0$         | 96.8    | 98.8    | 95.8 | 85.2 | 94.2 |
| $\pi_{0.5}$-Fast| 96.4    | 96.8    | 88.6 | 60.2 | 85.5 |
| $\pi_{0.5}$-KI  | 98.0    | 97.8    | **95.6** | 85.8 | 94.3 |
| InternVLA-M1    | **98.0**    | **99.0**    | 93.8 | **92.6** | **95.9** |

---

## ⬇️ 0. Download Checkpoints
First, download the checkpoints from 
- [LIBERO-Object](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Object)
- [LIBERO-Spatial](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Spatial)
- [LIBERO-Goal](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Goal)
- [LIBERO-Long](https://huggingface.co/InternRobotics/InternVLA-M1-LIBERO-Long)



## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `libero` environment.  

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **internvla_m1 environment**: runs the inference server.  
- **libero environment**: runs the simulation.  

### Step 1. Start the server (internvla_m1 environment)

In the first terminal, activate the `internvla_m1` conda environment and run:  

```bash
bash examples/libero/run_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/libero/run_server.sh`  


---

### Step 2. Start the simulation (libero environment)

In the second terminal, activate the `libero` conda environment and run:  

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
bash examples/libero/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/libero/eval_libero.sh`  

