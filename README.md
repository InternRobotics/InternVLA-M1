
# add log and demo video here


# Introduction
InternVLA-M1 is a open-source, end-to-end vision–language–action (VLA) framework. 

1. Dual-System and Dual-Supervision
InternVLA-M1 integrates both a language head and an action head within a single model. This enables training VLAs with multimodal data—particularly robotic perception data—enhancing interaction-following capabilities and generalization performance.

2. Efficient Training and Better Performance
The model supports standalone pretraining on large-scale multimodal datasets to learn spatial priors. Through spatial prompt post-training, these priors can be effectively transferred to downstream tasks. For instance, InternVLA-M1 achieves state-of-the-art performance on the OXE dataset with faster convergence (∼2.5 epochs) even without action-specific pretraining.

3. Modular and Extensible Framework
InternVLA-M1 draws inspiration from leading open-source works and emphasizes a highly modular codebase. Each major component is designed to be independently executable and easily modifiable, facilitating further research and development.


## 文件结构

```
InternVLA
├── model                # 模型相关代码
│   ├── framework        # 这里对应的是论文的主图， 模型， 数据流，loss 搭建都在这里
│   ├── modules   # 处理这里这了实现各种 InternVLA-M1 需要的模块
│   │   ├── vlm   # 处理这里这了实现各种VLM, LLM
│   │   ├── action_model     # 执行视觉语言动作
│   │   ├── projector        # 这里开发各个模块的 align moduless
│   │   ├── dino_model       # 提取视觉细节特征
│
├── dataloader           # 收据构建和预处理
│   ├── groot_lerobot    # lerobot 数据格式, 简洁 groot 做数据管理
├── training             # 训练相关代码
│   ├── train_vlm    # lerobot 数据格式, 简洁 groot 做数据管理
│   ├── train_vla    # lerobot 数据格式, 简洁 groot 做数据管理
│   ├── train_vla_withCotrain    # lerobot 数据格式, 简洁 groot 做数据管理
├── config                  # global的统一上实验配置文件

```


### setup envs

'''bash

conda create -n internVLA python=3.10

pip install -r requirements.txt

pip install -e .

<!-- hard to pip install flash_attn-->
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

'''




### prepare data
download lerobot format dataset (e.g.,[LIBERO](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot))

soft link your dataset to ./playground/Datasets/LEROBOT_LIBERO_DATA


### run vla only 

bash scripts/run_scripts/run_lerobot_datasets.sh # prepare OXE_LEROBOT_DATASET and QWenvl 3B to playground



### eval 

我们的评价采用 server的形式， 首先 
1. 讲本地模型部署为 soker

python /mnt/petrelfs/yejinhui/Projects/llavavla/real_deployment/deploy/server_policy.py

2. install LIBERO by following 

3. 


## 许可证

MIT License

