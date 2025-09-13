# LLaVA-VLA

## 特性


## 文件结构

```
LLaVA-VLA
├── model                # 模型相关代码
│   ├── vlm   # 处理这里这了实现各种VLM, LLM
│   ├── projector        # 这里开发各个模块的 align moduless
│   ├── action_model     # 执行视觉语言动作
│   ├── framework        # 这里对应的是论文的主图， 模型， 数据流，loss 搭建都在这里
│
├── dataloader           # 收据构建和预处理
│
├── training             # 训练相关代码
│
├── config                 # 配置文件
│
├── README.md            # 项目说明文件
├── requirements.txt     # 依赖包列表
```


### setup envs

'''bash

conda create -n llavavla python=3.10

pip install -r requirements.txt

pip install -e . --no-deps



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

