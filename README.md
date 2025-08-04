# LLaVA-VLA

## 特性

- **模块化设计**，包含 **视觉编码**、**语言处理** 和 **动作建模** 组件。
- **同时支持 VLM 和 VLA 任务**，可灵活适配不同应用场景。
- **开源可扩展** 遵循高内聚，低耦合，支持变体扩展。


## 文件结构

```
LLaVA-VLA
├── model                # 模型相关代码
│   ├── vlm   # 处理这里这了实现各种VLM, LLM
│   ├── projector        # 这里开发各个模块的 align moduless
│   ├── action_model     # 执行视觉语言动作
│   ├── framework        # 这里对应的是论文的主图， 模型， 数据流，loss 搭建都在这里
│   ├── openvla          # 这个是本地 cogact依赖， 在后续开发中会被移除
│
├── dataloader           # 收据构建和预处理
│
├── training             # 训练相关代码
│
├── conf                 # 配置文件
│
├── README.md            # 项目说明文件
├── requirements.txt     # 依赖包列表
```

# 最佳开发：
1. 在 framework 一个.py 就是一个论文的 model, 可以理由 其他文件夹的share模块或者自己在.py local 定义 modules (经过考虑后可以移动到share)

2. 全部模型参数 ，训练参数全部在 conf global 的方式分组管理。

# 愿景: 开发一个可以同时支持 VLM traning (System2) 和 VLA training 的框架


## 希望的feature 和 理想的脚本
1. Pretraining VLM
2. Pretraining DiT
3. align VLM with DiT (希望在 5 epcoh 内完成 alignment) # done 在1 epoch 就能完成


## 开发规划
1. 支持 QwenACT 的training (done)
2. 支持同时training VLA 和 VLM (done)
3. 支持同时 align VLA DiT and Qwen (done) 

4. 支持 单独 training VLM with own vision encode (pending) #直接用QWen
5. 支持 单独 training ACT with own vision encode (pending) # 直接用openVLA



### setup envs
'''bash

cd llavavla/model/openvla
conda create -n llavavla --python 3.10
pip install -r requirements.txt
pip install -e .
<!-- 他们的 pyproject.toml 里面已经有很多包的版本很难install， 比如python 版本绑定为 3.10 -->
<!-- 移除 presmiatic 之后将不需要 -->


<!-- hard to pip install flash_attn-->
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


'''

### run vla only 

bash /mnt/petrelfs/yejinhui/Projects/llavavla/scripts/run_scripts/qwenpi_bench200.sh # prepare OXE_LEROBOT_DATASET and QWenvl 3B to playground



### eval 

sample to https://github.com/microsoft/CogACT?tab=readme-ov-file#getting-started


## 许可证

MIT License


git remote add gitee https://gitee.pjlab.org.cn/L2/yejinhui/llavavla.git