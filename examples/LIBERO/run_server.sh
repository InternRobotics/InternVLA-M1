#!/bin/bash

your_ckpt=/mnt/petrelfs/yujunqiu/code/vla-baseline/intern-clean5/playground/Checkpoints/open_source/0903_libero_goal_augsteps_0_wo_flash_attention_wo_augsteps_two_view_action_chunk_8_pretrained_vlm/checkpoints/steps_30000_pytorch_model.pt
base_port=10093

python deployment/model_server/server_policy_M1.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16