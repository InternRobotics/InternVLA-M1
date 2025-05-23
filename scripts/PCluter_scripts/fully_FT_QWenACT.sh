
export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache
export HF_TOKEN=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

cd /mnt/petrelfs/yejinhui/Projects/llavavla
# conda activate llavavla310  # some bug here, plz activate at terminal

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
MODEL_PATH=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
data_root_dir=./playground/Datasets/OXE_openvla
run_root_dir=./playground/Checkpoints
run_id=produce_qwenact_bridge_rt
  # --pretrained_checkpoint ${MODEL_PATH} \

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train_qwen.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.base_vlm ${MODEL_PATH} \
  --vla.data_mix bridge_rt_1 \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 16 \
  --vla.learning_rate 2e-5 \
  --data_root_dir ${data_root_dir} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug True \
  --wandb_project llavavla \
  --wandb_entity jinhuiye \
  --hf_token HF_TOKEN \
  --save_interval 10000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False \
  --is_debug True



