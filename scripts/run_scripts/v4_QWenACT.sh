
export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache
export HF_TOKEN=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

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# 用于check save 的时候的通信
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # 超时时间设为 1 小时（单位：秒）

cd /mnt/petrelfs/yejinhui/Projects/llavavla
# conda activate llavavla310  # some bug here, plz activate at terminal

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
MODEL_PATH=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
data_root_dir=./playground/Datasets/OXE_openvla
run_root_dir=./playground/Checkpoints
run_id=0528_debug
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

  # --pretrained_checkpoint ${MODEL_PATH} \
# CUDA_VISIBLE_DEVICES=0 

accelerate launch \
  --config_file scripts/run_scripts/deepspeed_zero2_v2.yaml \
  --num_processes=8 llavavla/training/train_qwen_qformer_dit.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.base_vlm ${MODEL_PATH} \
  --vla.data_mix bridge \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 128 \
  --vla.per_device_batch_size 16 \
  --vla.freeze_modules "qwen_vl_interface" \
  --vla.learning_rate 2e-5 \
  --vla.qformer_start_layer 0 \
  --vla.qformer_end_layer 37 \
  --vla.max_steps 3000000 \
  --data_root_dir ${data_root_dir} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug True \
  --wandb_project llavavla \
  --wandb_entity jinhuiye \
  --hf_token HF_TOKEN \
  --save_interval 10 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False \
  --is_debug True



