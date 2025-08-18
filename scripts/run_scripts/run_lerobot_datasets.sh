
export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# 用于check save 的时候的通信
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # 超时时间设为 1 小时（单位：秒）

cd /mnt/petrelfs/yejinhui/Projects/InternVLA

MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # must be a local path, due to simpler will run in other where
data_root_dir=./playground/Datasets/OXE_LEROBOT_DATASET
run_root_dir=./playground/Checkpoints
run_id=0722_qwenpi

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

  # --pretrained_checkpoint ${MODEL_PATH} \
# export CUDA_VISIBLE_DEVICES=4,5,6,7

  # --datasets.vla_data.data_mix libero_goal \
  # --framework.framework_py qwenpi \
export pretrained_checkpoint=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0815_qwendino_vla/checkpoints/steps_10000_pytorch_model.pt

accelerate launch \
  --config_file llavavla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  llavavla/training/train_qwenvla.py \
  --config_yaml ./llavavla/config/lerobot_data/qwenvla_cotrain_oxe.yaml \
  --framework.qwenvl.base_vlm ${MODEL_PATH} \
  --data_root_dir ${data_root_dir} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 100000 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --trainer.save_interval 10000 \
  --trainer.eval_interval 10 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Internvla \
  --wandb_entity jinhuiye \
  # --is_debug True


