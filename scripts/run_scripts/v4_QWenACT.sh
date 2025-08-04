
export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# 用于check save 的时候的通信
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # 超时时间设为 1 小时（单位：秒）

cd /mnt/petrelfs/yejinhui/Projects/llavavla

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


accelerate launch \
  --config_file llavavla/conf/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  llavavla/training/train_qwenvla.py \
  --config_yaml ./llavavla/conf/qwenvla_cotrain_lerobot.yaml \
  --framework.qwenvl.base_vlm ${MODEL_PATH} \
  --data_root_dir ${data_root_dir} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project qwendino \
  --wandb_entity jinhuiye \
  # --is_debug True


