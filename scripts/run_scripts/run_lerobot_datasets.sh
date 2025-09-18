
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# 用于check save 的时候的通信
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # 超时时间设为 1 小时（单位：秒）

cd InternVLA-M1

Framework_name=InternVLA-M1
base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # must be a local path, due to simpler will run in other where

run_root_dir=./playground/Checkpoints
run_id=debug_0910_internM1

export WANDB_MODE=disabled



output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file scripts/run_scripts/deepspeed_zero2.yaml \
  --num_processes 8 \
  InternVLA/training/train_qwenvla.py \
  --config_yaml ./InternVLA/config/training/internvla_cotrain_oxe.yaml \
  --framework.framework_py ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.eval_interval 10 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project internvla \
  --wandb_entity jinhuiye \
  # --is_debug True


