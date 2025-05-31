#!/bin/bash
#SBATCH --job-name=qwenact_ds        # name
#SBATCH -p efm_p
#SBATCH -N 2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/mnt/petrelfs/yejinhui/Projects/llavavla/results/logs/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/yejinhui/Projects/llavavla/results/logs/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-49

# [8,34,47,49,93-94]

# source ~/.bashrc     # 确保 conda 命令可用
# source ~/.zshrc
# source ~/envs4jinhui.sh
# proxy_on

# conda activate llavavla310  # 替换为你的环境名

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))

export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache
export HF_TOKEN=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


cd /mnt/petrelfs/yejinhui/Projects/llavavla
# conda activate llavavla310
proxy_on

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
export MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # 必须是绝对路径，因为simper 会在其他工程测试，需要这个路径， @请在后续版本修复这个东西
export data_root_dir=./playground/Datasets/OXE_openvla
export run_root_dir=./results/Checkpoints
export lr=1e-1 # defualt export lr=1e-4

export run_id=0531_qwenact_fixqwen_16gpus_lr_${lr}

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
echo "Total GPUs: $TOTAL_GPUS"

#   --vla.expected_world_size ${TOTAL_GPUS} \ 后续这些要从代码中移除
#   --vla.global_batch_size 512 \
  # --num_processes=${TOTAL_GPUS} 是要说一共有多少卡，这个没有torchrun 直观， 之后改成torchrun 来管理
# 这个地方很😡直觉，需要check一下, 确认了官方的说法确实 total

srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --config_file scripts/run_scripts/deepspeed_zero2_v2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  llavavla/training/train_qwen_qformer_dit.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.base_vlm ${MODEL_PATH} \
  --vla.data_mix bridge \
  --vla.expected_world_size ${TOTAL_GPUS} \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 16 \
  --vla.learning_rate ${lr} \
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
  --is_resume False '

