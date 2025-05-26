#!/bin/bash
#SBATCH --job-name=qwenact_pd        # name
#SBATCH -p efm_p
#SBATCH -N 2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/mnt/petrelfs/yejinhui/Projects/llavavla/results/logs/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/yejinhui/Projects/llavavla/results/logs/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-77

# source ~/.bashrc     # 确保 conda 命令可用
# source ~/.zshrc
# source ~/envs4jinhui.sh
# proxy_on

# conda activate llavavla310  # 替换为你的环境名

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))

# @yangshuai: My keys are listed below, but they may no longer be usable.

# proxy_on
# to test if you can access ceph, you are expected to see:
#                            PRE open_x_embodiment_origin/

# to fix: libcudnn_ops_infer.so.8 with link time referencesymbol _ZN15TracebackLoggerC1EPKc
# export LD_LIBRARY_PATH=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export LD_PRELOAD=~/miniconda3/envs/openvla-simpler/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8
# export hf_token=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

# envs for llavavla

export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache
export HF_TOKEN=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


cd /mnt/petrelfs/yejinhui/Projects/llavavla
# conda activate llavavla310
proxy_on

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
export MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # 必须是绝对路径，因为simper 会在其他工程测试，需要这个路径， @请在后续版本修复这个东西
export data_root_dir=./playground/Datasets/OXE_openvla
export run_root_dir=./results/Checkpoints
export run_id=0523_pd_qwenact_bridge_rt_128gpus

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
echo "Total GPUs: $TOTAL_GPUS"

#   --vla.expected_world_size ${TOTAL_GPUS} \ 后续这些要从代码中移除
#   --vla.global_batch_size 512 \

srun --jobid $SLURM_JOBID bash -c 'torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
 scripts/train_qwen.py \
  --vla.base_vlm ${MODEL_PATH} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix bridge_rt_1 \
  --vla.expected_world_size 128 \
  --vla.global_batch_size 2048 \
  --vla.per_device_batch_size 16 \
  --vla.learning_rate 2e-5 \
  --data_root_dir ./playground/Datasets/OXE_openvla \
  --run_root_dir $run_root_dir \
  --run_id $run_id \
  --image_aug True \
  --wandb_project llavavla \
  --wandb_entity jinhuiye \
  --hf_token HF_TOKEN \
  --save_interval 10000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False '

