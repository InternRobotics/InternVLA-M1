#!/bin/bash
#SBATCH --job-name=fm_qwenact            # name
#SBATCH -p efm_p
#SBATCH -N 4                         # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/mnt/petrelfs/yejinhui/Projects/llavavla/results/logs/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/yejinhui/Projects/llavavla/results/logs/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-49

# [8,34,47,49,93-94]
# SH-IDCA1404-10-140-54-25 

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



cd /mnt/petrelfs/yejinhui/Projects/llavavla
# conda activate llavavla310
proxy_on

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
export MODEL_PATH=/mnt/petrelfs/share/efm_p/zhuyangkun/share_model/release_model/manip_sys2_genmanipdata_coco_0703 # 必须是绝对路径，因为simper 会在其他工程测试，需要这个路径， @请在后续版本修复这个东西
export data_root_dir=./playground/Datasets/OXE_openvla
export run_root_dir=./results/Checkpoints
export qformer_start_layer=36
export qformer_end_layer=37
export vlm_per_batch_size=4
export vla_per_device_batch_size=16

export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
export global_batch_size=$((TOTAL_GPUS * vla_per_device_batch_size)) # 512 is the default global batch size, you can change it if needed
echo "Total GPUs: $TOTAL_GPUS"

datasets_vlm=aokvqa_cauldron_llava_format,sharegpt4v_coco,sharegpt4v_knowledge,sharegpt4v_llava,sharegpt4v_sam
datasets_grounding=asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en
# ,${datasets_grounding}
datasets_genmanip_sys2=genmanip_sys2_14k_v2_action_plan%10,genmanip_sys2_14k_v2_grounding_rule_coco%20,genmanip_sys2_14k_v2_img_caption_rule%20,genmanip_sys2_14k_v2_obj_caption_rule%10,genmanip_sys2_14k_v2_qa_rule_llm_1obj_attr%10,genmanip_sys2_14k_v2_qa_rule_llm_1obj_nearby%10,genmanip_sys2_14k_v2_qa_rule_llm_1obj_senmatic%20

export system2_datasets="${datasets_vlm},${datasets_grounding}"

export llm_hook_weight=1 # 暂时不使用， 过于复炸， 效果不确定
# 其实如果能够生效，上面的方式是最直接的

# export lr=5e-5
# export qwen_vl_interface_lr=2e-5
# export action_model_lr=1e-4
export loss_scale_vla=1.0 # 1.0 is the default value, you can change it if needed
export loss_scale_vlm=0.1 # 1.0 is the default value, you can change it if needed

# 开始用 坤哥 的system2
export run_id=0911_qwenact_fm_debug

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

#   --vla.expected_world_size ${TOTAL_GPUS} \ 后续这些要从代码中移除
#   --vla.global_batch_size 512 \
  # --num_processes=${TOTAL_GPUS} 是要说一共有多少卡，这个没有torchrun 直观， 之后改成torchrun 来管理
# 这个地方很😡直觉，需要check一下, 确认了官方的说法确实 total

  #   --trainer.freeze_modules qwen_vl_interface \

export pretrained_checkpoint=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0815_qwendino_vla_cotrain/checkpoints/steps_70000_pytorch_model.pt
  # --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
# bridge_rt_1
# oxe_magic_soup_plus

  # --framework.dino.dino_backbone dinov2_vits14 \
  # --trainer.pretrained_checkpoint ${pretrained_checkpoint} \

# 换回原来朴素版的 image transform

# qwendino_cogactheader
Framework_name=qwenact_fmheader

srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --config_file scripts/run_scripts/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  llavavla/training/train_qwenvla.py \
  --config_yaml ./llavavla/config/lerobot_data/qwenvla_cotrain_oxe.yaml \
  --framework.qwenvl.base_vlm ${MODEL_PATH} \
  --framework.framework_py ${Framework_name} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.freeze_modules "" \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Internvla_V2 \
  --wandb_entity jinhuiye \
  --is_resume False '

