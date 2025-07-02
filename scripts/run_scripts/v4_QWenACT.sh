
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
run_id=0622_debug

datasets_grounding=asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

  # --pretrained_checkpoint ${MODEL_PATH} \
# export CUDA_VISIBLE_DEVICES=4,5,6,7

datasets_vlm=aokvqa_cauldron_llava_format,sharegpt4v_coco,sharegpt4v_knowledge,sharegpt4v_llava,sharegpt4v_sam
datasets_grounding=asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en
export system2_datasets="${datasets_vlm},${datasets_grounding}"
# ,${datasets_grounding}
  # --vlm_data.min_pixels 3136 \
  # --vlm_data.max_pixels 12845056 \
  # --vla.freeze_modules qwen_vl_interface.model.model.visual \

accelerate launch \
  --config_file scripts/run_scripts/deepspeed_zero2_v2.yaml \
  --num_processes 8 \
  llavavla/training/train_qwenvla.py \
  --config_yaml ./llavavla/conf/qwenvla.yaml \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.base_vlm ${MODEL_PATH} \
  --vlm_data.dataset_use ${system2_datasets} \
  --vla.per_device_batch_size 16 \
  --vlm_data.per_device_batch_size 4 \
  --trainer.learning_rate.base 5e-5 \
  --vla.qformer_start_layer 36 \
  --vla.qformer_end_layer 37 \
  --vlm_data.min_pixels 784 \
  --vlm_data.max_pixels 50176 \
  --vla.max_steps 10000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug True \
  --wandb_project llavavla \
  --wandb_entity jinhuiye \
  --save_interval 500 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False \
  # --is_debug True



