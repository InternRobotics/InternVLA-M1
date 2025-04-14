
export HF_HOME=/fs-computility/efm/shared/model_weights/huggingface_cache
export HF_TOKEN=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

cd /fs-computility/efm/yejinhui/Projects/CogACT

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
MODEL_PATH=/fs-computility/efm/yejinhui/Projects/CogACT/playground/Pretrained_models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt

torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint ${MODEL_PATH} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix bridge \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 16 \
  --vla.learning_rate 2e-5 \
  --data_root_dir /fs-computility/efm/yejinhui/Projects/CogACT/playground/Datasets/OXE_openvla \
  --run_root_dir /fs-computility/efm/yejinhui/Projects/CogACT/playground/Checkpoints \
  --run_id produce_cogact_FFT_bridge \
  --image_aug True \
  --wandb_project two_system_vla \
  --wandb_entity jinhuiye \
  --hf_token HF_TOKEN \
  --save_interval 10000 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False


# for i in {1..70}; do
#   torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
#     --vla.type prism-dinosiglip-224px+oxe+diffusion \
#     --vla.data_mix custom_finetuning \
#     --vla.expected_world_size 8 \
#     --vla.global_batch_size 256 \
#     --vla.per_device_batch_size 16 \
#     --vla.learning_rate 2e-5 \
#     --data_root_dir /fs-computility/efm/yejinhui/Projects/CogACT/playground/Datasets/GenManipDatasets \
#     --run_root_dir /fs-computility/efm/yejinhui/Projects/CogACT/playground/Checkpoints \
#     --run_id cogact_FFT_GenManipTiny \
#     --image_aug True \
#     --wandb_project two_system_vla \
#     --wandb_entity jinhuiye \
#     --hf_token HF_TOKEN \
#     --save_interval 1000 \
#     --repeated_diffusion_steps 8 \
#     --future_action_window_size 15 \
#     --action_model_type DiT-B \
#     --is_resume False
# done


