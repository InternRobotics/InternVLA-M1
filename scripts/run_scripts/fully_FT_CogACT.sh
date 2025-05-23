
export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache
export HF_TOKEN=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
export WANDB_MODE=disabled



cd /mnt/petrelfs/yejinhui/Projects/llavavla
conda activate llavavla310

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">
MODEL_PATH=./playground/Pretrained_models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt
MODEL_PATH=./playground/Checkpoints/produce_cogact_bridge_rt--image_aug/checkpoints/step-000200-epoch-00-loss=0.2031.pt

data_root_dir=./playground/Datasets/OXE_openvla
run_root_dir=./playground/Checkpoints
run_id=produce_cogact_bridge_rt

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint ${MODEL_PATH} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
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
  --save_interval 100 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume True \
  --resume_epoch 0 \
  --resume_step 200


  # --is_resume False \

