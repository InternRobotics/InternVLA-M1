# 4 (URDF 变体) × 1 (模型路径) = 4

MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0604_ftqwen_bridge_rt_32gpus_lr_5e-5_qformer_36_37_rp/checkpoints/steps_40000_pytorch_model.pt"
fi

cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir
# conda activate simpler_env4 # make sure you are in the right conda env
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/yejinhui/Projects/llavavla # make your llavavla seeable for SimplerEnv envs

policy_model=QwenACTAFormer

declare -a arr=(
  ${MODEL_PATH}
)
env_name=MoveNearGoogleBakedTexInScene-v0
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

# 轮转分配用的变量
total_gpus=8
count=0

for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

for urdf_version in "${urdf_version_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=$((count % total_gpus))
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model cogact --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
      --additional-env-build-kwargs urdf_version=${urdf_version} \
      --additional-env-save-tags baked_except_bpb_orange &
    count=$((count + 1))
  done
done

# 等待所有后台任务完成
wait

echo "所有任务已完成。"
