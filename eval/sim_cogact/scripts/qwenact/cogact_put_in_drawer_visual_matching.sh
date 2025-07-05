# 使用所有 8 个 GPU 并将 EvalOverlay 任务挂到后台运行
# 因此总共会实际运行 12 次 main_inference.py。

MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0604_ftqwen_bridge_rt_32gpus_lr_5e-5_qformer_36_37_rp/checkpoints/steps_40000_pytorch_model.pt"
fi


cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir
# conda activate simpler_env4 # make sure you are in the right conda env
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/yejinhui/Projects/llavavla

policy_model=Qwenpi

declare -a ckpt_paths=(
  ${MODEL_PATH}
)

declare -a env_names=(
  PlaceIntoClosedTopDrawerCustomInScene-v0
)

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

# 轮转分配用的变量
total_gpus=8
count=0

# EvalOverlay 函数：使用 ${gpu_id} 执行三次 main_inference，A0/B0/C0
EvalOverlay() {
  echo "${ckpt_path} ${env_name} (URDF=${urdf_version}) on GPU ${gpu_id}"

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name dummy_drawer \
    --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
    ${EXTRA_ARGS}

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name dummy_drawer \
    --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
    ${EXTRA_ARGS}

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name dummy_drawer \
    --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
    ${EXTRA_ARGS}
}

for urdf_version in "${urdf_version_arr[@]}"; do
  EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} model_ids=baked_apple_v2"

  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      gpu_id=$((count % total_gpus))
      EvalOverlay &
      count=$((count + 1))
    done
  done
done

# 等待所有后台任务完成
wait

echo "所有任务已完成。"
