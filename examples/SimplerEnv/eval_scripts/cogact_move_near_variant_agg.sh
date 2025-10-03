# 合计：1 + 1 + 2 + 2 + 2 + 2 = 10 个。

MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0604_ftqwen_bridge_rt_32gpus_lr_5e-5_qformer_36_37_rp/checkpoints/steps_40000_pytorch_model.pt"
fi

# 获取当前系统的 CUDA_VISIBLE_DEVICES 列表
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # 将逗号分隔的 GPU 列表转换为数组
NUM_GPUS=${#CUDA_DEVICES[@]}  # 获取可用 GPU 的数量


cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir

policy_model=Qwenpi

declare -a arr=(
  ${MODEL_PATH}
)

# 轮转分配用的变量
total_gpus=8
count=0

# CogACT/CogACT-Large CogACT/CogACT-Small
for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

# base setup
env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${arr[@]}"; do
  gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
  count=$((count + 1))
done

# distractor
for ckpt_path in "${arr[@]}"; do
  gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs no_distractor=True &
  count=$((count + 1))
done

# backgrounds
env_name=MoveNearGoogleInScene-v0
declare -a scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                      "google_pick_coke_can_1_v4_alt_background_2")

for scene_name in "${scene_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
    count=$((count + 1))
  done
done

# lighting
env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${arr[@]}"; do
  # 稍微更暗
  gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs slightly_darker_lighting=True &
  count=$((count + 1))

  # 稍微更亮
  gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs slightly_brighter_lighting=True &
  count=$((count + 1))
done

# table textures
env_name=MoveNearGoogleInScene-v0
declare -a table_scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                            "Baked_sc1_staging_objaverse_cabinet2_h870")

for scene_name in "${table_scene_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
    count=$((count + 1))
  done
done

# camera orientations
declare -a env_arr=("MoveNearAltGoogleCameraInScene-v0" \
                    "MoveNearAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for env_name in "${env_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
    count=$((count + 1))
  done
done

# 等待所有后台任务完成
wait

echo "所有任务已完成。"