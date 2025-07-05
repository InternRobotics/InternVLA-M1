# 3 + 6 + 6 + 6 + 6 + 6 = 33 次。

MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0604_ftqwen_bridge_rt_32gpus_lr_5e-5_qformer_36_37_rp/checkpoints/steps_40000_pytorch_model.pt"
fi


cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir
# conda activate simpler_env4 # make sure you are in the right conda env
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/yejinhui/Projects/llavavla # make your llavavla seeable for SimplerEnv envs

policy_model=Qwenpi

declare -a arr=(${MODEL_PATH})

# 轮转分配用的变量
total_gpus=8
count=0

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

# base setup
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=$((count % total_gpus))
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} &
    count=$((count + 1))
  done
done

# table textures
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                      "Baked_sc1_staging_objaverse_cabinet2_h870")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=$((count % total_gpus))
      CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      count=$((count + 1))
    done
  done
done

# distractors
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=$((count % total_gpus))
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} &
    count=$((count + 1))

    gpu_id=$((count % total_gpus))
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} distractor_config=more &
    count=$((count + 1))
  done
done

# backgrounds
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a bg_scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                         "google_pick_coke_can_1_v4_alt_background_2")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${bg_scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=$((count % total_gpus))
      CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      count=$((count + 1))
    done
  done
done

# lightings
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=$((count % total_gpus))
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True &
    count=$((count + 1))

    gpu_id=$((count % total_gpus))
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True &
    count=$((count + 1))
  done
done

# camera orientations
declare -a env_arr=("GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0" \
                    "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for env_name in "${env_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=$((count % total_gpus))
      CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      count=$((count + 1))
    done
  done
done

# 等待所有后台任务完成
wait

echo "所有任务已完成。"
