#!/bin/bash

echo `which python`

cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir

# export DEBUG=1

# 接收传入的模型路径参数
MODEL_PATH=$1
TSET_NUM=5

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0630_load_pretrained_v1s/checkpoints/steps_90000_pytorch_model.pt"
fi

policy_model=Qwenpi
ckpt_path=${MODEL_PATH} # CogACT/CogACT-Base CogACT/CogACT-Large CogACT/CogACT-Small


# 获取当前系统的 CUDA_VISIBLE_DEVICES 列表
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # 将逗号分隔的 GPU 列表转换为数组
NUM_GPUS=${#CUDA_DEVICES[@]}  # 获取可用 GPU 的数量

# 打印调试信息
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# 任务列表，每行指定一个 env-name
declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)


# 遍历任务，每个 env 执行 5 次，依次分配 GPU 并打 tag
# 遍历每个 env（通过下标 i）并执行多次 run
for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$((i % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # gpu_id=$((i  % 8))  # 假设 GPU 0–3 共 4 个，i 用来分配 GPU
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"  # 去掉 .pt 或 .bin 后缀

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py \
      --policy-model ${policy_model} \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &

  done
done

# V2 同理：PutEggplantInBasketScene-v0 也执行 5 次
declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$(((i + 3) % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # gpu_id=$(((i + 3) % 8))  # 假设 GPU 0–3 共 4 个，偏移 3
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching V2 task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py \
      --policy-model ${policy_model} \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      2>&1 | tee "${task_log}" &
  done
done

      # 2>&1 | tee "${task_log}" &
# 等待所有后台任务完成
wait
echo "✅ 所有测试完成"
