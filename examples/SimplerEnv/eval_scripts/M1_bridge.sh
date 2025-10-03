#!/bin/bash

echo `which python`
# 定义一个函数来启动服务
policyserver_pids=()
eval_pids=()

start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3

  echo "▶️ Starting service on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_policy_M1.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 &
  policyserver_pids+=($!)  # 保存服务进程的 PID
  echo $!  # 返回服务进程的 PID
}

# 定义一个函数来停止所有服务
stop_all_services() {
  echo "⛔ 正在停止所有服务..."

  # 等待所有评估任务完成
  if [ "${#eval_pids[@]}" -gt 0 ]; then
    echo "⏳ 等待评估任务完成..."
    for pid in "${eval_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        wait "$pid"
        status=$?
        if [ $status -ne 0 ]; then
          echo "⚠️ 评估任务 $pid 异常退出 (状态: $status)"
        else
          echo "✅ 评估任务 $pid 已完成"
        fi
      else
        echo "⚠️ 评估任务 $pid 已不存在 (可能已提前退出)"
      fi
    done
  fi

  # 停止所有服务
  if [ "${#policyserver_pids[@]}" -gt 0 ]; then
    echo "⏳ 停止服务进程..."
    for pid in "${policyserver_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "🔴 正在停止服务进程 PID: $pid"
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        echo "✅ 服务进程 $pid 已停止"
      else
        echo "⚠️ 服务进程 $pid 已不存在 (可能已提前退出)"
      fi
    done
  fi

  # 清空 PID 数组
  eval_pids=()
  policyserver_pids=()
  echo "✅ 所有服务和任务已停止"
}


export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}


MODEL_PATH=$1
TSET_NUM=1
run_count=0

if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1_need/0906_bestvla_retrain_sota2/checkpoints/steps_50000_pytorch_model.pt"
fi

ckpt_path=${MODEL_PATH}

# 获取当前系统的 CUDA_VISIBLE_DEVICES 列表
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # 将逗号分隔的 GPU 列表转换为数组
NUM_GPUS=${#CUDA_DEVICES[@]}  # 获取可用 GPU 的数量

# 打印调试信息
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# 任务列表，每行指定一个 env-name
declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)


for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}
    
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"  # 去掉 .pt 或 .bin 后缀

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"
    
    # 启动服务并获取服务进程的 PID
    port=$((10093 + run_count))
    server_pid=$(start_service ${gpu_id} ${ckpt_path} ${port})

    CUDA_VISIBLE_DEVICES=${gpu_id} python examples/SimplerEnv/start_simpler_env.py \
      --port $port \
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

    run_count=$((run_count + 1))
  done
done

# V2 同理：PutEggplantInBasketScene-v0 也执行 5 次
declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup

rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$(((run_count) % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching V2 task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

    # 启动服务并获取服务进程的 PID
    port=$((10093 + run_count))
    server_pid=$(start_service ${gpu_id} ${ckpt_path} ${port})


    CUDA_VISIBLE_DEVICES=${gpu_id} python examples/SimplerEnv/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port $port \
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

    run_count=$((run_count + 1))
  done
done

wait
echo "✅ 所有测试完成"


