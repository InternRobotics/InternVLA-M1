# 使用所有 8 个 GPU 并将任务挂到后台运行

# 原先的单 GPU 配置已移除
# 接收传入的模型路径参数
MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0611_noflash_vlm_bridge_rt_1_32gpus_vlm_4_0.1/checkpoints/steps_70000_pytorch_model.pt"
fi
cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir

# conda activate simpler_env4 # make sure you are in the right conda env
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/yejinhui/Projects/llavavla # make your llavavla seeable for SimplerEnv envs


policy_model=Qwenpi

declare -a ckpt_paths=(
  ${MODEL_PATH}
)
# CogACT/CogACT-Large CogACT/CogACT-Small
declare -a env_names=(
  CloseTopDrawerCustomInScene-v0
  CloseMiddleDrawerCustomInScene-v0
  CloseBottomDrawerCustomInScene-v0
  OpenTopDrawerCustomInScene-v0
  OpenMiddleDrawerCustomInScene-v0
  OpenBottomDrawerCustomInScene-v0
)

EXTRA_ARGS="--enable-raytracing"

# 使用 8 GPUs 轮转分配
total_gpus=8
count=0

# base setup
scene_name=frl_apartment_stage_simple

EvalSim() {
  echo "使用 GPU ${gpu_id} 运行: ${ckpt_path} ${env_name}"
  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS}
}

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    gpu_id=$((count % total_gpus))
    if (( (count + 1) == 32 )); then
      EvalSim
    else
      EvalSim &
    fi
    count=$((count + 1))
  done
done

# backgrounds
declare -a scene_names=(
  "modern_bedroom_no_roof"
  "modern_office_no_roof"
)

for scene_name in "${scene_names[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
      gpu_id=$((count % total_gpus))
      if (( (count + 1) == 32 )); then
        EvalSim
      else
        EvalSim &
      fi
      count=$((count + 1))
    done
  done
done

# lightings
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter"
    gpu_id=$((count % total_gpus))
    if (( (count + 1) == 32 )); then
      EvalSim
    else
      EvalSim &
    fi
    count=$((count + 1))

    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker"
    gpu_id=$((count % total_gpus))
    if (( (count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim &
    fi
    count=$((count + 1))
  done
done

# new cabinets
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    gpu_id=$((count % total_gpus))
    if (( (count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim &
    fi
    count=$((count + 1))

    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
    gpu_id=$((count % total_gpus))
    if (( (count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim &
    fi
    count=$((count + 1))
  done
done

# 等待所有后台任务完成
wait

echo "所有任务已完成。"
