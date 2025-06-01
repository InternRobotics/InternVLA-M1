
echo `which python`

cd /mnt/petrelfs/share/yejinhui/Projects/SimplerEnv # the SimplerEnv root dir

# export DEBUG=1

MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0523_pd_qwenact_bridge_rt--image_aug/checkpoints/step-020000-epoch-00-loss=0.0811.pt
MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0530_qwenact_ftqwen_32gpus_bridge_rt_1/checkpoints/steps_40000_pytorch_model.pt


gpu_id=6
policy_model=QwenACTAFormer
ckpt_path=${MODEL_PATH} # CogACT/CogACT-Base CogACT/CogACT-Large CogACT/CogACT-Small

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

# 遍历任务，依次分配 GPU
for i in "${!ENV_NAMES[@]}"; do
  gpu_id=$((i + 0 % 8))  # 假设 GPU 0–7 共 8 个
  ckpt_dir=$(dirname "${ckpt_path}")
  ckpt_base=$(basename "${ckpt_path}")
  
  ckpt_name="${ckpt_base%.*}"  # 去掉 .pt 或 .bin 后缀
  task_log="${ckpt_dir}/${ckpt_name}_infer_${ENV_NAMES[$i]}.log"


  echo "▶️ Launching task on GPU $gpu_id: ${ENV_NAMES[$i]}, log to ${task_log}"

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py \
    --policy-model ${policy_model} \
    --ckpt-path ${ckpt_path} \
    --robot ${robot} \
    --policy-setup widowx_bridge \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps 120 \
    --env-name "${ENV_NAMES[$i]}" \
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


# V2
declare -a ENV_NAMES=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06


# 遍历任务，依次分配 GPU
for i in "${!ENV_NAMES[@]}"; do
  gpu_id=$(((i + 3) % 8))  # 假设 GPU 0–7 共 8 个
  ckpt_dir=$(dirname "${ckpt_path}")
  ckpt_base=$(basename "${ckpt_path}")
  ckpt_name="${ckpt_base%.*}"  # 去掉 .pt 或 .bin 后缀
  task_log="${ckpt_dir}/${ckpt_name}_infer_${ENV_NAMES[$i]}.log"


  echo "▶️ Launching task on GPU $gpu_id: ${ENV_NAMES[$i]}, log to ${task_log}"

  CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py \
    --policy-model ${policy_model} \
    --ckpt-path ${ckpt_path} \
    --robot ${robot} \
    --policy-setup widowx_bridge \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps 120 \
    --env-name "${ENV_NAMES[$i]}" \
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

# 等待所有后台任务完成
wait
echo "✅ 所有测试完成"