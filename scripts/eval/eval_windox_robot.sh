#!/bin/bash

# test_all.sh
# 遍历所有以 0*e-3 开头的实验目录下的 checkpoints，若对应的日志缺失则通过 srun 启动测试

# 目录（Checkpoints 的父目录）
ROOT_BASE="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints"

# 日志文件名后缀（不含 steps_${step}_ 前缀）
LOG_SUFFIXES=(
  "pytorch_model_infer_PutCarrotOnPlateInScene-v0.log.run1"
  "pytorch_model_infer_PutEggplantInBasketScene-v0.log.run1"
  "pytorch_model_infer_PutSpoonOnTableClothInScene-v0.log.run1"
  "pytorch_model_infer_StackGreenCubeOnYellowCubeBakedTexInScene-v0.log.run1"
)

# 用于测试的脚本路径
SCRIPT_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_bridge.sh"

# 直接把通配路径写在 for 循环里，Bash 会帮你展开所有匹配的目录
# 0604_fixqwen_32gpus_lr_5e-5_qformer_36_37_rp
# /mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0604_ftqwen_32gpus_lr_5e-5_qformer_0_6


for checkpoints_dir in "$ROOT_BASE"/0817_qwendino_vla_rpsota_withlerobot/checkpoints; do
  # 确保 checklspoints 目录存在且是目录
  if [ -d "$checkpoints_dir" ]; then
    # 如果路径中包含 "without"，则跳过
    if [[ "$checkpoints_dir" == *"without"* ]]; then
      echo "Skipping directory (contains 'without'): $checkpoints_dir"
      continue
    fi
    echo "Processing directory: $checkpoints_dir"
    cd "$checkpoints_dir" || continue

    # 遍历所有以 steps_*_pytorch_model.pt 命名的 checkpoint 文件
    for pt_file in steps_*_pytorch_model.pt; do
      [ -e "$pt_file" ] || continue  # 如果没有匹配的文件，则跳过

      # 提取 step 编号（文件名格式假设为 steps_<step>_pytorch_model.pt）
      step=$(echo "$pt_file" | cut -d'_' -f2)

      # 检查对应的 4 个日志文件是否都存在
      all_logs_exist=true
      for suffix in "${LOG_SUFFIXES[@]}"; do
        log_file="steps_${step}_${suffix}"
        if [ ! -f "$log_file" ]; then
          all_logs_exist=false
          break
        fi
      done

      if $all_logs_exist; then
        echo "✔ All logs found for $pt_file — skipping"
        MODEL_PATH="$checkpoints_dir/$pt_file"
        nohup srun -p efm_p --gres=gpu:4 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" &
        sleep 10
        # rm $pt_file
      else
        echo "✘ Logs missing for $pt_file — launching test"
        MODEL_PATH="$checkpoints_dir/$pt_file"
        nohup srun -p efm_p --gres=gpu:4 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" &
        sleep 10
      fi
    done

    cd - >/dev/null
  fi
done


