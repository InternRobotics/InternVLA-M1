#!/bin/bash

# 目标根目录，建议在这个脚本所在位置运行或手动指定路径
ROOT_DIR="./"

# 日志文件名模板（不含steps前缀）
LOG_SUFFIXES=(
  "pytorch_model_infer_PutCarrotOnPlateInScene-v0.log.run1"
  "pytorch_model_infer_PutEggplantInBasketScene-v0.log.run1"
  "pytorch_model_infer_PutSpoonOnTableClothInScene-v0.log.run1"
  "pytorch_model_infer_StackGreenCubeOnYellowCubeBakedTexInScene-v0.log.run1"
)

cd /mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints

# 遍历所有以0601_开头的目录
for dir in "$ROOT_DIR"/0_bar/*/checkpoints; do
  # 确保 checkpoints 目录存在
  if [ -d "$dir" ]; then
    echo "Processing: $dir"

    cd "$dir" || continue

    # 查找所有模型文件
    for pt_file in steps_*_pytorch_model.pt; do
      [ -e "$pt_file" ] || continue  # 防止空匹配

      step=$(echo "$pt_file" | cut -d'_' -f2)

      all_logs_exist=true
      for suffix in "${LOG_SUFFIXES[@]}"; do
        log_pattern="steps_${step}_${suffix}*"  # 模糊匹配
        matched_logs=( $log_pattern )

        if [ ${#matched_logs[@]} -eq 0 ]; then
          all_logs_exist=false
          break
        fi
      done

      if $all_logs_exist; then
        echo "✔ All logs found for $pt_file — deleting..."
        # ls "$pt_file"
        rm "$pt_file"
      else
        echo "✘ Not all logs found for $pt_file — keeping."
        rm "$pt_file"

      fi
    done

    cd - >/dev/null
  fi
done
