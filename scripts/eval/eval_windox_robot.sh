#!/bin/bash

# test_all.sh
# traverse all checkpoints in experiments starting with 0*e-3, if the corresponding logs are missing, launch tests through srun

# directory (parent directory of Checkpoints)
ROOT_BASE="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints"

# log file name suffix (without steps_${step}_ prefix)
LOG_SUFFIXES=(
  "pytorch_model_infer_PutCarrotOnPlateInScene-v0.log.run1"
  "pytorch_model_infer_PutEggplantInBasketScene-v0.log.run1"
  "pytorch_model_infer_PutSpoonOnTableClothInScene-v0.log.run1"
  "pytorch_model_infer_StackGreenCubeOnYellowCubeBakedTexInScene-v0.log.run1"
)

# playground/Checkpoints/0905_qwenact_ft_vla_lerobot_cotrain_oxe

# script path for testing
SCRIPT_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/eval/sim_cogact/scripts/qwenact/cogact_bridge.sh"

# directly write the wildcard path in the for loop, Bash will expand all matching directories for you

# /mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0831_qwendact_vla_fm/checkpoints/steps_40000_pytorch_model.pt

for checkpoints_dir in "$ROOT_BASE"/0831_qwendact_vla_fm/checkpoints; do
  echo $checkpoints_dir
  # ensure checklspoints directory exists and is a directory
  if [ -d "$checkpoints_dir" ]; then
    # if the path contains "without", skip
    if [[ "$checkpoints_dir" == *"without"* ]]; then
      echo "Skipping directory (contains 'without'): $checkpoints_dir"
      continue
    fi
    echo "Processing directory: $checkpoints_dir"
    cd "$checkpoints_dir" || continue

    # traverse all checkpoint files named steps_*_pytorch_model.pt
    for pt_file in steps_*_pytorch_model.pt; do
      [ -e "$pt_file" ] || continue  # if there is no matching file, skip

      # extract step number (file name format assumed to be steps_<step>_pytorch_model.pt)
      step=$(echo "$pt_file" | cut -d'_' -f2)

      # check if all 4 corresponding log files exist
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


