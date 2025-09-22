#!/bin/bash
ckpt_path=/PATH/TO/CKPT/steps_30000_pytorch_model.pt
folder_name=$(echo "$ckpt_path" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"

host="127.0.0.1"
base_port=10095
unnorm_key="franka"

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}
export PYTHONPATH=$PYTHONPATH:$PWD/eval/LIBERO

python ./examples/libero/eval_libero.py \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"