# set env
export TZ=UTC-8
export PYTHONPATH=$PYTHONPATH:$PWD/3rd/LIBERO
# conda activate your_libero_env, which has been pip install .e 

# cd root to your libero repo

cd ./eval/LIBERO



# Put ckpts here
ckpts=(
    exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0090000
    exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0120000
    exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0150000
    exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0190000
)

# Server Info
host="0.0.0.0"
port=8000
port_lock_file=""

# Find available port
find_available_port() {
    while true; do
        port_lock_file="/tmp/port_${port}.lock"
        
        # Check if port is in use by netstat and lock file doesn't exist
        if ! netstat -tuln | grep -q ":${port} " && [ ! -f "$port_lock_file" ]; then
            # Create lock file
            echo $$ > "$port_lock_file"
            echo "Using port: $port (lock file created: $port_lock_file)"
            break
        else
            if netstat -tuln | grep -q ":${port} "; then
                echo "Port $port is in use by netstat"
            fi
            if [ -f "$port_lock_file" ]; then
                echo "Port $port lock file exists: $port_lock_file"
            fi
            port=$((port + 1))
        fi
    done
}

# Cleanup function to remove lock file
cleanup_port_lock() {
    if [ -n "$port_lock_file" ] && [ -f "$port_lock_file" ]; then
        echo "Removing port lock file: $port_lock_file"
        rm -f "$port_lock_file"
    fi
}

# Set trap to cleanup on exit or interrupt
trap cleanup_port_lock EXIT SIGINT SIGTERM

# Find and lock available port
find_available_port

# TTS args
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0

# s1s2 args
s1_replan_steps=8
s2_replan_steps=16

task_suite_name=libero_goal

test_name="s1-${s1_replan_steps}_"\
"s2-${s2_replan_steps}_"\
"s2cand-${s2_candidates_num}_"\
"ntl-${noise_temp_lower_bound}_"\
"ntu-${noise_temp_upper_bound}_"\
"ttl-${time_temp_lower_bound}_"\
"ttu-${time_temp_upper_bound}"

for ckpt in ${ckpts[@]}; do
    
    echo $ckpt
    run_ckpt_name=$(basename $(dirname $ckpt))_$(basename $ckpt)
    job_name=${run_ckpt_name}_${test_name}
    echo $job_name

    # launch polciy server in tmux
    session_name=$(echo $job_name | sed 's/\./_/g')
    tmux new -s "${session_name}" -d \
        "bash -c '
            source scripts/env.sh
            python -u src/hume/serve_policy.py --ckpt_path $ckpt --port $port
        '"
    
    # cleanup server
    close_sever() {
        echo "Closing policy server ${session_name}"
        tmux kill-session -t "${session_name}"
        cleanup_port_lock
    }
    trap close_sever EXIT SIGINT SIGTERM

    # start eval
    python ./eval/Eval_libero/eval_libero.py \
        --args.host $host \
        --args.port $port \
        --args.task-suite-name ${task_suite_name} \
        --args.job-name ${job_name} \
        --args.post-process-action \
        --args.num-trials-per-task 5 \
        --args.replan-steps ${s1_replan_steps} \
        --args.s2-replan-steps ${s2_replan_steps} \
        --args.s2-candidates-num ${s2_candidates_num} \
        --args.noise-temp-lower-bound ${noise_temp_lower_bound} \
        --args.noise-temp-upper-bound ${noise_temp_upper_bound} \
        --args.time-temp-lower-bound ${time_temp_lower_bound} \
        --args.time-temp-upper-bound ${time_temp_upper_bound}

    close_sever
done
