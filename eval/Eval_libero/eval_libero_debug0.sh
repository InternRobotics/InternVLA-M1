# set env
export TZ=UTC-8

# conda activate your_libero_env, which has been pip install .e 
# cd root to llavavla repo # TODO 这个应该是以评测平台为中心的

cd /mnt/petrelfs/yejinhui/Projects/llavavla
export PYTHONPATH=$PYTHONPATH:$PWD/eval/LIBERO
# Put ckpts here
ckpt_path=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0811_libero_goal/checkpoints/steps_60000_pytorch_model.pt
ckpts=(
    "$ckpt_path"
)


# Server Info
# http://10.140.54.23/
host="127.0.0.1" # local
port=10093
port_lock_file=""




# # run policy
# conda activate dinoact
# unnorm_key="franka"

    



export DEBUG=True
# TTS args
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0


task_suite_name=libero_goal
num_trials_per_task=5  # 每个任务的总试验次数

for ckpt in ${ckpts[@]}; do
    
    # 挂起模型推理服务
    find_available_port
    echo $ckpt
    run_ckpt_name=$(basename $(dirname $ckpt))_$(basename $ckpt)
    job_name=${run_ckpt_name}
    echo $job_name
    
    # /mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python /mnt/petrelfs/yejinhui/Projects/llavavla/real_deployment/deploy/server_policy.py \
    #     --ckpt_path ${ckpt_path} \
    #     --unnorm_key ${unnorm_key} \
    #     --port ${port} &


    # 取得父目录: /mnt/petrelfs/.../0806_libero_vla_alltask
    ckpt_dir=${ckpt_path%/*}          # 去掉文件名 -> .../checkpoints
    exp_dir=${ckpt_dir%/*}_debug            # 再去掉 checkpoints -> .../0806_libero_vla_alltask
    echo "exp_dir=${exp_dir}"

    # start eval
    /mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot/bin/python ./eval/Eval_libero/eval_libero.py \
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
done
