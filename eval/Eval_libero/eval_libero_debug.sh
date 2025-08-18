# set env
export TZ=UTC-8

# conda activate your_libero_env, which has been pip install .e 
# cd root to llavavla repo # TODO 这个应该是以评测平台为中心的

cd /mnt/petrelfs/yejinhui/Projects/llavavla
export PYTHONPATH=$PYTHONPATH:$PWD/eval/LIBERO
# Put ckpts here
ckpt_path="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0811_libero_object/final_model/pytorch_model.pt"

ckpts=(
    "$ckpt_path"
)


# Server Info
# http://10.140.54.23/
host="127.0.0.1" # local
port=10095
port_lock_file=""




# # run policy
# conda activate dinoact
unnorm_key="franka"


# export DEBUG=True

task_suite_name=libero_goal
num_trials_per_task=50  # 每个任务的总试验次数

for ckpt in ${ckpts[@]}; do
    
    # 挂起模型推理服务
    # find_available_port
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
        --args.num-trials-per-task ${num_trials_per_task} \
        --args.video_out_path ${exp_dir} \
        # --args.trials_start_index 0 \
        # --args.trials_end_index 2 \
done

