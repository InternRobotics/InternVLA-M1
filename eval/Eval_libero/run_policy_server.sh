# set env
export TZ=UTC-8

# conda activate your_libero_env, which has been pip install .e 
# cd root to llavavla repo # TODO 这个应该是以评测平台为中心的

cd /mnt/petrelfs/yejinhui/Projects/llavavla
export PYTHONPATH=$PYTHONPATH:$PWD/eval/LIBERO
# Put ckpts here
ckpt_path="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0811_libero_goal/checkpoints/steps_60000_pytorch_model.pt"

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
num_trials_per_task=5  # 每个任务的总试验次数



echo $ckpt_path
run_ckpt_name=$(basename $(dirname $ckpt_path))_$(basename $ckpt_path)
job_name=${run_ckpt_name}
echo $job_name

# python /mnt/petrelfs/yejinhui/Projects/llavavla/real_deployment/deploy/server_policy.py \
#     --ckpt_path ${ckpt_path} \
#     --unnorm_key ${unnorm_key} \
#     --port ${port}

/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python /mnt/petrelfs/yejinhui/Projects/llavavla/real_deployment/deploy/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --unnorm_key ${unnorm_key} \
    --port ${port}


