#!/bin/bash
# LIBERO 多模型并行评估脚本 - 固定每个服务处理 2 次试验
# 用法: 
#   1. 修改开头的配置变量
#   2. 运行: bash parallel_libero_eval.sh

# ====================== 用户配置区域 ======================
# 设置时区
export TZ=UTC-8

# 项目根目录 (llavavla 仓库)
PROJECT_ROOT="/mnt/petrelfs/yejinhui/Projects/llavavla"
cd "$PROJECT_ROOT" || { echo "错误: 无法进入项目目录 $PROJECT_ROOT"; exit 1; }

# Python 环境路径
DINOACT_ENV="/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact"
LEROBO_ENV="/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot"

# 模型检查点路径
ckpt_dir="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/0811_libero_goal/checkpoints"

ckpts=(
    "$ckpt_dir/steps_10000_pytorch_model.pt"
    "$ckpt_dir/steps_20000_pytorch_model.pt"
    "$ckpt_dir/steps_30000_pytorch_model.pt"
    "$ckpt_dir/steps_40000_pytorch_model.pt"
    "$ckpt_dir/steps_50000_pytorch_model.pt"
    # "$ckpt_dir/steps_60000_pytorch_model.pt"
    "$ckpt_dir/steps_70000_pytorch_model.pt"
    # "$ckpt_dir/steps_80000_pytorch_model.pt"
)

# 服务器配置
host="127.0.0.1"  # 本地主机
base_port=10093    # 起始端口号
port_lock_file=""  # 端口锁文件（可选）

# 评估参数
unnorm_key="franka"          # 归一化键
task_suite_name="libero_goal" # 任务套件名称
num_trials_per_task=50       # 每个任务的总试验次数
num_gpu=8                    # 可用 GPU 数量
num_threads_per_gpu=4        # 每个 GPU 的线程数

# 固定每个服务处理的试验次数
trials_per_server=2

# 调试模式 (取消注释启用)
# export DEBUG=True

# ====================== 计算派生参数 ======================
max_servers=$((num_gpu * num_threads_per_gpu))  # 最大服务数
num_servers_needed=$(( (num_trials_per_task + trials_per_server - 1) / trials_per_server )) # 需要的服务数
num_servers_to_run=$((num_servers_needed < max_servers ? num_servers_needed : max_servers)) # 实际运行的服务数

# 设置 PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$PWD/eval/LIBERO"

# ====================== 函数定义 ======================
# 检查端口是否可用
port_is_available() {
    local port=$1
    if ! command -v lsof &> /dev/null; then
        echo "警告: lsof 未安装，无法检查端口 $port" >&2
        return 0
    fi
    if lsof -i :"$port" > /dev/null; then
        return 1 # 端口被占用
    else
        return 0 # 端口可用
    fi
}

# 查找可用端口
find_available_port() {
    local port=$base_port
    while ! port_is_available "$port"; do
        ((port++))
        if [ $port -gt $((base_port + 1000)) ]; then
            echo "错误: 找不到可用端口 (起始: $base_port)" >&2
            exit 1
        fi
    done
    echo "$port"
}

# 清理后台进程
cleanup() {
    echo "正在清理..."
    pkill -f "server_policy.py"
    pkill -f "eval_libero.py"
}

# 注册退出时的清理函数
trap cleanup EXIT INT TERM

# ====================== 主评估流程 ======================
for ckpt in "${ckpts[@]}"; do
    # 获取检查点基本信息
    ckpt_dir=${ckpt%/*}       # 检查点目录
    exp_dir=${ckpt_dir%/*}    # 实验目录
    run_ckpt_name=$(basename "$exp_dir")_$(basename "$ckpt")
    job_name="${run_ckpt_name}"
    
    echo "===================================================="
    echo "开始评估模型: $job_name"
    echo "实验目录: $exp_dir"
    echo "总试验次数: $num_trials_per_task"
    echo "每个服务处理试验次数: $trials_per_server"
    echo "需要服务数量: $num_servers_needed"
    echo "实际运行服务数量: $num_servers_to_run"
    echo "最大可用服务数: $max_servers"
    echo "===================================================="
    
    # 存储服务端口和PID
    declare -a server_ports
    declare -a server_pids
    
    # 步骤1: 启动模型服务
    echo "启动 $num_servers_to_run 个模型服务..."
    for ((i=0; i<num_servers_to_run; i++)); do
        # 计算GPU分配 (循环分配)
        gpu_index=$((i % num_gpu))
        
        # 获取可用端口
        port=$(find_available_port)
        server_ports[i]=$port
        
        # 启动服务 (指定GPU)
        echo "  服务 $((i+1)): GPU $gpu_index, 端口 $port"
        CUDA_VISIBLE_DEVICES=$gpu_index \
        "$DINOACT_ENV/bin/python" real_deployment/deploy/server_policy.py \
            --ckpt_path "$ckpt" \
            --unnorm_key "$unnorm_key" \
            --port "$port" > "server_${port}.log" 2>&1 &
        server_pids[i]=$!
        
        # 更新端口基值
        base_port=$((port + 1))
        sleep 5 # 避免端口冲突
    done
    
    echo "等待服务启动(10秒)..."
    sleep 10
    
    # 步骤2: 启动评估任务
    echo "启动评估任务..."
    declare -a eval_pids
    for ((i=0; i<num_servers_to_run; i++)); do
        port=${server_ports[i]}
        start_idx=$((i * trials_per_server))
        end_idx=$((start_idx + trials_per_server))
        
        # 最后一个服务处理剩余试验
        if [ $i -eq $((num_servers_to_run - 1)) ]; then
            end_idx=$num_trials_per_task
        fi
        
        echo "  评估任务 $((i+1)): 端口 $port, 试验 [$start_idx-$end_idx]"
        
        # 启动评估
        "$LEROBO_ENV/bin/python" ./eval/Eval_libero/eval_libero.py \
            --args.host "$host" \
            --args.port "$port" \
            --args.task-suite-name "$task_suite_name" \
            --args.job-name "${job_name}" \
            --args.video_out_path "${exp_dir}/videos" \
            --args.num-trials-per-task "$num_trials_per_task" \
            --args.trials_start_index "$start_idx" \
            --args.trials_end_index "$end_idx" > "eval_${port}.log" 2>&1 &
        eval_pids[i]=$!
    done
    
    # 等待所有评估完成
    echo "等待评估任务完成..."
    for pid in "${eval_pids[@]}"; do
        wait "$pid"
        status=$?
        if [ $status -ne 0 ]; then
            echo "警告: 评估任务 $pid 异常退出 (状态: $status)"
        fi
    done
    
    # 关闭模型服务
    echo "关闭模型服务..."
    for pid in "${server_pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
    
    # 确保服务关闭
    sleep 2
    echo "评估完成: $job_name"
    echo "端口已释放: ${server_ports[*]}"
    echo "===================================================="
done

echo "所有模型评估完成!"