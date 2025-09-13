# 用法： ./run_grep_in_checkpoints.sh /your/root/path


# # 写在 heredoc 中更清晰
ROOT_BASE=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints
ROOT_BASE=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints




echo "🔍 Searching in base directory: $ROOT_BASE"
echo "==========================================="

# 遍历匹配目录

script_file=/mnt/petrelfs/yejinhui/Projects/llavavla/scripts/eval/analyze_success_windowx.sh

# 设置 del_file 参数，默认为 false
del_file=${1:-false}


# 遍历一级子目录
for dir in "$ROOT_BASE"/0831_qwendact_vla_fm*; do
  if [ -d "$dir" ]; then
    echo "📂 Entering: $dir"
    (cd "$dir" && bash $script_file $dir $del_file)
    echo ""
  fi
done

echo "✅ Done."