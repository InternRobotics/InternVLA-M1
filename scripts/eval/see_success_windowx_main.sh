# usage: ./run_grep_in_checkpoints.sh /your/root/path


# # written in heredoc is clearer
ROOT_BASE=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints
ROOT_BASE=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints




echo "🔍 Searching in base directory: $ROOT_BASE"
echo "==========================================="

# traverse matching directories

script_file=/mnt/petrelfs/yejinhui/Projects/llavavla/scripts/eval/analyze_success_windowx.sh

# set del_file parameter, default is false
del_file=${1:-false}


# traverse first level subdirectories
for dir in "$ROOT_BASE"/0831_qwendact_vla_fm*; do
  if [ -d "$dir" ]; then
    echo "📂 Entering: $dir"
    (cd "$dir" && bash $script_file $dir $del_file)
    echo ""
  fi
done

echo "✅ Done."