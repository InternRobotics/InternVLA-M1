#!/bin/bash

# 用法: ./analyze_success_all.sh <log_dir> <output_dir> [true|false]
LOG_DIR="$1/checkpoints"
OUT_DIR="$1/success_summary"
mkdir -p "$OUT_DIR"
RM_LOGS="${2:-false}"

RAW_TXT="$OUT_DIR/raw_success.txt"
CSV_OUT="$OUT_DIR/success_summary.csv"
PNG_OUT="$OUT_DIR/success_plot.png"

# 检查输入
if [ -z "$LOG_DIR" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 <log_directory> <output_directory> [true|false]"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "📁 Logs: $LOG_DIR"
echo "📤 Output: $OUT_DIR"

Step 1: 提取成功率日志
echo "🔍 Extracting success scores..."
OUTPUT=""
for file in "$LOG_DIR"/*.log.*; do
  if [ -f "$file" ]; then
    success=$(grep -E "Average success" "$file" | awk '{print $NF}')
    if [ -n "$success" ]; then
      line="$(basename "$file") → Average success: $success"
      echo "$line"
      OUTPUT+="$line"$'\n'
    else
      echo "$(basename "$file") → ❌ Not found"
      if ${RM_LOGS}; then
        rm -f "$file"
        echo "🗑️  Deleted: $file"
      fi
    fi
  fi
done

# ✅ 保存日志到 TXT 文件（修复关键点）
echo "$OUTPUT" > "$RAW_TXT"

export PYTHON_SCRIPT="/mnt/petrelfs/yejinhui/Projects/llavavla/scripts/eval/vis_results_windox.py"
# Step 3: 执行 Python 分析脚本
echo "🐍 Running Python analysis..."
python3 "$PYTHON_SCRIPT" "$RAW_TXT" "$CSV_OUT" "$PNG_OUT"
