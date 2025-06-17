#!/bin/bash

# 用法: ./parse_success.sh /path/to/logs
log_dir="$1"
RM_LOGS="${2:-false}"  # 第二个参数决定是否删除日志文件，默认为 false


if [ -z "$log_dir" ]; then
  echo "Usage: $0 <log_directory>"
  exit 1
fi

echo "📋 Checking logs in: $log_dir"
echo "----------------------------------------"

for file in "$log_dir"/*.log.*; do
  if [ -f "$file" ]; then
    success=$(grep -E "Average success" "$file" | awk '{print $NF}')
    if [ -n "$success" ]; then
      echo "$(basename "$file") → Average success: $success"
    else
      echo "$(basename "$file") → ❌ Not found"
      if ${RM_LOGS}; then
        rm -f "$file"
        echo "🗑️  已删除: $file"
      fi
    fi
  fi
done


