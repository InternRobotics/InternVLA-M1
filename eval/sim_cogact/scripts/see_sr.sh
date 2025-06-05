#!/bin/bash

# 用法: ./parse_success.sh /path/to/logs
log_dir="$1"

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
    fi
  fi
done
