#!/bin/bash
set -euo pipefail

ROOT_DIR="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints"
DIR_GLOB="08*"          # 要匹配的目录前缀或通配：如 0822* 或 0822_best*
FILE_GLOB="*5*.pt"          # 要匹配的文件通配：如 '*.pt' 或 'steps_*_pytorch_model.pt' 或 '*pytorch_model*.pt'

# dry-run: 1 = 只列出，0 = 真删
DRY_RUN=0

echo "ROOT_DIR = $ROOT_DIR"
echo "DIR_GLOB = $DIR_GLOB"
echo "FILE_GLOB = $FILE_GLOB"
echo "DRY_RUN  = $DRY_RUN"
echo

for dir in "$ROOT_DIR"/$DIR_GLOB; do
  [ -d "$dir" ] || continue
  echo "Processing directory: $dir"
  # find 所有匹配的文件，安全处理文件名带空格
  while IFS= read -r -d '' f; do
    if [ "$DRY_RUN" -eq 1 ]; then
      printf "WILL DELETE: %s\n" "$f"
    else
      printf "DELETING: %s\n" "$f"
      rm -v -- "$f"
    fi
  done < <(find "$dir" -type f -name "$FILE_GLOB" -print0)
done

echo "Done."