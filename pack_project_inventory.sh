#!/bin/bash
set -e

ROOT="/share/home/u2515283028/caries_project"
OUT="$ROOT/project_inventory_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

cd "$ROOT"

echo "[1/8] 保存当前路径和顶层目录..."
pwd > "$OUT/pwd.txt"
ls -lah > "$OUT/top_ls_lah.txt"

echo "[2/8] 生成目录树清单（排除大文件目录）..."
find . \
  -path "./.git" -prune -o \
  -path "./data" -prune -o \
  -path "./dataset" -prune -o \
  -path "./datasets" -prune -o \
  -path "./runs/*/checkpoints" -prune -o \
  -path "./runs/*/pred*" -prune -o \
  -path "./runs/*/vis*" -prune -o \
  -path "./__pycache__" -prune -o \
  -type f \
  | sed 's#^\./##' \
  | sort \
  > "$OUT/file_list.txt"

echo "[3/8] 统计目录大小..."
du -h --max-depth=2 . 2>/dev/null | sort -h > "$OUT/du_depth2.txt" || true

echo "[4/8] 收集 Python / shell / 配置文件..."
mkdir -p "$OUT/code_snapshot"
find . \
  -path "./.git" -prune -o \
  -path "./data" -prune -o \
  -path "./dataset" -prune -o \
  -path "./datasets" -prune -o \
  -path "./runs" -prune -o \
  -type f \( \
    -name "*.py" -o \
    -name "*.sh" -o \
    -name "*.yaml" -o \
    -name "*.yml" -o \
    -name "*.json" -o \
    -name "*.txt" -o \
    -name "*.md" \
  \) \
  -exec cp --parents {} "$OUT/code_snapshot/" \; 2>/dev/null || true

echo "[5/8] 收集 runs 下的小型指标文件..."
mkdir -p "$OUT/run_metrics"
find ./runs \
  -type f \( \
    -name "*metrics*.json" -o \
    -name "*metrics*.csv" -o \
    -name "history.csv" -o \
    -name "config.json" -o \
    -name "args.json" -o \
    -name "best*.json" -o \
    -name "*.log" -o \
    -name "*.out" \
  \) \
  -size -5M \
  -exec cp --parents {} "$OUT/run_metrics/" \; 2>/dev/null || true

echo "[6/8] 收集环境信息..."
{
  echo "===== date ====="
  date
  echo
  echo "===== hostname ====="
  hostname
  echo
  echo "===== module list ====="
  module list 2>&1 || true
  echo
  echo "===== conda envs ====="
  module load anaconda3/4.12.0 2>/dev/null || true
  conda env list 2>/dev/null || true
  echo
  echo "===== current python ====="
  which python 2>/dev/null || true
  python --version 2>/dev/null || true
  echo
  echo "===== git status ====="
  git status 2>/dev/null || true
} > "$OUT/env_info.txt"

echo "[7/8] 尝试记录 PyTorch / CUDA 信息..."
{
  module load anaconda3/4.12.0 2>/dev/null || true
  source activate caries-train 2>/dev/null || true
  python - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)
    print("gpu_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}:", torch.cuda.get_device_name(i))
except Exception as e:
    print("torch_info_error:", repr(e))
PY
} > "$OUT/torch_info.txt" 2>&1 || true

echo "[8/8] 打包..."
tar -czf "${OUT}.tar.gz" -C "$ROOT" "$(basename "$OUT")"

echo
echo "完成！请把这个文件下载后发给我："
echo "${OUT}.tar.gz"
ls -lh "${OUT}.tar.gz"
