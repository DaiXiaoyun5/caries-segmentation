#!/bin/bash
set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

# If pip cannot connect, try:
# module load http-proxy
# pip install --proxy=http://211.67.63.75:3128 ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

python - <<'PY'
import ultralytics
print("ultralytics version:", ultralytics.__version__)
PY
