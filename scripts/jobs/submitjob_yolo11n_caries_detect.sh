#!/bin/bash
#SBATCH -J yolo11n_caries
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/yolo11n_caries_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs runs_yolo

echo "===== Prepare YOLO caries detection dataset ====="
python src/prepare_yolo_caries_detection.py --overwrite

echo
echo "===== Check ultralytics ====="
python - <<'PY'
import ultralytics
print("ultralytics version:", ultralytics.__version__)
PY

echo
echo "===== Train YOLO11n caries detection ====="
yolo detect train \
  model=yolo11n.pt \
  data=data/yolo_caries_detect/data.yaml \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  workers=2 \
  device=0 \
  project=runs_yolo \
  name=yolo11n_caries_detect \
  exist_ok=True \
  seed=42

echo
echo "===== Validate YOLO11n on test split with Ultralytics metrics ====="
yolo detect val \
  model=runs_yolo/yolo11n_caries_detect/weights/best.pt \
  data=data/yolo_caries_detect/data.yaml \
  split=test \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs_yolo \
  name=yolo11n_caries_detect_testval \
  exist_ok=True

echo
echo "===== Custom low-conf recall evaluation ====="
python src/eval_yolo_caries_recall_conf.py \
  --model runs_yolo/yolo11n_caries_detect/weights/best.pt \
  --data-root data/yolo_caries_detect \
  --split test \
  --run-name yolo11n_caries_detect \
  --imgsz 640 \
  --device 0 \
  --predict-conf 0.001 \
  --nms-iou 0.7 \
  --eval-iou 0.5 \
  --conf-thresholds 0.05,0.10,0.15,0.25

echo
echo "YOLO11n caries detection baseline done."
