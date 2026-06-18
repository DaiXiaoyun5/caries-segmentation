#!/bin/bash
#SBATCH -J medsam_gtbox
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/medsam_gtbox_%j.out

set -e
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
cd /share/home/u2515283028/caries_project
mkdir -p runs/logs runs_medsam

export PYTHONPATH=/share/home/u2515283028/caries_project/external_models/MedSAM:$PYTHONPATH

echo "===== GPU check ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "===== Eval MedSAM GT-box oracle baseline ====="
python src/eval_medsam_gtbox_caries.py \
  --checkpoint external_models/medsam_vit_b.pth \
  --medsam-repo external_models/MedSAM \
  --model-type vit_b \
  --split test \
  --run-name medsam_gtbox_test_pad005 \
  --device cuda:0 \
  --image-size 1024 \
  --box-pad-ratio 0.05 \
  --mask-threshold 0.5 \
  --save-preds 0 \
  --num-preview 20

echo "===== summary ====="
cat runs_medsam/medsam_gtbox_test_pad005/summary_metrics.json
