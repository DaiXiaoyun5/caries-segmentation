#!/bin/bash
set -e
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
cd /share/home/u2515283028/caries_project
mkdir -p external_models
module load http-proxy || true

echo "===== Clone MedSAM repo ====="
if [ ! -d external_models/MedSAM ]; then
  git clone https://github.com/bowang-lab/MedSAM.git external_models/MedSAM
else
  echo "external_models/MedSAM already exists, skip clone."
fi

echo "===== Download MedSAM ViT-B checkpoint from Zenodo ====="
CKPT=external_models/medsam_vit_b.pth
if [ ! -f "$CKPT" ]; then
  wget -c -O "$CKPT" "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
else
  echo "$CKPT already exists, skip download."
fi

ls -lh "$CKPT"

PYTHONPATH=/share/home/u2515283028/caries_project/external_models/MedSAM:$PYTHONPATH python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
from segment_anything import sam_model_registry
print("segment_anything imported.")
print("registries:", sorted(sam_model_registry.keys()))
PY

echo "MedSAM setup done."
