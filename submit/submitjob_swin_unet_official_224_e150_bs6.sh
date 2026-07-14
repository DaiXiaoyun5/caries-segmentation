#!/bin/bash
#SBATCH -J swin_unet
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/swin_unet_official_%j.out

set -e

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p runs/logs external_models

ASSET_DIR=/share/home/u2515283028/caries_project/external_assets
ZIP_PATH=${ASSET_DIR}/Swin-Unet-main.zip
CKPT_PATH=${ASSET_DIR}/swin_tiny_patch4_window7_224.pth
SOURCE_DIR=/share/home/u2515283028/caries_project/external_models/Swin-Unet-main

test -f "${ZIP_PATH}" || { echo "Missing ${ZIP_PATH}"; exit 2; }
test -f "${CKPT_PATH}" || { echo "Missing ${CKPT_PATH}"; exit 2; }

echo "0dff0004b71514fb3509b4fabc6b5ffd76d33aee283ba9fca8e0b1ceeb6215dc  ${ZIP_PATH}" | sha256sum -c -
echo "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3  ${CKPT_PATH}" | sha256sum -c -

if [ ! -f "${SOURCE_DIR}/networks/vision_transformer.py" ]; then
  unzip -oq "${ZIP_PATH}" -d /share/home/u2515283028/caries_project/external_models
fi

echo "===== SWIN-UNET OFFICIAL 224 ====="
which python
python --version
nvidia-smi

python - <<'PY'
import einops
import timm
import yacs
print("einops:", einops.__version__)
print("timm:", timm.__version__)
print("yacs: OK")
PY

python -m py_compile \
  src/official_baseline_common.py \
  src/train_swin_unet_official.py \
  "${SOURCE_DIR}/config.py" \
  "${SOURCE_DIR}/networks/vision_transformer.py" \
  "${SOURCE_DIR}/networks/swin_transformer_unet_skip_expand_decoder_sys.py"

python src/train_swin_unet_official.py \
  --run-name swin_unet_official_224_e150_bs6 \
  --image-size 224 \
  --batch-size 6 \
  --val-batch-size 6 \
  --epochs 150 \
  --lr 0.0125 \
  --num-workers 2 \
  --seed 1234 \
  --swin-source-dir "${SOURCE_DIR}" \
  --pretrained-checkpoint "${CKPT_PATH}"

cat runs/swin_unet_official_224_e150_bs6/test_metrics.json
cat runs/swin_unet_official_224_e150_bs6/summary_metrics.json

