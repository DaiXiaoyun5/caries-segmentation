#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SCRIPT="src/train_resnet34_unet_brr_b9_dcf_e200.py"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: cannot find $SCRIPT"
  echo "Please generate B9-DCF script first."
  exit 1
fi

mkdir -p runs/logs

echo "===== Python syntax check ====="
python -m py_compile "$SCRIPT"

echo "===== Create smoke script: B9-DCF-lite ====="
cat > submitjob_smoke_b9_dcf_lite_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b9l
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b9_dcf_lite_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== ENV ====="
which python
python --version
nvidia-smi

python -m py_compile src/train_resnet34_unet_brr_b9_dcf_e200.py

echo "===== SMOKE RUN B9-DCF-lite: ctx64 + dcf16, 3 epochs ====="
python src/train_resnet34_unet_brr_b9_dcf_e200.py \
  --run-name smoke_b9_dcf_lite_e3 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 2 \
  --epochs 3 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --context-branch-channels 64 \
  --context-dropout 0.0 \
  --context-init-scale 0.10 \
  --dcf-channels 16 \
  --dcf-beta 0.20

echo "===== DONE SMOKE B9-DCF-lite ====="
EOF

echo "===== Create full training script: B9-DCF-lite ====="
cat > submitjob_b9_dcf_lite_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b9_dcf_lite
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b9_dcf_lite_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== ENV ====="
which python
python --version
nvidia-smi

python -m py_compile src/train_resnet34_unet_brr_b9_dcf_e200.py

echo "===== RUN B9-DCF-lite: B8B ctx64 + DCF dcf16 ====="
python src/train_resnet34_unet_brr_b9_dcf_e200.py \
  --run-name b9_dcf_lite_ctx64_auglite_e200_constlr_bs6 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 200 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --context-branch-channels 64 \
  --context-dropout 0.0 \
  --context-init-scale 0.10 \
  --dcf-channels 16 \
  --dcf-beta 0.20

echo "===== DONE B9-DCF-lite ====="
EOF

chmod +x submitjob_smoke_b9_dcf_lite_e3.sh
chmod +x submitjob_b9_dcf_lite_auglite_e200_constlr_bs6.sh

echo "===== Shell syntax check ====="
bash -n submitjob_smoke_b9_dcf_lite_e3.sh
bash -n submitjob_b9_dcf_lite_auglite_e200_constlr_bs6.sh

echo "===== Created files ====="
ls -lh submitjob_smoke_b9_dcf_lite_e3.sh
ls -lh submitjob_b9_dcf_lite_auglite_e200_constlr_bs6.sh

echo "===== Next step ====="
echo "1) Submit smoke:"
echo "   sbatch submitjob_smoke_b9_dcf_lite_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b9_dcf_lite_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_b9_dcf_lite_auglite_e200_constlr_bs6.sh"
