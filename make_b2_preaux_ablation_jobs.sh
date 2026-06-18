#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SCRIPT="src/train_resnet34_unet_brr_b2_preaux_e200.py"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: cannot find $SCRIPT"
  echo "Please make sure B2-preaux script has been generated."
  exit 1
fi

mkdir -p runs/logs

echo "===== Python syntax check ====="
python -m py_compile "$SCRIPT"

echo "===== Create smoke script: B2-preaux-lite ====="
cat > submitjob_smoke_b2_preaux_lite_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_pal
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b2_preaux_lite_%j.out

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

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== SMOKE RUN: B2-preaux-lite, 3 epochs ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name smoke_b2_preaux_lite_e3 \
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
  --pre-boundary-weight 0.05 \
  --pre-rim-weight 0.025

echo "===== DONE SMOKE B2-preaux-lite ====="
EOF

echo "===== Create smoke script: B2-prebd-only ====="
cat > submitjob_smoke_b2_prebd_only_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_pbd
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b2_prebd_only_%j.out

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

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== SMOKE RUN: B2-prebd-only, 3 epochs ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name smoke_b2_prebd_only_e3 \
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
  --pre-boundary-weight 0.05 \
  --pre-rim-weight 0.00

echo "===== DONE SMOKE B2-prebd-only ====="
EOF

echo "===== Create full script: B2-preaux-lite ====="
cat > submitjob_b2_preaux_lite_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b2_pal
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b2_preaux_lite_%j.out

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

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== RUN B2-preaux-lite: pre_boundary=0.05, pre_rim=0.025 ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name b2_preaux_lite_pb005_pr0025_auglite_e200_constlr_bs6 \
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
  --pre-boundary-weight 0.05 \
  --pre-rim-weight 0.025

echo "===== DONE B2-preaux-lite ====="
EOF

echo "===== Create full script: B2-prebd-only ====="
cat > submitjob_b2_prebd_only_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b2_pbd
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b2_prebd_only_%j.out

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

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== RUN B2-prebd-only: pre_boundary=0.05, pre_rim=0.00 ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name b2_prebd_only_pb005_pr000_auglite_e200_constlr_bs6 \
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
  --pre-boundary-weight 0.05 \
  --pre-rim-weight 0.00

echo "===== DONE B2-prebd-only ====="
EOF

chmod +x submitjob_smoke_b2_preaux_lite_e3.sh
chmod +x submitjob_smoke_b2_prebd_only_e3.sh
chmod +x submitjob_b2_preaux_lite_auglite_e200_constlr_bs6.sh
chmod +x submitjob_b2_prebd_only_auglite_e200_constlr_bs6.sh

echo "===== Shell syntax check ====="
bash -n submitjob_smoke_b2_preaux_lite_e3.sh
bash -n submitjob_smoke_b2_prebd_only_e3.sh
bash -n submitjob_b2_preaux_lite_auglite_e200_constlr_bs6.sh
bash -n submitjob_b2_prebd_only_auglite_e200_constlr_bs6.sh

echo "===== Created files ====="
ls -lh submitjob_smoke_b2_preaux_lite_e3.sh
ls -lh submitjob_smoke_b2_prebd_only_e3.sh
ls -lh submitjob_b2_preaux_lite_auglite_e200_constlr_bs6.sh
ls -lh submitjob_b2_prebd_only_auglite_e200_constlr_bs6.sh

echo "===== Next step ====="
echo "1) Submit smoke tests:"
echo "   sbatch submitjob_smoke_b2_preaux_lite_e3.sh"
echo "   sbatch submitjob_smoke_b2_prebd_only_e3.sh"
echo
echo "2) Watch smoke logs:"
echo "   tail -f runs/logs/smoke_b2_preaux_lite_*.out"
echo "   tail -f runs/logs/smoke_b2_prebd_only_*.out"
echo
echo "3) If both smoke tests pass, submit full runs:"
echo "   sbatch submitjob_b2_preaux_lite_auglite_e200_constlr_bs6.sh"
echo "   sbatch submitjob_b2_prebd_only_auglite_e200_constlr_bs6.sh"
