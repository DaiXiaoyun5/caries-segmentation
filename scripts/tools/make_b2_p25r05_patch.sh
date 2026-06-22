#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b2_p25r05_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B2-p25r05 ====="
cp -p "$SRC" "$DST"

echo "===== Patch lambda_pos=0.25, lambda_rim=0.05 ====="
python - <<'PY'
from pathlib import Path
import re

p = Path("src/train_resnet34_unet_brr_b2_p25r05_e200.py")
s = p.read_text(encoding="utf-8")

# 1) 修改 RBSGLite 默认参数：lambda_pos 0.20 -> 0.25, lambda_rim 0.10 -> 0.05
s, n_pos = re.subn(r"(lambda_pos\s*=\s*)0\.20", r"\g<1>0.25", s)
s, n_rim = re.subn(r"(lambda_rim\s*=\s*)0\.10", r"\g<1>0.05", s)

# 如果代码里是 0.2 / 0.1 写法，也兼容一下
if n_pos == 0:
    s, n_pos = re.subn(r"(lambda_pos\s*=\s*)0\.2([^0-9])", r"\g<1>0.25\2", s)
if n_rim == 0:
    s, n_rim = re.subn(r"(lambda_rim\s*=\s*)0\.1([^0-9])", r"\g<1>0.05\2", s)

if n_pos == 0 or n_rim == 0:
    raise RuntimeError(
        f"Failed to patch lambda values: n_pos={n_pos}, n_rim={n_rim}. "
        "Please inspect RBSGLite definition."
    )

# 2) 修改实验说明
s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B2-p25r05: B2 BRR-RBSG with lambda_pos=0.25 and lambda_rim=0.05"',
)

s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B2-p25r05 keeps the original B2 architecture, rb=3 and rr=9, but changes RBSG-lite strength to lambda_pos=0.25 and lambda_rim=0.05. It strengthens positive boundary/lesion enhancement and weakens rim suppression."',
)

# 3) 修改默认 run name
s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b2_p25r05_rbsg_auglite_e200_constlr_bs6")',
)

# 4) 修改打印提示
s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B2-p25r05 BRR RBSG-lite ResNet34-U-Net")',
)

# 5) 在配置里记录 lambda
old_loss_block = '''    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim"
    }
'''
new_loss_block = '''    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
        "rbsg_lambda_pos": 0.25,
        "rbsg_lambda_rim": 0.05,
        "boundary_rb": 3,
        "rim_rr": 9
    }
'''
if old_loss_block in s:
    s = s.replace(old_loss_block, new_loss_block)
else:
    print("[WARN] Could not find original config loss block. Lambda values were still patched.")

p.write_text(s, encoding="utf-8")

print(f"Patched file: {p}")
print(f"lambda_pos replacements: {n_pos}")
print(f"lambda_rim replacements: {n_rim}")

# 简单检查
text = p.read_text(encoding="utf-8")
for key in ["lambda_pos", "lambda_rim"]:
    lines = [line.strip() for line in text.splitlines() if key in line]
    print(f"\n[{key}]")
    for line in lines[:10]:
        print(line)
PY

echo "===== Create full training sbatch script ====="
cat > scripts/jobs/submitjob_b2_p25r05_rbsg_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b2_p25r05
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b2_p25r05_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_p25r05_e200.py

echo "===== RUN B2-p25r05: lambda_pos=0.25, lambda_rim=0.05 ====="
python src/train_resnet34_unet_brr_b2_p25r05_e200.py \
  --run-name b2_p25r05_rbsg_auglite_e200_constlr_bs6 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 200 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet

echo "===== DONE B2-p25r05 ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > scripts/jobs/submitjob_smoke_b2_p25r05_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b2p
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b2_p25r05_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_p25r05_e200.py

echo "===== SMOKE RUN B2-p25r05: 3 epochs ====="
python src/train_resnet34_unet_brr_b2_p25r05_e200.py \
  --run-name smoke_b2_p25r05_e3 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 2 \
  --epochs 3 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet

echo "===== DONE SMOKE B2-p25r05 ====="
EOF

chmod +x scripts/jobs/submitjob_b2_p25r05_rbsg_auglite_e200_constlr_bs6.sh
chmod +x scripts/jobs/submitjob_smoke_b2_p25r05_e3.sh

echo "===== Shell syntax check ====="
bash -n scripts/jobs/submitjob_b2_p25r05_rbsg_auglite_e200_constlr_bs6.sh
bash -n scripts/jobs/submitjob_smoke_b2_p25r05_e3.sh

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_b2_p25r05_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b2_p25r05_e200.py
ls -lh scripts/jobs/submitjob_b2_p25r05_rbsg_auglite_e200_constlr_bs6.sh
ls -lh scripts/jobs/submitjob_smoke_b2_p25r05_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch scripts/jobs/submitjob_smoke_b2_p25r05_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b2_p25r05_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch scripts/jobs/submitjob_b2_p25r05_rbsg_auglite_e200_constlr_bs6.sh"
