#!/bin/bash
set -e

ROOT="/share/home/u2515283028/caries_project"
OUT="$ROOT/m1_m5_readiness.txt"

cd "$ROOT"

echo "===== M1-M5 readiness check =====" > "$OUT"
echo "date: $(date)" >> "$OUT"
echo "pwd: $(pwd)" >> "$OUT"
echo >> "$OUT"

echo "===== 1. top level =====" >> "$OUT"
ls -lah >> "$OUT"
echo >> "$OUT"

echo "===== 2. src files related to unet / cbsa / simam / laspp / aspp =====" >> "$OUT"
find src -maxdepth 3 -type f | sort | grep -Ei "unet|cbsa|simam|laspp|aspp|smp|resnet" >> "$OUT" || true
echo >> "$OUT"

echo "===== 3. train scripts =====" >> "$OUT"
find src -maxdepth 2 -type f -name "train*.py" | sort >> "$OUT"
echo >> "$OUT"

echo "===== 4. class names possibly useful =====" >> "$OUT"
grep -R "class .*UNet\|class .*Unet\|class .*CBSA\|class .*SimAM\|class .*ASPP\|class .*Lite" -n src --include="*.py" >> "$OUT" || true
echo >> "$OUT"

echo "===== 5. function names possibly useful =====" >> "$OUT"
grep -R "def .*cbsa\|def .*simam\|def .*aspp\|def .*unet\|def .*build\|def .*create" -ni src --include="*.py" >> "$OUT" || true
echo >> "$OUT"

echo "===== 6. existing U-Net / UNet++ / ResNet34 runs =====" >> "$OUT"
find runs -maxdepth 2 -type f \( \
  -name "test_metrics.json" -o \
  -name "per_case_metrics_summary.json" -o \
  -name "config.json" -o \
  -name "history.csv" \
\) | sort | grep -Ei "unet|unetpp|resnet34|cbsa|simam|laspp|aspp" >> "$OUT" || true
echo >> "$OUT"

echo "===== 7. existing run dirs =====" >> "$OUT"
find runs -maxdepth 1 -type d | sort | grep -Ei "unet|unetpp|resnet34|cbsa|simam|laspp|aspp" >> "$OUT" || true
echo >> "$OUT"

echo "===== 8. environment =====" >> "$OUT"
{
  module load anaconda3/4.12.0 2>/dev/null || true
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate caries-train 2>/dev/null || true

  echo "which python:"
  which python || true
  echo "python version:"
  python --version || true

  echo "torch info:"
  python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)
except Exception as e:
    print("torch_error:", repr(e))
PY

  echo "installed possible libs:"
  python - <<'PY'
mods = ["segmentation_models_pytorch", "timm", "torchvision", "pandas", "numpy", "cv2"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "NO", repr(e))
PY
} >> "$OUT" 2>&1 || true

echo "检查完成：$OUT"
cat "$OUT"
