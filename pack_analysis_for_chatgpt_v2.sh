#!/bin/bash
set -e

ROOT="/share/home/u2515283028/caries_project"
OUT="$ROOT/chatgpt_analysis_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

cd "$ROOT"

echo "[1/7] 保存项目结构清单..."
pwd > "$OUT/pwd.txt"
ls -lah > "$OUT/top_ls_lah.txt"

find . \
  -path "./.git" -prune -o \
  -path "./data" -prune -o \
  -path "./dataset" -prune -o \
  -path "./datasets" -prune -o \
  -path "./nnunet_caries" -prune -o \
  -path "./project_inventory_*" -prune -o \
  -path "./chatgpt_analysis_pack_*" -prune -o \
  -path "./chatgpt_backup_*" -prune -o \
  -path "./rerun_backup" -prune -o \
  -path "./runs/*/checkpoints" -prune -o \
  -path "./runs/*/preds" -prune -o \
  -path "./runs/*/preds_per_case" -prune -o \
  -path "./runs/*/vis" -prune -o \
  -path "./runs/*/overlay" -prune -o \
  -type f \
  | sed 's#^\./##' \
  | sort > "$OUT/file_list_clean.txt"

echo "[2/7] 复制 src 当前代码..."
mkdir -p "$OUT/src"
rsync -av \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  src/ "$OUT/src/" >/dev/null

echo "[3/7] 复制提交脚本..."
mkdir -p "$OUT/submit_scripts"
find . -maxdepth 1 -type f \( -name "submit*.sh" -o -name "*.sh" \) \
  -exec cp {} "$OUT/submit_scripts/" \; 2>/dev/null || true

echo "[4/7] 复制核心实验指标..."
mkdir -p "$OUT/runs"

EXPS=(
  "resnet34_unet3p_cbsa_laspp_boundaryloss_e50_bs6"
  "resnet34_unet3p_cbsa_laspp_csftl_e50_bs6"
  "resnet34_unet3p_cbsa_laspp_e50_bs6"
  "resnet34_unet3p_cbsa_ppm_e50_bs6"
  "resnet34_unet3p_cbsa_ocr_lite_e50_bs6"
  "resnet34_unet3p_cbsa_e50_bs6"
  "resnet34_unet3p_edge_simam_e50_bs6"
  "resnet34_unet3p_edge_e50"
  "resnet34_unet3p_baseline_e50"
  "resnet34_unet3p_cbrm_rs_sad_e50_bs6"
  "resnet34_unet3p_cbrm_rs_e50_bs6"
  "resnet34_unet3p_brm_e50_bs6"
)

for exp in "${EXPS[@]}"; do
  if [ -d "runs/$exp" ]; then
    mkdir -p "$OUT/runs/$exp"
    for f in \
      config.json \
      history.csv \
      test_metrics.json \
      best_val_metrics.json \
      best_train_metrics.json \
      summary_metrics.json \
      per_case_metrics.csv \
      per_case_metrics_summary.json
    do
      if [ -f "runs/$exp/$f" ]; then
        cp "runs/$exp/$f" "$OUT/runs/$exp/"
      fi
    done
  fi
done

echo "[5/7] 复制分组分析结果..."
if [ -d "runs/group_analysis" ]; then
  mkdir -p "$OUT/runs/group_analysis"
  rsync -av \
    --include="*/" \
    --include="*.csv" \
    --include="*.json" \
    --exclude="*" \
    runs/group_analysis/ "$OUT/runs/group_analysis/" >/dev/null
fi

echo "[6/7] 复制训练日志..."
mkdir -p "$OUT/runs/logs"
if [ -d "runs/logs" ]; then
  find runs/logs -type f \( -name "*.out" -o -name "*.log" \) -size -5M \
    -exec cp {} "$OUT/runs/logs/" \; 2>/dev/null || true
fi

echo "[7/7] 保存环境信息并打包..."
{
  echo "===== date ====="
  date
  echo
  echo "===== hostname ====="
  hostname
  echo
  echo "===== conda env list ====="
  module load anaconda3/4.12.0 2>/dev/null || true
  conda env list 2>/dev/null || true
  echo
  echo "===== caries-train torch info ====="
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate caries-train 2>/dev/null || true
  which python || true
  python -V || true
  python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)
    print("gpu_count:", torch.cuda.device_count())
except Exception as e:
    print("torch_error:", repr(e))
PY
} > "$OUT/env_info.txt" 2>&1 || true

tar -czf "${OUT}.tar.gz" -C "$ROOT" "$(basename "$OUT")"

echo
echo "完成，请上传这个文件："
echo "${OUT}.tar.gz"
ls -lh "${OUT}.tar.gz"
