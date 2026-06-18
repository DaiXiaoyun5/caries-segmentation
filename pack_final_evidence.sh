#!/bin/bash
set -e

ROOT="/share/home/u2515283028/caries_project"
cd "$ROOT"

OUT="$ROOT/final_paper_evidence_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

mkdir -p "$OUT/src" "$OUT/runs" "$OUT/logs" "$OUT/submit_scripts"

# 新增脚本
for f in \
  src/analyze_final_paper_evidence.py \
  src/measure_final_model_complexity.py \
  src/make_final_visual_cases.py
 do
  [ -f "$f" ] && cp --parents "$f" "$OUT/"
 done

# 最终分析结果
[ -d runs/group_analysis/top_cbsa_final ] && cp -r runs/group_analysis/top_cbsa_final "$OUT/runs/"
[ -d runs/model_complexity_final ] && cp -r runs/model_complexity_final "$OUT/runs/"
[ -d runs/final_visual_cases ] && cp -r runs/final_visual_cases "$OUT/runs/"

# 关键 run 的指标文件
EXPS=(
  "unet3p_baseline_e50_bs6_dp2_fair"
  "unet3p_baseline_e50_bs6_fair"
  "resnet34_unet3p_baseline_e50"
  "resnet34_unet3p_cbsa_e50_bs6"
  "resnet34_unet3p_cbsa_laspp_e50_bs6"
  "resnet34_unet3p_cbsa_ppm_e50_bs6"
  "resnet34_unet3p_cbsa_ocr_lite_e50_bs6"
  "resnet34_unet3p_cbsa_laspp_boundaryloss_e50_bs6"
  "resnet34_unet3p_cbsa_laspp_csftl_e50_bs6"
  "resnet34_unet3p_edge_simam_e50_bs6"
  "resnet34_unet3p_brm_e50_bs6"
)

for exp in "${EXPS[@]}"; do
  if [ -d "runs/$exp" ]; then
    mkdir -p "$OUT/runs/$exp"
    for f in config.json history.csv test_metrics.json best_val_metrics.json best_train_metrics.json summary_metrics.json per_case_metrics.csv per_case_metrics_summary.json; do
      [ -f "runs/$exp/$f" ] && cp "runs/$exp/$f" "$OUT/runs/$exp/"
    done
  fi
done

# 日志
find runs/logs -type f \( -name "*.out" -o -name "*.log" \) -size -5M -exec cp {} "$OUT/logs/" \; 2>/dev/null || true
find . -maxdepth 1 -type f -name "submit*.sh" -exec cp {} "$OUT/submit_scripts/" \; 2>/dev/null || true

# 环境信息
{
  echo "===== date ====="; date
  echo "===== pwd ====="; pwd
  echo "===== hostname ====="; hostname
  echo "===== final dirs ====="
  find runs/group_analysis/top_cbsa_final runs/model_complexity_final runs/final_visual_cases -maxdepth 2 -type f 2>/dev/null | sort || true
} > "$OUT/pack_info.txt"

tar -czf "${OUT}.tar.gz" -C "$ROOT" "$(basename "$OUT")"

echo "完成，请上传：${OUT}.tar.gz"
ls -lh "${OUT}.tar.gz"
