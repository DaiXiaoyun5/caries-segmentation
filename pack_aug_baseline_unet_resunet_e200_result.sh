#!/bin/bash
set -e
cd /share/home/u2515283028/caries_project
OUT=aug_baseline_unet_resunet_e200_result_for_chatgpt_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT/runs" "$OUT/src" "$OUT/submit_scripts" "$OUT/runs/logs"
cp -r runs/unet_auglite_e200_constlr_bs6 "$OUT/runs/" 2>/dev/null || true
cp -r runs/classic_resunet_auglite_e200_constlr_bs6 "$OUT/runs/" 2>/dev/null || true
cp src/train_aug_baseline_unet_resunet_e200.py "$OUT/src/" 2>/dev/null || true
cp submitjob_unet_auglite_e200_constlr_bs6.sh "$OUT/submit_scripts/" 2>/dev/null || true
cp submitjob_classic_resunet_auglite_e200_constlr_bs6.sh "$OUT/submit_scripts/" 2>/dev/null || true
cp runs/logs/unet_auglite_e200_const_*.out "$OUT/runs/logs/" 2>/dev/null || true
cp runs/logs/classic_resunet_auglite_e200_const_*.out "$OUT/runs/logs/" 2>/dev/null || true
find "$OUT" -type f | sort > "$OUT/file_list.txt"
tar -czf "${OUT}.tar.gz" "$OUT"
ls -lh "${OUT}.tar.gz"
