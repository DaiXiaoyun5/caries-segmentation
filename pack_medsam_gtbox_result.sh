#!/bin/bash
set -e
cd /share/home/u2515283028/caries_project
OUT=medsam_gtbox_result_for_chatgpt_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
mkdir -p "$OUT/runs_medsam"
cp -r runs_medsam/medsam_gtbox_test_pad005 "$OUT/runs_medsam/" 2>/dev/null || true
mkdir -p "$OUT/src" "$OUT/submit_scripts" "$OUT/runs/logs"
cp src/eval_medsam_gtbox_caries.py "$OUT/src/" 2>/dev/null || true
cp setup_medsam_caries.sh "$OUT/submit_scripts/" 2>/dev/null || true
cp submitjob_medsam_gtbox_caries.sh "$OUT/submit_scripts/" 2>/dev/null || true
cp runs/logs/medsam_gtbox_*.out "$OUT/runs/logs/" 2>/dev/null || true
find "$OUT" -type f | sort > "$OUT/file_list.txt"
tar -czf "${OUT}.tar.gz" "$OUT"
ls -lh "${OUT}.tar.gz"
