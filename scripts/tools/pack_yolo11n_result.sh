cd /share/home/u2515283028/caries_project

OUT=yolo11n_caries_detect_result_for_chatgpt_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

mkdir -p "$OUT/runs_yolo"
cp -r runs_yolo/yolo11n_caries_detect "$OUT/runs_yolo/" 2>/dev/null || true
cp -r runs_yolo/yolo11n_caries_detect_testval "$OUT/runs_yolo/" 2>/dev/null || true

mkdir -p "$OUT/data_preview"
cp data/yolo_caries_detect/data.yaml "$OUT/data_preview/" 2>/dev/null || true
cp data/yolo_caries_detect/train.txt "$OUT/data_preview/" 2>/dev/null || true
cp data/yolo_caries_detect/val.txt "$OUT/data_preview/" 2>/dev/null || true
cp data/yolo_caries_detect/test.txt "$OUT/data_preview/" 2>/dev/null || true
find data/yolo_caries_detect/labels/test -type f | sort | head -20 | xargs -I{} cp {} "$OUT/data_preview/" 2>/dev/null || true

mkdir -p "$OUT/src" "$OUT/submit_scripts" "$OUT/runs/logs"
cp src/prepare_yolo_caries_detection.py "$OUT/src/" 2>/dev/null || true
cp src/eval_yolo_caries_recall_conf.py "$OUT/src/" 2>/dev/null || true
cp scripts/jobs/submitjob_yolo11n_caries_detect.sh "$OUT/submit_scripts/" 2>/dev/null || true
cp runs/logs/yolo11n_caries_*.out "$OUT/runs/logs/" 2>/dev/null || true

find "$OUT" -type f | sort > "$OUT/file_list.txt"
tar -czf "${OUT}.tar.gz" "$OUT"
ls -lh "${OUT}.tar.gz"
