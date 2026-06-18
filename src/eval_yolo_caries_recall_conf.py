import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")


def box_iou_xyxy(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def read_yolo_bbox_label(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes

    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return boxes

    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = parts
        xc, yc, w, h = map(float, [xc, yc, w, h])

        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def evaluate_at_threshold(pred_records, conf_thr, iou_thr):
    tp = 0
    fp = 0
    fn = 0
    missed_images = 0
    total_gt_images = 0

    for rec in pred_records:
        gt_boxes = rec["gt_boxes"]
        pred_boxes = [
            box for box, conf in zip(rec["pred_boxes"], rec["pred_confs"])
            if conf >= conf_thr
        ]

        if len(gt_boxes) > 0:
            total_gt_images += 1

        matched_gt = set()
        pred_boxes_sorted = pred_boxes

        for pb in pred_boxes_sorted:
            ious = box_iou_xyxy(pb, gt_boxes)
            if len(ious) == 0:
                fp += 1
                continue

            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_thr and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        cur_fn = len(gt_boxes) - len(matched_gt)
        fn += cur_fn

        if len(gt_boxes) > 0 and cur_fn == len(gt_boxes):
            missed_images += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    miss_image_rate = missed_images / (total_gt_images + 1e-9)

    return {
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "missed_images": int(missed_images),
        "total_gt_images": int(total_gt_images),
        "miss_image_rate": float(miss_image_rate),
    }


def main(args):
    from ultralytics import YOLO

    data_root = PROJECT_ROOT / args.data_root
    split_file = data_root / f"{args.split}.txt"
    label_dir = data_root / "labels" / args.split
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    image_paths = [Path(x.strip()) for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    print("model:", model_path)
    print("split:", args.split)
    print("num images:", len(image_paths))

    model = YOLO(str(model_path))

    pred_records = []

    for idx, img_path in enumerate(image_paths, start=1):
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = read_yolo_bbox_label(label_path, img_w, img_h)

        # Use low conf during prediction, then evaluate multiple thresholds manually.
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.predict_conf,
            iou=args.nms_iou,
            verbose=False,
            device=args.device,
        )

        pred_boxes = []
        pred_confs = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            for b, c in zip(xyxy, confs):
                pred_boxes.append([float(v) for v in b])
                pred_confs.append(float(c))

        pred_records.append({
            "image": str(img_path),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_confs": pred_confs,
        })

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] done")

    thresholds = [float(x) for x in args.conf_thresholds.split(",")]
    summaries = [evaluate_at_threshold(pred_records, t, args.eval_iou) for t in thresholds]

    out_dir = PROJECT_ROOT / "runs_yolo" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary = out_dir / f"custom_recall_eval_{args.split}.json"
    out_detail = out_dir / f"custom_recall_predictions_{args.split}.json"

    result_obj = {
        "model": str(model_path),
        "data_root": str(data_root),
        "split": args.split,
        "num_images": len(image_paths),
        "predict_conf": args.predict_conf,
        "nms_iou": args.nms_iou,
        "eval_iou": args.eval_iou,
        "summaries": summaries,
    }

    out_summary.write_text(json.dumps(result_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    out_detail.write_text(json.dumps(pred_records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== custom recall summary =====")
    print(json.dumps(result_obj, indent=2, ensure_ascii=False))
    print("Saved:", out_summary)
    print("Saved:", out_detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/yolo_caries_detect")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-name", type=str, default="yolo11n_caries_detect")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--conf-thresholds", type=str, default="0.05,0.10,0.15,0.25")
    args = parser.parse_args()
    main(args)
