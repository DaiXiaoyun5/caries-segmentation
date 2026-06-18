#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
NNUNET_BASE = PROJECT_ROOT / "nnunet_caries"
META_PATH = NNUNET_BASE / "caries_nnunet_meta.json"


def read_mask(path: Path):
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8)


def compute_one(pred, gt, eps=1e-7):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_pred(pred_dir: Path, case_id: str):
    candidates = [
        pred_dir / f"{case_id}.png",
        pred_dir / f"{case_id}_0000.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = list(pred_dir.glob(f"{case_id}*.png"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find prediction for case_id={case_id} in {pred_dir}")


def main(pred_dir):
    pred_dir = Path(pred_dir)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_ids = meta["test_ids"]
    gt_dir = Path(meta["test_gt_dir"])

    rows = []
    missing = []

    for case_id in test_ids:
        gt_path = gt_dir / f"{case_id}.png"
        try:
            pred_path = find_pred(pred_dir, case_id)
        except FileNotFoundError:
            missing.append(case_id)
            continue

        pred = read_mask(pred_path)
        gt = read_mask(gt_path)

        if pred.shape != gt.shape:
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = (np.array(pred_img) > 0).astype(np.uint8)

        m = compute_one(pred, gt)
        m["case_id"] = case_id
        m["pred_path"] = str(pred_path)
        m["gt_path"] = str(gt_path)
        rows.append(m)

    if len(rows) == 0:
        raise RuntimeError("No predictions evaluated.")

    summary = {}
    for k in ["dice", "iou", "precision", "recall"]:
        summary[k] = float(np.mean([r[k] for r in rows]))

    summary["num_cases"] = len(rows)
    summary["missing_cases"] = missing
    summary["pred_dir"] = str(pred_dir)

    out_json = pred_dir / "nnunet_test_metrics.json"
    out_csv = pred_dir / "nnunet_per_case_metrics.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("case_id,dice,iou,precision,recall,tp,fp,fn,pred_path,gt_path\n")
        for r in rows:
            f.write(
                f"{r['case_id']},{r['dice']},{r['iou']},{r['precision']},{r['recall']},"
                f"{r['tp']},{r['fp']},{r['fn']},{r['pred_path']},{r['gt_path']}\n"
            )

    print("=" * 80)
    print("nnU-Net test evaluation done")
    print(json.dumps(summary, indent=2))
    print("Saved:", out_json)
    print("Saved:", out_csv)
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, required=True)
    args = parser.parse_args()

    main(args.pred_dir)
