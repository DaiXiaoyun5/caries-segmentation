from pathlib import Path
import argparse
import csv
import json

import torch
from torch.utils.data import DataLoader
import numpy as np

from train_resunet_cbsa_mkdc_ori1_pad640 import (
    ResNet34UNetCBSAMKDC,
    OriginalScalePadCropCariesMaskDataset,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_binary_metrics(pred, target, eps=1e-7):
    pred = pred.bool()
    target = target.bool()

    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, torch.logical_not(target)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred), target).sum().item()

    pred_area = pred.sum().item()
    gt_area = target.sum().item()

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
        "pred_area": int(pred_area),
        "gt_area": int(gt_area),
    }


def summarize_rows(rows):
    metrics = ["dice", "iou", "precision", "recall", "pred_area", "gt_area"]
    summary = {"num_cases": len(rows)}

    for m in metrics:
        values = np.array([r[m] for r in rows], dtype=np.float64)
        summary[f"mean_{m}"] = float(values.mean()) if len(values) else None
        summary[f"median_{m}"] = float(np.median(values)) if len(values) else None
        summary[f"std_{m}"] = float(values.std()) if len(values) else None
        summary[f"min_{m}"] = float(values.min()) if len(values) else None
        summary[f"max_{m}"] = float(values.max()) if len(values) else None

    # Lesion-size quartile analysis by gt_area
    if len(rows) > 0:
        sorted_rows = sorted(rows, key=lambda x: x["gt_area"])
        chunks = np.array_split(sorted_rows, 4)
        quartiles = {}
        for i, chunk in enumerate(chunks, start=1):
            chunk = list(chunk)
            if not chunk:
                continue
            quartiles[f"Q{i}"] = {
                "num_cases": len(chunk),
                "gt_area_min": int(min(r["gt_area"] for r in chunk)),
                "gt_area_max": int(max(r["gt_area"] for r in chunk)),
                "mean_dice": float(np.mean([r["dice"] for r in chunk])),
                "mean_iou": float(np.mean([r["iou"] for r in chunk])),
                "mean_precision": float(np.mean([r["precision"] for r in chunk])),
                "mean_recall": float(np.mean([r["recall"] for r in chunk])),
                "mean_pred_area": float(np.mean([r["pred_area"] for r in chunk])),
                "mean_gt_area": float(np.mean([r["gt_area"] for r in chunk])),
            }
        summary["quartiles_by_gt_area"] = quartiles

    return summary


def main(args):
    project_root = Path("/share/home/u2515283028/caries_project")
    run_dir = project_root / "runs" / args.run_name
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best.pth"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    config = load_json(config_path)
    image_size = int(config.get("image_size", args.image_size))

    split_path = project_root / "data" / "splits" / f"{args.split}.txt"
    dataset = OriginalScalePadCropCariesMaskDataset(split_path, target_size=(image_size, image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    kernel_sizes = tuple(int(v) for v in str(config.get("mk_kernel_sizes", "1,3,5")).split(","))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNetCBSAMKDC(
        encoder_name=config.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=float(config.get("cbsa_lambda", 1e-4)),
        cbsa_alpha=float(config.get("cbsa_alpha", 1.0)),
        cbsa_beta=float(config.get("cbsa_beta", 0.1)),
        mk_expansion_factor=int(config.get("mk_expansion_factor", 2)),
        mk_kernel_sizes=kernel_sizes,
        mk_depth=int(config.get("mk_depth", 1)),
        mk_add=bool(config.get("mk_add", True)),
        mk_activation=config.get("mk_activation", "relu6"),
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for images, masks, img_paths, mask_paths in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = probs > args.threshold

            for b in range(images.shape[0]):
                m = compute_binary_metrics(preds[b, 0].cpu(), masks[b, 0].cpu())
                m["image_path"] = img_paths[b]
                m["mask_path"] = mask_paths[b]
                rows.append(m)

    out_csv = run_dir / f"per_case_metrics_{args.split}_ori1.csv"
    out_summary = run_dir / f"per_case_metrics_summary_{args.split}_ori1.json"

    fieldnames = [
        "image_path", "mask_path",
        "dice", "iou", "precision", "recall",
        "tp", "fp", "fn", "pred_area", "gt_area",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    summary = summarize_rows(rows)
    summary["run_name"] = args.run_name
    summary["split"] = args.split
    summary["threshold"] = args.threshold
    summary["image_size"] = image_size
    summary["input_mode"] = "original-scale center pad/crop"
    save_json(summary, out_summary)

    # Also copy to standard names for easier packaging/comparison.
    if args.split == "test":
        standard_csv = run_dir / "per_case_metrics.csv"
        standard_summary = run_dir / "per_case_metrics_summary.json"
        standard_csv.write_text(out_csv.read_text(encoding="utf-8"), encoding="utf-8")
        save_json(summary, standard_summary)

    print("Saved:", out_csv)
    print("Saved:", out_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
