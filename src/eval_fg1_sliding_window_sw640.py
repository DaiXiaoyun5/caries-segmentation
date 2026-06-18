import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_resunet_cbsa_mkdc_fg1_crop import ResNet34UNetCBSAMKDC  # noqa: E402


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_window_starts(full_size, patch_size, stride):
    """
    Generate sliding-window start positions that always cover [0, full_size).
    For full_size=640, patch_size=512, stride=128 -> [0, 128].
    """
    if full_size <= patch_size:
        return [0]

    starts = list(range(0, full_size - patch_size + 1, stride))
    last = full_size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def compute_binary_metrics(pred, target, eps=1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    pred_area = pred.sum()
    gt_area = target.sum()

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
    summary = {"num_cases": len(rows)}
    metrics = ["dice", "iou", "precision", "recall", "pred_area", "gt_area"]

    for m in metrics:
        values = np.array([r[m] for r in rows], dtype=np.float64)
        summary[f"mean_{m}"] = float(values.mean()) if len(values) else None
        summary[f"median_{m}"] = float(np.median(values)) if len(values) else None
        summary[f"std_{m}"] = float(values.std()) if len(values) else None
        summary[f"min_{m}"] = float(values.min()) if len(values) else None
        summary[f"max_{m}"] = float(values.max()) if len(values) else None

    if rows:
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


def load_split(split_file):
    items = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, mask_path = line.split("\t")
            items.append((Path(img_path), Path(mask_path)))
    return items


def build_model_from_config(config, device):
    kernel_sizes = tuple(int(v) for v in str(config.get("mk_kernel_sizes", "1,3,5")).split(","))

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

    return model


@torch.no_grad()
def sliding_window_predict(model, image_np, patch_size=512, stride=128, batch_size=4, device="cuda"):
    """
    image_np: RGB float32 numpy, shape [H, W, 3], range [0, 1]
    returns prob_map: float32 numpy, shape [H, W], range [0, 1]
    """
    h, w = image_np.shape[:2]
    y_starts = get_window_starts(h, patch_size, stride)
    x_starts = get_window_starts(w, patch_size, stride)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patches = []
    coords = []

    def flush():
        nonlocal patches, coords, prob_sum, count_map
        if not patches:
            return

        batch_np = np.stack(patches, axis=0)  # [B, H, W, 3]
        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(device)

        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]  # [B, patch, patch]

        for prob, (yy, xx) in zip(probs, coords):
            prob_sum[yy:yy + patch_size, xx:xx + patch_size] += prob.astype(np.float32)
            count_map[yy:yy + patch_size, xx:xx + patch_size] += 1.0

        patches = []
        coords = []

    for yy in y_starts:
        for xx in x_starts:
            patch = image_np[yy:yy + patch_size, xx:xx + patch_size, :]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                raise RuntimeError(f"Invalid patch shape: {patch.shape}, yy={yy}, xx={xx}")

            patches.append(patch)
            coords.append((yy, xx))

            if len(patches) >= batch_size:
                flush()

    flush()

    count_map = np.maximum(count_map, 1.0)
    prob_map = prob_sum / count_map
    return prob_map


def main(args):
    run_dir = PROJECT_ROOT / "runs" / args.run_name
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    config = load_json(config_path)

    split_file = PROJECT_ROOT / "data" / "splits" / f"{args.split}.txt"
    items = load_split(split_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("run_name:", args.run_name)
    print("split:", args.split)
    print("num_cases:", len(items))
    print("eval_size:", args.eval_size)
    print("patch_size:", args.patch_size)
    print("stride:", args.stride)
    print("threshold:", args.threshold)

    model = build_model_from_config(config, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []

    for idx, (img_path, mask_path) in enumerate(items, start=1):
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # FG-style sliding-window evaluation:
        # resize image and mask to 640, infer with 512 patch sliding window.
        image = image.resize((args.eval_size, args.eval_size), resample=Image.BILINEAR)
        mask = mask.resize((args.eval_size, args.eval_size), resample=Image.NEAREST)

        image_np = np.asarray(image).astype(np.float32) / 255.0
        mask_np = (np.asarray(mask).astype(np.float32) > 0).astype(np.uint8)

        prob_map = sliding_window_predict(
            model,
            image_np,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
        )
        pred_np = (prob_map > args.threshold).astype(np.uint8)

        m = compute_binary_metrics(pred_np, mask_np)
        m["image_path"] = str(img_path)
        m["mask_path"] = str(mask_path)
        rows.append(m)

        if idx % 50 == 0 or idx == len(items):
            print(f"[{idx}/{len(items)}] mean dice so far = {np.mean([r['dice'] for r in rows]):.4f}")

    tag = f"sw{args.eval_size}_p{args.patch_size}_s{args.stride}_{args.split}"
    out_csv = run_dir / f"sliding_window_metrics_{tag}.csv"
    out_summary = run_dir / f"sliding_window_summary_{tag}.json"

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
    summary.update({
        "run_name": args.run_name,
        "split": args.split,
        "eval_mode": "resize640_sliding_window_patch512",
        "eval_size": args.eval_size,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "threshold": args.threshold,
        "checkpoint": str(ckpt_path),
        "num_windows_per_image": len(get_window_starts(args.eval_size, args.patch_size, args.stride)) ** 2,
    })

    save_json(summary, out_summary)

    print("Saved:", out_csv)
    print("Saved:", out_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="resunet_cbsa_mkdc_fg1_crop_e80_bs6")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval-size", type=int, default=640)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
