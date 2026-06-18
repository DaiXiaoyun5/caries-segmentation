#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import csv
import random
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class CariesDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512)):
        self.items = []
        self.target_size = target_size

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, mask_path = line.split("\t")
                self.items.append((Path(img_path), Path(mask_path)))

        if len(self.items) == 0:
            raise RuntimeError(f"No samples found in {split_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return image_tensor, mask_tensor, str(img_path), str(mask_path)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=True
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PlainUNet(nn.Module):
    """
    普通 U-Net baseline：
    不使用预训练编码器，用来作为最基础的经典分割模型对比。
    """
    def __init__(self, in_channels=3, num_classes=1, base_ch=64):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)

    return 1.0 - dice.mean()


def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


def compute_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


def run_one_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    num_batches = 0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=True
                )

            loss = bce_dice_loss(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

        metrics = compute_metrics_from_logits(logits.detach(), masks.detach())

        total["loss"] += loss.item()
        for k in ["dice", "iou", "precision", "recall"]:
            total[k] += metrics[k]

        num_batches += 1

    for k in total:
        total[k] /= max(num_batches, 1)

    return total


@torch.no_grad()
def eval_per_case(model, loader, device, out_csv, out_json, threshold=0.5, eps=1e-7):
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for images, masks, img_paths, mask_paths in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=True
            )

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i in range(images.size(0)):
            pred = preds[i].view(-1)
            gt = masks[i].view(-1)

            tp = (pred * gt).sum().item()
            fp = (pred * (1 - gt)).sum().item()
            fn = ((1 - pred) * gt).sum().item()

            dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)

            rows.append({
                "case_id": Path(img_paths[i]).stem,
                "img_path": img_paths[i],
                "mask_path": mask_paths[i],
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            })

    fieldnames = [
        "case_id",
        "img_path",
        "mask_path",
        "dice",
        "iou",
        "precision",
        "recall",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_cases": len(rows),
        "dice": float(np.mean([r["dice"] for r in rows])),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
    }

    save_json(summary, out_json)
    return summary


@torch.no_grad()
def save_prediction_panel(model, loader, device, out_path, max_items=4):
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images, masks, _, _ = next(iter(loader))
    images = images.to(device)

    logits = model(images)

    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=True
        )

    preds = (torch.sigmoid(logits).cpu() > 0.5).float()

    images_np = images.cpu().numpy()
    masks_np = masks.numpy()
    preds_np = preds.numpy()

    n = min(images_np.shape[0], max_items)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img = np.transpose(images_np[i], (1, 2, 0))
        gt = masks_np[i, 0]
        pred = preds_np[i, 0]

        overlay = img.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] * 0.6 + pred * 0.4, 0, 1)

        panels = [
            (img, "Image", None),
            (gt, "GT Mask", "gray"),
            (pred, "Pred Mask", "gray"),
            (overlay, "Overlay", None),
        ]

        for j, (arr, title, cmap) in enumerate(panels):
            axes[i, j].imshow(arr, cmap=cmap)
            axes[i, j].set_title(title)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def try_profile_gflops(model, image_size, device):
    try:
        from thop import profile
    except Exception as e:
        return "", f"thop import failed: {repr(e)}"

    try:
        model = model.to(device)
        model.eval()
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9, ""
    except Exception as e:
        return "", f"profile failed: {repr(e)}"


def build_model(args):
    if args.model_type == "plain_unet":
        return PlainUNet(
            in_channels=3,
            num_classes=1,
            base_ch=args.base_channels
        )

    if args.model_type == "resunet":
        return smp.Unet(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=3,
            classes=1,
        )

    raise ValueError(f"Unsupported model_type: {args.model_type}")


def main(args):
    set_seed(args.seed)

    train_split = PROJECT_ROOT / "data/splits/train.txt"
    val_split = PROJECT_ROOT / "data/splits/val.txt"
    test_split = PROJECT_ROOT / "data/splits/test.txt"

    run_dir = PROJECT_ROOT / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "preds"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["script"] = "train_unet_resunet_baselines.py"
    save_json(config, run_dir / "config.json")

    train_ds = CariesDataset(
        train_split,
        target_size=(args.image_size, args.image_size)
    )
    val_ds = CariesDataset(
        val_split,
        target_size=(args.image_size, args.image_size)
    )
    test_ds = CariesDataset(
        test_split,
        target_size=(args.image_size, args.image_size)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("=" * 80)
    print("Run name:", args.run_name)
    print("Model type:", args.model_type)
    print("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    model = build_model(args).to(device)

    params_m = count_params_m(model)
    gflops, profile_error = try_profile_gflops(model, args.image_size, device)

    save_json({
        "model_type": args.model_type,
        "run_name": args.run_name,
        "params_m": params_m,
        "gflops_512": gflops,
        "profile_error": profile_error,
    }, run_dir / "model_info.json")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sample_images, _, _, _ = next(iter(train_loader))
    sample_images = sample_images.to(device)

    with torch.no_grad():
        sample_logits = model(sample_images)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", tuple(sample_images.shape))
    print("logits.shape       =", tuple(sample_logits.shape))
    print("params_m           =", params_m)
    print("gflops_512         =", gflops)
    print("profile_error      =", profile_error)
    print("=" * 80)

    history_path = run_dir / "history.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "train_dice",
        "train_iou",
        "train_precision",
        "train_recall",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
    ]

    best_val_dice = -1.0
    best_epoch = -1
    best_train_metrics = None
    best_val_metrics = None

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train=True
            )
            val_metrics = run_one_epoch(
                model,
                val_loader,
                None,
                device,
                train=False
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }

            writer.writerow(row)
            f.flush()

            print(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_dice={train_metrics['dice']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_dice={val_metrics['dice']:.4f} "
                f"val_iou={val_metrics['iou']:.4f} "
                f"val_recall={val_metrics['recall']:.4f}"
            )

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "args": config,
            }

            torch.save(ckpt, ckpt_dir / "last.pth")

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_epoch = epoch
                best_train_metrics = train_metrics
                best_val_metrics = val_metrics

                torch.save(ckpt, ckpt_dir / "best.pth")
                save_json(best_train_metrics, run_dir / "best_train_metrics.json")
                save_json(best_val_metrics, run_dir / "best_val_metrics.json")

                print(f"  -> New best val dice: {best_val_dice:.4f} at epoch {best_epoch}")

    print("=" * 80)
    print("Loading best checkpoint for test...")

    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = run_one_epoch(
        model,
        test_loader,
        None,
        device,
        train=False
    )
    save_json(test_metrics, run_dir / "test_metrics.json")

    per_case_summary = eval_per_case(
        model,
        test_loader,
        device,
        run_dir / "per_case_metrics.csv",
        run_dir / "per_case_metrics_summary.json",
        threshold=0.5,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics_batch_mean": test_metrics,
        "test_metrics_per_case_mean": per_case_summary,
    }
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(
        model,
        test_loader,
        device,
        pred_dir / "test_preview.png"
    )

    print("=" * 80)
    print("Finished.")
    print("Best epoch:", best_epoch)
    print("Best val dice:", best_val_dice)
    print("Batch-mean test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Per-case test metrics, preferred for paper:")
    print(json.dumps(per_case_summary, indent=2))
    print("Outputs saved to:", run_dir)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["plain_unet", "resunet"]
    )

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    args = parser.parse_args()
    main(args)
