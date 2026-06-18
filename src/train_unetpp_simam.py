#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UNet++ + SimAM for caries segmentation.

基于 train_unetpp.py 风格：
- segmentation_models_pytorch.UnetPlusPlus
- ResNet34 encoder
- BCE + Dice loss
- Dice / IoU / Precision / Recall
- 保存 config.json / history.csv / best.pth / last.pth / test_metrics.json / summary_metrics.json / test_preview.png

SimAM 插入位置：
encoder -> UNet++ decoder -> SimAM -> segmentation_head -> logits
"""

from pathlib import Path
import json
import csv
import random
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def compute_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (1, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)

    dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size, resample=Image.BILINEAR)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask, str(img_path), str(mask_path)


class SimAMBlock(nn.Module):
    """
    SimAM: parameter-free attention.
    用于增强 decoder 最后一级高分辨率特征响应。
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        if n <= 0:
            return x

        x_minus_mu_square = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * torch.sigmoid(y)


class UnetPlusPlusSimAM(nn.Module):
    """
    SMP UnetPlusPlus + SimAM.

    不是把 SimAM 放在最终 1 通道 mask 上，
    而是放在 UNet++ decoder 输出的高分辨率特征上。
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        simam_lambda=1e-4,
    ):
        super().__init__()

        self.net = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        self.simam = SimAMBlock(e_lambda=simam_lambda)

    def forward(self, x):
        features = self.net.encoder(x)
        decoder_output = self.net.decoder(*features)
        decoder_output = self.simam(decoder_output)
        logits = self.net.segmentation_head(decoder_output)
        return logits


def run_one_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_batches = 0

    for images, masks, _, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = bce_loss_fn(logits, masks) + dice_loss_from_logits(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

        metrics = compute_metrics_from_logits(logits.detach(), masks)

        total_loss += loss.item()
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "iou": total_iou / total_batches,
        "precision": total_precision / total_batches,
        "recall": total_recall / total_batches,
    }


@torch.no_grad()
def save_prediction_panel(model, loader, device, out_path, max_items=4):
    model.eval()

    batch = next(iter(loader))
    images, masks, _, _ = batch

    images = images.to(device)
    logits = model(images)
    preds = (torch.sigmoid(logits) > 0.5).float()

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    b = min(images.shape[0], max_items)

    fig, axes = plt.subplots(b, 4, figsize=(12, 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).numpy()
        gt = masks[i, 0].numpy()
        pd = preds[i, 0].numpy()

        overlay = img.copy()
        overlay[..., 0] = np.where(pd > 0, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(pd > 0, 0.0, overlay[..., 1])
        overlay[..., 2] = np.where(pd > 0, 0.0, overlay[..., 2])
        overlay = 0.6 * img + 0.4 * overlay

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("GT Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pd, cmap="gray")
        axes[i, 2].set_title("Pred Mask")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    set_seed(args.seed)

    project_root = Path("/share/home/u2515283028/caries_project")
    train_split = project_root / "data" / "splits" / "train.txt"
    val_split = project_root / "data" / "splits" / "val.txt"
    test_split = project_root / "data" / "splits" / "test.txt"

    run_dir = project_root / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "preds"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["model_name"] = "UnetPlusPlusSimAM"
    config["simam_position"] = "decoder_output_before_segmentation_head"
    save_json(config, run_dir / "config.json")

    train_ds = CariesDataset(train_split, target_size=(args.image_size, args.image_size))
    val_ds = CariesDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("=" * 80)
    print("Run name:", args.run_name)
    print("Model   : UNet++ + SimAM")
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    model = UnetPlusPlusSimAM(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
        simam_lambda=args.simam_lambda,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # forward check
    sample_batch = next(iter(train_loader))
    sample_images, _, _, _ = sample_batch
    sample_images = sample_images.to(device)
    with torch.no_grad():
        sample_logits = model(sample_images)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", sample_images.shape)
    print("logits.shape       =", sample_logits.shape)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall"
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, train=False)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch(model, train_loader, optimizer=None, device=device, train=False)
    best_val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, train=False)
    test_metrics = run_one_epoch(model, test_loader, optimizer=None, device=device, train=False)

    print("=" * 80)
    print("Best checkpoint summary")
    print("best_epoch:", best_epoch)
    print("best_val_dice_during_training:", best_val_dice)
    print("best_train_metrics:", best_train_metrics)
    print("best_val_metrics  :", best_val_metrics)
    print("test_metrics      :", test_metrics)
    print("=" * 80)

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")

    print("=" * 80)
    print("Finished.")
    print("Outputs saved to:", run_dir)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="unetpp_simam_v1_e30_bs6_fix")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--simam-lambda", type=float, default=1e-4)

    args = parser.parse_args()
    main(args)
