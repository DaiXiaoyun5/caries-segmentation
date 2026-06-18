from pathlib import Path
import json
import csv
import random
import argparse

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# 基础工具
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


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


# =========================
# Dataset
# =========================
class CariesMaskDataset(Dataset):
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

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32)
        mask_np = (mask_np > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return image_tensor, mask_tensor, str(img_path), str(mask_path)


# =========================
# UNet3+ 模块
# =========================
class ConvBlock(nn.Module):
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


class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3Plus(nn.Module):
    """
    简化但标准思路的 UNet3+：
    - encoder 五层
    - decoder 使用 full-scale skip fusion
    - 可选 deep supervision
    """
    def __init__(self, in_channels=3, num_classes=1, base_ch=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        filters = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        cat_ch = base_ch
        up_ch = cat_ch * 5

        # Encoder
        self.conv1 = ConvBlock(in_channels, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # hd4: target size = h4
        self.h1_PT_hd4 = nn.Sequential(nn.MaxPool2d(8, 8), Conv1x1BNReLU(filters[0], cat_ch))
        self.h2_PT_hd4 = nn.Sequential(nn.MaxPool2d(4, 4), Conv1x1BNReLU(filters[1], cat_ch))
        self.h3_PT_hd4 = nn.Sequential(nn.MaxPool2d(2, 2), Conv1x1BNReLU(filters[2], cat_ch))
        self.h4_Cat_hd4 = Conv1x1BNReLU(filters[3], cat_ch)
        self.h5_UT_hd4 = Conv1x1BNReLU(filters[4], cat_ch)
        self.conv_hd4 = ConvBlock(up_ch, up_ch)

        # hd3: target size = h3
        self.h1_PT_hd3 = nn.Sequential(nn.MaxPool2d(4, 4), Conv1x1BNReLU(filters[0], cat_ch))
        self.h2_PT_hd3 = nn.Sequential(nn.MaxPool2d(2, 2), Conv1x1BNReLU(filters[1], cat_ch))
        self.h3_Cat_hd3 = Conv1x1BNReLU(filters[2], cat_ch)
        self.h4_UT_hd3 = Conv1x1BNReLU(up_ch, cat_ch)
        self.h5_UT_hd3 = Conv1x1BNReLU(filters[4], cat_ch)
        self.conv_hd3 = ConvBlock(up_ch, up_ch)

        # hd2: target size = h2
        self.h1_PT_hd2 = nn.Sequential(nn.MaxPool2d(2, 2), Conv1x1BNReLU(filters[0], cat_ch))
        self.h2_Cat_hd2 = Conv1x1BNReLU(filters[1], cat_ch)
        self.h3_UT_hd2 = Conv1x1BNReLU(up_ch, cat_ch)
        self.h4_UT_hd2 = Conv1x1BNReLU(up_ch, cat_ch)
        self.h5_UT_hd2 = Conv1x1BNReLU(filters[4], cat_ch)
        self.conv_hd2 = ConvBlock(up_ch, up_ch)

        # hd1: target size = h1
        self.h1_Cat_hd1 = Conv1x1BNReLU(filters[0], cat_ch)
        self.h2_UT_hd1 = Conv1x1BNReLU(up_ch, cat_ch)
        self.h3_UT_hd1 = Conv1x1BNReLU(up_ch, cat_ch)
        self.h4_UT_hd1 = Conv1x1BNReLU(up_ch, cat_ch)
        self.h5_UT_hd1 = Conv1x1BNReLU(filters[4], cat_ch)
        self.conv_hd1 = ConvBlock(up_ch, up_ch)

        # 输出头
        self.outconv1 = nn.Conv2d(up_ch, num_classes, kernel_size=1)

        if self.deep_supervision:
            self.outconv2 = nn.Conv2d(up_ch, num_classes, kernel_size=1)
            self.outconv3 = nn.Conv2d(up_ch, num_classes, kernel_size=1)
            self.outconv4 = nn.Conv2d(up_ch, num_classes, kernel_size=1)
            self.outconv5 = nn.Conv2d(filters[4], num_classes, kernel_size=1)

    def up_to(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        h1 = self.conv1(x)                  # H
        h2 = self.conv2(self.maxpool(h1))   # H/2
        h3 = self.conv3(self.maxpool(h2))   # H/4
        h4 = self.conv4(self.maxpool(h3))   # H/8
        h5 = self.conv5(self.maxpool(h4))   # H/16

        # Decoder hd4
        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        h5_UT_hd4 = self.up_to(self.h5_UT_hd4(h5), h4)
        hd4 = self.conv_hd4(torch.cat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, h5_UT_hd4], dim=1))

        # Decoder hd3
        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        h4_UT_hd3 = self.up_to(self.h4_UT_hd3(hd4), h3)
        h5_UT_hd3 = self.up_to(self.h5_UT_hd3(h5), h3)
        hd3 = self.conv_hd3(torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, h4_UT_hd3, h5_UT_hd3], dim=1))

        # Decoder hd2
        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2(h2)
        h3_UT_hd2 = self.up_to(self.h3_UT_hd2(hd3), h2)
        h4_UT_hd2 = self.up_to(self.h4_UT_hd2(hd4), h2)
        h5_UT_hd2 = self.up_to(self.h5_UT_hd2(h5), h2)
        hd2 = self.conv_hd2(torch.cat([h1_PT_hd2, h2_Cat_hd2, h3_UT_hd2, h4_UT_hd2, h5_UT_hd2], dim=1))

        # Decoder hd1
        h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        h2_UT_hd1 = self.up_to(self.h2_UT_hd1(hd2), h1)
        h3_UT_hd1 = self.up_to(self.h3_UT_hd1(hd3), h1)
        h4_UT_hd1 = self.up_to(self.h4_UT_hd1(hd4), h1)
        h5_UT_hd1 = self.up_to(self.h5_UT_hd1(h5), h1)
        hd1 = self.conv_hd1(torch.cat([h1_Cat_hd1, h2_UT_hd1, h3_UT_hd1, h4_UT_hd1, h5_UT_hd1], dim=1))

        d1 = self.outconv1(hd1)

        if not self.deep_supervision:
            return d1

        d2 = self.up_to(self.outconv2(hd2), d1)
        d3 = self.up_to(self.outconv3(hd3), d1)
        d4 = self.up_to(self.outconv4(hd4), d1)
        d5 = self.up_to(self.outconv5(h5), d1)

        return [d1, d2, d3, d4, d5]


# =========================
# 可视化
# =========================
def save_prediction_panel(model, loader, device, out_path, max_items=4):
    model.eval()
    batch = next(iter(loader))
    images, masks, _, _ = batch
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, list) else outputs
        preds = (torch.sigmoid(logits) > 0.5).float()

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    b = min(images.shape[0], max_items)

    fig, axes = plt.subplots(b, 4, figsize=(12, 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).detach().numpy()
        gt = masks[i, 0].detach().numpy()
        pd = preds[i, 0].detach().numpy()

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
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# train / val / test
# =========================
def get_main_logits(outputs):
    return outputs[0] if isinstance(outputs, list) else outputs


def compute_loss(outputs, masks, deep_supervision=True):
    if isinstance(outputs, list):
        if deep_supervision:
            losses = [bce_dice_loss(out, masks) for out in outputs]
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            loss = sum(w * l for w, l in zip(weights, losses)) / sum(weights)
            return loss
        return bce_dice_loss(outputs[0], masks)

    return bce_dice_loss(outputs, masks)


def run_one_epoch(model, loader, optimizer, device, deep_supervision=True, train=True):
    if train:
        model.train()
    else:
        model.eval()

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
            outputs = model(images)
            loss = compute_loss(outputs, masks, deep_supervision=deep_supervision)

            if train:
                loss.backward()
                optimizer.step()

        logits = get_main_logits(outputs)
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
    save_json(config, run_dir / "config.json")

    train_ds = CariesMaskDataset(train_split, target_size=(args.image_size, args.image_size))
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

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
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    deep_supervision = bool(args.deep_supervision)

    model = UNet3Plus(
        in_channels=3,
        num_classes=1,
        base_ch=args.base_channels,
        deep_supervision=deep_supervision,
    ).to(device)

    # Multi-GPU patch: use DataParallel when more than one GPU is visible.
    # With batch_size=6 and 2 GPUs, each GPU processes about 3 images.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("=" * 80)
        print(f"Using {torch.cuda.device_count()} GPUs with torch.nn.DataParallel")
        print("Total batch size will be split across visible GPUs.")
        print("=" * 80)
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # forward check
    sample_batch = next(iter(train_loader))
    sample_images, sample_masks, _, _ = sample_batch
    sample_images = sample_images.to(device)

    with torch.no_grad():
        outputs = model(sample_images)
        main_logits = get_main_logits(outputs)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", sample_images.shape)
    print("main_logits.shape  =", main_logits.shape)
    if isinstance(outputs, list):
        print("deep outputs       =", [tuple(o.shape) for o in outputs])
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

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            deep_supervision=deep_supervision,
            train=True
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            deep_supervision=deep_supervision,
            train=False
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        train=False
    )

    print("=" * 80)
    print("Test metrics:")
    print(test_metrics)
    print("=" * 80)

    save_json(test_metrics, run_dir / "test_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="smoke_unet3p")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--deep-supervision", type=int, default=1)

    args = parser.parse_args()
    main(args)
