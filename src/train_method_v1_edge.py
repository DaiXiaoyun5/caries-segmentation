from pathlib import Path
import json
import csv
import random
import argparse

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# =========================
# 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_edge_from_mask(mask_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_bin = (mask_array > 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    edge = dilated - eroded
    edge = (edge > 0).astype(np.float32)

    return edge


# =========================
# 数据集
# =========================
class CariesEdgeDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512), edge_kernel_size=3):
        self.items = []
        self.target_size = target_size
        self.edge_kernel_size = edge_kernel_size

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

        edge_np = make_edge_from_mask(mask_np, kernel_size=self.edge_kernel_size)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)   # [3,H,W]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)         # [1,H,W]
        edge_tensor = torch.from_numpy(edge_np).unsqueeze(0)         # [1,H,W]

        return image_tensor, mask_tensor, edge_tensor, str(img_path), str(mask_path)


# =========================
# 模型：U-Net++ + Edge Branch
# =========================
class UnetPPWithEdge(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
        super().__init__()

        base_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

        decoder_out_channels = self.segmentation_head[0].in_channels

        self.edge_head = nn.Conv2d(decoder_out_channels, 1, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        seg_logits = self.segmentation_head(decoder_output)
        edge_logits = self.edge_head(decoder_output)

        if edge_logits.shape[-2:] != seg_logits.shape[-2:]:
            edge_logits = torch.nn.functional.interpolate(
                edge_logits,
                size=seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return seg_logits, edge_logits


# =========================
# 可视化
# =========================
def save_prediction_panel(model, loader, device, out_path, max_items=4):
    model.eval()

    batch = next(iter(loader))
    images, masks, edges, _, _ = batch

    images = images.to(device)

    seg_logits, edge_logits = model(images)
    seg_preds = (torch.sigmoid(seg_logits) > 0.5).float()
    edge_preds = (torch.sigmoid(edge_logits) > 0.5).float()

    images = images.cpu()
    masks = masks.cpu()
    edges = edges.cpu()
    seg_preds = seg_preds.cpu()
    edge_preds = edge_preds.cpu()

    b = min(images.shape[0], max_items)

    fig, axes = plt.subplots(b, 6, figsize=(18, 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).numpy()
        gt = masks[i, 0].numpy()
        edge_gt = edges[i, 0].numpy()
        seg_pd = seg_preds[i, 0].numpy()
        edge_pd = edge_preds[i, 0].numpy()

        overlay = img.copy()
        overlay[..., 0] = np.where(seg_pd > 0, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(seg_pd > 0, 0.0, overlay[..., 1])
        overlay[..., 2] = np.where(seg_pd > 0, 0.0, overlay[..., 2])
        overlay = 0.6 * img + 0.4 * overlay

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("GT Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(edge_gt, cmap="gray")
        axes[i, 2].set_title("GT Edge")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(seg_pd, cmap="gray")
        axes[i, 3].set_title("Pred Mask")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(edge_pd, cmap="gray")
        axes[i, 4].set_title("Pred Edge")
        axes[i, 4].axis("off")

        axes[i, 5].imshow(overlay)
        axes[i, 5].set_title("Overlay")
        axes[i, 5].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# 训练 / 验证 / 测试
# =========================
def run_one_epoch(model, loader, optimizer, device, lambda_edge=0.3, train=True):
    if train:
        model.train()
    else:
        model.eval()

    seg_bce_loss_fn = nn.BCEWithLogitsLoss()
    edge_bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_edge_loss = 0.0

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_batches = 0

    for images, masks, edges, _, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        edges = edges.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            seg_logits, edge_logits = model(images)

            seg_loss = seg_bce_loss_fn(seg_logits, masks) + dice_loss_from_logits(seg_logits, masks)
            edge_loss = edge_bce_loss_fn(edge_logits, edges)

            loss = seg_loss + lambda_edge * edge_loss

            if train:
                loss.backward()
                optimizer.step()

        metrics = compute_metrics_from_logits(seg_logits.detach(), masks)

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_edge_loss += edge_loss.item()

        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_batches += 1

    return {
        "loss": total_loss / total_batches,
        "seg_loss": total_seg_loss / total_batches,
        "edge_loss": total_edge_loss / total_batches,
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

    train_ds = CariesEdgeDataset(train_split, target_size=(args.image_size, args.image_size), edge_kernel_size=3)
    val_ds = CariesEdgeDataset(val_split, target_size=(args.image_size, args.image_size), edge_kernel_size=3)
    test_ds = CariesEdgeDataset(test_split, target_size=(args.image_size, args.image_size), edge_kernel_size=3)

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

    model = UnetPPWithEdge(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 先做一次前向检查
    sample_batch = next(iter(train_loader))
    sample_images, _, _, _, _ = sample_batch
    sample_images = sample_images.to(device)

    with torch.no_grad():
        seg_logits, edge_logits = model(sample_images)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", sample_images.shape)
    print("seg_logits.shape   =", seg_logits.shape)
    print("edge_logits.shape  =", edge_logits.shape)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_seg_loss", "train_edge_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_seg_loss", "val_edge_loss", "val_dice", "val_iou", "val_precision", "val_recall"
        ])

    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, lambda_edge=args.lambda_edge, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, lambda_edge=args.lambda_edge, train=False)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_seg={train_metrics['seg_loss']:.4f} "
            f"train_edge={train_metrics['edge_loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_seg={val_metrics['seg_loss']:.4f} "
            f"val_edge={val_metrics['edge_loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["seg_loss"], train_metrics["edge_loss"], train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["seg_loss"], val_metrics["edge_loss"], val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics = run_one_epoch(model, test_loader, optimizer=None, device=device, lambda_edge=args.lambda_edge, train=False)

    print("=" * 80)
    print("Test metrics:")
    print(test_metrics)
    print("=" * 80)

    save_json(test_metrics, run_dir / "test_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="unetpp_v1_edge")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--lambda-edge", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
