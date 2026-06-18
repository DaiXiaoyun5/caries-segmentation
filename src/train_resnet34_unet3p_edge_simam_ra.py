#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ResNet34 UNet3+ + Edge + SimAM + RA
保持 caries_project 现有实验风格：
- CariXray split: data/splits/train.txt / val.txt / test.txt
- Dataset 返回 image, mask, edge, img_path, mask_path
- 主分割 loss: BCE + Dice
- edge loss: 带 pos_weight 的 BCE
- total loss = seg_loss + lambda_edge * edge_loss
- 输出 runs/{run_name}/config.json, history.csv, checkpoints, metrics, test_preview.png
"""

import os
import csv
import json
import random
from pathlib import Path

import cv2
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_edge_from_mask(mask_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    mask_array: H,W, 0/1 或 0/255
    return: H,W, 0/1
    """
    mask_bin = (mask_array > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    edge = (dilated - eroded) > 0
    return edge.astype(np.float32)


class CariesEdgeDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512), edge_kernel_size=3):
        self.split_file = Path(split_file)
        self.target_size = target_size
        self.edge_kernel_size = edge_kernel_size
        self.items = []

        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, mask_path = line.split("\t")
                self.items.append((Path(img_path), Path(mask_path)))

        if len(self.items) == 0:
            raise RuntimeError(f"No samples found in split file: {self.split_file}")

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
        edge_np = make_edge_from_mask(mask_np, kernel_size=self.edge_kernel_size)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        edge_tensor = torch.from_numpy(edge_np).unsqueeze(0).float()

        return image_tensor, mask_tensor, edge_tensor, str(img_path), str(mask_path)


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


class SimAMBlock(nn.Module):
    """
    SimAM: parameter-free attention.
    e_lambda 默认 1e-4。
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


class ReverseAttentionRefine(nn.Module):
    """
    RA: Reverse Attention refine.
    用粗分割结果的反向区域，补充模型没有充分关注的区域。
    """
    def __init__(self, channels, scale=1.0):
        super().__init__()
        self.scale = scale
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, feat, coarse_logits):
        reverse_att = 1.0 - torch.sigmoid(coarse_logits)
        refined_feat = feat * reverse_att
        residual_logits = self.refine(refined_feat)
        return coarse_logits + self.scale * residual_logits


class ResNet34UNet3PlusEdgeSimAMRA(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_channels=64,
        deep_supervision=True,
        simam_lambda=1e-4,
        simam_position="all",
        ra_scale=1.0,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        enc_ch = self.encoder.out_channels
        # enc_ch: [3, 64, 64, 128, 256, 512] for resnet34

        self.num_classes = num_classes
        self.cat_channels = cat_channels
        self.up_ch = cat_channels * 5
        self.deep_supervision = bool(deep_supervision)
        self.simam_position = simam_position

        # hd4, target size = h4
        self.h1_PT_hd4 = nn.Sequential(nn.MaxPool2d(8, 8, ceil_mode=True), Conv1x1BNReLU(enc_ch[1], cat_channels))
        self.h2_PT_hd4 = nn.Sequential(nn.MaxPool2d(4, 4, ceil_mode=True), Conv1x1BNReLU(enc_ch[2], cat_channels))
        self.h3_PT_hd4 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), Conv1x1BNReLU(enc_ch[3], cat_channels))
        self.h4_Cat_hd4 = Conv1x1BNReLU(enc_ch[4], cat_channels)
        self.h5_UT_hd4 = Conv1x1BNReLU(enc_ch[5], cat_channels)
        self.conv_hd4 = ConvBlock(self.up_ch, self.up_ch)

        # hd3, target size = h3
        self.h1_PT_hd3 = nn.Sequential(nn.MaxPool2d(4, 4, ceil_mode=True), Conv1x1BNReLU(enc_ch[1], cat_channels))
        self.h2_PT_hd3 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), Conv1x1BNReLU(enc_ch[2], cat_channels))
        self.h3_Cat_hd3 = Conv1x1BNReLU(enc_ch[3], cat_channels)
        self.hd4_UT_hd3 = Conv1x1BNReLU(self.up_ch, cat_channels)
        self.h5_UT_hd3 = Conv1x1BNReLU(enc_ch[5], cat_channels)
        self.conv_hd3 = ConvBlock(self.up_ch, self.up_ch)

        # hd2, target size = h2
        self.h1_PT_hd2 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), Conv1x1BNReLU(enc_ch[1], cat_channels))
        self.h2_Cat_hd2 = Conv1x1BNReLU(enc_ch[2], cat_channels)
        self.hd3_UT_hd2 = Conv1x1BNReLU(self.up_ch, cat_channels)
        self.hd4_UT_hd2 = Conv1x1BNReLU(self.up_ch, cat_channels)
        self.h5_UT_hd2 = Conv1x1BNReLU(enc_ch[5], cat_channels)
        self.conv_hd2 = ConvBlock(self.up_ch, self.up_ch)

        # hd1, target size = h1
        self.h1_Cat_hd1 = Conv1x1BNReLU(enc_ch[1], cat_channels)
        self.hd2_UT_hd1 = Conv1x1BNReLU(self.up_ch, cat_channels)
        self.hd3_UT_hd1 = Conv1x1BNReLU(self.up_ch, cat_channels)
        self.hd4_UT_hd1 = Conv1x1BNReLU(self.up_ch, cat_channels)
        self.h5_UT_hd1 = Conv1x1BNReLU(enc_ch[5], cat_channels)
        self.conv_hd1 = ConvBlock(self.up_ch, self.up_ch)

        self.simam_hd4 = SimAMBlock(simam_lambda)
        self.simam_hd3 = SimAMBlock(simam_lambda)
        self.simam_hd2 = SimAMBlock(simam_lambda)
        self.simam_hd1 = SimAMBlock(simam_lambda)

        self.edge_head = nn.Conv2d(self.up_ch, num_classes, kernel_size=1)

        self.outconv1 = nn.Conv2d(self.up_ch, num_classes, kernel_size=1)
        self.outconv2 = nn.Conv2d(self.up_ch, num_classes, kernel_size=1)
        self.outconv3 = nn.Conv2d(self.up_ch, num_classes, kernel_size=1)
        self.outconv4 = nn.Conv2d(self.up_ch, num_classes, kernel_size=1)
        self.outconv5 = nn.Conv2d(enc_ch[5], num_classes, kernel_size=1)

        self.ra_refine = ReverseAttentionRefine(self.up_ch, scale=ra_scale)

    def _resize(self, x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

    def _maybe_simam(self, name, x):
        pos = str(self.simam_position).lower()
        if pos == "none":
            return x
        if pos == "all" or pos == name:
            if name == "hd4":
                return self.simam_hd4(x)
            if name == "hd3":
                return self.simam_hd3(x)
            if name == "hd2":
                return self.simam_hd2(x)
            if name == "hd1":
                return self.simam_hd1(x)
        return x

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        h1, h2, h3, h4, h5 = features[1], features[2], features[3], features[4], features[5]

        # hd4
        target4 = h4.shape[2:]
        h1_hd4 = self._resize(self.h1_PT_hd4(h1), target4)
        h2_hd4 = self._resize(self.h2_PT_hd4(h2), target4)
        h3_hd4 = self._resize(self.h3_PT_hd4(h3), target4)
        h4_hd4 = self.h4_Cat_hd4(h4)
        h5_hd4 = self._resize(self.h5_UT_hd4(h5), target4)
        hd4 = self.conv_hd4(torch.cat([h1_hd4, h2_hd4, h3_hd4, h4_hd4, h5_hd4], dim=1))
        hd4 = self._maybe_simam("hd4", hd4)

        # hd3
        target3 = h3.shape[2:]
        h1_hd3 = self._resize(self.h1_PT_hd3(h1), target3)
        h2_hd3 = self._resize(self.h2_PT_hd3(h2), target3)
        h3_hd3 = self.h3_Cat_hd3(h3)
        hd4_hd3 = self._resize(self.hd4_UT_hd3(hd4), target3)
        h5_hd3 = self._resize(self.h5_UT_hd3(h5), target3)
        hd3 = self.conv_hd3(torch.cat([h1_hd3, h2_hd3, h3_hd3, hd4_hd3, h5_hd3], dim=1))
        hd3 = self._maybe_simam("hd3", hd3)

        # hd2
        target2 = h2.shape[2:]
        h1_hd2 = self._resize(self.h1_PT_hd2(h1), target2)
        h2_hd2 = self.h2_Cat_hd2(h2)
        hd3_hd2 = self._resize(self.hd3_UT_hd2(hd3), target2)
        hd4_hd2 = self._resize(self.hd4_UT_hd2(hd4), target2)
        h5_hd2 = self._resize(self.h5_UT_hd2(h5), target2)
        hd2 = self.conv_hd2(torch.cat([h1_hd2, h2_hd2, hd3_hd2, hd4_hd2, h5_hd2], dim=1))
        hd2 = self._maybe_simam("hd2", hd2)

        # hd1
        target1 = h1.shape[2:]
        h1_hd1 = self.h1_Cat_hd1(h1)
        hd2_hd1 = self._resize(self.hd2_UT_hd1(hd2), target1)
        hd3_hd1 = self._resize(self.hd3_UT_hd1(hd3), target1)
        hd4_hd1 = self._resize(self.hd4_UT_hd1(hd4), target1)
        h5_hd1 = self._resize(self.h5_UT_hd1(h5), target1)
        hd1 = self.conv_hd1(torch.cat([h1_hd1, hd2_hd1, hd3_hd1, hd4_hd1, h5_hd1], dim=1))

        # edge branch 保持从 hd1 出边缘
        edge_logits = self.edge_head(hd1)
        edge_logits = self._resize(edge_logits, input_size)

        # SimAM + RA 用于主分割 d1
        hd1_att = self._maybe_simam("hd1", hd1)
        coarse_d1 = self.outconv1(hd1_att)
        refined_d1 = self.ra_refine(hd1_att, coarse_d1)
        d1 = self._resize(refined_d1, input_size)

        if not self.deep_supervision:
            return d1, edge_logits

        d2 = self._resize(self.outconv2(hd2), input_size)
        d3 = self._resize(self.outconv3(hd3), input_size)
        d4 = self._resize(self.outconv4(hd4), input_size)
        d5 = self._resize(self.outconv5(h5), input_size)

        return [d1, d2, d3, d4, d5], edge_logits


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


def get_main_logits(seg_outputs):
    if isinstance(seg_outputs, (list, tuple)):
        return seg_outputs[0]
    return seg_outputs


def compute_seg_loss(seg_outputs, masks, deep_supervision=True):
    if isinstance(seg_outputs, (list, tuple)):
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        losses = []
        used_weights = []
        for out, w in zip(seg_outputs, weights):
            if out.shape[2:] != masks.shape[2:]:
                out = F.interpolate(out, size=masks.shape[2:], mode="bilinear", align_corners=True)
            losses.append(w * bce_dice_loss(out, masks))
            used_weights.append(w)
        return sum(losses) / sum(used_weights)
    else:
        if seg_outputs.shape[2:] != masks.shape[2:]:
            seg_outputs = F.interpolate(seg_outputs, size=masks.shape[2:], mode="bilinear", align_corners=True)
        return bce_dice_loss(seg_outputs, masks)


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


def estimate_edge_pos_weight(dataset, max_samples=800):
    total_pos = 0.0
    total_pixels = 0.0
    n = min(len(dataset), max_samples)

    for i in range(n):
        _, _, edge, _, _ = dataset[i]
        e = edge.numpy()
        total_pos += float(e.sum())
        total_pixels += float(e.size)

    total_neg = total_pixels - total_pos
    if total_pos <= 0:
        pos_weight = 1.0
    else:
        pos_weight = total_neg / total_pos

    pos_weight = max(1.0, min(50.0, pos_weight))
    pos_ratio = total_pos / max(total_pixels, 1.0)

    return {
        "samples_used": n,
        "pos_pixels": total_pos,
        "total_pixels": total_pixels,
        "pos_ratio": pos_ratio,
        "pos_weight": pos_weight,
    }


def run_one_epoch(model, loader, optimizer, device, edge_criterion, lambda_edge=0.3, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "edge_loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    num_batches = 0

    for images, masks, edges, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            seg_outputs, edge_logits = model(images)

            if edge_logits.shape[2:] != edges.shape[2:]:
                edge_logits = F.interpolate(edge_logits, size=edges.shape[2:], mode="bilinear", align_corners=True)

            seg_loss = compute_seg_loss(seg_outputs, masks)
            edge_loss = edge_criterion(edge_logits, edges)
            loss = seg_loss + lambda_edge * edge_loss

            if train:
                loss.backward()
                optimizer.step()

        main_logits = get_main_logits(seg_outputs)
        metrics = compute_metrics_from_logits(main_logits.detach(), masks.detach())

        total["loss"] += loss.item()
        total["seg_loss"] += seg_loss.item()
        total["edge_loss"] += edge_loss.item()
        for k in ["dice", "iou", "precision", "recall"]:
            total[k] += metrics[k]

        num_batches += 1

    for k in total:
        total[k] /= max(num_batches, 1)

    return total


@torch.no_grad()
def save_prediction_panel(model, loader, device, out_path: Path, edge_vis_threshold=0.2, max_items=4):
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader))
    images, masks, edges, img_paths, _ = batch
    images = images.to(device)

    seg_outputs, edge_logits = model(images)
    main_logits = get_main_logits(seg_outputs)

    pred_probs = torch.sigmoid(main_logits).cpu().numpy()
    pred_masks = (pred_probs > 0.5).astype(np.float32)
    edge_probs = torch.sigmoid(edge_logits).cpu().numpy()
    edge_bins = (edge_probs > edge_vis_threshold).astype(np.float32)

    images_np = images.cpu().numpy()
    masks_np = masks.numpy()
    edges_np = edges.numpy()

    n = min(images_np.shape[0], max_items)
    cols = 7
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3.0, n * 3.0))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img = np.transpose(images_np[i], (1, 2, 0))
        gt = masks_np[i, 0]
        gt_edge = edges_np[i, 0]
        pred = pred_masks[i, 0]
        edge_prob = edge_probs[i, 0]
        edge_bin = edge_bins[i, 0]

        overlay = img.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] * 0.6 + pred * 0.4, 0, 1)

        show_items = [
            (img, "Image", None),
            (gt, "GT Mask", "gray"),
            (gt_edge, "GT Edge", "gray"),
            (pred, "Pred Mask", "gray"),
            (edge_prob, "Pred Edge Prob", "gray"),
            (edge_bin, "Pred Edge Bin", "gray"),
            (overlay, "Overlay", None),
        ]

        for j, (arr, title, cmap) in enumerate(show_items):
            ax = axes[i, j]
            ax.imshow(arr, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


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

    print("=" * 80)
    print("Run name:", args.run_name)
    print("Run dir :", run_dir)
    print("Train split:", train_split)
    print("Val split  :", val_split)
    print("Test split :", test_split)
    print("=" * 80)

    train_ds = CariesEdgeDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        edge_kernel_size=args.edge_kernel_size,
    )
    val_ds = CariesEdgeDataset(
        val_split,
        target_size=(args.image_size, args.image_size),
        edge_kernel_size=args.edge_kernel_size,
    )
    test_ds = CariesEdgeDataset(
        test_split,
        target_size=(args.image_size, args.image_size),
        edge_kernel_size=args.edge_kernel_size,
    )

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    edge_stats = estimate_edge_pos_weight(train_ds, max_samples=args.edge_stat_samples)
    config["estimated_edge_stats"] = edge_stats
    save_json(config, run_dir / "config.json")

    print("Estimated edge stats:")
    print(json.dumps(edge_stats, indent=2))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ResNet34UNet3PlusEdgeSimAMRA(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cat_channels=args.cat_channels,
        deep_supervision=bool(args.deep_supervision),
        simam_lambda=args.simam_lambda,
        simam_position=args.simam_position,
        ra_scale=args.ra_scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    edge_pos_weight = torch.tensor([edge_stats["pos_weight"]], dtype=torch.float32, device=device)
    edge_criterion = nn.BCEWithLogitsLoss(pos_weight=edge_pos_weight)

    # forward check
    print("=" * 80)
    print("Forward check...")
    batch = next(iter(train_loader))
    images, masks, edges, _, _ = batch
    images = images.to(device)
    with torch.no_grad():
        seg_outputs, edge_logits = model(images)
    if isinstance(seg_outputs, (list, tuple)):
        print("Deep supervision outputs:", [tuple(o.shape) for o in seg_outputs])
    else:
        print("Seg output:", tuple(seg_outputs.shape))
    print("Edge logits:", tuple(edge_logits.shape))
    print("=" * 80)

    history_path = run_dir / "history.csv"
    fieldnames = [
        "epoch",
        "train_loss", "train_seg_loss", "train_edge_loss",
        "train_dice", "train_iou", "train_precision", "train_recall",
        "val_loss", "val_seg_loss", "val_edge_loss",
        "val_dice", "val_iou", "val_precision", "val_recall",
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
                edge_criterion=edge_criterion,
                lambda_edge=args.lambda_edge,
                train=True,
            )
            val_metrics = run_one_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                edge_criterion=edge_criterion,
                lambda_edge=args.lambda_edge,
                train=False,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_seg_loss": train_metrics["seg_loss"],
                "train_edge_loss": train_metrics["edge_loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_loss": val_metrics["loss"],
                "val_seg_loss": val_metrics["seg_loss"],
                "val_edge_loss": val_metrics["edge_loss"],
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

            last_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "args": config,
            }
            torch.save(last_ckpt, ckpt_dir / "last.pth")

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_epoch = epoch
                best_train_metrics = train_metrics
                best_val_metrics = val_metrics

                torch.save(last_ckpt, ckpt_dir / "best.pth")
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
        optimizer=None,
        device=device,
        edge_criterion=edge_criterion,
        lambda_edge=args.lambda_edge,
        train=False,
    )
    save_json(test_metrics, run_dir / "test_metrics.json")

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(
        model,
        test_loader,
        device,
        pred_dir / "test_preview.png",
        edge_vis_threshold=args.edge_vis_threshold,
        max_items=4,
    )

    print("=" * 80)
    print("Finished.")
    print("Best epoch:", best_epoch)
    print("Best val dice:", best_val_dice)
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Outputs saved to:", run_dir)
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="resnet34_unet3p_edge_simam_ra_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--cat-channels", type=int, default=64)
    parser.add_argument("--deep-supervision", type=int, default=1)

    parser.add_argument("--lambda-edge", type=float, default=0.3)
    parser.add_argument("--edge-kernel-size", type=int, default=3)
    parser.add_argument("--edge-stat-samples", type=int, default=800)
    parser.add_argument("--edge-vis-threshold", type=float, default=0.2)

    parser.add_argument("--simam-lambda", type=float, default=1e-4)
    parser.add_argument(
        "--simam-position",
        type=str,
        default="all",
        choices=["all", "hd1", "hd2", "hd3", "hd4", "none"],
    )
    parser.add_argument("--ra-scale", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
