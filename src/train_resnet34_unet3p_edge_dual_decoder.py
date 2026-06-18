from pathlib import Path
import json
import csv
import random
import argparse

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

# =========================
# 1. 基础工具与数据加载 (新增高斯热力图生成)
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
    return 1.0 - dice.mean()

def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice

def compute_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (1, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1.0 - targets)).sum(dims)
    fn = ((1.0 - preds) * targets).sum(dims)

    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}

def make_edge_from_mask(mask_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_bin = (mask_array > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    edge = dilated - eroded
    return (edge > 0).astype(np.float32)

def make_center_heatmap_from_mask(mask_array: np.ndarray, sigma=15) -> np.ndarray:
    """自动基于病灶 Mask 生成多中心高斯热力图"""
    heatmap = np.zeros_like(mask_array, dtype=np.float32)
    mask_bin = (mask_array > 0).astype(np.uint8)
    # 寻找所有的独立病灶连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    for i in range(1, num_labels): # 0是背景，跳过
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        # 在病灶中心生成高斯分布
        y, x = np.ogrid[-cy:heatmap.shape[0]-cy, -cx:heatmap.shape[1]-cx]
        g = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        heatmap = np.maximum(heatmap, g)
    return heatmap

def estimate_edge_pos_weight(split_file, target_size=(512, 512), edge_kernel_size=3, max_samples=800):
    lines = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: lines.append(line)
    random.seed(42)
    chosen = random.sample(lines, min(max_samples, len(lines)))
    pos_pixels, total_pixels = 0.0, 0.0
    for line in chosen:
        _, mask_path = line.split("\t")
        mask = Image.open(mask_path).convert("L").resize(target_size, resample=Image.NEAREST)
        mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.float32)
        edge_np = make_edge_from_mask(mask_np, kernel_size=edge_kernel_size)
        pos_pixels += float(edge_np.sum())
        total_pixels += float(edge_np.size)
    neg_pixels = total_pixels - pos_pixels
    pos_ratio = pos_pixels / max(total_pixels, 1.0)
    pos_weight = max(1.0, min(float(neg_pixels / max(pos_pixels, 1.0)), 50.0))
    return {"samples_used": len(chosen), "pos_pixels": pos_pixels, "total_pixels": total_pixels, "pos_ratio": pos_ratio, "pos_weight": pos_weight}

class CariesDualDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512), edge_kernel_size=3, heatmap_sigma=15):
        self.items = []
        self.target_size = target_size
        self.edge_kernel_size = edge_kernel_size
        self.heatmap_sigma = heatmap_sigma
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, mask_path = line.split("\t")
                    self.items.append((Path(img_path), Path(mask_path)))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        image = Image.open(img_path).convert("RGB").resize(self.target_size, resample=Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.target_size, resample=Image.NEAREST)
        
        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.float32)
        
        edge_np = make_edge_from_mask(mask_np, kernel_size=self.edge_kernel_size)
        center_np = make_center_heatmap_from_mask(mask_np, sigma=self.heatmap_sigma)
        
        return (torch.from_numpy(image_np).permute(2, 0, 1), 
                torch.from_numpy(mask_np).unsqueeze(0), 
                torch.from_numpy(edge_np).unsqueeze(0),
                torch.from_numpy(center_np).unsqueeze(0), 
                str(img_path), str(mask_path))

# =========================
# 2. 网络组件与 Dual Decoder 架构
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class ResNet34UNet3PlusDualDecoder(nn.Module):
    """ Edge + Center Dual Decoder 缝合架构 """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, num_classes=1, cat_channels=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights)
        enc_channels = list(self.encoder.out_channels)
        ch1, ch2, ch3, ch4, ch5 = enc_channels[1], enc_channels[2], enc_channels[3], enc_channels[4], enc_channels[5]

        self.cat_channels = cat_channels
        self.up_channels = cat_channels * 5

        # 降采样分支
        self.h1_PT_hd4 = nn.Sequential(nn.MaxPool2d(8, 8), Conv1x1BNReLU(ch1, cat_channels))
        self.h2_PT_hd4 = nn.Sequential(nn.MaxPool2d(4, 4), Conv1x1BNReLU(ch2, cat_channels))
        self.h3_PT_hd4 = nn.Sequential(nn.MaxPool2d(2, 2), Conv1x1BNReLU(ch3, cat_channels))
        self.h4_Cat_hd4 = Conv1x1BNReLU(ch4, cat_channels)
        self.h5_UT_hd4 = Conv1x1BNReLU(ch5, cat_channels)
        self.conv_hd4 = ConvBlock(self.up_channels, self.up_channels)

        self.h1_PT_hd3 = nn.Sequential(nn.MaxPool2d(4, 4), Conv1x1BNReLU(ch1, cat_channels))
        self.h2_PT_hd3 = nn.Sequential(nn.MaxPool2d(2, 2), Conv1x1BNReLU(ch2, cat_channels))
        self.h3_Cat_hd3 = Conv1x1BNReLU(ch3, cat_channels)
        self.h4_UT_hd3 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h5_UT_hd3 = Conv1x1BNReLU(ch5, cat_channels)
        self.conv_hd3 = ConvBlock(self.up_channels, self.up_channels)

        self.h1_PT_hd2 = nn.Sequential(nn.MaxPool2d(2, 2), Conv1x1BNReLU(ch1, cat_channels))
        self.h2_Cat_hd2 = Conv1x1BNReLU(ch2, cat_channels)
        self.h3_UT_hd2 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h4_UT_hd2 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h5_UT_hd2 = Conv1x1BNReLU(ch5, cat_channels)
        self.conv_hd2 = ConvBlock(self.up_channels, self.up_channels)

        self.h1_Cat_hd1 = Conv1x1BNReLU(ch1, cat_channels)
        self.h2_UT_hd1 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h3_UT_hd1 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h4_UT_hd1 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h5_UT_hd1 = Conv1x1BNReLU(ch5, cat_channels)
        self.conv_hd1 = ConvBlock(self.up_channels, self.up_channels)

        # 【核心缝合点】：在最终高清特征 hd1 上，建立三个独立的输出分支
        self.outconv1 = nn.Conv2d(self.up_channels, num_classes, kernel_size=1)  # 分割分支
        self.edge_head = nn.Conv2d(self.up_channels, 1, kernel_size=1)           # 边缘分支
        
        # Dual Decoder 新增：中心点检测分支（预测高斯热力图）
        self.center_head = nn.Sequential(
            nn.Conv2d(self.up_channels, self.up_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.up_channels // 2, 1, kernel_size=1)
        )

        if self.deep_supervision:
            self.outconv2 = nn.Conv2d(self.up_channels, num_classes, kernel_size=1)
            self.outconv3 = nn.Conv2d(self.up_channels, num_classes, kernel_size=1)
            self.outconv4 = nn.Conv2d(self.up_channels, num_classes, kernel_size=1)
            self.outconv5 = nn.Conv2d(ch5, num_classes, kernel_size=1)

    def up_to(self, x, ref): return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    def up_to_size(self, x, size_hw): return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x)
        h1, h2, h3, h4, h5 = features[1], features[2], features[3], features[4], features[5]

        hd4 = self.conv_hd4(torch.cat([self.h1_PT_hd4(h1), self.h2_PT_hd4(h2), self.h3_PT_hd4(h3), self.h4_Cat_hd4(h4), self.up_to(self.h5_UT_hd4(h5), h4)], dim=1))
        hd3 = self.conv_hd3(torch.cat([self.h1_PT_hd3(h1), self.h2_PT_hd3(h2), self.h3_Cat_hd3(h3), self.up_to(self.h4_UT_hd3(hd4), h3), self.up_to(self.h5_UT_hd3(h5), h3)], dim=1))
        hd2 = self.conv_hd2(torch.cat([self.h1_PT_hd2(h1), self.h2_Cat_hd2(h2), self.up_to(self.h3_UT_hd2(hd3), h2), self.up_to(self.h4_UT_hd2(hd4), h2), self.up_to(self.h5_UT_hd2(h5), h2)], dim=1))
        hd1 = self.conv_hd1(torch.cat([self.h1_Cat_hd1(h1), self.up_to(self.h2_UT_hd1(hd2), h1), self.up_to(self.h3_UT_hd1(hd3), h1), self.up_to(self.h4_UT_hd1(hd4), h1), self.up_to(self.h5_UT_hd1(h5), h1)], dim=1))

        # 三分支输出
        d1 = self.up_to_size(self.outconv1(hd1), input_size)
        edge_logits = self.up_to_size(self.edge_head(hd1), input_size)
        center_logits = self.up_to_size(self.center_head(hd1), input_size)

        if not self.deep_supervision: return d1, edge_logits, center_logits

        d2 = self.up_to_size(self.outconv2(hd2), input_size)
        d3 = self.up_to_size(self.outconv3(hd3), input_size)
        d4 = self.up_to_size(self.outconv4(hd4), input_size)
        d5 = self.up_to_size(self.outconv5(h5), input_size)
        return [d1, d2, d3, d4, d5], edge_logits, center_logits

# =========================
# 3. 训练流程与可视化
# =========================
def get_main_logits(seg_outputs): return seg_outputs[0] if isinstance(seg_outputs, list) else seg_outputs

def compute_seg_loss(seg_outputs, masks, deep_supervision=True):
    if isinstance(seg_outputs, list):
        if deep_supervision:
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            losses = [bce_dice_loss(out, masks) for out in seg_outputs]
            return sum(w * loss for w, loss in zip(weights, losses)) / sum(weights)
        return bce_dice_loss(seg_outputs[0], masks)
    return bce_dice_loss(seg_outputs, masks)

def run_one_epoch(model, loader, optimizer, device, deep_supervision=True, lambda_edge=0.3, lambda_center=0.5, edge_pos_weight=1.0, train=True):
    if train: model.train()
    else: model.eval()

    edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([edge_pos_weight], device=device))
    center_loss_fn = nn.MSELoss() # 热力图采用均方误差
    
    total_loss = total_seg_loss = total_edge_loss = total_center_loss = 0.0
    total_dice = total_iou = total_precision = total_recall = total_batches = 0

    for images, masks, edges, centers, _, _ in loader:
        images, masks, edges, centers = images.to(device), masks.to(device), edges.to(device), centers.to(device)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            seg_outputs, edge_logits, center_logits = model(images)
            
            # 损失计算
            seg_loss = compute_seg_loss(seg_outputs, masks, deep_supervision)
            edge_loss = edge_loss_fn(edge_logits, edges)
            center_loss = center_loss_fn(torch.sigmoid(center_logits), centers)
            
            # 总损失包含：主干 + 边缘惩罚 + 热力图惩罚
            loss = seg_loss + (lambda_edge * edge_loss) + (lambda_center * center_loss)
            
            if train:
                loss.backward()
                optimizer.step()

        metrics = compute_metrics_from_logits(get_main_logits(seg_outputs).detach(), masks)
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_edge_loss += edge_loss.item()
        total_center_loss += center_loss.item()
        
        total_dice += metrics["dice"]
        total_iou += metrics["iou"]
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_batches += 1

    return {"loss": total_loss/total_batches, "seg_loss": total_seg_loss/total_batches, 
            "edge_loss": total_edge_loss/total_batches, "center_loss": total_center_loss/total_batches,
            "dice": total_dice/total_batches, "iou": total_iou/total_batches, 
            "precision": total_precision/total_batches, "recall": total_recall/total_batches}

def save_prediction_panel(model, loader, device, out_path, edge_vis_threshold=0.2, max_items=4):
    model.eval()
    batch = next(iter(loader))
    images, masks, edges, centers, _, _ = batch
    images = images.to(device)
    with torch.no_grad():
        seg_outputs, edge_logits, center_logits = model(images)
        main_logits = get_main_logits(seg_outputs)
        seg_pd = (torch.sigmoid(main_logits) > 0.5).float().cpu()
        
        edge_pd = (torch.sigmoid(edge_logits).cpu() > edge_vis_threshold).float()
        center_pd = torch.sigmoid(center_logits).cpu()

    images, masks, centers = images.cpu(), masks.cpu(), centers.cpu()
    b = min(images.shape[0], max_items)
    # 画布增加一列展示中心热力图
    fig, axes = plt.subplots(b, 8, figsize=(24, 3 * b))
    if b == 1: axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).detach().numpy()
        overlay = img.copy()
        overlay[..., 0] = np.where(seg_pd[i, 0].numpy() > 0, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(seg_pd[i, 0].numpy() > 0, 0.0, overlay[..., 1])
        overlay[..., 2] = np.where(seg_pd[i, 0].numpy() > 0, 0.0, overlay[..., 2])
        overlay = 0.6 * img + 0.4 * overlay
        
        axes[i, 0].imshow(img); axes[i, 0].set_title("Input"); axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray"); axes[i, 1].set_title("GT Mask"); axes[i, 1].axis("off")
        axes[i, 2].imshow(centers[i, 0].numpy(), cmap="hot"); axes[i, 2].set_title("GT Center"); axes[i, 2].axis("off")
        axes[i, 3].imshow(center_pd[i, 0].numpy(), cmap="hot"); axes[i, 3].set_title("Pred Center"); axes[i, 3].axis("off")
        axes[i, 4].imshow(edge_pd[i, 0].numpy(), cmap="gray"); axes[i, 4].set_title("Pred Edge"); axes[i, 4].axis("off")
        axes[i, 5].imshow(seg_pd[i, 0].numpy(), cmap="gray"); axes[i, 5].set_title("Pred Mask"); axes[i, 5].axis("off")
        axes[i, 6].imshow(overlay); axes[i, 6].set_title("Overlay"); axes[i, 6].axis("off")
        # 错误图
        err = np.abs(masks[i, 0].numpy() - seg_pd[i, 0].numpy())
        axes[i, 7].imshow(err, cmap="Reds"); axes[i, 7].set_title("Error"); axes[i, 7].axis("off")
        
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def main(args):
    set_seed(args.seed)
    project_root = Path("/share/home/u2515283028/caries_project")
    run_dir = project_root / "runs" / args.run_name
    ckpt_dir, pred_dir = run_dir / "checkpoints", run_dir / "preds"
    for d in [run_dir, ckpt_dir, pred_dir]: d.mkdir(parents=True, exist_ok=True)

    edge_stats = estimate_edge_pos_weight(project_root / "data/splits/train.txt", target_size=(args.image_size, args.image_size), edge_kernel_size=args.edge_kernel_size, max_samples=args.edge_stat_samples)
    save_json({**vars(args), "estimated_edge_stats": edge_stats}, run_dir / "config.json")

    ds_kwargs = {"target_size": (args.image_size, args.image_size), "edge_kernel_size": args.edge_kernel_size, "heatmap_sigma": args.heatmap_sigma}
    train_loader = DataLoader(CariesDualDataset(project_root / "data/splits/train.txt", **ds_kwargs), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(CariesDualDataset(project_root / "data/splits/val.txt", **ds_kwargs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(CariesDualDataset(project_root / "data/splits/test.txt", **ds_kwargs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNet3PlusDualDecoder(
        encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, 
        num_classes=1, cat_channels=args.cat_channels, deep_supervision=bool(args.deep_supervision)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_seg_loss", "train_edge_loss", "train_center_loss", "train_dice", "train_iou", "train_precision", "train_recall", "val_loss", "val_seg_loss", "val_edge_loss", "val_center_loss", "val_dice", "val_iou", "val_precision", "val_recall"])

    best_val_dice, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        tm = run_one_epoch(model, train_loader, optimizer, device, bool(args.deep_supervision), args.lambda_edge, args.lambda_center, edge_stats["pos_weight"], True)
        vm = run_one_epoch(model, val_loader, None, device, bool(args.deep_supervision), args.lambda_edge, args.lambda_center, edge_stats["pos_weight"], False)
        print(f"[Epoch {epoch:03d}/{args.epochs}] train_dice={tm['dice']:.4f} val_dice={vm['dice']:.4f} val_recall={vm['recall']:.4f}")

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, tm["loss"], tm["seg_loss"], tm["edge_loss"], tm["center_loss"], tm["dice"], tm["iou"], tm["precision"], tm["recall"], vm["loss"], vm["seg_loss"], vm["edge_loss"], vm["center_loss"], vm["dice"], vm["iou"], vm["precision"], vm["recall"]])
        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if vm["dice"] > best_val_dice:
            best_val_dice, best_epoch = vm["dice"], epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))
    best_tm = run_one_epoch(model, train_loader, None, device, bool(args.deep_supervision), args.lambda_edge, args.lambda_center, edge_stats["pos_weight"], False)
    best_vm = run_one_epoch(model, val_loader, None, device, bool(args.deep_supervision), args.lambda_edge, args.lambda_center, edge_stats["pos_weight"], False)
    test_m = run_one_epoch(model, test_loader, None, device, bool(args.deep_supervision), args.lambda_edge, args.lambda_center, edge_stats["pos_weight"], False)

    save_json(best_tm, run_dir / "best_train_metrics.json")
    save_json(best_vm, run_dir / "best_val_metrics.json")
    save_json(test_m, run_dir / "test_metrics.json")
    save_json({"best_epoch": best_epoch, "best_val_dice": best_val_dice, "best_train_metrics": best_tm, "best_val_metrics": best_vm, "test_metrics": test_m}, run_dir / "summary_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png", args.edge_vis_threshold)
    print(f"=== Done! Best Epoch: {best_epoch} | Test Dice: {test_m['dice']:.4f} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="resnet34_unet3p_edge_dual_decoder_e50_bs6")
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
    
    # 边缘检测参数
    parser.add_argument("--lambda-edge", type=float, default=0.3)
    parser.add_argument("--edge-kernel-size", type=int, default=3)
    parser.add_argument("--edge-stat-samples", type=int, default=800)
    parser.add_argument("--edge-vis-threshold", type=float, default=0.2)
    
    # Dual Decoder 中心检测专有参数
    parser.add_argument("--lambda-center", type=float, default=0.5)
    parser.add_argument("--heatmap-sigma", type=int, default=15)
    
    main(parser.parse_args())
