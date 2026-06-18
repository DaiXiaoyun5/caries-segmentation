from pathlib import Path
import json
import csv
import random
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =========================
# 1. 基础工具与修正后的数据加载
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
    return (edge > 0).astype(np.float32) # 【修复】确保是 float32

def estimate_edge_pos_weight(split_file, target_size=(512, 512), edge_kernel_size=3, max_samples=800):
    lines =[]
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: lines.append(line)
    random.seed(42)
    chosen = random.sample(lines, min(max_samples, len(lines)))
    pos_pixels, total_pixels = 0.0, 0.0
    for line in chosen:
        _, mask_path = line.split("\t")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.float32)
        edge_np = make_edge_from_mask(mask_np, kernel_size=edge_kernel_size)
        pos_pixels += float(edge_np.sum())
        total_pixels += float(edge_np.size)
    neg_pixels = total_pixels - pos_pixels
    pos_ratio = pos_pixels / max(total_pixels, 1.0)
    pos_weight = max(1.0, min(float(neg_pixels / max(pos_pixels, 1.0)), 50.0))
    return {"pos_weight": pos_weight}

class CariesEdgeDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512), edge_kernel_size=3, is_train=True):
        self.items =[]
        self.edge_kernel_size = edge_kernel_size
        self.is_train = is_train
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, mask_path = line.split("\t")
                    self.items.append((Path(img_path), Path(mask_path)))

        # 【修复】使用 Albumentations 替代之前的无增强加载
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0)

        edge_np = make_edge_from_mask(mask_tensor[0].numpy(), kernel_size=self.edge_kernel_size)
        edge_tensor = torch.from_numpy(edge_np).unsqueeze(0).float() # 【修复】强制 float32

        return image_tensor, mask_tensor, edge_tensor, str(img_path), str(mask_path)

# =========================
# 2. 核心网络组件与 DCN 模块
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

class DCNv2Block(nn.Module):
    """ DCNv2 (可变形卷积) 模块 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_mask_conv = nn.Conv2d(in_channels, 27, kernel_size=3, padding=1)
        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

        self.dcn = ops.DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.offset_mask_conv(x)
        offset = out[:, :18, :, :]
        mask = torch.sigmoid(out[:, 18:, :, :])

        x = self.dcn(x, offset, mask)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class ResNet34UNet3PlusEdgeDCN(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, num_classes=1, cat_channels=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights)
        enc_channels = list(self.encoder.out_channels)
        ch1, ch2, ch3, ch4, ch5 = enc_channels[1], enc_channels[2], enc_channels[3], enc_channels[4], enc_channels[5]

        self.cat_channels = cat_channels
        self.up_channels = cat_channels * 5

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
        
        # DCN缝合点1
        self.conv_hd2 = DCNv2Block(self.up_channels, self.up_channels)

        self.h1_Cat_hd1 = Conv1x1BNReLU(ch1, cat_channels)
        self.h2_UT_hd1 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h3_UT_hd1 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h4_UT_hd1 = Conv1x1BNReLU(self.up_channels, cat_channels)
        self.h5_UT_hd1 = Conv1x1BNReLU(ch5, cat_channels)
        
        # DCN缝合点2
        self.conv_hd1 = DCNv2Block(self.up_channels, self.up_channels)

        self.outconv1 = nn.Conv2d(self.up_channels, num_classes, kernel_size=1)
        self.edge_head = nn.Conv2d(self.up_channels, 1, kernel_size=1)

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

        d1 = self.up_to_size(self.outconv1(hd1), input_size)
        edge_logits = self.up_to_size(self.edge_head(hd1), input_size)

        if not self.deep_supervision: return d1, edge_logits

        d2 = self.up_to_size(self.outconv2(hd2), input_size)
        d3 = self.up_to_size(self.outconv3(hd3), input_size)
        d4 = self.up_to_size(self.outconv4(hd4), input_size)
        d5 = self.up_to_size(self.outconv5(h5), input_size)
        return [d1, d2, d3, d4, d5], edge_logits

# =========================
# 3. 训练流程
# =========================
def get_main_logits(seg_outputs): return seg_outputs[0] if isinstance(seg_outputs, list) else seg_outputs

def compute_seg_loss(seg_outputs, masks, deep_supervision=True):
    if isinstance(seg_outputs, list) and deep_supervision:
        weights =[1.0, 0.8, 0.6, 0.4, 0.2]
        losses =[bce_dice_loss(out, masks) for out in seg_outputs]
        return sum(w * loss for w, loss in zip(weights, losses)) / sum(weights)
    return bce_dice_loss(seg_outputs[0] if isinstance(seg_outputs, list) else seg_outputs, masks)

def run_one_epoch(model, loader, optimizer, device, deep_supervision=True, lambda_edge=0.3, edge_pos_weight=1.0, train=True):
    if train: model.train()
    else: model.eval()

    # 【修复】强制 dtype 为 torch.float32，完美解决 Found dtype Double 报错
    edge_weight_tensor = torch.tensor([edge_pos_weight], device=device, dtype=torch.float32)
    edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight=edge_weight_tensor)
    
    total_loss = total_seg_loss = total_edge_loss = 0.0
    total_dice = total_recall = total_batches = 0

    for images, masks, edges, _, _ in loader:
        images, masks, edges = images.to(device), masks.to(device), edges.to(device)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            seg_outputs, edge_logits = model(images)
            seg_loss = compute_seg_loss(seg_outputs, masks, deep_supervision)
            edge_loss = edge_loss_fn(edge_logits, edges)
            loss = seg_loss + lambda_edge * edge_loss
            if train:
                loss.backward()
                optimizer.step()

        metrics = compute_metrics_from_logits(get_main_logits(seg_outputs).detach(), masks)
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_edge_loss += edge_loss.item()
        total_dice += metrics["dice"]
        total_recall += metrics["recall"]
        total_batches += 1

    return {"loss": total_loss/total_batches, "seg_loss": total_seg_loss/total_batches, "edge_loss": total_edge_loss/total_batches, "dice": total_dice/total_batches, "recall": total_recall/total_batches}

def main(args):
    set_seed(args.seed)
    project_root = Path("/share/home/u2515283028/caries_project")
    run_dir = project_root / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    for d in [run_dir, ckpt_dir]: d.mkdir(parents=True, exist_ok=True)

    edge_stats = estimate_edge_pos_weight(project_root / "data/splits/train.txt", target_size=(args.image_size, args.image_size))
    
    ds_kwargs = {"target_size": (args.image_size, args.image_size), "edge_kernel_size": args.edge_kernel_size}
    train_loader = DataLoader(CariesEdgeDataset(project_root / "data/splits/train.txt", is_train=True, **ds_kwargs), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(CariesEdgeDataset(project_root / "data/splits/val.txt", is_train=False, **ds_kwargs), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNet34UNet3PlusEdgeDCN(
        encoder_name=args.encoder_name, encoder_weights=args.encoder_weights, in_channels=3, 
        cat_channels=args.cat_channels, deep_supervision=bool(args.deep_supervision)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        tm = run_one_epoch(model, train_loader, optimizer, device, bool(args.deep_supervision), args.lambda_edge, edge_stats["pos_weight"], True)
        vm = run_one_epoch(model, val_loader, None, device, bool(args.deep_supervision), args.lambda_edge, edge_stats["pos_weight"], False)
        print(f"[Epoch {epoch:03d}/{args.epochs}] Train Loss={tm['loss']:.4f} | Val Dice={vm['dice']:.4f} | Val Recall={vm['recall']:.4f}")

        if vm["dice"] > best_val_dice:
            best_val_dice = vm["dice"]
            torch.save(model.state_dict(), ckpt_dir / "best.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="unet3p_edge_dcn_fixed")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--cat-channels", type=int, default=64)
    parser.add_argument("--deep-supervision", type=int, default=1)
    parser.add_argument("--lambda-edge", type=float, default=0.3)
    parser.add_argument("--edge-kernel-size", type=int, default=3)
    main(parser.parse_args())
