#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ResNet34 UNet3+ + Edge + SimAM + SCSE-hd1

说明：
- 以 train_resnet34_unet3p_edge.py 为基础风格
- 复用原有 Dataset / loss / metrics / run_one_epoch / save_prediction_panel
- 新增 SimAMBlock
- 新增 SCSEBlock
- SimAM 可选作用于 hd1/hd2/hd3/hd4/all/none
- SCSE 固定只作用于 hd1
- forward 输出保持：seg_outputs, edge_logits
- loss 保持：seg_loss + lambda_edge * edge_loss
"""

from pathlib import Path
import json
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_resnet34_unet3p_edge import (
    set_seed,
    save_json,
    estimate_edge_pos_weight,
    CariesEdgeDataset,
    ResNet34UNet3PlusEdge,
    get_main_logits,
    run_one_epoch,
    save_prediction_panel,
)


class SimAMBlock(nn.Module):
    """
    SimAM: Parameter-free attention.
    作用：增强显著区域响应，不明显增加参数量。
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


class SCSEBlock(nn.Module):
    """
    SCSE: Spatial and Channel Squeeze & Excitation.
    作用：
    - cSE：通道注意力，强调有用通道
    - sSE：空间注意力，强调有用空间位置
    本实验只加在 hd1，也就是最高分辨率 decoder feature。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(channels // reduction, 1)

        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        c_att = self.cse(x)
        s_att = self.sse(x)
        return x * c_att + x * s_att


class ResNet34UNet3PlusEdgeSimAMSCSEHD1(ResNet34UNet3PlusEdge):
    """
    在 ResNet34UNet3PlusEdge 基础上：
    - hd4/hd3/hd2/hd1 可插入 SimAM
    - hd1 额外插入 SCSE
    - edge_head 仍然从 hd1 原始高分辨率特征输出
    - 主分割 d1 使用 SimAM + SCSE 后的 hd1 输出
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        simam_lambda=1e-4,
        simam_position="all",
        scse_reduction=16,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            cat_ch=cat_ch,
            deep_supervision=deep_supervision,
        )

        self.simam_position = str(simam_position).lower()

        self.simam_hd4 = SimAMBlock(e_lambda=simam_lambda)
        self.simam_hd3 = SimAMBlock(e_lambda=simam_lambda)
        self.simam_hd2 = SimAMBlock(e_lambda=simam_lambda)
        self.simam_hd1 = SimAMBlock(e_lambda=simam_lambda)

        self.scse_hd1 = SCSEBlock(self.up_ch, reduction=scse_reduction)

    def maybe_simam(self, name, x):
        if self.simam_position == "none":
            return x

        if self.simam_position == "all" or self.simam_position == name:
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
        input_size = x.shape[-2:]

        features = self.encoder(x)
        h1, h2, h3, h4, h5 = features[1], features[2], features[3], features[4], features[5]

        # =========================
        # hd4
        # =========================
        hd4 = self.conv_hd4(torch.cat([
            self.h1_PT_hd4(h1),
            self.h2_PT_hd4(h2),
            self.h3_PT_hd4(h3),
            self.h4_Cat_hd4(h4),
            self.up_to(self.h5_UT_hd4(h5), h4),
        ], dim=1))
        hd4 = self.maybe_simam("hd4", hd4)

        # =========================
        # hd3
        # =========================
        hd3 = self.conv_hd3(torch.cat([
            self.h1_PT_hd3(h1),
            self.h2_PT_hd3(h2),
            self.h3_Cat_hd3(h3),
            self.up_to(self.h4_UT_hd3(hd4), h3),
            self.up_to(self.h5_UT_hd3(h5), h3),
        ], dim=1))
        hd3 = self.maybe_simam("hd3", hd3)

        # =========================
        # hd2
        # =========================
        hd2 = self.conv_hd2(torch.cat([
            self.h1_PT_hd2(h1),
            self.h2_Cat_hd2(h2),
            self.up_to(self.h3_UT_hd2(hd3), h2),
            self.up_to(self.h4_UT_hd2(hd4), h2),
            self.up_to(self.h5_UT_hd2(h5), h2),
        ], dim=1))
        hd2 = self.maybe_simam("hd2", hd2)

        # =========================
        # hd1
        # =========================
        hd1 = self.conv_hd1(torch.cat([
            self.h1_Cat_hd1(h1),
            self.up_to(self.h2_UT_hd1(hd2), h1),
            self.up_to(self.h3_UT_hd1(hd3), h1),
            self.up_to(self.h4_UT_hd1(hd4), h1),
            self.up_to(self.h5_UT_hd1(h5), h1),
        ], dim=1))

        # edge branch 保持从 hd1 原始高分辨率特征输出
        edge_logits = self.up_to_size(self.edge_head(hd1), input_size)

        # 主分割分支：hd1 -> SimAM -> SCSE-hd1 -> outconv1
        hd1_att = self.maybe_simam("hd1", hd1)
        hd1_att = self.scse_hd1(hd1_att)
        d1 = self.up_to_size(self.outconv1(hd1_att), input_size)

        if not self.deep_supervision:
            return d1, edge_logits

        d2 = self.up_to_size(self.outconv2(hd2), input_size)
        d3 = self.up_to_size(self.outconv3(hd3), input_size)
        d4 = self.up_to_size(self.outconv4(hd4), input_size)
        d5 = self.up_to_size(self.outconv5(h5), input_size)

        return [d1, d2, d3, d4, d5], edge_logits


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

    edge_stats = estimate_edge_pos_weight(
        train_split,
        target_size=(args.image_size, args.image_size),
        edge_kernel_size=args.edge_kernel_size,
        max_samples=args.edge_stat_samples,
    )

    print("=" * 80)
    print("Run name:", args.run_name)
    print("Run dir :", run_dir)
    print("Model   : ResNet34 UNet3+ Edge + SimAM + SCSE-hd1")
    print("Estimated edge statistics:")
    print(edge_stats)
    print("=" * 80)

    config = vars(args).copy()
    config["model_name"] = "ResNet34UNet3PlusEdgeSimAMSCSEHD1"
    config["estimated_edge_stats"] = edge_stats
    save_json(config, run_dir / "config.json")

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

    model = ResNet34UNet3PlusEdgeSimAMSCSEHD1(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cat_ch=args.cat_channels,
        deep_supervision=deep_supervision,
        simam_lambda=args.simam_lambda,
        simam_position=args.simam_position,
        scse_reduction=args.scse_reduction,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # forward check
    sample_batch = next(iter(train_loader))
    sample_images, _, _, _, _ = sample_batch
    sample_images = sample_images.to(device)

    with torch.no_grad():
        seg_outputs, edge_logits = model(sample_images)
        main_logits = get_main_logits(seg_outputs)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", sample_images.shape)
    print("main_logits.shape  =", main_logits.shape)
    print("edge_logits.shape  =", edge_logits.shape)
    if isinstance(seg_outputs, list):
        print("deep outputs       =", [tuple(o.shape) for o in seg_outputs])
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_seg_loss", "train_edge_loss",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_seg_loss", "val_edge_loss",
            "val_dice", "val_iou", "val_precision", "val_recall"
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            deep_supervision=deep_supervision,
            lambda_edge=args.lambda_edge,
            edge_pos_weight=edge_stats["pos_weight"],
            train=True,
        )

        val_metrics = run_one_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            deep_supervision=deep_supervision,
            lambda_edge=args.lambda_edge,
            edge_pos_weight=edge_stats["pos_weight"],
            train=False,
        )

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
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["seg_loss"], train_metrics["edge_loss"],
                train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["seg_loss"], val_metrics["edge_loss"],
                val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    # 加载 best.pth 后，分别评估 train / val / test
    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch(
        model,
        train_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        lambda_edge=args.lambda_edge,
        edge_pos_weight=edge_stats["pos_weight"],
        train=False,
    )

    best_val_metrics = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        lambda_edge=args.lambda_edge,
        edge_pos_weight=edge_stats["pos_weight"],
        train=False,
    )

    test_metrics = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        lambda_edge=args.lambda_edge,
        edge_pos_weight=edge_stats["pos_weight"],
        train=False,
    )

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
    print("Outputs saved to:", run_dir)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="resnet34_unet3p_edge_simam_scse_hd1_e50_bs6")
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

    parser.add_argument("--scse-reduction", type=int, default=16)

    args = parser.parse_args()
    main(args)
