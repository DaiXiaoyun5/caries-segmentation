from pathlib import Path
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


class WaveletHighFreqEnhance(nn.Module):
    """
    Haar-wavelet inspired high-frequency enhancement block.

    It extracts LH / HL / HH high-frequency components from a feature map,
    upsamples them back to the original resolution,
    projects them back to the original channel dimension,
    and enhances the original feature through a residual connection.
    """
    def __init__(self, channels, init_scale=0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def haar_high_frequency(self, x):
        b, c, h, w = x.shape

        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            x_pad = x

        x00 = x_pad[:, :, 0::2, 0::2]
        x01 = x_pad[:, :, 0::2, 1::2]
        x10 = x_pad[:, :, 1::2, 0::2]
        x11 = x_pad[:, :, 1::2, 1::2]

        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = ( x00 - x01 - x10 + x11) * 0.5

        target_size = x_pad.shape[-2:]

        lh = F.interpolate(lh, size=target_size, mode="nearest")
        hl = F.interpolate(hl, size=target_size, mode="nearest")
        hh = F.interpolate(hh, size=target_size, mode="nearest")

        if pad_h != 0 or pad_w != 0:
            lh = lh[:, :, :h, :w]
            hl = hl[:, :, :h, :w]
            hh = hh[:, :, :h, :w]

        return torch.cat([lh, hl, hh], dim=1)

    def forward(self, x):
        high = self.haar_high_frequency(x)
        high = self.proj(high)

        out = x + self.scale * high
        out = self.act(self.norm(out))

        return out


class ResNet34UNet3PlusEdgeWavelet(ResNet34UNet3PlusEdge):
    """
    ResNet34-UNet3+ + Edge Branch + Wavelet high-frequency enhancement.

    wavelet_position:
      hd1  : only enhance final high-resolution decoder feature
      hd2  : only enhance hd2
      hd3  : only enhance hd3
      hd4  : only enhance hd4
      all  : enhance all decoder features
      none : no wavelet enhancement
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        wavelet_position="hd1",
        wavelet_scale=0.1,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            cat_ch=cat_ch,
            deep_supervision=deep_supervision,
        )

        self.wavelet_position = wavelet_position

        self.wavelet_hd4 = WaveletHighFreqEnhance(self.up_ch, init_scale=wavelet_scale)
        self.wavelet_hd3 = WaveletHighFreqEnhance(self.up_ch, init_scale=wavelet_scale)
        self.wavelet_hd2 = WaveletHighFreqEnhance(self.up_ch, init_scale=wavelet_scale)
        self.wavelet_hd1 = WaveletHighFreqEnhance(self.up_ch, init_scale=wavelet_scale)

    def maybe_wavelet(self, name, x):
        if self.wavelet_position == "none":
            return x

        if self.wavelet_position == "all":
            return getattr(self, f"wavelet_{name}")(x)

        if self.wavelet_position == name:
            return getattr(self, f"wavelet_{name}")(x)

        return x

    def forward(self, x):
        input_size = x.shape[-2:]

        features = self.encoder(x)
        h1, h2, h3, h4, h5 = features[1], features[2], features[3], features[4], features[5]

        hd4 = self.conv_hd4(torch.cat([
            self.h1_PT_hd4(h1),
            self.h2_PT_hd4(h2),
            self.h3_PT_hd4(h3),
            self.h4_Cat_hd4(h4),
            self.up_to(self.h5_UT_hd4(h5), h4),
        ], dim=1))
        hd4 = self.maybe_wavelet("hd4", hd4)

        hd3 = self.conv_hd3(torch.cat([
            self.h1_PT_hd3(h1),
            self.h2_PT_hd3(h2),
            self.h3_Cat_hd3(h3),
            self.up_to(self.h4_UT_hd3(hd4), h3),
            self.up_to(self.h5_UT_hd3(h5), h3),
        ], dim=1))
        hd3 = self.maybe_wavelet("hd3", hd3)

        hd2 = self.conv_hd2(torch.cat([
            self.h1_PT_hd2(h1),
            self.h2_Cat_hd2(h2),
            self.up_to(self.h3_UT_hd2(hd3), h2),
            self.up_to(self.h4_UT_hd2(hd4), h2),
            self.up_to(self.h5_UT_hd2(h5), h2),
        ], dim=1))
        hd2 = self.maybe_wavelet("hd2", hd2)

        hd1 = self.conv_hd1(torch.cat([
            self.h1_Cat_hd1(h1),
            self.up_to(self.h2_UT_hd1(hd2), h1),
            self.up_to(self.h3_UT_hd1(hd3), h1),
            self.up_to(self.h4_UT_hd1(hd4), h1),
            self.up_to(self.h5_UT_hd1(h5), h1),
        ], dim=1))
        hd1 = self.maybe_wavelet("hd1", hd1)

        d1 = self.up_to_size(self.outconv1(hd1), input_size)
        edge_logits = self.up_to_size(self.edge_head(hd1), input_size)

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
    print("Estimated edge statistics")
    print(edge_stats)
    print("=" * 80)

    config = vars(args).copy()
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

    model = ResNet34UNet3PlusEdgeWavelet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cat_ch=args.cat_channels,
        deep_supervision=deep_supervision,
        wavelet_position=args.wavelet_position,
        wavelet_scale=args.wavelet_scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
    print("wavelet_position   =", args.wavelet_position)
    print("wavelet_scale      =", args.wavelet_scale)
    print("=" * 80)

    history_path = run_dir / "history.csv"

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_seg_loss", "train_edge_loss",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_seg_loss", "val_edge_loss",
            "val_dice", "val_iou", "val_precision", "val_recall",
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
            f"val_iou={val_metrics['iou']:.4f}"
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="smoke_resnet34_unet3p_edge_wavelet")

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--cat-channels", type=int, default=64)
    parser.add_argument("--deep-supervision", type=int, default=1)

    parser.add_argument("--lambda-edge", type=float, default=0.3)
    parser.add_argument("--edge-kernel-size", type=int, default=3)
    parser.add_argument("--edge-stat-samples", type=int, default=800)
    parser.add_argument("--edge-vis-threshold", type=float, default=0.2)

    parser.add_argument(
        "--wavelet-position",
        type=str,
        default="hd1",
        choices=["hd1", "hd2", "hd3", "hd4", "all", "none"],
    )

    parser.add_argument("--wavelet-scale", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
