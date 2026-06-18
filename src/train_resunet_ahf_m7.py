from pathlib import Path
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    run_one_epoch,
    save_prediction_panel,
)


class DoubleConv(nn.Module):
    """
    Official AHF-U-Net style DoubleConv:
    Conv-BN-ReLU twice.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class AFF(nn.Module):
    """
    Official AHF-U-Net AFF module.

    Inputs:
    x             : enhanced skip feature, I_out_i
    outI          : upsampled decoder feature
    feature_level : original encoder feature at the same level

    Official fusion:
    xa = x + outI + feature_level
    local attention + global attention -> weight
    output = 2*x*weight + 2*outI*(1-weight)
    """
    def __init__(self, channels, r=4):
        super().__init__()

        inter_channels = max(int(channels // r), 1)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, outI, feature_level):
        xa = x + outI + feature_level
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        wei = self.sigmoid(xl + xg)

        xo = 2.0 * x * wei + 2.0 * outI * (1.0 - wei)
        return xo


class HAEBlock(nn.Module):
    """
    Hierarchical Attention-Enhanced skip generation.

    This follows the official AHF-U-Net idea:
    1. For each encoder feature, calculate spatial/channel interaction:
       mul_i = GAP(feature_i) * channel_mean(feature_i)
    2. Upsample deeper mul_{i+1} to current level.
    3. Fuse current and deeper information.
    4. Multiply with original encoder feature to obtain I_out_i.

    Adapted to ResNet34 encoder channel layout:
    f1: 64,  H/2
    f2: 64,  H/4
    f3: 128, H/8
    f4: 256, H/16
    f5: 512, H/32
    """
    def __init__(self, channels):
        super().__init__()

        c1, c2, c3, c4, c5 = channels

        self.spatial_avgpool = nn.AdaptiveAvgPool2d(1)

        self.trans1 = nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trans2 = nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trans3 = nn.ConvTranspose2d(c4, c3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trans4 = nn.ConvTranspose2d(c5, c4, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def make_mul(self, x):
        # Official-style: spatial avg pooling * channel mean
        s = self.spatial_avgpool(x)
        c = torch.mean(x, dim=1, keepdim=True)
        return s * c

    def align_like(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, f1, f2, f3, f4, f5):
        mul1 = self.make_mul(f1)
        mul2 = self.make_mul(f2)
        mul3 = self.make_mul(f3)
        mul4 = self.make_mul(f4)
        mul5 = self.make_mul(f5)

        t1 = self.align_like(self.trans1(mul2), mul1)
        i1 = self.conv1(mul1 + t1)
        i1 = i1 * f1

        t2 = self.align_like(self.trans2(mul3), mul2)
        i2 = self.conv2(mul2 + t2)
        i2 = i2 * f2

        t3 = self.align_like(self.trans3(mul4), mul3)
        i3 = self.conv3(mul3 + t3)
        i3 = i3 * f3

        t4 = self.align_like(self.trans4(mul5), mul4)
        i4 = self.conv4(mul4 + t4)
        i4 = i4 * f4

        return i1, i2, i3, i4


class UpAHF(nn.Module):
    """
    Official AHF decoder idea:
    upsample deeper decoder feature,
    fuse with enhanced skip and original encoder feature by AFF,
    then DoubleConv.
    """
    def __init__(self, in_channels, skip_channels, out_channels, aff_reduction=4):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.aff = AFF(skip_channels, r=aff_reduction)
        self.conv = DoubleConv(skip_channels, out_channels)

    def align_like(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x_deep, skip_enhanced, skip_raw):
        x_up = self.up(x_deep)
        x_up = self.align_like(x_up, skip_enhanced)

        fused = self.aff(skip_enhanced, x_up, skip_raw)
        out = self.conv(fused)
        return out


class ResNet34UNetAHF(nn.Module):
    """
    M7:
    ResNet34-U-Net + full AHF-Fusion.

    The original AHF-U-Net uses a plain U-Net encoder.
    Here we keep the official AHF fusion logic and replace the encoder with ResNet34.

    Encoder features:
    f1: H/2,  64 channels
    f2: H/4,  64 channels
    f3: H/8,  128 channels
    f4: H/16, 256 channels
    f5: H/32, 512 channels

    Decoder:
    f5 -> AHF(f4) -> AHF(f3) -> AHF(f2) -> AHF(f1) -> final upsample -> output
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        aff_reduction=4,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        # For smp resnet34 encoder: [3, 64, 64, 128, 256, 512]
        encoder_channels = list(self.encoder.out_channels)
        feature_channels = encoder_channels[1:]  # [64, 64, 128, 256, 512]

        if len(feature_channels) != 5:
            raise RuntimeError(f"Expected 5 feature levels, got {feature_channels}")

        c1, c2, c3, c4, c5 = feature_channels

        self.hae = HAEBlock([c1, c2, c3, c4, c5])

        self.up4 = UpAHF(c5, c4, c4, aff_reduction=aff_reduction)
        self.up3 = UpAHF(c4, c3, c3, aff_reduction=aff_reduction)
        self.up2 = UpAHF(c3, c2, c2, aff_reduction=aff_reduction)
        self.up1 = UpAHF(c2, c1, c1, aff_reduction=aff_reduction)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            DoubleConv(c1, c1),
        )

        self.outc = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)

        # features[0] is usually the input-scale tensor.
        f1, f2, f3, f4, f5 = features[1], features[2], features[3], features[4], features[5]

        i1, i2, i3, i4 = self.hae(f1, f2, f3, f4, f5)

        x = self.up4(f5, i4, f4)
        x = self.up3(x, i3, f3)
        x = self.up2(x, i2, f2)
        x = self.up1(x, i1, f1)

        x = self.final_up(x)

        if x.shape[-2:] != features[0].shape[-2:]:
            x = F.interpolate(x, size=features[0].shape[-2:], mode="bilinear", align_corners=False)

        logits = self.outc(x)
        return logits


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
    config["model_class"] = "ResNet34UNetAHF"
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

    model = ResNet34UNetAHF(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        aff_reduction=args.aff_reduction,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=" * 80)
    print("Model ready: M7 ResNet34-U-Net + full AHF-Fusion")
    print("run_name       =", args.run_name)
    print("encoder_name   =", args.encoder_name)
    print("encoder_weights=", args.encoder_weights)
    print("aff_reduction  =", args.aff_reduction)
    print("batch_size     =", args.batch_size)
    print("epochs         =", args.epochs)
    print("lr             =", args.lr)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            deep_supervision=False,
            train=True,
        )

        val_metrics = run_one_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            deep_supervision=False,
            train=False,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
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

    best_train_metrics = run_one_epoch(
        model,
        train_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )
    best_val_metrics = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )
    test_metrics = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    print("=" * 80)
    print("Best checkpoint summary")
    print(summary)
    print("=" * 80)

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="resunet_ahf_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--aff-reduction", type=int, default=4)

    args = parser.parse_args()
    main(args)
