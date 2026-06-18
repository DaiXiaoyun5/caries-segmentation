from pathlib import Path
import csv
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    run_one_epoch,
    save_prediction_panel,
)


class ResidualConvBlock(nn.Module):
    """
    Residual convolution block for bridge and decoder.

    Main path:
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
    Shortcut path:
        Identity if channels match, otherwise Conv1x1 + BN
    Output:
        ReLU(main + shortcut)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResDecoderUpBlock(nn.Module):
    """
    Residual decoder upsampling block.

    Input:
        x    : lower-resolution decoder feature
        skip : same-scale encoder feature

    Flow:
        transposed conv upsamples x
        concatenate with skip feature
        residual conv block refines fused feature
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.res_block = ResidualConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Safety for non-perfectly divisible shapes.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class ResNet34ResDecoderUNet(nn.Module):
    """
    B1: ImageNet-pretrained ResNet34 encoder + residual bridge/decoder U-Net.

    Purpose:
    - Fairly compare ResNet34-U-Net with a residual decoder variant.
    - Encoder is standard ResNet34 from SMP and can load ImageNet pretrained weights.
    - Bridge and decoder use ResidualConvBlock.
    - This is not the same as Classic ResUNet from scratch; it is a pretrained
      ResNet34 encoder with a ResUNet-style residual decoder.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        decoder_channels=(256, 128, 64, 64, 32),
        **kwargs,
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        encoder_channels = list(self.encoder.out_channels)
        # For ResNet34 encoder in SMP, this is usually:
        # [3, 64, 64, 128, 256, 512]
        if len(encoder_channels) != 6:
            raise ValueError(f"Expected 6 encoder feature levels, got {encoder_channels}")

        self.encoder_channels = encoder_channels

        c4, c3, c2, c1, c0 = decoder_channels

        # Bridge at the deepest encoder feature.
        self.bridge = ResidualConvBlock(encoder_channels[5], encoder_channels[5])

        # Decoder: progressively use ResNet34 skip features.
        # features[5]: H/32, 512
        # features[4]: H/16, 256
        # features[3]: H/8,  128
        # features[2]: H/4,   64
        # features[1]: H/2,   64
        self.up4 = ResDecoderUpBlock(encoder_channels[5], encoder_channels[4], c4)  # H/16
        self.up3 = ResDecoderUpBlock(c4, encoder_channels[3], c3)                   # H/8
        self.up2 = ResDecoderUpBlock(c3, encoder_channels[2], c2)                   # H/4
        self.up1 = ResDecoderUpBlock(c2, encoder_channels[1], c1)                   # H/2

        # Restore full input resolution. No raw-image skip is used here to avoid
        # injecting unfiltered image noise directly into the decoder.
        self.final_up = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2)
        self.final_res = ResidualConvBlock(c0, c0)

        self.segmentation_head = nn.Conv2d(c0, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        # features:
        # f0: H,   W,   3
        # f1: H/2, W/2, 64
        # f2: H/4, W/4, 64
        # f3: H/8, W/8, 128
        # f4: H/16,W/16,256
        # f5: H/32,W/32,512
        f0, f1, f2, f3, f4, f5 = features

        b = self.bridge(f5)

        d4 = self.up4(b, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        out = self.final_up(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(
                out,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        out = self.final_res(out)

        masks = self.segmentation_head(out)
        return masks


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
    config["model_class"] = "ResNet34ResDecoderUNet"
    config["experiment"] = "B1: ImageNet-pretrained ResNet34 encoder + residual bridge/decoder U-Net"
    config["note"] = (
        "This model keeps a standard ImageNet-pretrained ResNet34 encoder, "
        "but replaces the plain U-Net decoder with residual decoder blocks. "
        "It tests whether a ResUNet-style bridge/decoder improves over ResNet34-U-Net."
    )
    save_json(config, run_dir / "config.json")

    train_ds = CariesMaskDataset(train_split, target_size=(args.image_size, args.image_size))
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("=" * 80)
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    decoder_channels = tuple(int(v) for v in args.decoder_channels.split(","))

    model = ResNet34ResDecoderUNet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        decoder_channels=decoder_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B1 ResNet34 encoder + ResBlock decoder")
    print("run_name         =", args.run_name)
    print("image_size       =", args.image_size)
    print("batch_size       =", args.batch_size)
    print("epochs           =", args.epochs)
    print("lr               =", args.lr)
    print("weight_decay     =", args.weight_decay)
    print("encoder_name     =", args.encoder_name)
    print("encoder_weights  =", args.encoder_weights)
    print("decoder_channels =", decoder_channels)
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
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, deep_supervision=False, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, deep_supervision=False, train=False)

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

    best_train_metrics = run_one_epoch(model, train_loader, optimizer=None, device=device, deep_supervision=False, train=False)
    best_val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, deep_supervision=False, train=False)
    test_metrics = run_one_epoch(model, test_loader, optimizer=None, device=device, deep_supervision=False, train=False)

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

    parser.add_argument("--run-name", type=str, default="resnet34_resdecoder_b1_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--decoder-channels", type=str, default="256,128,64,64,32")

    args = parser.parse_args()
    main(args)
