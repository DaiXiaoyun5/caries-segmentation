from pathlib import Path
import csv
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    run_one_epoch,
    save_prediction_panel,
)


class ResidualConvBlock(nn.Module):
    """
    Classic residual convolution block used in ResUNet.

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


class ResUNetUpBlock(nn.Module):
    """
    Decoder upsampling block.

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

        # Safety for images whose size is not perfectly divisible.
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


class ClassicResUNet(nn.Module):
    """
    Classic ResUNet baseline.

    Difference from ResNet34-U-Net:
    - This model does NOT use an ImageNet-pretrained ResNet34 encoder.
    - Encoder, bridge, and decoder blocks are all residual convolution blocks.
    - It is trained from scratch unless you explicitly load external weights.
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_channels=64,
        encoder_name=None,
        encoder_weights=None,
        **kwargs,
    ):
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, c)          # H, W
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualConvBlock(c, c * 2)                # H/2, W/2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResidualConvBlock(c * 2, c * 4)            # H/4, W/4
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ResidualConvBlock(c * 4, c * 8)            # H/8, W/8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge / bottleneck
        self.bridge = ResidualConvBlock(c * 8, c * 16)         # H/16, W/16

        # Decoder
        self.up4 = ResUNetUpBlock(c * 16, c * 8, c * 8)        # H/8, W/8
        self.up3 = ResUNetUpBlock(c * 8, c * 4, c * 4)         # H/4, W/4
        self.up2 = ResUNetUpBlock(c * 4, c * 2, c * 2)         # H/2, W/2
        self.up1 = ResUNetUpBlock(c * 2, c, c)                 # H, W

        self.segmentation_head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bridge
        b = self.bridge(self.pool4(e4))

        # Decoder path with skip connections
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        masks = self.segmentation_head(d1)
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
    config["model_class"] = "ClassicResUNet"
    config["experiment"] = "B0: Classic ResUNet baseline with residual encoder, bridge, and decoder blocks"
    config["note"] = (
        "Classic ResUNet baseline trained from scratch. "
        "Unlike ResNet34-U-Net, this model does not load ImageNet-pretrained encoder weights. "
        "Encoder, bridge, and decoder are all residual convolution blocks."
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

    model = ClassicResUNet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B0 Classic ResUNet")
    print("run_name       =", args.run_name)
    print("image_size     =", args.image_size)
    print("base_channels  =", args.base_channels)
    print("batch_size     =", args.batch_size)
    print("epochs         =", args.epochs)
    print("lr             =", args.lr)
    print("weight_decay   =", args.weight_decay)
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

    parser.add_argument("--run-name", type=str, default="classic_resunet_b0_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=64)

    args = parser.parse_args()
    main(args)
