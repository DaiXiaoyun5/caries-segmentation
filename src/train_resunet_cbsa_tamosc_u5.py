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


class CBSABlock(nn.Module):
    def __init__(self, e_lambda=1e-4, alpha=1.0, beta=0.1):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = alpha
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def simam_attention(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        mean = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mean).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / max(n, 1)
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return self.sigmoid(e_inv)

    def boundary_gradient(self, x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))

        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))

        g = gx + gy
        g = g.mean(dim=1, keepdim=True)

        g_min = g.amin(dim=(2, 3), keepdim=True)
        g_max = g.amax(dim=(2, 3), keepdim=True)
        g = (g - g_min) / (g_max - g_min + 1e-6)
        return g

    def forward(self, x):
        simam = self.simam_attention(x)
        boundary = self.boundary_gradient(x)
        att = simam * (1.0 + self.alpha * boundary)
        return x + self.beta * x * att


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TAMoSCLite(nn.Module):
    """
    TA-MoSC-lite: Task-Adaptive Mixture of Skip Connections.

    It keeps the original SMP U-Net decoder unchanged.
    Only encoder skip features are adaptively mixed before being sent to decoder.
    """
    def __init__(
        self,
        skip_channels,
        context_channels,
        mid_channels=16,
        scale_init=0.1,
        dropout=0.0,
    ):
        super().__init__()
        self.skip_channels = list(skip_channels)
        self.num_skips = len(self.skip_channels)

        self.to_common = nn.ModuleList([
            ConvBNReLU(ch, mid_channels, kernel_size=1, padding=0)
            for ch in self.skip_channels
        ])
        self.to_target = nn.ModuleList([
            ConvBNReLU(mid_channels, ch, kernel_size=1, padding=0)
            for ch in self.skip_channels
        ])

        hidden = max(context_channels // 4, 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(context_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden, self.num_skips * self.num_skips, kernel_size=1, bias=True),
        )

        self.res_scale = nn.Parameter(torch.full((self.num_skips,), float(scale_init)))

    def forward(self, skips, context):
        if len(skips) != self.num_skips:
            raise ValueError(f"Expected {self.num_skips} skip features, got {len(skips)}")

        b = context.shape[0]
        common_feats = [proj(feat) for proj, feat in zip(self.to_common, skips)]

        logits = self.gate(context).view(b, self.num_skips, self.num_skips)
        weights = torch.softmax(logits, dim=2)

        enhanced_skips = []
        for target_idx, target_feat in enumerate(skips):
            target_size = target_feat.shape[-2:]
            mixed = None

            for source_idx, source_feat in enumerate(common_feats):
                source_resized = F.interpolate(
                    source_feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                w = weights[:, target_idx, source_idx].view(b, 1, 1, 1)
                mixed = source_resized * w if mixed is None else mixed + source_resized * w

            delta = self.to_target[target_idx](mixed)
            scale = self.res_scale[target_idx].view(1, 1, 1, 1)
            enhanced_skips.append(target_feat + scale * delta)

        return enhanced_skips


class ResNet34UNetCBSATAMoSC(nn.Module):
    """
    U5:
    ResNet34-U-Net + CBSA + TA-MoSC-lite.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cbsa_lambda=1e-4,
        cbsa_alpha=1.0,
        cbsa_beta=0.1,
        tamosc_mid_channels=16,
        tamosc_scale_init=0.1,
        tamosc_dropout=0.0,
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        encoder_channels = list(self.unet.encoder.out_channels)
        skip_channels = encoder_channels[1:-1]
        context_channels = encoder_channels[-1]

        self.tamosc = TAMoSCLite(
            skip_channels=skip_channels,
            context_channels=context_channels,
            mid_channels=tamosc_mid_channels,
            scale_init=tamosc_scale_init,
            dropout=tamosc_dropout,
        )

        self.cbsa = CBSABlock(
            e_lambda=cbsa_lambda,
            alpha=cbsa_alpha,
            beta=cbsa_beta,
        )

    def forward(self, x):
        features = list(self.unet.encoder(x))
        features[1:-1] = self.tamosc(features[1:-1], features[-1])
        decoder_output = self.unet.decoder(*features)
        decoder_output = self.cbsa(decoder_output)
        masks = self.unet.segmentation_head(decoder_output)
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
    config["model_class"] = "ResNet34UNetCBSATAMoSC"
    config["experiment"] = "U5: ResNet34-U-Net + CBSA + TA-MoSC-lite"
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

    model = ResNet34UNetCBSATAMoSC(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        tamosc_mid_channels=args.tamosc_mid_channels,
        tamosc_scale_init=args.tamosc_scale_init,
        tamosc_dropout=args.tamosc_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: U5 ResNet34-U-Net + CBSA + TA-MoSC-lite")
    print("run_name             =", args.run_name)
    print("encoder_name         =", args.encoder_name)
    print("encoder_weights      =", args.encoder_weights)
    print("cbsa_lambda          =", args.cbsa_lambda)
    print("cbsa_alpha           =", args.cbsa_alpha)
    print("cbsa_beta            =", args.cbsa_beta)
    print("tamosc_mid_channels  =", args.tamosc_mid_channels)
    print("tamosc_scale_init    =", args.tamosc_scale_init)
    print("tamosc_dropout       =", args.tamosc_dropout)
    print("batch_size           =", args.batch_size)
    print("epochs               =", args.epochs)
    print("lr                   =", args.lr)
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

    parser.add_argument("--run-name", type=str, default="resunet_cbsa_tamosc_u5_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--cbsa-lambda", type=float, default=1e-4)
    parser.add_argument("--cbsa-alpha", type=float, default=1.0)
    parser.add_argument("--cbsa-beta", type=float, default=0.1)

    parser.add_argument("--tamosc-mid-channels", type=int, default=16)
    parser.add_argument("--tamosc-scale-init", type=float, default=0.1)
    parser.add_argument("--tamosc-dropout", type=float, default=0.0)

    args = parser.parse_args()
    main(args)
