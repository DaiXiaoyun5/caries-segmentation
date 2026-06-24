import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)
from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset
from train_resnet34_unet_brr_b1_aux_e200 import run_one_epoch_b1
from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


def _valid_group_count(channels: int, max_groups: int = 4) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class PPMLiteBottleneck(nn.Module):
    """
    PPM-lite on the deepest ResNet34 encoder feature.

    This module follows the useful signal from this project's earlier
    ResNet34-UNet3+ experiments, where CBSA+PPM performed best on small-lesion
    groups. Unlike the previous B8B dilated-conv context bottleneck, PPM uses
    pooled regional priors and should be less sensitive to local noisy texture.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        pool_sizes=(1, 2, 3, 6),
        dropout=0.0,
        init_scale=0.08,
    ):
        super().__init__()
        self.pool_sizes = tuple(int(s) for s in pool_sizes)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_valid_group_count(hidden_channels), hidden_channels),
                nn.SiLU(inplace=True),
            )
            for size in self.pool_sizes
        ])

        total_channels = in_channels + hidden_channels * len(self.pool_sizes)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
        )
        nn.init.zeros_(self.project[1].weight)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [x]
        for branch in self.branches:
            pooled = branch(x)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(pooled)
        context = self.project(torch.cat(feats, dim=1))
        return x + torch.clamp(self.scale, 0.0, 1.0) * context


class ResNet34UNetBRRB14PPM(ResNet34UNetBRRB2RBSG):
    """
    B14: B2 + PPM-lite bottleneck.

    B2 is kept unchanged:
      ResNet34-U-Net + boundary/rim auxiliary supervision + RBSG-lite.

    B14 only changes the deepest encoder feature before the SMP decoder:
      features[-1] = features[-1] + scale * PPM(features[-1])

    It tests whether regional/global context, which helped small lesions in
    the older UNet3+ line, can complement B2 without adding more boundary/rim
    supervision or suppression.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        ppm_hidden_channels=64,
        ppm_pool_sizes=(1, 2, 3, 6),
        ppm_dropout=0.0,
        ppm_init_scale=0.08,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        enc_channels = list(self.unet.encoder.out_channels)
        self.ppm_lite = PPMLiteBottleneck(
            in_channels=enc_channels[-1],
            hidden_channels=ppm_hidden_channels,
            pool_sizes=ppm_pool_sizes,
            dropout=ppm_dropout,
            init_scale=ppm_init_scale,
        )

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        features[-1] = self.ppm_lite(features[-1])
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)
        refined = self.rbsg(decoder_output, b_pre, r_pre)

        mask_logits = self.mask_head(refined)
        if not return_aux:
            return mask_logits

        return {
            "mask": mask_logits,
            "boundary": self.boundary_head(refined),
            "rim": self.rim_head(refined),
        }

    @property
    def learned_ppm_scale(self):
        return float(torch.clamp(self.ppm_lite.scale, 0.0, 1.0).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB14PPM"
    config["experiment"] = "B14: B2 + PPM-lite bottleneck context"
    config["note"] = (
        "B14 keeps the complete B2 BRR-RBSG framework and adds one PPM-lite "
        "regional context module on the deepest encoder feature. This is chosen "
        "because this project's previous UNet3+ analysis found CBSA+PPM strongest "
        "on small-lesion groups, while B10/B11 showed that more selective attention "
        "can hurt recall."
    )
    config["dataset_observation"] = {
        "test_cases": 358,
        "small_group_cases": 119,
        "small_area_ratio_mean_512": 0.011539875960149685,
        "overall_area_ratio_mean_512": 0.03290683597159785,
        "implication": "foreground is sparse; regional context should help disambiguate weak local texture without relying on stronger rim suppression",
    }
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["ppm_lite"] = {
        "position": "deepest ResNet34 encoder feature before decoder",
        "pool_sizes": args.ppm_pool_sizes,
        "hidden_channels": args.ppm_hidden_channels,
        "dropout": args.ppm_dropout,
        "init_scale": args.ppm_init_scale,
        "residual_projection_init": "project BatchNorm scale initialized to zero",
        "uses_boundary_or_rim": False,
        "extra_supervision": False,
    }
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        profile=args.augment_profile,
    )
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNetBRRB14PPM(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        ppm_hidden_channels=args.ppm_hidden_channels,
        ppm_pool_sizes=tuple(args.ppm_pool_sizes),
        ppm_dropout=args.ppm_dropout,
        ppm_init_scale=args.ppm_init_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B14 = B2 BRR-RBSG + PPM-lite bottleneck")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("ppm_pool_sizes  =", args.ppm_pool_sizes)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "ppm_scale",
        ])

    best_val_dice = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)
        ppm_scale = model.learned_ppm_scale

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} ppm_scale={ppm_scale:.4f}"
        )
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                ppm_scale,
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))
    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)
    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)
    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ppm_parameters": sum(p.numel() for p in model.ppm_lite.parameters()),
        "learned_ppm_scale": model.learned_ppm_scale,
    }

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="b14_ppm_bottleneck_auglite_e200_constlr_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--augment-profile", type=str, default="lite", choices=["lite", "strong", "none"])
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--ppm-hidden-channels", type=int, default=64)
    parser.add_argument("--ppm-pool-sizes", type=int, nargs="+", default=[1, 2, 3, 6])
    parser.add_argument("--ppm-dropout", type=float, default=0.0)
    parser.add_argument("--ppm-init-scale", type=float, default=0.08)
    main(parser.parse_args())
