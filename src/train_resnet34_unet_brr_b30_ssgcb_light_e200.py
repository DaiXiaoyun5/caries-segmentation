import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)

from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset

from train_resnet34_unet_brr_b1_aux_e200 import run_one_epoch_b1
from train_resnet34_unet_brr_b2_rbsg_e200 import RBSGLite


def parse_dilations(dilations):
    if isinstance(dilations, str):
        return tuple(int(x.strip()) for x in dilations.split(",") if x.strip())
    return tuple(int(x) for x in dilations)


class DepthwiseContextBranch(nn.Module):
    """
    Lightweight depthwise-separable context branch used by SSGCB.
    """
    def __init__(self, channels, dilation):
        super().__init__()
        dilation = int(dilation)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SelectiveSpatialGatedContextBottleneck(nn.Module):
    """
    Selective Spatial-Gated Context Bottleneck (SSGCB).

    Input:
      encoder bottleneck feature F in R^{C x H x W}

    Steps:
      1) reduce channels C -> c_mid with 1x1 conv
      2) extract multi-scale context with depthwise dilated branches
         and one global pooling branch
      3) adaptively choose branch scales with SK-style softmax weights
      4) apply lightweight spatial gate on the fused context
      5) project back and residual-fuse with a bounded learnable scale

    This differs from B8B's fixed concat-fusion context bottleneck because
    SSGCB selects the effective receptive field per input.
    """
    def __init__(
        self,
        channels,
        branch_channels=96,
        dilations=(1, 3, 5),
        dropout=0.0,
        scale_init=0.05,
        scale_max=0.30,
        use_global=True,
        use_spatial_gate=True,
    ):
        super().__init__()

        channels = int(channels)
        branch_channels = int(branch_channels)
        self.channels = channels
        self.branch_channels = branch_channels
        self.dilations = parse_dilations(dilations)
        self.use_global = bool(use_global)
        self.use_spatial_gate = bool(use_spatial_gate)
        self.scale_max = float(scale_max)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )

        self.local_branches = nn.ModuleList([
            DepthwiseContextBranch(branch_channels, dilation=d)
            for d in self.dilations
        ])

        if self.use_global:
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=True),
                nn.SiLU(inplace=True),
            )
        else:
            self.global_branch = None

        self.num_branches = len(self.local_branches) + (1 if self.use_global else 0)
        mlp_hidden = max(branch_channels // 4, 16)
        self.scale_mlp = nn.Sequential(
            nn.Linear(branch_channels, mlp_hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, self.num_branches, bias=True),
        )

        if self.use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None

        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(branch_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        init_ratio = min(max(float(scale_init) / max(self.scale_max, 1e-6), 1e-4), 1.0 - 1e-4)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio))))

    def residual_scale(self):
        return self.scale_max * torch.sigmoid(self.scale_logit)

    def forward(self, x):
        size = x.shape[2:]
        x_mid = self.reduce(x)

        branch_outputs = [branch(x_mid) for branch in self.local_branches]
        if self.global_branch is not None:
            global_context = self.global_branch(x_mid)
            global_context = F.interpolate(
                global_context,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            branch_outputs.append(global_context)

        u = torch.stack(branch_outputs, dim=0).sum(dim=0)
        z = F.adaptive_avg_pool2d(u, 1).flatten(1)
        weights = torch.softmax(self.scale_mlp(z), dim=1)

        stacked = torch.stack(branch_outputs, dim=1)
        fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)

        if self.spatial_gate is not None:
            avg_map = fused.mean(dim=1, keepdim=True)
            max_map = fused.max(dim=1, keepdim=True)[0]
            gate = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
            fused = fused * gate

        fused = self.dropout(fused)
        residual = self.project(fused)
        return x + self.residual_scale() * residual


class ResNet34UNetBRRB30SSGCBLight(nn.Module):
    """
    B30-light: B2 + SSGCB on the deepest encoder bottleneck feature.

    Keeps the original B2 boundary/rim auxiliary supervision and RBSG-lite.
    Adds a lightweight SK-style dynamic multi-scale context module before
    the decoder.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        ssgcb_branch_channels=96,
        ssgcb_dilations=(1, 3, 5),
        ssgcb_dropout=0.0,
        ssgcb_scale_init=0.05,
        ssgcb_scale_max=0.30,
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels

        try:
            bottleneck_ch = int(self.unet.encoder.out_channels[-1])
        except Exception:
            bottleneck_ch = 512

        self.ssgcb = SelectiveSpatialGatedContextBottleneck(
            channels=bottleneck_ch,
            branch_channels=ssgcb_branch_channels,
            dilations=ssgcb_dilations,
            dropout=ssgcb_dropout,
            scale_init=ssgcb_scale_init,
            scale_max=ssgcb_scale_max,
            use_global=True,
            use_spatial_gate=True,
        )

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        features[-1] = self.ssgcb(features[-1])

        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)
        refined = self.rbsg(decoder_output, b_pre, r_pre)

        mask_logits = self.mask_head(refined)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }


def build_augmentation_config(args):
    return {
        "type": "online",
        "profile": args.augment_profile,
        "train_only": True,
        "geometry": {
            "rotation_deg": [-15.0, 15.0],
            "scale": [0.85, 1.20],
            "translate_ratio": 0.05,
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.0,
        },
        "photometric": {
            "brightness_factor": [0.85, 1.15],
            "contrast_factor": [0.85, 1.15],
            "gamma": [0.80, 1.25],
            "gaussian_noise_p": 0.10,
            "noise_std": [0.005, 0.020],
            "gaussian_blur_p": 0.10,
            "blur_radius": [0.2, 0.8],
        },
    }


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

    ssgcb_dilations = parse_dilations(args.ssgcb_dilations)

    config = vars(args).copy()
    config["ssgcb_dilations"] = list(ssgcb_dilations)
    config["model_class"] = "ResNet34UNetBRRB30SSGCBLight"
    config["experiment"] = "B30-light: B2 BRR-RBSG + Selective Spatial-Gated Context Bottleneck"
    config["note"] = (
        "B30-light keeps the original B2 boundary/rim auxiliary supervision and "
        "RBSG-lite. It adds SSGCB on the deepest encoder feature: 1x1 channel "
        "reduction, depthwise dilated branches with dynamic SK-style scale "
        "selection, a lightweight spatial gate, and bounded residual fusion."
    )
    config["aux_targets"] = {
        "boundary": "dilate(mask,3)-erode(mask,3)",
        "rim": "dilate(mask,9)-dilate(mask,3)",
    }
    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation",
    }
    config["ssgcb"] = {
        "enabled": True,
        "position": "deepest encoder bottleneck feature before decoder",
        "branch_channels": args.ssgcb_branch_channels,
        "dilations": list(ssgcb_dilations),
        "global_branch": True,
        "dynamic_scale_selection": "softmax(MLP(GAP(sum(branches))))",
        "spatial_gate": "sigmoid(Conv7x7([mean(B), max(B)]))",
        "dropout": args.ssgcb_dropout,
        "scale_init": args.ssgcb_scale_init,
        "scale_max": args.ssgcb_scale_max,
        "fusion": "F_out = F + bounded_scale * Conv1x1(B_g)",
    }
    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["augmentation"] = build_augmentation_config(args)
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        profile=args.augment_profile,
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34UNetBRRB30SSGCBLight(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        ssgcb_branch_channels=args.ssgcb_branch_channels,
        ssgcb_dilations=ssgcb_dilations,
        ssgcb_dropout=args.ssgcb_dropout,
        ssgcb_scale_init=args.ssgcb_scale_init,
        ssgcb_scale_max=args.ssgcb_scale_max,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B30-light SSGCB + B2 BRR-RBSG ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("ssgcb_branch_channels =", args.ssgcb_branch_channels)
    print("ssgcb_dilations       =", ssgcb_dilations)
    print("ssgcb_scale_init      =", args.ssgcb_scale_init)
    print("ssgcb_scale_max       =", args.ssgcb_scale_max)
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
        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
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

    parser.add_argument("--run-name", type=str, default="b30_ssgcb_light_auglite_e200_constlr_bs6")
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

    parser.add_argument("--ssgcb-branch-channels", type=int, default=96)
    parser.add_argument("--ssgcb-dilations", type=str, default="1,3,5")
    parser.add_argument("--ssgcb-dropout", type=float, default=0.0)
    parser.add_argument("--ssgcb-scale-init", type=float, default=0.05)
    parser.add_argument("--ssgcb-scale-max", type=float, default=0.30)

    args = parser.parse_args()
    main(args)
