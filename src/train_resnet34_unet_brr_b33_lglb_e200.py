import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
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
from train_resnet34_unet_brr_b30_ssgcb_light_e200 import build_augmentation_config
from train_resnet34_unet_brr_b32_cglb_e200 import GlobalResponseNorm2d


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        padding = (kernel_size // 2) * dilation

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
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


class LocalGlobalLocalBottleneck(nn.Module):
    """
    LGLB-lite: Local-Global-Local large-kernel bottleneck context block.

    It is slightly stronger than B32 CGLB:
      local detail aggregation -> dilated large-kernel context -> local recovery.

    The design is still convolution-only and dependency-free, so it keeps the
    ResNet34-U-Net/B2 training recipe stable.
    """
    def __init__(
        self,
        channels,
        hidden_channels=128,
        local_kernel=5,
        global_kernel=7,
        global_dilation=3,
        dropout=0.0,
        scale_init=0.05,
        scale_max=0.30,
    ):
        super().__init__()

        channels = int(channels)
        hidden_channels = int(hidden_channels)
        self.scale_max = float(scale_max)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.local_in = DepthwiseSeparableConv(hidden_channels, kernel_size=local_kernel, dilation=1)
        self.global_context = DepthwiseSeparableConv(
            hidden_channels,
            kernel_size=global_kernel,
            dilation=global_dilation,
        )
        self.grn = GlobalResponseNorm2d(hidden_channels)
        self.local_out = DepthwiseSeparableConv(hidden_channels, kernel_size=local_kernel, dilation=1)

        self.project = nn.Sequential(
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        init_ratio = min(max(float(scale_init) / max(self.scale_max, 1e-6), 1e-4), 1.0 - 1e-4)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio))))

    def residual_scale(self):
        return self.scale_max * torch.sigmoid(self.scale_logit)

    def forward(self, x):
        context = self.reduce(x)
        context = self.local_in(context)
        context = self.global_context(context)
        context = self.grn(context)
        context = self.local_out(context)
        context = self.project(context)
        return x + self.residual_scale() * context


class ResNet34UNetBRRB33LGLB(nn.Module):
    """
    B33: B2 + LGLB-lite on the deepest encoder bottleneck feature.

    Keeps the original B2 boundary/rim auxiliary supervision and RBSG-lite.
    Adds local-global-local large-kernel context before the decoder.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        lglb_hidden_channels=128,
        lglb_local_kernel=5,
        lglb_global_kernel=7,
        lglb_global_dilation=3,
        lglb_dropout=0.0,
        lglb_scale_init=0.05,
        lglb_scale_max=0.30,
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

        self.lglb = LocalGlobalLocalBottleneck(
            channels=bottleneck_ch,
            hidden_channels=lglb_hidden_channels,
            local_kernel=lglb_local_kernel,
            global_kernel=lglb_global_kernel,
            global_dilation=lglb_global_dilation,
            dropout=lglb_dropout,
            scale_init=lglb_scale_init,
            scale_max=lglb_scale_max,
        )

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        features[-1] = self.lglb(features[-1])

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
    config["model_class"] = "ResNet34UNetBRRB33LGLB"
    config["experiment"] = "B33: B2 BRR-RBSG + Local-Global-Local Bottleneck"
    config["note"] = (
        "B33 keeps the original B2 boundary/rim auxiliary supervision and "
        "RBSG-lite. It adds a local-global-local large-kernel bottleneck context "
        "module on the deepest encoder feature before the decoder."
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
    config["lglb"] = {
        "enabled": True,
        "position": "deepest encoder bottleneck feature before decoder",
        "hidden_channels": args.lglb_hidden_channels,
        "local_kernel": args.lglb_local_kernel,
        "global_kernel": args.lglb_global_kernel,
        "global_dilation": args.lglb_global_dilation,
        "grn": True,
        "dropout": args.lglb_dropout,
        "scale_init": args.lglb_scale_init,
        "scale_max": args.lglb_scale_max,
        "fusion": "F_out = F + bounded_scale * LGLB(F)",
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34UNetBRRB33LGLB(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        lglb_hidden_channels=args.lglb_hidden_channels,
        lglb_local_kernel=args.lglb_local_kernel,
        lglb_global_kernel=args.lglb_global_kernel,
        lglb_global_dilation=args.lglb_global_dilation,
        lglb_dropout=args.lglb_dropout,
        lglb_scale_init=args.lglb_scale_init,
        lglb_scale_max=args.lglb_scale_max,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B33 LGLB + B2 BRR-RBSG ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("lglb_hidden_channels =", args.lglb_hidden_channels)
    print("lglb_local_kernel    =", args.lglb_local_kernel)
    print("lglb_global_kernel   =", args.lglb_global_kernel)
    print("lglb_global_dilation =", args.lglb_global_dilation)
    print("lglb_scale_init      =", args.lglb_scale_init)
    print("lglb_scale_max       =", args.lglb_scale_max)
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

    parser.add_argument("--run-name", type=str, default="b33_lglb_auglite_e200_constlr_bs6")
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

    parser.add_argument("--lglb-hidden-channels", type=int, default=128)
    parser.add_argument("--lglb-local-kernel", type=int, default=5)
    parser.add_argument("--lglb-global-kernel", type=int, default=7)
    parser.add_argument("--lglb-global-dilation", type=int, default=3)
    parser.add_argument("--lglb-dropout", type=float, default=0.0)
    parser.add_argument("--lglb-scale-init", type=float, default=0.05)
    parser.add_argument("--lglb-scale-max", type=float, default=0.30)

    args = parser.parse_args()
    main(args)
