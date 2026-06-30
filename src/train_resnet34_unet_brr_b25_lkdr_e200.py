import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
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


def _valid_group_count(channels, max_groups=4):
    for groups in range(min(int(max_groups), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class LargeKernelDecoderRefiner(nn.Module):
    """
    Large-kernel inverted bottleneck refiner for the final decoder feature.

    It borrows the efficient CNN spirit from CMUNeXt/EMCAD-style medical
    segmentation blocks: pointwise expansion, cheap depthwise large-kernel
    context, then pointwise projection. It is deliberately inserted before B2's
    pre-boundary/pre-rim heads and RBSG-lite so the original B2 prior can still
    decide how to use the enriched feature.
    """
    def __init__(
        self,
        channels,
        hidden_channels=48,
        expansion=3.0,
        local_kernel=5,
        context_kernel=7,
        context_dilation=2,
        init_scale=0.05,
        max_scale=0.50,
    ):
        super().__init__()
        channels = int(channels)
        hidden_channels = max(int(hidden_channels), int(round(channels * float(expansion))))
        self.max_scale = float(max_scale)

        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.local_context = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=local_kernel,
                padding=local_kernel // 2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )

        context_padding = (context_kernel // 2) * int(context_dilation)
        self.long_context = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, context_kernel),
                padding=(0, context_padding),
                dilation=(1, context_dilation),
                groups=hidden_channels,
                bias=False,
            ),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(context_kernel, 1),
                padding=(context_padding, 0),
                dilation=(context_dilation, 1),
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_valid_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x):
        expanded = self.expand(x)
        local = self.local_context(expanded)
        long = self.long_context(expanded)
        residual = self.fuse(torch.cat([local, long], dim=1))
        scale = torch.clamp(self.residual_scale, min=0.0, max=self.max_scale)
        return x + scale * residual


class ResNet34UNetBRRB25LKDR(ResNet34UNetBRRB2RBSG):
    """
    B25: B2 + Large-Kernel Decoder Refiner.

    The architecture change is strictly located on the final decoder feature
    before B2's auxiliary heads and RBSG-lite. It does not introduce additional
    boundary/rim losses, post-mask residuals, or transformer/mamba backbones.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        lkdr_hidden_channels=48,
        lkdr_expansion=3.0,
        lkdr_local_kernel=5,
        lkdr_context_kernel=7,
        lkdr_context_dilation=2,
        lkdr_init_scale=0.05,
        lkdr_max_scale=0.50,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        seg_conv = self.unet.segmentation_head[0]
        dec_ch = int(seg_conv.in_channels)
        self.lkdr = LargeKernelDecoderRefiner(
            channels=dec_ch,
            hidden_channels=lkdr_hidden_channels,
            expansion=lkdr_expansion,
            local_kernel=lkdr_local_kernel,
            context_kernel=lkdr_context_kernel,
            context_dilation=lkdr_context_dilation,
            init_scale=lkdr_init_scale,
            max_scale=lkdr_max_scale,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        decoder_output = self.lkdr(decoder_output)

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
    def learned_lkdr_scale(self):
        return float(torch.clamp(self.lkdr.residual_scale, 0.0, self.lkdr.max_scale).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB25LKDR"
    config["experiment"] = "B25: B2 + Large-Kernel Decoder Refiner"
    config["note"] = (
        "B25 keeps B2 boundary/rim auxiliary supervision and RBSG-lite unchanged. "
        "It inserts a lightweight large-kernel inverted bottleneck refiner on the "
        "final decoder feature before B2's pre-boundary/pre-rim heads and RBSG-lite. "
        "This provides local-to-long decoder context without adding another "
        "boundary/rim gate or output-logit rescue branch."
    )
    config["design_reflection"] = {
        "why_decoder": "final decoder features are closest to pixel-level caries decisions and have only 16 channels, so large-kernel context is cheap",
        "why_before_rbsg": "B2's RBSG-lite can recalibrate the refined decoder feature using its established boundary/rim priors",
        "why_not_dynamic_skip": "B10 dynamic skip selection did not beat B2; B25 uses a fixed residual block instead of hard branch selection",
        "why_not_channel_only": "B21 channel-only IFE learned scale 0.0, so B25 uses spatial depthwise large kernels with near-identity residual initialization",
    }
    config["design_references"] = [
        {
            "title": "CMUNeXt-style large-kernel CNN modules for medical image segmentation",
            "adopted_idea": "large-kernel depthwise context with efficient inverted bottleneck",
        },
        {
            "title": "EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation",
            "adopted_idea": "decoder-side lightweight convolutional context can improve segmentation without replacing the encoder backbone",
        },
        {
            "title": "PRAD: A Benchmark Dataset for Periapical Radiograph Analysis and Diagnosis",
            "adopted_idea": "dental periapical radiographs favor lightweight CNN-compatible modules over full backbone replacement",
        },
    ]
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["lkdr"] = {
        "position": "final SMP decoder feature before pre-boundary/pre-rim heads and RBSG-lite",
        "hidden_channels": args.lkdr_hidden_channels,
        "expansion": args.lkdr_expansion,
        "local_kernel": args.lkdr_local_kernel,
        "context_kernel": args.lkdr_context_kernel,
        "context_dilation": args.lkdr_context_dilation,
        "init_scale": args.lkdr_init_scale,
        "max_scale": args.lkdr_max_scale,
        "final_projection_bn_gamma_init": 0.0,
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
    model = ResNet34UNetBRRB25LKDR(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        lkdr_hidden_channels=args.lkdr_hidden_channels,
        lkdr_expansion=args.lkdr_expansion,
        lkdr_local_kernel=args.lkdr_local_kernel,
        lkdr_context_kernel=args.lkdr_context_kernel,
        lkdr_context_dilation=args.lkdr_context_dilation,
        lkdr_init_scale=args.lkdr_init_scale,
        lkdr_max_scale=args.lkdr_max_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B25 LKDR + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "learned_lkdr_scale",
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
            f"val_recall={val_metrics['recall']:.4f} lkdr_scale={model.learned_lkdr_scale:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                model.learned_lkdr_scale,
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
        "learned_lkdr_scale": model.learned_lkdr_scale,
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

    parser.add_argument("--run-name", type=str, default="b25_lkdr_auglite_e200_constlr_bs6")
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

    parser.add_argument("--lkdr-hidden-channels", type=int, default=48)
    parser.add_argument("--lkdr-expansion", type=float, default=3.0)
    parser.add_argument("--lkdr-local-kernel", type=int, default=5)
    parser.add_argument("--lkdr-context-kernel", type=int, default=7)
    parser.add_argument("--lkdr-context-dilation", type=int, default=2)
    parser.add_argument("--lkdr-init-scale", type=float, default=0.05)
    parser.add_argument("--lkdr-max-scale", type=float, default=0.50)

    args = parser.parse_args()
    main(args)
