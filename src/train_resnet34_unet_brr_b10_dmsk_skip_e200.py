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


class DynamicDepthwiseKernelBank(nn.Module):
    """
    Per-image dynamic selection from a depthwise convolution kernel bank.
    """

    def __init__(self, channels, kernel_sizes, dilation=1, hard_selection=True):
        super().__init__()
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.dilation = int(dilation)
        self.hard_selection = bool(hard_selection)

        self.branches = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=(kernel_size // 2) * self.dilation,
                dilation=self.dilation,
                groups=channels,
                bias=False,
            )
            for kernel_size in self.kernel_sizes
        ])
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, len(self.kernel_sizes), kernel_size=1, bias=True),
        )

    def forward(self, x):
        scores = self.selector(x).flatten(1)
        probabilities = torch.softmax(scores, dim=1)

        if self.hard_selection:
            indices = probabilities.argmax(dim=1, keepdim=True)
            hard_weights = torch.zeros_like(probabilities).scatter_(
                1,
                indices,
                1.0,
            )
            # Hard branch selection in forward, softmax gradient in backward.
            weights = hard_weights - probabilities.detach() + probabilities
        else:
            weights = probabilities

        output = torch.zeros_like(x)
        for branch_index, branch in enumerate(self.branches):
            weight = weights[:, branch_index].view(-1, 1, 1, 1)
            output = output + weight * branch(x)

        return output, weights


class DMSKSkipLite(nn.Module):
    """
    Dynamic Multi-Scale Kernel skip enhancement for the encoder 1/4 feature.

    The module is inspired by the DMSK component of Dynamic Skip Connections
    (Medical Image Analysis, 2026, DOI: 10.1016/j.media.2026.104010).

    Compared with the full DSC block:
      - no test-time-training dependency is introduced;
      - only the 1/4 ResNet34 skip is enhanced;
      - the output is injected through a small learnable residual scale.

    Local bank:   3x3 / 5x5 depthwise kernels.
    Context bank: 7x7 / 9x9 / 11x11 depthwise kernels with dilation 3.
    """

    def __init__(
        self,
        channels,
        hidden_channels=None,
        local_kernel_sizes=(3, 5),
        context_kernel_sizes=(7, 9, 11),
        context_dilation=3,
        init_scale=0.10,
        hard_selection=True,
    ):
        super().__init__()

        channels = int(channels)
        hidden_channels = (
            max(channels // 2, 16)
            if hidden_channels is None
            else int(hidden_channels)
        )

        self.input_projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.local_bank = DynamicDepthwiseKernelBank(
            hidden_channels,
            kernel_sizes=local_kernel_sizes,
            dilation=1,
            hard_selection=hard_selection,
        )
        self.context_bank = DynamicDepthwiseKernelBank(
            hidden_channels,
            kernel_sizes=context_kernel_sizes,
            dilation=context_dilation,
            hard_selection=hard_selection,
        )

        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Start exactly from the original B2 mapping. The residual branch is
        # enabled gradually as the final BN scale learns away from zero.
        nn.init.zeros_(self.output_projection[-1].weight)

        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

        # Exposed for optional post-hoc interpretability; not used in the loss.
        self.latest_local_weights = None
        self.latest_context_weights = None

    def forward(self, x):
        projected = self.input_projection(x)
        local_feature, local_weights = self.local_bank(projected)
        context_feature, context_weights = self.context_bank(local_feature)

        merged = torch.cat([local_feature, context_feature], dim=1)
        avg_map = merged.mean(dim=1, keepdim=True)
        max_map = merged.amax(dim=1, keepdim=True)
        spatial_weights = self.spatial_mixer(torch.cat([avg_map, max_map], dim=1))

        weighted = torch.cat([
            local_feature * spatial_weights[:, 0:1],
            context_feature * spatial_weights[:, 1:2],
        ], dim=1)
        residual = self.output_projection(weighted)

        # Keep detached choices for analysis without retaining the graph.
        self.latest_local_weights = local_weights.detach()
        self.latest_context_weights = context_weights.detach()

        scale = torch.clamp(self.residual_scale, min=0.0, max=1.0)
        return x + scale * residual


class ResNet34UNetBRRB10DMSKSkip(ResNet34UNetBRRB2RBSG):
    """
    B10: original B2 + one DMSK-Skip-lite module.

    The DMSK module is applied only to encoder feature[2], the 1/4-resolution
    ResNet34 skip feature (64 channels for the standard encoder).

    Everything else is inherited from B2:
      - original SMP ResNet34-U-Net encoder/decoder;
      - boundary/rim auxiliary heads;
      - final RBSG-lite;
      - rb=3, rr=9;
      - lambda_pos=0.20, lambda_rim=0.10;
      - original two-stage auxiliary loss schedule.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        dmsk_hidden_channels=32,
        dmsk_context_dilation=3,
        dmsk_init_scale=0.10,
        dmsk_hard_selection=True,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        encoder_channels = list(self.unet.encoder.out_channels)
        if len(encoder_channels) <= 2:
            raise RuntimeError(
                "B10 requires the encoder 1/4 skip at feature index 2. "
                f"Detected encoder channels: {encoder_channels}"
            )

        skip_channels = int(encoder_channels[2])
        self.dmsk_skip = DMSKSkipLite(
            channels=skip_channels,
            hidden_channels=dmsk_hidden_channels,
            local_kernel_sizes=(3, 5),
            context_kernel_sizes=(7, 9, 11),
            context_dilation=dmsk_context_dilation,
            init_scale=dmsk_init_scale,
            hard_selection=dmsk_hard_selection,
        )

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))

        # The sole B10 architecture change: content-adaptive scale selection
        # on the high-resolution encoder skip before the original B2 decoder.
        features[2] = self.dmsk_skip(features[2])

        decoder_output = self.unet.decoder(*features)
        boundary_pre = self.boundary_head_pre(decoder_output)
        rim_pre = self.rim_head_pre(decoder_output)
        refined = self.rbsg(decoder_output, boundary_pre, rim_pre)

        mask_logits = self.mask_head(refined)
        if not return_aux:
            return mask_logits

        return {
            "mask": mask_logits,
            "boundary": self.boundary_head(refined),
            "rim": self.rim_head(refined),
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
    config["model_class"] = "ResNet34UNetBRRB10DMSKSkip"
    config["experiment"] = "B10: B2 + DMSK-Skip-lite at encoder 1/4 feature"
    config["note"] = (
        "B10 keeps the complete B2 framework and adds one morphology/scale-adaptive "
        "dynamic kernel module to the ResNet34 1/4 skip feature. It introduces no "
        "new boundary/rim supervision, no additional segmentation loss, and no "
        "multi-scale RBSG gate."
    )
    config["design_references"] = [
        {
            "title": "Enhancing feature fusion of U-like networks with dynamic skip connections",
            "journal": "Medical Image Analysis",
            "year": 2026,
            "doi": "10.1016/j.media.2026.104010",
            "adopted_idea": "content-adaptive dynamic multi-scale kernel selection",
        },
        {
            "title": "I2U-Net: A dual-path U-Net with rich information interaction",
            "journal": "Medical Image Analysis",
            "year": 2024,
            "doi": "10.1016/j.media.2024.103241",
            "adopted_idea": "reuse high-resolution detail and semantic information in U-like networks",
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
    config["dmsk_skip"] = {
        "position": "ResNet34 encoder feature[2], 1/4-resolution skip",
        "local_kernel_sizes": [3, 5],
        "context_kernel_sizes": [7, 9, 11],
        "context_dilation": args.dmsk_context_dilation,
        "hidden_channels": args.dmsk_hidden_channels,
        "initial_residual_scale": args.dmsk_init_scale,
        "residual_projection_init": "final BatchNorm scale initialized to zero",
        "hard_selection": bool(args.dmsk_hard_selection),
        "extra_supervision": False,
        "uses_boundary_or_rim": False,
    }
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        profile=args.augment_profile,
    )
    train_eval_ds = CariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
    )
    val_ds = CariesMaskDataset(
        val_split,
        target_size=(args.image_size, args.image_size),
    )
    test_ds = CariesMaskDataset(
        test_split,
        target_size=(args.image_size, args.image_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
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
    model = ResNet34UNetBRRB10DMSKSkip(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        dmsk_hidden_channels=args.dmsk_hidden_channels,
        dmsk_context_dilation=args.dmsk_context_dilation,
        dmsk_init_scale=args.dmsk_init_scale,
        dmsk_hard_selection=bool(args.dmsk_hard_selection),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=" * 80)
    print("Model ready: B10 = B2 + DMSK-Skip-lite")
    print("run_name               =", args.run_name)
    print("device                 =", device)
    print("augment_profile        =", args.augment_profile)
    print("batch_size             =", args.batch_size)
    print("epochs                 =", args.epochs)
    print("lr                     =", args.lr)
    print("dmsk_hidden_channels   =", args.dmsk_hidden_channels)
    print("dmsk_context_dilation  =", args.dmsk_context_dilation)
    print("dmsk_init_scale        =", args.dmsk_init_scale)
    print("dmsk_hard_selection    =", bool(args.dmsk_hard_selection))
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b1(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
        )
        val_metrics = run_one_epoch_b1(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
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

        with open(history_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["dice"],
                train_metrics["iou"],
                train_metrics["precision"],
                train_metrics["recall"],
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(
                f"  -> best model updated, epoch={best_epoch}, "
                f"val_dice={best_val_dice:.4f}"
            )

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    best_train_metrics = run_one_epoch_b1(
        model,
        train_eval_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )
    best_val_metrics = run_one_epoch_b1(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )
    test_metrics = run_one_epoch_b1(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "dmsk_parameters": sum(
            p.numel() for p in model.dmsk_skip.parameters()
        ),
        "learned_dmsk_residual_scale": float(
            torch.clamp(
                model.dmsk_skip.residual_scale.detach().cpu(),
                min=0.0,
                max=1.0,
            )
        ),
    }

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(
        model,
        test_loader,
        device,
        pred_dir / "test_preview.png",
    )

    print("=" * 80)
    print("Best checkpoint summary")
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        type=str,
        default="b10_dmsk_skip_auglite_e200_constlr_bs6",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--augment-profile",
        type=str,
        default="lite",
        choices=["lite", "strong", "none"],
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--dmsk-hidden-channels", type=int, default=32)
    parser.add_argument("--dmsk-context-dilation", type=int, default=3)
    parser.add_argument("--dmsk-init-scale", type=float, default=0.10)
    parser.add_argument("--dmsk-hard-selection", type=int, default=1, choices=[0, 1])

    main(parser.parse_args())
