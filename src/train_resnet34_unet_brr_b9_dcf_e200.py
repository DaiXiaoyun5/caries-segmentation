import argparse
import csv
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

from train_resnet34_unet_brr_b1_aux_e200 import (
    run_one_epoch_b1,
)


class RBSGLite(nn.Module):
    """
    Rim-Boundary Skip Gate lite.

    First B2 version:
      - does not rewrite SMP decoder internals
      - applies boundary/rim-guided gate on final decoder feature
      - inference still uses only mask logits
    """
    def __init__(self, channels, lambda_pos=0.20, lambda_rim=0.10):
        super().__init__()
        self.lambda_pos = float(lambda_pos)
        self.lambda_rim = float(lambda_rim)

        self.pos_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.rim_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def feature_gradient(x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))
        g = (gx + gy).mean(dim=1, keepdim=True)

        g_min = g.amin(dim=(2, 3), keepdim=True)
        g_max = g.amax(dim=(2, 3), keepdim=True)
        return (g - g_min) / (g_max - g_min + 1e-6)

    def forward(self, feat, boundary_logits, rim_logits):
        avg_map = feat.mean(dim=1, keepdim=True)
        max_map = feat.max(dim=1, keepdim=True)[0]
        grad_map = self.feature_gradient(feat)

        # detach aux maps to avoid unstable early mutual feedback
        b_prob = torch.sigmoid(boundary_logits.detach())
        r_prob = torch.sigmoid(rim_logits.detach())

        pos = self.pos_gate(torch.cat([avg_map, max_map, b_prob, grad_map], dim=1))
        rim = self.rim_gate(torch.cat([avg_map, max_map, r_prob, grad_map], dim=1))

        feat_pos = feat * (1.0 + self.lambda_pos * pos)
        feat_rim = feat * (1.0 - self.lambda_rim * rim)

        return self.out(torch.cat([feat, feat_pos, feat_rim], dim=1))




class MultiScaleContextBottleneck(nn.Module):
    """
    Lightweight multi-scale context bottleneck.

    Same as B8B:
      - dilation=1 / 3 / 5 branches
      - global average pooling branch
      - residual fusion

    It adds semantic multi-scale context before the decoder.
    """
    def __init__(
        self,
        channels,
        branch_channels=128,
        dilations=(1, 3, 5),
        dropout=0.0,
        init_scale=0.10,
    ):
        super().__init__()

        channels = int(channels)
        branch_channels = int(branch_channels)
        self.dilations = tuple(int(d) for d in dilations)

        branches = []
        for d in self.dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        fuse_layers = [
            nn.Conv2d(
                branch_channels * (len(self.dilations) + 1),
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        if float(dropout) > 0:
            fuse_layers.append(nn.Dropout2d(float(dropout)))

        self.fuse = nn.Sequential(*fuse_layers)

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        size = x.shape[2:]

        outs = [branch(x) for branch in self.branches]

        gp = self.global_branch(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        outs.append(gp)

        context = self.fuse(torch.cat(outs, dim=1))
        return x + self.scale * context


class DetailAwareCrossLayerDecoderFusion(nn.Module):
    """
    DCF: Detail-aware Cross-Layer Decoder Fusion.

    Inputs:
      - final refined decoder feature after RBSG-lite
      - decoder 1/4 feature: more spatial detail
      - decoder 1/8 feature: local semantic + shape context

    Unlike MS-RBSG-2, DCF does NOT add boundary/rim gates at multiple scales.
    It only fuses decoder features, so it should not directly strengthen
    rim suppression or make the model overly conservative.

    Output:
      F_out = F_refined + beta * DCF(F_refined, F_1/4, F_1/8)
    """
    def __init__(
        self,
        final_channels,
        ch_1_8,
        ch_1_4,
        dcf_channels=16,
        beta=0.20,
    ):
        super().__init__()

        final_channels = int(final_channels)
        ch_1_8 = int(ch_1_8)
        ch_1_4 = int(ch_1_4)
        dcf_channels = int(dcf_channels)

        self.beta = float(beta)

        self.final_proj = nn.Sequential(
            nn.Conv2d(final_channels, dcf_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dcf_channels),
            nn.ReLU(inplace=True),
        )

        self.proj_1_8 = nn.Sequential(
            nn.Conv2d(ch_1_8, dcf_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dcf_channels),
            nn.ReLU(inplace=True),
        )

        self.proj_1_4 = nn.Sequential(
            nn.Conv2d(ch_1_4, dcf_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dcf_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(dcf_channels * 3, dcf_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dcf_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(dcf_channels * 2, dcf_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dcf_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(dcf_channels * 2, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    def forward(self, final_refined, feat_1_8, feat_1_4):
        target_size = final_refined.shape[2:]

        f_final = self.final_proj(final_refined)

        f_1_8 = self.proj_1_8(feat_1_8)
        f_1_8 = F.interpolate(
            f_1_8,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        f_1_4 = self.proj_1_4(feat_1_4)
        f_1_4 = F.interpolate(
            f_1_4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        residual = self.fuse(torch.cat([f_final, f_1_4, f_1_8], dim=1))
        return final_refined + self.beta * residual


class ResNet34UNetBRRB9DCF(nn.Module):
    """
    B9-DCF:
      B8B Context Bottleneck
      + final-stage RBSG-lite
      + Detail-aware Cross-Layer Decoder Fusion.

    Keeps:
      lambda_pos = 0.20
      lambda_rim = 0.10
      boundary rb = 3
      rim rr = 9
      original B2 boundary/rim auxiliary supervision
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        context_branch_channels=128,
        context_dropout=0.0,
        context_init_scale=0.10,
        dcf_channels=16,
        dcf_beta=0.20,
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

        self.context_bottleneck = MultiScaleContextBottleneck(
            channels=bottleneck_ch,
            branch_channels=int(context_branch_channels),
            dilations=(1, 3, 5),
            dropout=float(context_dropout),
            init_scale=float(context_init_scale),
        )

        decoder_channels = self._get_decoder_channels()
        if len(decoder_channels) < 3:
            # SMP default for resnet34 Unet decoder is usually [256, 128, 64, 32, 16]
            decoder_channels = [256, 128, 64, 32, dec_ch]

        ch_1_8 = decoder_channels[1]
        ch_1_4 = decoder_channels[2]

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.dcf = DetailAwareCrossLayerDecoderFusion(
            final_channels=dec_ch,
            ch_1_8=ch_1_8,
            ch_1_4=ch_1_4,
            dcf_channels=int(dcf_channels),
            beta=float(dcf_beta),
        )

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def _get_decoder_channels(self):
        channels = []

        for block in self.unet.decoder.blocks:
            out_ch = None

            conv2 = getattr(block, "conv2", None)
            if conv2 is not None:
                for m in conv2.modules():
                    if isinstance(m, nn.Conv2d):
                        out_ch = m.out_channels

            if out_ch is None:
                conv1 = getattr(block, "conv1", None)
                if conv1 is not None:
                    for m in conv1.modules():
                        if isinstance(m, nn.Conv2d):
                            out_ch = m.out_channels

            if out_ch is None:
                for m in reversed(list(block.modules())):
                    if isinstance(m, nn.Conv2d):
                        out_ch = m.out_channels
                        break

            if out_ch is not None:
                channels.append(int(out_ch))

        return channels

    @staticmethod
    def _run_decoder_block(block, x, skip):
        """
        Compatible with common SMP Unet decoder block signatures.
        """
        try:
            return block(x, skip)
        except TypeError:
            pass

        if skip is not None:
            height, width = skip.shape[2], skip.shape[3]
        else:
            height, width = x.shape[2] * 2, x.shape[3] * 2

        try:
            return block(x, height, width, skip)
        except TypeError:
            pass

        try:
            return block(x, height=height, width=width, skip_connection=skip)
        except TypeError as e:
            raise RuntimeError(
                "Cannot run SMP decoder block. Please send the smoke log to ChatGPT."
            ) from e

    def _decode_collect(self, features):
        """
        Manually run SMP Unet decoder so that we can collect 1/8 and 1/4 decoder features.

        SMP Unet decoder block index mapping for ResNet34-like U-Net:
          block 0: 1/16
          block 1: 1/8
          block 2: 1/4
          block 3: 1/2
          block 4: full
        """
        decoder = self.unet.decoder

        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = decoder.center(head)
        collected = {}

        for i, decoder_block in enumerate(decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = self._run_decoder_block(decoder_block, x, skip)

            if i == 1:
                collected["1_8"] = x
            elif i == 2:
                collected["1_4"] = x

        return x, collected

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))

        # Same as B8B: add multi-scale context on the deepest encoder feature.
        features[-1] = self.context_bottleneck(features[-1])

        decoder_output, collected = self._decode_collect(features)

        if "1_8" not in collected or "1_4" not in collected:
            raise RuntimeError("DCF could not collect decoder 1/8 and 1/4 features.")

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)

        # New B9 DCF: fuse final refined feature with decoder 1/8 and 1/4 detail features.
        fused = self.dcf(
            refined,
            collected["1_8"],
            collected["1_4"],
        )

        mask_logits = self.mask_head(fused)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(fused)
        rim_logits = self.rim_head(fused)

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
    config["model_class"] = "ResNet34UNetBRRB9DCF"
    config["experiment"] = "B9-DCF: B8B Context Bottleneck + Detail-aware Cross-Layer Decoder Fusion"
    config["note"] = "B9-DCF keeps B8B multi-scale context bottleneck and final RBSG-lite, and adds a lightweight detail-aware cross-layer decoder fusion module. DCF fuses decoder 1/8, decoder 1/4 and final refined features without adding extra boundary/rim gates, aiming to improve small-lesion detail while avoiding over-conservative rim suppression."
    config["aux_targets"] = {
        "boundary": "dilate(mask,3)-erode(mask,3)",
        "rim": "dilate(mask,9)-dilate(mask,3)"
    }
    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
    config["context_bottleneck"] = {
        "enabled": True,
        "position": "deepest encoder feature before decoder",
        "branches": ["dilation=1", "dilation=3", "dilation=5", "global average pooling"],
        "branch_channels": args.context_branch_channels,
        "dropout": args.context_dropout,
        "init_scale": args.context_init_scale,
        "fusion": "residual: x + learnable_scale * context(x)"
    }
    config["dcf"] = {
        "enabled": True,
        "position": "after final RBSG-lite before output heads",
        "features": ["decoder_1/8", "decoder_1/4", "final_refined"],
        "dcf_channels": args.dcf_channels,
        "dcf_beta": args.dcf_beta,
        "fusion": "F_out = F_refined + beta * DCF(F_refined, F_1/4, F_1/8)",
        "note": "DCF fuses multi-level decoder features without additional boundary/rim suppression."
    }
    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim"
    }
    config["augmentation"] = {
        "type": "online",
        "profile": args.augment_profile,
        "train_only": True,
        "geometry": {
            "rotation_deg": [-15.0, 15.0],
            "scale": [0.85, 1.20],
            "translate_ratio": 0.05,
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.0
        },
        "photometric": {
            "brightness_factor": [0.85, 1.15],
            "contrast_factor": [0.85, 1.15],
            "gamma": [0.80, 1.25],
            "gaussian_noise_p": 0.10,
            "noise_std": [0.005, 0.020],
            "gaussian_blur_p": 0.10,
            "blur_radius": [0.2, 0.8]
        }
    }
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

    model = ResNet34UNetBRRB9DCF(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        context_branch_channels=args.context_branch_channels,
        context_dropout=args.context_dropout,
        context_init_scale=args.context_init_scale,
        dcf_channels=args.dcf_channels,
        dcf_beta=args.dcf_beta,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B9-DCF BRR-RBSG Context + Decoder Fusion ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("context_branch_channels =", args.context_branch_channels)
    print("context_dropout         =", args.context_dropout)
    print("context_init_scale      =", args.context_init_scale)
    print("dcf_channels            =", args.dcf_channels)
    print("dcf_beta                =", args.dcf_beta)
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

    parser.add_argument("--run-name", type=str, default="b9_dcf_b8b_context_auglite_e200_constlr_bs6")
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

    parser.add_argument("--context-branch-channels", type=int, default=128)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--context-init-scale", type=float, default=0.10)

    parser.add_argument("--dcf-channels", type=int, default=16)
    parser.add_argument("--dcf-beta", type=float, default=0.20)

    args = parser.parse_args()
    main(args)
