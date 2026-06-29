import argparse
import csv
import math
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


class IFELite(nn.Module):
    """
    Instructive Feature Enhancement lite.

    This module is deliberately channel-oriented rather than another spatial
    attention/gating block. It estimates which final decoder channels carry
    useful local texture variation and spatially concentrated information, then
    applies a small learnable residual amplification.

    The statistics are computed from detached features to keep the branch cheap
    and avoid unstable feedback. Gradients still flow through the enhanced
    feature and the learnable residual scale.
    """
    def __init__(
        self,
        init_scale=0.0,
        max_scale=0.30,
        curvature_weight=0.60,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.max_scale = float(max_scale)
        self.curvature_weight = float(curvature_weight)

    @staticmethod
    def _normalize_channel_score(score):
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True, unbiased=False)
        return (score - mean) / (std + 1e-6)

    @staticmethod
    def _local_curvature_score(x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
        return (gx + gy).mean(dim=(2, 3))

    @staticmethod
    def _entropy_concentration_score(x):
        flat = torch.abs(x).flatten(2)
        prob = torch.softmax(flat, dim=2)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=2)
        entropy = entropy / max(math.log(max(flat.shape[-1], 2)), 1e-6)
        return 1.0 - entropy

    def forward(self, x):
        stat_x = x.detach()
        curvature = self._normalize_channel_score(
            self._local_curvature_score(stat_x)
        )
        concentration = self._normalize_channel_score(
            self._entropy_concentration_score(stat_x)
        )
        score = (
            self.curvature_weight * curvature
            + (1.0 - self.curvature_weight) * concentration
        )
        gate = torch.sigmoid(score).unsqueeze(-1).unsqueeze(-1)
        scale = torch.clamp(self.scale, min=0.0, max=self.max_scale)
        return x + scale * x * gate


class ResNet34UNetBRRB21IFELite(ResNet34UNetBRRB2RBSG):
    """
    B21: B2 + IFE-lite channel feature enhancement.

    B2 already models boundary/rim priors. B21 therefore avoids adding another
    boundary/rim or multi-scale spatial gate. It only recalibrates final decoder
    channels before B2's pre-boundary/pre-rim heads and RBSG-lite.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        ife_init_scale=0.0,
        ife_max_scale=0.30,
        ife_curvature_weight=0.60,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        self.ife_lite = IFELite(
            init_scale=ife_init_scale,
            max_scale=ife_max_scale,
            curvature_weight=ife_curvature_weight,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        decoder_output = self.ife_lite(decoder_output)

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
    def learned_ife_scale(self):
        return float(torch.clamp(self.ife_lite.scale, 0.0, self.ife_lite.max_scale).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB21IFELite"
    config["experiment"] = "B21: B2 + IFE-lite local-curvature/global-entropy channel enhancement"
    config["note"] = (
        "B21 keeps the B2 auxiliary losses and RBSG-lite unchanged, then inserts "
        "a lightweight IFE-style channel enhancement on the final decoder feature. "
        "It is designed for low-contrast X-ray texture discrimination and avoids "
        "adding more boundary/rim supervision, rim suppression, multi-scale gates, "
        "or post-mask residual rescue."
    )
    config["failure_reflection"] = {
        "boundary_rim_reinforcement": "pre-rim, stronger rim/gate, and consistency-style variants tended to raise precision but lower recall",
        "context_or_rescue_modules": "PPM/detail/wavelet-style additions did not consistently beat B2 on held-out test results",
        "design_implication": "try a near-identity channel-only enhancement that complements B2's spatial prior instead of competing with it",
    }
    config["design_references"] = [
        {
            "title": "Instructive Feature Enhancement for Dichotomous Medical Image Segmentation",
            "year": 2023,
            "adopted_idea": "use local curvature and global entropy/information cues to enhance discriminative medical-image features",
        },
        {
            "title": "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation",
            "year": 2024,
            "adopted_idea": "make conservative architecture changes on top of a strong U-Net baseline",
        },
        {
            "title": "When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation",
            "year": 2025,
            "adopted_idea": "favor lightweight CNN-compatible modules for dental caries radiographs",
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
    config["ife_lite"] = {
        "position": "final decoder feature before B2 pre-boundary/pre-rim heads and RBSG-lite",
        "local_score": "mean absolute first-order feature differences as a curvature/texture proxy",
        "global_score": "1 - normalized spatial entropy of absolute channel activations",
        "mode": "channel-only residual amplification; detached statistics; no extra spatial attention map",
        "init_scale": args.ife_init_scale,
        "max_scale": args.ife_max_scale,
        "curvature_weight": args.ife_curvature_weight,
    }
    config["augmentation"] = {
        "type": "online",
        "profile": args.augment_profile,
        "train_only": True,
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
    model = ResNet34UNetBRRB21IFELite(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        ife_init_scale=args.ife_init_scale,
        ife_max_scale=args.ife_max_scale,
        ife_curvature_weight=args.ife_curvature_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=" * 80)
    print("Model ready: B21 IFE-lite + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("ife_scale       =", args.ife_init_scale)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "learned_ife_scale",
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
            f"val_recall={val_metrics['recall']:.4f} "
            f"ife_scale={model.learned_ife_scale:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                model.learned_ife_scale,
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
        "learned_ife_scale": model.learned_ife_scale,
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

    parser.add_argument("--run-name", type=str, default="b21_ife_lite_auglite_e200_constlr_bs6")
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

    parser.add_argument("--ife-init-scale", type=float, default=0.0)
    parser.add_argument("--ife-max-scale", type=float, default=0.30)
    parser.add_argument("--ife-curvature-weight", type=float, default=0.60)

    args = parser.parse_args()
    main(args)
