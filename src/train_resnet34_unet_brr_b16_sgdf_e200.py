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


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SemanticGuidedDetailFusion(nn.Module):
    """
    Semantic-Guided Detail Fusion (SGDF).

    B10's dynamic skip tried to select multi-scale kernels on a shallow skip and
    became less recall-friendly. SGDF keeps the skip path deterministic: it
    injects a small residual computed from shallow detail + deepest semantic
    context, with zero-initialized output so the model starts exactly as B2.
    """
    def __init__(self, skip_channels, semantic_channels, hidden_channels=32, init_scale=0.05):
        super().__init__()
        self.detail_proj = nn.Sequential(
            ConvGNAct(skip_channels, hidden_channels, kernel_size=3),
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
        )
        self.semantic_proj = nn.Sequential(
            ConvGNAct(semantic_channels, hidden_channels, kernel_size=1),
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
        )
        self.out = nn.Conv2d(hidden_channels * 3, skip_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, skip, semantic):
        semantic = F.interpolate(
            semantic,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail_feat = self.detail_proj(skip)
        semantic_feat = self.semantic_proj(semantic)
        fused = torch.cat(
            [
                detail_feat,
                semantic_feat,
                detail_feat * torch.sigmoid(semantic_feat),
            ],
            dim=1,
        )
        delta = self.out(fused)
        return skip + torch.clamp(self.scale, 0.0, 0.50) * delta


class ResNet34UNetBRRB16SGDF(ResNet34UNetBRRB2RBSG):
    """
    B16: B2 + Semantic-Guided Detail Fusion on shallow skips.

    It is designed for sparse, low-contrast caries masks:
      - shallow skips preserve fine dental X-ray texture;
      - deepest features provide lesion-level semantic context;
      - zero-init residual keeps the starting network identical to B2.

    This is a skip interaction module, not another boundary/rim gate.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        sgdf_hidden_channels=32,
        sgdf_init_scale=0.05,
        sgdf_use_skip1=True,
        sgdf_use_skip2=True,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        enc_channels = list(self.unet.encoder.out_channels)
        deep_channels = enc_channels[-1]
        self.sgdf_use_skip1 = bool(sgdf_use_skip1)
        self.sgdf_use_skip2 = bool(sgdf_use_skip2)

        self.sgdf_skip1 = SemanticGuidedDetailFusion(
            skip_channels=enc_channels[1],
            semantic_channels=deep_channels,
            hidden_channels=sgdf_hidden_channels,
            init_scale=sgdf_init_scale,
        ) if self.sgdf_use_skip1 else nn.Identity()
        self.sgdf_skip2 = SemanticGuidedDetailFusion(
            skip_channels=enc_channels[2],
            semantic_channels=deep_channels,
            hidden_channels=sgdf_hidden_channels,
            init_scale=sgdf_init_scale,
        ) if self.sgdf_use_skip2 else nn.Identity()

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        semantic = features[-1]
        if self.sgdf_use_skip1:
            features[1] = self.sgdf_skip1(features[1], semantic)
        if self.sgdf_use_skip2:
            features[2] = self.sgdf_skip2(features[2], semantic)

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
    def learned_sgdf_scales(self):
        scales = {}
        if self.sgdf_use_skip1:
            scales["skip1"] = float(torch.clamp(self.sgdf_skip1.scale, 0.0, 0.50).detach().cpu())
        if self.sgdf_use_skip2:
            scales["skip2"] = float(torch.clamp(self.sgdf_skip2.scale, 0.0, 0.50).detach().cpu())
        return scales


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
    config["model_class"] = "ResNet34UNetBRRB16SGDF"
    config["experiment"] = "B16: B2 + Semantic-Guided Detail Fusion"
    config["note"] = (
        "B16 keeps B2 unchanged and injects deepest semantic context into shallow "
        "encoder skips through zero-initialized residual fusion. It is motivated "
        "by recent U-like skip-fusion literature and by the project's observation "
        "that dynamic/attention modules often hurt recall."
    )
    config["dataset_observation"] = {
        "test_cases": 358,
        "small_group_cases": 119,
        "small_area_ratio_mean_512": 0.011539875960149685,
        "overall_area_ratio_mean_512": 0.03290683597159785,
        "implication": "small sparse lesions need high-resolution detail, but shallow detail should be constrained by lesion-level semantics",
    }
    config["design_references"] = [
        {
            "title": "Enhancing Feature Fusion of U-like Networks with Dynamic Skip Connections",
            "year": 2025,
            "adopted_idea": "skip connections should interact semantics and spatial details rather than pass static features",
        },
        {
            "title": "PRAD: Periapical Radiograph Analysis Dataset and Benchmark Model Development",
            "year": 2025,
            "adopted_idea": "periapical radiographs retain detailed local lesions but are affected by resolution limits and artifacts",
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
    config["sgdf"] = {
        "position": "ResNet34 encoder skip features[1] and features[2] before SMP decoder",
        "hidden_channels": args.sgdf_hidden_channels,
        "init_scale": args.sgdf_init_scale,
        "use_skip1": bool(args.sgdf_use_skip1),
        "use_skip2": bool(args.sgdf_use_skip2),
        "output_init": "zero conv, so initial network is exactly B2",
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
    model = ResNet34UNetBRRB16SGDF(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        sgdf_hidden_channels=args.sgdf_hidden_channels,
        sgdf_init_scale=args.sgdf_init_scale,
        sgdf_use_skip1=bool(args.sgdf_use_skip1),
        sgdf_use_skip2=bool(args.sgdf_use_skip2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B16 = B2 BRR-RBSG + Semantic-Guided Detail Fusion")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("sgdf_hidden     =", args.sgdf_hidden_channels)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "sgdf_skip1_scale", "sgdf_skip2_scale",
        ])

    best_val_dice = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)
        scales = model.learned_sgdf_scales

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} sgdf={scales}"
        )
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                scales.get("skip1", 0.0), scales.get("skip2", 0.0),
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
        "sgdf_parameters": sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("sgdf_")
        ),
        "learned_sgdf_scales": model.learned_sgdf_scales,
    }
    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="b16_sgdf_auglite_e200_constlr_bs6")
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
    parser.add_argument("--sgdf-hidden-channels", type=int, default=32)
    parser.add_argument("--sgdf-init-scale", type=float, default=0.05)
    parser.add_argument("--sgdf-use-skip1", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sgdf-use-skip2", type=int, default=1, choices=[0, 1])
    main(parser.parse_args())
