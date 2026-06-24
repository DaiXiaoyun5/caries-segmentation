import argparse
import csv
import math
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


def _scale_to_logit(scale: float, max_delta: float) -> float:
    ratio = max(min(float(scale) / max(float(max_delta), 1e-6), 0.999), 1e-4)
    return math.log(ratio / (1.0 - ratio))


def _valid_group_count(channels: int, max_groups: int = 4) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_valid_group_count(out_channels),
                num_channels=out_channels,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EdgeSimAMRescueHead(nn.Module):
    """
    Edge-SimAM Detail Rescue head.

    Motivation comes from this project's previous UNet3+ evidence: Edge+SimAM
    improved hard/small lesion recall. Here it is adapted to B2 safely:
      - image Sobel and local contrast provide low-level dental X-ray detail;
      - SimAM-weighted pre-RBSG decoder feature provides semantic context;
      - the branch predicts a bounded non-negative logit residual.

    Therefore it can raise weak caries responses but cannot suppress B2 logits.
    """
    def __init__(
        self,
        decoder_channels,
        hidden_channels=16,
        max_delta=0.60,
        init_scale=0.10,
        bias_init=-4.0,
        contrast_kernel=7,
        simam_lambda=1e-4,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.contrast_kernel = int(contrast_kernel)
        self.simam_lambda = float(simam_lambda)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.edge_stem = nn.Sequential(
            ConvGNAct(3, hidden_channels, kernel_size=3),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )
        self.context_stem = nn.Sequential(
            ConvGNAct(decoder_channels, hidden_channels, kernel_size=3),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(hidden_channels * 2, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        final_conv = self.fuse[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, float(bias_init))

        self.scale_logit = nn.Parameter(
            torch.tensor(_scale_to_logit(init_scale, self.max_delta), dtype=torch.float32)
        )

    def simam_attention(self, x):
        b, c, h, w = x.shape
        n = max(h * w - 1, 1)
        mean = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mean).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / n
        e_inv = d / (4.0 * (v + self.simam_lambda)) + 0.5
        return torch.sigmoid(e_inv)

    def normalize_map(self, x):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def edge_inputs(self, image):
        gray = image.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        grad = self.normalize_map(grad)

        k = self.contrast_kernel
        if k > 1:
            local_mean = F.avg_pool2d(
                gray,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                count_include_pad=False,
            )
            contrast = self.normalize_map(torch.abs(gray - local_mean))
        else:
            contrast = torch.zeros_like(gray)

        return torch.cat([gray, grad, contrast], dim=1)

    def forward(self, image, decoder_feature):
        edge_feature = self.edge_stem(self.edge_inputs(image))
        attended_decoder = decoder_feature * self.simam_attention(decoder_feature)
        context_feature = self.context_stem(attended_decoder)
        raw_delta = self.fuse(torch.cat([edge_feature, context_feature], dim=1))
        scale = self.max_delta * torch.sigmoid(self.scale_logit)
        return scale * torch.sigmoid(raw_delta)


class ResNet34UNetBRRB15EdgeSimAMRescue(ResNet34UNetBRRB2RBSG):
    """
    B15: B2 + Edge-SimAM Detail Rescue.

    This is a local-detail complement to B14:
      B14 tests regional context; B15 tests explicit image-edge/contrast detail.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        esdr_hidden_channels=16,
        esdr_max_delta=0.60,
        esdr_init_scale=0.10,
        esdr_bias_init=-4.0,
        esdr_contrast_kernel=7,
        esdr_simam_lambda=1e-4,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels
        self.esdr = EdgeSimAMRescueHead(
            decoder_channels=dec_ch,
            hidden_channels=esdr_hidden_channels,
            max_delta=esdr_max_delta,
            init_scale=esdr_init_scale,
            bias_init=esdr_bias_init,
            contrast_kernel=esdr_contrast_kernel,
            simam_lambda=esdr_simam_lambda,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)
        refined = self.rbsg(decoder_output, b_pre, r_pre)

        b2_mask_logits = self.mask_head(refined)
        mask_logits = b2_mask_logits + self.esdr(x, decoder_output)

        if not return_aux:
            return mask_logits

        return {
            "mask": mask_logits,
            "boundary": self.boundary_head(refined),
            "rim": self.rim_head(refined),
        }

    @property
    def learned_esdr_scale(self):
        return float((self.esdr.max_delta * torch.sigmoid(self.esdr.scale_logit)).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB15EdgeSimAMRescue"
    config["experiment"] = "B15: B2 + Edge-SimAM Detail Rescue"
    config["note"] = (
        "B15 keeps B2 unchanged and adds an image-edge/local-contrast + SimAM "
        "detail branch that predicts a bounded non-negative logit residual. It "
        "is motivated by this project's earlier Edge+SimAM gains on hard/small "
        "groups and by the B10/B11 observation that selective modules tend to "
        "hurt recall."
    )
    config["dataset_observation"] = {
        "test_cases": 358,
        "small_group_cases": 119,
        "small_area_ratio_mean_512": 0.011539875960149685,
        "hard_group_cases": 119,
        "older_edge_simam_hard_recall_delta_vs_resnet34_unet3p": 0.10910301183119309,
        "implication": "explicit image-edge/detail evidence is worth retesting on top of B2, but as a positive rescue residual rather than a suppressive attention gate",
    }
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["edge_simam_rescue"] = {
        "position": "parallel non-negative residual added to final B2 mask logits",
        "inputs": ["grayscale image", "Sobel magnitude", "local contrast", "SimAM-weighted pre-RBSG decoder feature"],
        "hidden_channels": args.esdr_hidden_channels,
        "max_delta_logit": args.esdr_max_delta,
        "initial_learned_scale": args.esdr_init_scale,
        "head_bias_init": args.esdr_bias_init,
        "contrast_kernel": args.esdr_contrast_kernel,
        "simam_lambda": args.esdr_simam_lambda,
        "uses_boundary_or_rim": False,
        "extra_supervision": False,
        "non_negative_residual": True,
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
    model = ResNet34UNetBRRB15EdgeSimAMRescue(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        esdr_hidden_channels=args.esdr_hidden_channels,
        esdr_max_delta=args.esdr_max_delta,
        esdr_init_scale=args.esdr_init_scale,
        esdr_bias_init=args.esdr_bias_init,
        esdr_contrast_kernel=args.esdr_contrast_kernel,
        esdr_simam_lambda=args.esdr_simam_lambda,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B15 = B2 BRR-RBSG + Edge-SimAM Detail Rescue")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("esdr_scale_init =", args.esdr_init_scale)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "esdr_scale",
        ])

    best_val_dice = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)
        esdr_scale = model.learned_esdr_scale

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} esdr_scale={esdr_scale:.4f}"
        )
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                esdr_scale,
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
        "esdr_parameters": sum(p.numel() for p in model.esdr.parameters()),
        "learned_esdr_scale": model.learned_esdr_scale,
    }

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="b15_edge_simam_rescue_auglite_e200_constlr_bs6")
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
    parser.add_argument("--esdr-hidden-channels", type=int, default=16)
    parser.add_argument("--esdr-max-delta", type=float, default=0.60)
    parser.add_argument("--esdr-init-scale", type=float, default=0.10)
    parser.add_argument("--esdr-bias-init", type=float, default=-4.0)
    parser.add_argument("--esdr-contrast-kernel", type=int, default=7)
    parser.add_argument("--esdr-simam-lambda", type=float, default=1e-4)
    main(parser.parse_args())
