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


class CEMIALiteFinalGate(nn.Module):
    """
    Final-stage boundary + uncertainty gate.

    This is a conservative CEMIA-inspired block for the final B2 feature:
    - use pre-boundary to locate weak boundary regions;
    - use pre-mask uncertainty to emphasize ambiguous lesion pixels;
    - do not add rim suppression;
    - do not add new loss terms.
    """
    def __init__(self, alpha=0.10, beta=0.10):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, refined_feature, boundary_pre_logits, mask_pre_logits):
        f_mean = refined_feature.mean(dim=1, keepdim=True)
        f_max = refined_feature.amax(dim=1, keepdim=True)

        boundary_pre = torch.sigmoid(boundary_pre_logits)
        mask_pre = torch.sigmoid(mask_pre_logits)
        uncertainty_pre = 4.0 * mask_pre * (1.0 - mask_pre)

        cue = torch.cat(
            [
                f_mean,
                f_max,
                boundary_pre,
                uncertainty_pre,
                boundary_pre * uncertainty_pre,
            ],
            dim=1,
        )
        spatial = self.spatial_gate(cue)
        gain = 1.0 + self.alpha * boundary_pre + self.beta * boundary_pre * uncertainty_pre
        return refined_feature + refined_feature * spatial * gain


class ResNet34UNetBRRB26CEMIALite(ResNet34UNetBRRB2RBSG):
    """
    B26: B2 + CEMIA-lite final-stage boundary/uncertainty gate.

    B2 components remain unchanged:
      boundary_head_pre, rim_head_pre, RBSG-lite, mask/boundary/rim heads.

    New:
      mask_head_pre from raw decoder feature for uncertainty estimation only.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cemia_alpha=0.10,
        cemia_beta=0.10,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        seg_conv = self.unet.segmentation_head[0]
        dec_ch = int(seg_conv.in_channels)
        self.mask_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.cemia_lite = CEMIALiteFinalGate(
            alpha=cemia_alpha,
            beta=cemia_beta,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)
        m_pre = self.mask_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)
        refined_cemia = self.cemia_lite(refined, b_pre, m_pre)

        mask_logits = self.mask_head(refined_cemia)
        if not return_aux:
            return mask_logits

        return {
            "mask": mask_logits,
            "boundary": self.boundary_head(refined_cemia),
            "rim": self.rim_head(refined_cemia),
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
            "mask_pre": m_pre,
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
    config["model_class"] = "ResNet34UNetBRRB26CEMIALite"
    config["experiment"] = "B26: B2 + CEMIA-lite final-stage boundary/uncertainty gate"
    config["note"] = (
        "B26 keeps B2 boundary/rim auxiliary supervision and RBSG-lite unchanged. "
        "It adds a final-stage CEMIA-inspired gate after RBSG-lite and before the "
        "final mask/boundary/rim heads. The gate uses pre-boundary and pre-mask "
        "uncertainty cues only; it does not add pre-mask supervision, rim "
        "suppression, a second decoder, or a new loss."
    )
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["cemia_lite"] = {
        "position": "after RBSG-lite, before final mask/boundary/rim heads",
        "inputs": "RBSG-refined feature, pre-boundary logits, pre-mask logits",
        "uncertainty": "U_pre = 4 * sigmoid(mask_pre) * (1 - sigmoid(mask_pre))",
        "spatial_gate": "sigmoid(Conv7x7([f_mean, f_max, B_pre, U_pre, B_pre*U_pre]))",
        "gain": "1 + alpha*B_pre + beta*B_pre*U_pre",
        "alpha": args.cemia_alpha,
        "beta": args.cemia_beta,
        "extra_loss": False,
        "extra_rim_suppression": False,
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNetBRRB26CEMIALite(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cemia_alpha=args.cemia_alpha,
        cemia_beta=args.cemia_beta,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B26 CEMIA-lite + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("cemia_alpha     =", args.cemia_alpha)
    print("cemia_beta      =", args.cemia_beta)
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

    parser.add_argument("--run-name", type=str, default="b26_cemia_lite_auglite_e200_constlr_bs6")
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

    parser.add_argument("--cemia-alpha", type=float, default=0.10)
    parser.add_argument("--cemia-beta", type=float, default=0.10)

    args = parser.parse_args()
    main(args)
