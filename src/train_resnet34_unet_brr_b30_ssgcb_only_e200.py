import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    run_one_epoch,
    save_prediction_panel,
)

from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset
from train_resnet34_unet_brr_b30_ssgcb_light_e200 import (
    SelectiveSpatialGatedContextBottleneck,
    build_augmentation_config,
    parse_dilations,
)


class ResNet34UNetB30SSGCBOnly(nn.Module):
    """
    B30 ablation: plain ResNet34-U-Net + SSGCB only.

    This isolates the SSGCB bottleneck context module from the full B30-light
    pipeline. It does NOT use:
      - boundary/rim auxiliary supervision
      - RBSG-lite
      - boundary/rim heads

    Training loss is the same mask-only BCEWithLogits + Dice used by the
    plain ResNet34-U-Net baseline.
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

        self.mask_head = self.unet.segmentation_head

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        features[-1] = self.ssgcb(features[-1])

        decoder_output = self.unet.decoder(*features)
        mask_logits = self.mask_head(decoder_output)

        if not return_aux:
            return mask_logits

        return {"mask": mask_logits}


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
    config["model_class"] = "ResNet34UNetB30SSGCBOnly"
    config["experiment"] = "B30 ablation: plain ResNet34-U-Net + SSGCB only"
    config["note"] = (
        "This is the module-only ablation for B30-light. It inserts the same "
        "Selective Spatial-Gated Context Bottleneck on the deepest encoder "
        "feature of a plain ResNet34-U-Net, while removing boundary/rim "
        "auxiliary supervision and RBSG-lite. Training recipe is unchanged."
    )
    config["ablation_of"] = "B30-light: B2 + SSGCB"
    config["br_auxiliary_supervision"] = False
    config["rbsg"] = False
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
        "boundary": None,
        "rim": None,
        "total": "L_seg only",
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

    model = ResNet34UNetB30SSGCBOnly(
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
    print("Model ready: B30 ablation ResNet34-U-Net + SSGCB only")
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
    print("loss                    = L_seg only")
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
        train_metrics = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            deep_supervision=False,
            train=True,
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            deep_supervision=False,
            train=False,
        )

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

    best_train_metrics = run_one_epoch(
        model,
        train_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )
    best_val_metrics = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )
    test_metrics = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )

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

    parser.add_argument("--run-name", type=str, default="b30_ssgcb_only_auglite_e200_constlr_bs6")
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
