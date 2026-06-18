from pathlib import Path
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    ResNet34UNet3Plus,
    get_main_logits,
    run_one_epoch,
    save_prediction_panel,
)


class LiteASPPBlock(nn.Module):
    """
    Lite-ASPP: lightweight atrous spatial pyramid module.

    This is the single-module ablation experiment:
    ResNet34-UNet3+ + Lite-ASPP, without CBSA.

    Lite-ASPP is placed only on the deepest encoder feature h5.
    """
    def __init__(
        self,
        in_ch,
        hidden_ch=128,
        dilations=(1, 3, 5),
        dropout=0.0,
        residual_scale=0.1,
    ):
        super().__init__()

        self.branches = nn.ModuleList()

        # 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
            )
        )

        # atrous branches
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        hidden_ch,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # global context branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        total_ch = hidden_ch * (len(self.branches) + 1)

        self.project = nn.Sequential(
            nn.Conv2d(total_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        h, w = x.shape[-2:]

        feats = [branch(x) for branch in self.branches]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(gp)

        out = self.project(torch.cat(feats, dim=1))

        scale = torch.clamp(self.residual_scale, 0.0, 1.0)
        return x + scale * out


class ResNet34UNet3PlusLiteASPP(ResNet34UNet3Plus):
    """
    ResNet34-UNet3+ + Lite-ASPP.

    Lite-ASPP is applied after encoder h5.
    The decoder remains the original UNet3+ decoder.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        aspp_hidden=128,
        aspp_dilations=(1, 3, 5),
        aspp_dropout=0.0,
        aspp_res_scale=0.1,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            cat_ch=cat_ch,
            deep_supervision=deep_supervision,
        )

        enc_ch = list(self.encoder.out_channels)
        ch5 = enc_ch[5]

        self.lite_aspp = LiteASPPBlock(
            in_ch=ch5,
            hidden_ch=aspp_hidden,
            dilations=aspp_dilations,
            dropout=aspp_dropout,
            residual_scale=aspp_res_scale,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        features = self.encoder(x)
        h1, h2, h3, h4, h5 = features[1], features[2], features[3], features[4], features[5]

        # Lite-ASPP only on deepest semantic feature.
        h5 = self.lite_aspp(h5)

        # hd4
        hd4 = self.conv_hd4(torch.cat([
            self.h1_PT_hd4(h1),
            self.h2_PT_hd4(h2),
            self.h3_PT_hd4(h3),
            self.h4_Cat_hd4(h4),
            self.up_to(self.h5_UT_hd4(h5), h4),
        ], dim=1))

        # hd3
        hd3 = self.conv_hd3(torch.cat([
            self.h1_PT_hd3(h1),
            self.h2_PT_hd3(h2),
            self.h3_Cat_hd3(h3),
            self.up_to(self.h4_UT_hd3(hd4), h3),
            self.up_to(self.h5_UT_hd3(h5), h3),
        ], dim=1))

        # hd2
        hd2 = self.conv_hd2(torch.cat([
            self.h1_PT_hd2(h1),
            self.h2_Cat_hd2(h2),
            self.up_to(self.h3_UT_hd2(hd3), h2),
            self.up_to(self.h4_UT_hd2(hd4), h2),
            self.up_to(self.h5_UT_hd2(h5), h2),
        ], dim=1))

        # hd1
        hd1 = self.conv_hd1(torch.cat([
            self.h1_Cat_hd1(h1),
            self.up_to(self.h2_UT_hd1(hd2), h1),
            self.up_to(self.h3_UT_hd1(hd3), h1),
            self.up_to(self.h4_UT_hd1(hd4), h1),
            self.up_to(self.h5_UT_hd1(h5), h1),
        ], dim=1))

        d1 = self.up_to_size(self.outconv1(hd1), input_size)

        if not self.deep_supervision:
            return d1

        d2 = self.up_to_size(self.outconv2(hd2), input_size)
        d3 = self.up_to_size(self.outconv3(hd3), input_size)
        d4 = self.up_to_size(self.outconv4(hd4), input_size)
        d5 = self.up_to_size(self.outconv5(h5), input_size)

        return [d1, d2, d3, d4, d5]


def parse_dilations(text):
    values = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(int(x))
    if not values:
        raise ValueError("aspp_dilations cannot be empty")
    return tuple(values)


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
    save_json(config, run_dir / "config.json")

    train_ds = CariesMaskDataset(train_split, target_size=(args.image_size, args.image_size))
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

    print("=" * 80)
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    deep_supervision = bool(args.deep_supervision)
    aspp_dilations = parse_dilations(args.aspp_dilations)

    model = ResNet34UNet3PlusLiteASPP(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cat_ch=args.cat_channels,
        deep_supervision=deep_supervision,
        aspp_hidden=args.aspp_hidden,
        aspp_dilations=aspp_dilations,
        aspp_dropout=args.aspp_dropout,
        aspp_res_scale=args.aspp_res_scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # forward check
    sample_batch = next(iter(train_loader))
    sample_images, _, _, _ = sample_batch
    sample_images = sample_images.to(device)

    with torch.no_grad():
        outputs = model(sample_images)
        main_logits = get_main_logits(outputs)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", sample_images.shape)
    print("main_logits.shape  =", main_logits.shape)
    if isinstance(outputs, list):
        print("deep outputs       =", [tuple(o.shape) for o in outputs])
    print("aspp_hidden        =", args.aspp_hidden)
    print("aspp_dilations     =", aspp_dilations)
    print("aspp_dropout       =", args.aspp_dropout)
    print("aspp_res_scale     =", args.aspp_res_scale)
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
            deep_supervision=deep_supervision,
            train=True,
        )

        val_metrics = run_one_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            deep_supervision=deep_supervision,
            train=False,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch(
        model,
        train_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        train=False,
    )
    best_val_metrics = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        train=False,
    )
    test_metrics = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        deep_supervision=deep_supervision,
        train=False,
    )

    print("=" * 80)
    print("Best checkpoint summary")
    print("best_epoch:", best_epoch)
    print("best_val_dice_during_training:", best_val_dice)
    print("best_train_metrics:", best_train_metrics)
    print("best_val_metrics  :", best_val_metrics)
    print("test_metrics      :", test_metrics)
    print("=" * 80)

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="smoke_resnet34_unet3p_laspp")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--cat-channels", type=int, default=64)
    parser.add_argument("--deep-supervision", type=int, default=1)

    parser.add_argument("--aspp-hidden", type=int, default=128)
    parser.add_argument("--aspp-dilations", type=str, default="1,3,5")
    parser.add_argument("--aspp-dropout", type=float, default=0.0)
    parser.add_argument("--aspp-res-scale", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
