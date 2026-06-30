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
from train_resnet34_unet_brr_b1_aux_e200 import compute_b1_loss
from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


class ResNet34UNetBRRB23RAMLite(ResNet34UNetBRRB2RBSG):
    """
    B23 uses the exact B2 architecture.

    The experimental change is training-time Fourier Random Amplitude Mixup.
    A distinct class name is kept so eval_per_case_metrics can reconstruct the
    model from config/run name without special-casing it as vanilla B2.
    """


def _center_low_frequency_mask(height, width, ratio, device):
    ratio = max(min(float(ratio), 0.50), 0.01)
    radius_h = max(int(round(height * ratio * 0.5)), 1)
    radius_w = max(int(round(width * ratio * 0.5)), 1)

    yy = torch.arange(height, device=device) - (height // 2)
    xx = torch.arange(width, device=device) - (width // 2)
    mask = (yy.abs().view(height, 1) <= radius_h) & (xx.abs().view(1, width) <= radius_w)
    return mask.float().view(1, 1, height, width)


def random_amplitude_mixup(images, prob=0.50, max_strength=0.20, low_freq_ratio=0.08):
    """
    Fourier Random Amplitude Mixup for X-ray style robustness.

    We preserve each image's phase, which mainly carries anatomical structure,
    and softly mix only low-frequency amplitude with another image in the same
    batch, which mainly changes global exposure/contrast/style. Masks remain
    paired with the original image.
    """
    if images.shape[0] < 2:
        return images, False
    if torch.rand((), device=images.device) >= float(prob):
        return images, False

    batch, _, height, width = images.shape
    perm = torch.randperm(batch, device=images.device)
    if bool(torch.all(perm == torch.arange(batch, device=images.device)).item()):
        perm = torch.roll(perm, shifts=1)

    fft = torch.fft.fft2(images, dim=(-2, -1), norm="ortho")
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)

    amplitude_shift = torch.fft.fftshift(amplitude, dim=(-2, -1))
    amplitude_perm_shift = torch.fft.fftshift(amplitude[perm], dim=(-2, -1))

    low_mask = _center_low_frequency_mask(
        height,
        width,
        ratio=low_freq_ratio,
        device=images.device,
    )
    strength = torch.empty(
        (batch, 1, 1, 1),
        device=images.device,
        dtype=images.dtype,
    ).uniform_(0.0, float(max_strength))

    mixed_shift = amplitude_shift * (1.0 - low_mask * strength) + amplitude_perm_shift * (low_mask * strength)
    mixed_amplitude = torch.fft.ifftshift(mixed_shift, dim=(-2, -1))
    mixed_fft = torch.polar(mixed_amplitude, phase)
    mixed_images = torch.fft.ifft2(mixed_fft, dim=(-2, -1), norm="ortho").real

    return mixed_images.clamp(0.0, 1.0), True


def run_one_epoch_b23_ram(model, loader, optimizer, device, train=True, epoch=1, args=None):
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    ram_batches = 0
    n_batches = 0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train and args is not None:
            images, applied = random_amplitude_mixup(
                images,
                prob=args.ram_prob,
                max_strength=args.ram_max_strength,
                low_freq_ratio=args.ram_low_freq_ratio,
            )
            ram_batches += int(applied)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, _ = compute_b1_loss(outputs, masks, epoch=epoch)

            if train:
                loss.backward()
                optimizer.step()

        logits = outputs["mask"].detach()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        gt = (masks > 0.5).float()

        tp = (preds * gt).sum()
        fp = (preds * (1.0 - gt)).sum()
        fn = ((1.0 - preds) * gt).sum()
        tn = ((1.0 - preds) * (1.0 - gt)).sum()

        total_loss += float(loss.detach().cpu())
        total_tp += float(tp.detach().cpu())
        total_fp += float(fp.detach().cpu())
        total_fn += float(fn.detach().cpu())
        total_tn += float(tn.detach().cpu())
        n_batches += 1

    eps = 1e-6
    dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "ram_apply_ratio": ram_batches / max(n_batches, 1),
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
    config["model_class"] = "ResNet34UNetBRRB23RAMLite"
    config["experiment"] = "B23: B2 unchanged + Fourier Random Amplitude Mixup training"
    config["note"] = (
        "B23 keeps the exact B2 network and losses. The only change is applying "
        "low-frequency Fourier amplitude mixup during training to simulate "
        "radiograph exposure/contrast/style shifts while preserving phase-driven "
        "anatomical structure and the original mask label."
    )
    config["result_reflection"] = {
        "b18": "EMA reduced loss and almost matched B2, implying the base model is strong but may need better style-level generalization",
        "b19": "flip consistency over-regularized ambiguous lesions",
        "b20": "distance supervision was dominated by near-zero background and did not improve lesion recall",
        "b21": "feature enhancement branch learned no useful scale",
        "design_implication": "avoid more branches; improve robustness of the already-good B2 decision boundary through realistic X-ray style perturbation",
    }
    config["design_references"] = [
        {
            "title": "Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration",
            "adopted_idea": "Fourier amplitude perturbation can improve robustness to acquisition/style variation",
        },
        {
            "title": "When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation",
            "adopted_idea": "dental caries radiograph segmentation benefits from strong CNN baselines and careful validation more than unnecessary architectural complexity",
        },
    ]
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L = L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L = L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["ram_lite"] = {
        "position": "training batch only, before forward pass",
        "phase": "preserved from the original image",
        "amplitude": "low-frequency amplitude softly mixed with another batch image",
        "mask": "unchanged",
        "prob": args.ram_prob,
        "max_strength": args.ram_max_strength,
        "low_freq_ratio": args.ram_low_freq_ratio,
    }
    config["augmentation"] = {
        "type": "online + RAM-lite",
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
    model = ResNet34UNetBRRB23RAMLite(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B23 RAM-lite + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("ram_prob        =", args.ram_prob)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall", "train_ram_apply_ratio",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b23_ram(
            model, train_loader, optimizer, device, train=True, epoch=epoch, args=args
        )
        val_metrics = run_one_epoch_b23_ram(
            model, val_loader, optimizer=None, device=device, train=False, epoch=epoch, args=args
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"train_ram={train_metrics['ram_apply_ratio']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"], train_metrics["ram_apply_ratio"],
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

    best_train_metrics = run_one_epoch_b23_ram(
        model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args
    )
    best_val_metrics = run_one_epoch_b23_ram(
        model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args
    )
    test_metrics = run_one_epoch_b23_ram(
        model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args
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

    parser.add_argument("--run-name", type=str, default="b23_ram_lite_auglite_e200_constlr_bs6")
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

    parser.add_argument("--ram-prob", type=float, default=0.50)
    parser.add_argument("--ram-max-strength", type=float, default=0.20)
    parser.add_argument("--ram-low-freq-ratio", type=float, default=0.08)

    args = parser.parse_args()
    main(args)
