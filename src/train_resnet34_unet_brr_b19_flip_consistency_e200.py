import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)
from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset
from train_resnet34_unet_brr_b1_aux_e200 import (
    compute_b1_loss,
    run_one_epoch_b1,
)
from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


class ResNet34UNetBRRB19FlipConsistency(ResNet34UNetBRRB2RBSG):
    """
    B19 uses the exact B2 architecture.

    The experiment adds a candidate/uncertainty-weighted horizontal flip
    consistency regularizer during training. It is intentionally not a new
    attention or boundary/rim module: the hypothesis is that B2 needs better
    invariance/generalization rather than stronger feature suppression.
    """
    pass


def consistency_weight(epoch, max_weight=0.05, warmup_epochs=30):
    if warmup_epochs <= 0:
        return float(max_weight)
    ramp = min(max(float(epoch) / float(warmup_epochs), 0.0), 1.0)
    return float(max_weight) * ramp


def flip_consistency_loss(logits, logits_flip_back, candidate_tau=0.15):
    prob = torch.sigmoid(logits)
    prob_flip = torch.sigmoid(logits_flip_back)

    with torch.no_grad():
        candidate = torch.maximum(prob, prob_flip)
        uncertainty = torch.maximum(
            1.0 - torch.abs(2.0 * prob - 1.0),
            1.0 - torch.abs(2.0 * prob_flip - 1.0),
        )
        candidate_mask = torch.sigmoid((candidate - float(candidate_tau)) * 12.0)
        weight = 0.10 + 0.45 * candidate_mask + 0.45 * uncertainty

    return (weight * (prob - prob_flip).pow(2)).mean()


def train_one_epoch_b19(model, loader, optimizer, device, epoch=1, args=None):
    model.train(True)

    total_loss = 0.0
    total_sup_loss = 0.0
    total_cons_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    cw = consistency_weight(
        epoch,
        max_weight=args.consistency_max_weight,
        warmup_epochs=args.consistency_warmup_epochs,
    )

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images, return_aux=True)
        sup_loss, _ = compute_b1_loss(outputs, masks, epoch=epoch)

        images_flip = torch.flip(images, dims=[3])
        logits_flip = model(images_flip, return_aux=False)
        logits_flip_back = torch.flip(logits_flip, dims=[3])
        cons_loss = flip_consistency_loss(
            outputs["mask"],
            logits_flip_back,
            candidate_tau=args.consistency_candidate_tau,
        )

        loss = sup_loss + cw * cons_loss
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
        total_sup_loss += float(sup_loss.detach().cpu())
        total_cons_loss += float(cons_loss.detach().cpu())
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
        "sup_loss": total_sup_loss / max(n_batches, 1),
        "consistency_loss": total_cons_loss / max(n_batches, 1),
        "consistency_weight": cw,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
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
    config["model_class"] = "ResNet34UNetBRRB19FlipConsistency"
    config["experiment"] = "B19: B2 architecture + candidate-weighted flip consistency"
    config["note"] = (
        "B19 keeps the exact B2 architecture and adds a mild horizontal-flip "
        "consistency regularizer. The consistency term is weighted toward B2 "
        "candidate and uncertain pixels so sparse foreground is not overwhelmed "
        "by easy background."
    )
    config["design_references"] = [
        {
            "title": "Mean teachers are better role models / consistency regularization line",
            "adopted_idea": "prediction consistency under perturbations can improve generalization without changing architecture",
        },
        {
            "title": "When CNNs Outperform Transformers and Mambas",
            "adopted_idea": "for dental caries segmentation, lightweight CNN priors and robust training can be preferable to heavier modules",
        },
        {
            "title": "PRAD periapical radiograph benchmark",
            "adopted_idea": "periapical radiographs contain artifacts and resolution limits, so invariance to benign geometric perturbation is valuable",
        },
    ]
    config["failure_reflection"] = {
        "b10_b16_skip_modules": "skip/detail modules did not beat B2, suggesting more structural fusion is not the bottleneck",
        "b12_b15_rescue_heads": "positive residual heads often traded recall/precision without improving global Dice",
        "b17_wavelet": "validation improvement did not transfer to test, suggesting robustness/generalization needs attention",
        "b19_hypothesis": "mild candidate-weighted flip consistency improves B2 robustness while preserving its architecture",
    }
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
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
    model = ResNet34UNetBRRB19FlipConsistency(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B19 = B2 architecture + candidate-weighted flip consistency")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("consistency_max =", args.consistency_max_weight)
    print("epochs          =", args.epochs)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_sup_loss", "train_consistency_loss", "train_consistency_weight",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_b19(model, train_loader, optimizer, device, epoch=epoch, args=args)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"cons={train_metrics['consistency_loss']:.5f} cw={train_metrics['consistency_weight']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["sup_loss"],
                train_metrics["consistency_loss"], train_metrics["consistency_weight"],
                train_metrics["dice"], train_metrics["iou"],
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
        "consistency_max_weight": args.consistency_max_weight,
        "consistency_warmup_epochs": args.consistency_warmup_epochs,
        "consistency_candidate_tau": args.consistency_candidate_tau,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="b19_flip_consistency_auglite_e200_constlr_bs6")
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
    parser.add_argument("--consistency-max-weight", type=float, default=0.05)
    parser.add_argument("--consistency-warmup-epochs", type=int, default=30)
    parser.add_argument("--consistency-candidate-tau", type=float, default=0.15)
    main(parser.parse_args())
