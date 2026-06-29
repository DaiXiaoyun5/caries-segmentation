import argparse
import copy
import csv
from pathlib import Path

import torch
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


class ResNet34UNetBRRB18EMA(ResNet34UNetBRRB2RBSG):
    """
    B18 uses the exact B2 architecture.

    The experiment changes only the training/inference protocol:
    maintain an exponential moving average (EMA) of model weights and select
    checkpoints by EMA validation Dice. This tests whether B2's strong but
    slightly noisy 200-epoch training can benefit from weight averaging without
    adding another attention/gating module.
    """
    pass


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


def train_one_epoch_b18_ema(model, ema_helper, loader, optimizer, device, epoch=1):
    model.train(True)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, return_aux=True)
        loss, _ = compute_b1_loss(outputs, masks, epoch=epoch)
        loss.backward()
        optimizer.step()
        ema_helper.update(model)

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
    config["model_class"] = "ResNet34UNetBRRB18EMA"
    config["experiment"] = "B18: B2 architecture + EMA weight averaging"
    config["note"] = (
        "B18 keeps the exact B2 ResNet34-U-Net + BRR/RBSG architecture and changes "
        "only the training protocol. Multiple previous B2 extensions improved one "
        "axis but failed to beat B2 on test Dice, suggesting that generalization "
        "stability may be more valuable than another module."
    )
    config["design_references"] = [
        {
            "title": "Mean teachers are better role models",
            "adopted_idea": "EMA of weights provides smoother, more stable evaluation targets without changing the architecture",
        },
        {
            "title": "nnU-Net Revisited",
            "adopted_idea": "strong CNN/U-Net baselines should be improved with rigorous, minimal, well-controlled changes",
        },
        {
            "title": "When CNNs Outperform Transformers and Mambas",
            "adopted_idea": "dental caries segmentation benefits from task-aligned CNN inductive bias under limited data",
        },
    ]
    config["failure_reflection"] = {
        "b10_dmsk": "test Dice and Recall dropped, suggesting dynamic skip selection overfit or became conservative",
        "b12_tdr": "higher Precision but lower Recall, not enough Dice gain",
        "b17_wavelet": "best validation Dice but lower test Dice, suggesting validation/checkpoint instability",
        "b18_hypothesis": "EMA may reduce checkpoint variance without changing B2's precision/recall balance",
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
    model = ResNet34UNetBRRB18EMA(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
    ema_helper = ModelEMA(model, decay=args.ema_decay)
    ema_helper.ema.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B18 = B2 architecture + EMA weight averaging")
    print("run_name   =", args.run_name)
    print("device     =", device)
    print("ema_decay  =", args.ema_decay)
    print("epochs     =", args.epochs)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "ema_val_loss", "ema_val_dice", "ema_val_iou", "ema_val_precision", "ema_val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_b18_ema(
            model,
            ema_helper,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
        )
        val_metrics = run_one_epoch_b1(
            ema_helper.ema,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"ema_val_loss={val_metrics['loss']:.4f} ema_val_dice={val_metrics['dice']:.4f} "
            f"ema_val_iou={val_metrics['iou']:.4f} ema_val_precision={val_metrics['precision']:.4f} "
            f"ema_val_recall={val_metrics['recall']:.4f}"
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

        torch.save(model.state_dict(), ckpt_dir / "last_raw.pth")
        torch.save(ema_helper.ema.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(ema_helper.ema.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best EMA model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    ema_helper.ema.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    best_train_metrics = run_one_epoch_b1(ema_helper.ema, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)
    best_val_metrics = run_one_epoch_b1(ema_helper.ema, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)
    test_metrics = run_one_epoch_b1(ema_helper.ema, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "ema_decay": args.ema_decay,
        "model_parameters": sum(p.numel() for p in ema_helper.ema.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(ema_helper.ema, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="b18_ema_auglite_e200_constlr_bs6")
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
    parser.add_argument("--ema-decay", type=float, default=0.999)
    main(parser.parse_args())
