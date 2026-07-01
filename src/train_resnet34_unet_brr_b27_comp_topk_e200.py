import argparse
import csv
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
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


class ResNet34UNetBRRB27CompTopK(ResNet34UNetBRRB2RBSG):
    """
    B27: exact B2 architecture + component-level top-k recall loss.

    This wrapper keeps eval_per_case_metrics dynamic loading simple while the
    architecture remains identical to B2.
    """


def connected_components_2d(mask_np):
    """
    8-connected components for a binary 2D numpy mask.

    Preference:
      scipy.ndimage.label -> cv2.connectedComponents -> pure numpy BFS.
    """
    mask_np = mask_np.astype(bool)
    if not mask_np.any():
        return np.zeros(mask_np.shape, dtype=np.int32), 0

    try:
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=np.int8)
        labels, num = ndimage.label(mask_np, structure=structure)
        return labels.astype(np.int32, copy=False), int(num)
    except Exception:
        pass

    try:
        import cv2

        num, labels = cv2.connectedComponents(mask_np.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32, copy=False), int(num - 1)
    except Exception:
        pass

    height, width = mask_np.shape
    labels = np.zeros((height, width), dtype=np.int32)
    component_id = 0
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    for y in range(height):
        for x in range(width):
            if not mask_np[y, x] or labels[y, x] != 0:
                continue
            component_id += 1
            queue = deque([(y, x)])
            labels[y, x] = component_id

            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if mask_np[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = component_id
                        queue.append((ny, nx))

    return labels, component_id


def component_topk_recall_loss(logits, masks, topk_ratio=0.30, gamma=2.0):
    probs = torch.sigmoid(logits)
    batch_losses = []

    for batch_index in range(probs.shape[0]):
        prob = probs[batch_index, 0]
        gt_np = (masks[batch_index, 0].detach().cpu().numpy() > 0.5)
        labels, num_components = connected_components_2d(gt_np)

        if num_components <= 0:
            batch_losses.append(prob.sum() * 0.0)
            continue

        component_losses = []
        component_weights = []
        for component_id in range(1, num_components + 1):
            component_np = labels == component_id
            area = int(component_np.sum())
            if area <= 0:
                continue

            component_mask = torch.as_tensor(
                component_np,
                dtype=torch.bool,
                device=prob.device,
            )
            values = prob[component_mask]
            if values.numel() == 0:
                continue

            k = max(1, int(math.floor(float(topk_ratio) * float(area))))
            k = min(k, int(values.numel()))
            topk_mean = values.topk(k).values.mean()
            component_losses.append((1.0 - topk_mean).pow(float(gamma)))
            component_weights.append(1.0 / math.sqrt(float(area) + 1e-6))

        if not component_losses:
            batch_losses.append(prob.sum() * 0.0)
            continue

        weight_tensor = torch.as_tensor(
            component_weights,
            dtype=prob.dtype,
            device=prob.device,
        )
        weight_tensor = weight_tensor / (weight_tensor.sum() + 1e-6)
        loss_tensor = torch.stack(component_losses)
        batch_losses.append((weight_tensor * loss_tensor).sum())

    return torch.stack(batch_losses).mean()


def compute_b27_loss(outputs, masks, epoch, args):
    base_loss, details = compute_b1_loss(outputs, masks, epoch=epoch)

    if epoch < args.component_start_epoch:
        lambda_comp = 0.0
        loss_comp = outputs["mask"].sum() * 0.0
    else:
        lambda_comp = float(args.component_weight)
        loss_comp = component_topk_recall_loss(
            outputs["mask"],
            masks,
            topk_ratio=args.component_topk_ratio,
            gamma=args.component_gamma,
        )

    total = base_loss + lambda_comp * loss_comp
    details = details.copy()
    details.update(
        {
            "loss_comp": float(loss_comp.detach().cpu()),
            "lambda_comp": lambda_comp,
        }
    )
    return total, details


def run_one_epoch_b27(model, loader, optimizer, device, train=True, epoch=1, args=None):
    model.train(train)

    total_loss = 0.0
    total_loss_comp = 0.0
    total_lambda_comp = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, details = compute_b27_loss(outputs, masks, epoch=epoch, args=args)

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
        total_loss_comp += float(details["loss_comp"])
        total_lambda_comp += float(details["lambda_comp"])
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
        "loss_comp": total_loss_comp / max(n_batches, 1),
        "lambda_comp": total_lambda_comp / max(n_batches, 1),
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
    config["model_class"] = "ResNet34UNetBRRB27CompTopK"
    config["experiment"] = "B27: B2 + component-level top-k recall loss"
    config["note"] = (
        "B27 keeps the exact B2 architecture and B2 boundary/rim auxiliary loss. "
        "It adds a lightweight component-level top-k recall loss after warm-up to "
        "encourage each GT component, especially small lesions, to contain at "
        "least confident predicted pixels. The loss is intentionally weak and "
        "does not use Tversky, SDF, consistency, or additional modules."
    )
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["component_topk_loss"] = {
        "definition": "For each GT connected component, maximize mean top-k predicted probabilities inside the component.",
        "topk_ratio": args.component_topk_ratio,
        "gamma": args.component_gamma,
        "weight": args.component_weight,
        "start_epoch": args.component_start_epoch,
        "area_weight": "1/sqrt(area), normalized per image",
        "component_labeling": "scipy.ndimage.label -> cv2.connectedComponents -> numpy BFS fallback",
        "objective": "reduce missed small lesion components without forcing a global precision/recall shift",
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
    model = ResNet34UNetBRRB27CompTopK(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B27 CompTopK + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name              =", args.run_name)
    print("device                =", device)
    print("augment_profile       =", args.augment_profile)
    print("batch_size            =", args.batch_size)
    print("epochs                =", args.epochs)
    print("lr                    =", args.lr)
    print("component_topk_ratio  =", args.component_topk_ratio)
    print("component_weight      =", args.component_weight)
    print("component_start_epoch =", args.component_start_epoch)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_loss_comp", "train_lambda_comp",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_loss_comp", "val_lambda_comp",
            "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b27(model, train_loader, optimizer, device, train=True, epoch=epoch, args=args)
        val_metrics = run_one_epoch_b27(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch, args=args)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_comp={val_metrics['loss_comp']:.4f} lambda={val_metrics['lambda_comp']:.3f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["loss_comp"], train_metrics["lambda_comp"],
                train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["loss_comp"], val_metrics["lambda_comp"],
                val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    best_train_metrics = run_one_epoch_b27(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args)
    best_val_metrics = run_one_epoch_b27(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args)
    test_metrics = run_one_epoch_b27(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args)

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

    parser.add_argument("--run-name", type=str, default="b27_comp_topk_auglite_e200_constlr_bs6")
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

    parser.add_argument("--component-topk-ratio", type=float, default=0.30)
    parser.add_argument("--component-gamma", type=float, default=2.0)
    parser.add_argument("--component-weight", type=float, default=0.03)
    parser.add_argument("--component-start-epoch", type=int, default=40)

    args = parser.parse_args()
    main(args)
