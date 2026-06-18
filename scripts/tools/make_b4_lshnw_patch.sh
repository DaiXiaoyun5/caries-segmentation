#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b4_lshnw_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B4 ====="
cp -p "$SRC" "$DST"

echo "===== Patch B4 script: add Lesion-size and Hard-negative Weighted Loss ====="
python - <<'PY'
from pathlib import Path

p = Path("src/train_resnet34_unet_brr_b4_lshnw_e200.py")
s = p.read_text(encoding="utf-8")

old_import = '''from train_resnet34_unet_brr_b1_aux_e200 import (
    run_one_epoch_b1,
)
'''
new_import = '''from train_resnet34_unet_brr_b1_aux_e200 import (
    bce_dice_loss,
    mask_to_boundary_and_rim,
)
'''

if old_import not in s:
    raise RuntimeError("Cannot find old B1 import block. Please check B2 script structure.")
s = s.replace(old_import, new_import)

insert_code = r'''

def lesion_size_adaptive_weight(
    masks,
    ref_ratio=0.01,
    gamma=0.5,
    max_weight=2.0,
    eps=1e-6,
):
    """
    Compute per-sample weights according to lesion area ratio.

    Smaller lesion -> larger loss weight.
    masks: [B,1,H,W], values in {0,1}

    weight = clamp((ref_ratio / area_ratio)^gamma, 1, max_weight)

    Notes:
      - If lesion is larger than ref_ratio, weight becomes 1.
      - If lesion is very small, weight is capped by max_weight.
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    area_ratio = masks.float().flatten(1).mean(dim=1)  # [B]
    raw_weight = (float(ref_ratio) / (area_ratio + eps)).pow(float(gamma))
    weights = torch.clamp(raw_weight, min=1.0, max=float(max_weight))

    # Avoid strange behavior for empty masks, if any.
    weights = torch.where(
        area_ratio > eps,
        weights,
        torch.ones_like(weights),
    )
    return weights.detach()


def weighted_bce_dice_loss(
    logits,
    targets,
    sample_weights=None,
    pixel_weights=None,
    eps=1e-6,
):
    """
    BCEWithLogits + Dice with optional sample-level and pixel-level weights.

    sample_weights:
      [B], used to emphasize small-lesion samples.

    pixel_weights:
      [B,1,H,W], used to emphasize rim hard-negative regions.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    if pixel_weights is not None:
        pixel_weights = pixel_weights.float()
        bce_map = bce_map * pixel_weights

    bce_per_sample = bce_map.flatten(1).mean(dim=1)  # [B]

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice_per_sample = 1.0 - (2.0 * inter + eps) / (denom + eps)

    if sample_weights is not None:
        sample_weights = sample_weights.float()
        loss = (
            sample_weights * (bce_per_sample + dice_per_sample)
        ).sum() / (sample_weights.sum() + eps)
    else:
        loss = (bce_per_sample + dice_per_sample).mean()

    return loss


def compute_b4_loss(
    outputs,
    masks,
    epoch,
    warmup_epochs=20,
    small_ref_ratio=0.01,
    small_gamma=0.5,
    small_max_weight=2.0,
    rim_hard_weight=0.5,
):
    """
    B4 loss = B2 loss with lesion-size and rim hard-negative weighting.

    B2:
      L_seg + wb * L_boundary + wr * L_rim

    B4:
      weighted L_seg + wb * L_boundary + wr * L_rim

    The main segmentation loss is modified by:
      1) sample-level small-lesion weight
      2) pixel-level rim hard-negative weight
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    sample_weights = lesion_size_adaptive_weight(
        masks,
        ref_ratio=small_ref_ratio,
        gamma=small_gamma,
        max_weight=small_max_weight,
    )

    # Warm up rim hard-negative weighting to avoid early over-suppression.
    if epoch <= warmup_epochs:
        effective_rim_hard_weight = min(float(rim_hard_weight), 0.25)
    else:
        effective_rim_hard_weight = float(rim_hard_weight)

    # rim_gt is outside lesion and close to lesion: hard negative region.
    pixel_weights = 1.0 + effective_rim_hard_weight * rim_gt.float()

    loss_seg = weighted_bce_dice_loss(
        mask_logits,
        masks,
        sample_weights=sample_weights,
        pixel_weights=pixel_weights,
    )

    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
    else:
        wb = 0.20
        wr = 0.10

    total = loss_seg + wb * loss_boundary + wr * loss_rim

    return total, {
        "loss_seg_weighted": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "sample_weight_mean": float(sample_weights.mean().detach().cpu()),
        "sample_weight_max": float(sample_weights.max().detach().cpu()),
        "rim_hard_weight": effective_rim_hard_weight,
        "wb": wb,
        "wr": wr,
    }


def run_one_epoch_b4(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    small_ref_ratio=0.01,
    small_gamma=0.5,
    small_max_weight=2.0,
    rim_hard_weight=0.5,
):
    """
    Train/eval one epoch for B4.

    Metrics are still computed only from final mask logits.
    Small-lesion and hard-negative weighting only affects training loss.
    """
    model.train(train)

    total_loss = 0.0
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
            loss, _ = compute_b4_loss(
                outputs,
                masks,
                epoch=epoch,
                small_ref_ratio=small_ref_ratio,
                small_gamma=small_gamma,
                small_max_weight=small_max_weight,
                rim_hard_weight=rim_hard_weight,
            )

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
    }
'''

marker = "\n\ndef main(args):"
if marker not in s:
    raise RuntimeError("Cannot find def main(args) marker.")
s = s.replace(marker, insert_code + marker)

s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B4: B2 BRR-RBSG + Lesion-size and Hard-negative Weighted Loss"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B4 keeps B2 BRR-RBSG architecture and modifies the main segmentation loss with lesion-size adaptive sample weighting and rim hard-negative pixel weighting. No extra inference parameter is introduced."',
)

old_loss_block = '''    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim"
    }
'''
new_loss_block = '''    config["loss"] = {
        "main": "Weighted BCEWithLogits + Weighted Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "small_lesion_weight": "sample weight = clamp((small_ref_ratio / lesion_area_ratio)^small_gamma, 1, small_max_weight)",
        "rim_hard_negative_weight": "pixel weight = 1 + rim_hard_weight * rim_gt",
        "epoch_1_to_20": "weighted L_seg + 0.10*L_boundary + 0.05*L_rim, rim_hard_weight capped at 0.25",
        "epoch_21_to_end": "weighted L_seg + 0.20*L_boundary + 0.10*L_rim",
        "small_ref_ratio": args.small_ref_ratio,
        "small_gamma": args.small_gamma,
        "small_max_weight": args.small_max_weight,
        "rim_hard_weight": args.rim_hard_weight
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B4 BRR-RBSG + Lesion-size and Hard-negative Weighted Loss")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("small_ref_ratio =", args.small_ref_ratio)\n    print("small_gamma     =", args.small_gamma)\n    print("small_max_weight=", args.small_max_weight)\n    print("rim_hard_weight =", args.rim_hard_weight)\n',
)

repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b4(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            small_ref_ratio=args.small_ref_ratio,
            small_gamma=args.small_gamma,
            small_max_weight=args.small_max_weight,
            rim_hard_weight=args.rim_hard_weight,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b4(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            small_ref_ratio=args.small_ref_ratio,
            small_gamma=args.small_gamma,
            small_max_weight=args.small_max_weight,
            rim_hard_weight=args.rim_hard_weight,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b4(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        small_ref_ratio=args.small_ref_ratio,
        small_gamma=args.small_gamma,
        small_max_weight=args.small_max_weight,
        rim_hard_weight=args.rim_hard_weight,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b4(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        small_ref_ratio=args.small_ref_ratio,
        small_gamma=args.small_gamma,
        small_max_weight=args.small_max_weight,
        rim_hard_weight=args.rim_hard_weight,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b4(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        small_ref_ratio=args.small_ref_ratio,
        small_gamma=args.small_gamma,
        small_max_weight=args.small_max_weight,
        rim_hard_weight=args.rim_hard_weight,
    )''',
}

for old, new in repls.items():
    if old not in s:
        raise RuntimeError(f"Cannot find call block: {old[:80]}")
    s = s.replace(old, new)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--small-ref-ratio", type=float, default=0.01)
    parser.add_argument("--small-gamma", type=float, default=0.5)
    parser.add_argument("--small-max-weight", type=float, default=2.0)
    parser.add_argument("--rim-hard-weight", type=float, default=0.5)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse insertion marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Create full training sbatch script ====="
cat > scripts/jobs/submitjob_b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b4_lshnw
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b4_lshnw_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== ENV ====="
which python
python --version
nvidia-smi

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b4_lshnw_e200.py

echo "===== RUN B4: BRR-RBSG + Lesion-size and Hard-negative Weighted Loss ====="
python src/train_resnet34_unet_brr_b4_lshnw_e200.py \
  --run-name b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 200 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --small-ref-ratio 0.01 \
  --small-gamma 0.5 \
  --small-max-weight 2.0 \
  --rim-hard-weight 0.5

echo "===== DONE B4 ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > scripts/jobs/submitjob_smoke_b4_brr_rbsg_lshnw_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b4_lshnw_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== ENV ====="
which python
python --version
nvidia-smi

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b4_lshnw_e200.py

echo "===== SMOKE RUN B4: 3 epochs, batch size 2 ====="
python src/train_resnet34_unet_brr_b4_lshnw_e200.py \
  --run-name smoke_b4_brr_rbsg_lshnw_e3 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 2 \
  --epochs 3 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --small-ref-ratio 0.01 \
  --small-gamma 0.5 \
  --small-max-weight 2.0 \
  --rim-hard-weight 0.5

echo "===== DONE SMOKE B4 ====="
EOF

chmod +x scripts/jobs/submitjob_b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6.sh
chmod +x scripts/jobs/submitjob_smoke_b4_brr_rbsg_lshnw_e3.sh

echo "===== Shell syntax check ====="
bash -n scripts/jobs/submitjob_b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6.sh
bash -n scripts/jobs/submitjob_smoke_b4_brr_rbsg_lshnw_e3.sh

echo "===== Python syntax check ====="
module load anaconda3/4.12.0 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
python -m py_compile src/train_resnet34_unet_brr_b4_lshnw_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b4_lshnw_e200.py
ls -lh scripts/jobs/submitjob_b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6.sh
ls -lh scripts/jobs/submitjob_smoke_b4_brr_rbsg_lshnw_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch scripts/jobs/submitjob_smoke_b4_brr_rbsg_lshnw_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b4_lshnw_*.out"
echo
echo "3) If smoke passes, submit full B4:"
echo "   sbatch scripts/jobs/submitjob_b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6.sh"
