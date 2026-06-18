#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b3_mbc_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B3 ====="
cp -p "$SRC" "$DST"

echo "===== Patch B3 script: add Mask-Boundary Consistency Loss ====="
python - <<'PY'
from pathlib import Path

p = Path("src/train_resnet34_unet_brr_b3_mbc_e200.py")
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

def dice_loss_from_probs(probs, targets, eps=1e-6):
    """
    Dice loss for probability maps.
    probs:   [B,1,H,W], values in [0,1]
    targets: [B,1,H,W], values in {0,1}
    """
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def soft_mask_to_boundary(mask_prob, rb=3):
    """
    Differentiable soft boundary from predicted mask probability.

    mask_prob: [B,1,H,W], values in [0,1]
    boundary = soft_dilate(mask_prob) - soft_erode(mask_prob)

    This keeps gradients from boundary consistency loss flowing back to mask logits.
    """
    if mask_prob.dim() == 3:
        mask_prob = mask_prob.unsqueeze(1)

    mask_prob = mask_prob.float()
    k = 2 * rb + 1

    dil = F.max_pool2d(mask_prob, kernel_size=k, stride=1, padding=rb)
    ero = -F.max_pool2d(-mask_prob, kernel_size=k, stride=1, padding=rb)

    boundary = (dil - ero).clamp(0.0, 1.0)
    return boundary


def mask_boundary_consistency_loss(mask_logits, boundary_gt, rb=3, eps=1e-6):
    """
    Mask-Boundary Consistency Loss.

    It enforces the soft boundary derived from predicted mask to match
    the GT boundary target derived from GT mask.

    Important:
      - boundary_gt is used only as training/evaluation target.
      - No GT boundary/rim is used as model input.
      - This loss adds no inference parameter.
    """
    mask_prob = torch.sigmoid(mask_logits)
    pred_boundary = soft_mask_to_boundary(mask_prob, rb=rb)

    pred_boundary = pred_boundary.clamp(eps, 1.0 - eps)
    boundary_gt = boundary_gt.float()

    bce = F.binary_cross_entropy(pred_boundary, boundary_gt)
    dice = dice_loss_from_probs(pred_boundary, boundary_gt)

    return bce + dice


def compute_b3_loss(outputs, masks, epoch, warmup_epochs=20, mbc_weight=0.05, mbc_rb=3):
    """
    B3 loss = B2 loss + Mask-Boundary Consistency Loss.

    B2:
      L_seg + wb * L_boundary + wr * L_rim

    B3:
      L_seg + wb * L_boundary + wr * L_rim + wc * L_mbc
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)
    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    loss_mbc = mask_boundary_consistency_loss(
        mask_logits,
        boundary_gt,
        rb=mbc_rb,
    )

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
        wc = min(float(mbc_weight), 0.02)
    else:
        wb = 0.20
        wr = 0.10
        wc = float(mbc_weight)

    total = loss_seg + wb * loss_boundary + wr * loss_rim + wc * loss_mbc

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "loss_mbc": float(loss_mbc.detach().cpu()),
        "wb": wb,
        "wr": wr,
        "wc": wc,
    }


def run_one_epoch_b3(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    mbc_weight=0.05,
    mbc_rb=3,
):
    """
    Train/eval one epoch for B3.

    Metrics are still computed only from final mask logits.
    Boundary/rim/MBC only affect loss.
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
            loss, _ = compute_b3_loss(
                outputs,
                masks,
                epoch=epoch,
                mbc_weight=mbc_weight,
                mbc_rb=mbc_rb,
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

# Update config description.
s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B3: B2 BRR-RBSG + Mask-Boundary Consistency Loss"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B3 keeps B2 BRR-RBSG architecture and adds Mask-Boundary Consistency Loss. It enforces the soft boundary derived from predicted mask to match GT boundary. No extra inference parameter is introduced."',
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
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "mask_boundary_consistency": "BCE + Dice between soft boundary from predicted mask and GT boundary",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim + 0.02*L_mbc",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim + mbc_weight*L_mbc",
        "mbc_weight": args.mbc_weight,
        "mbc_rb": args.mbc_rb
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

# Update print header.
s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B3 BRR-RBSG + Mask-Boundary Consistency Loss")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("mbc_weight      =", args.mbc_weight)\n    print("mbc_rb          =", args.mbc_rb)\n',
)

# Replace train/val/test calls.
repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b3(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            mbc_weight=args.mbc_weight,
            mbc_rb=args.mbc_rb,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b3(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            mbc_weight=args.mbc_weight,
            mbc_rb=args.mbc_rb,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b3(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        mbc_weight=args.mbc_weight,
        mbc_rb=args.mbc_rb,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b3(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        mbc_weight=args.mbc_weight,
        mbc_rb=args.mbc_rb,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b3(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        mbc_weight=args.mbc_weight,
        mbc_rb=args.mbc_rb,
    )''',
}

for old, new in repls.items():
    if old not in s:
        raise RuntimeError(f"Cannot find call block: {old[:80]}")
    s = s.replace(old, new)

# Update default run name.
s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b3_brr_rbsg_mbc_auglite_e200_constlr_bs6")',
)

# Add argparse parameters.
arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--mbc-weight", type=float, default=0.05)
    parser.add_argument("--mbc-rb", type=int, default=3)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse insertion marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Create full training sbatch script ====="
cat > scripts/jobs/submitjob_b3_brr_rbsg_mbc_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b3_brr_mbc
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b3_brr_mbc_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b3_mbc_e200.py

echo "===== RUN B3: BRR-RBSG + Mask-Boundary Consistency Loss ====="
python src/train_resnet34_unet_brr_b3_mbc_e200.py \
  --run-name b3_brr_rbsg_mbc_auglite_e200_constlr_bs6 \
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
  --mbc-weight 0.05 \
  --mbc-rb 3

echo "===== DONE B3 ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > scripts/jobs/submitjob_smoke_b3_brr_rbsg_mbc_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b3_mbc_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b3_mbc_e200.py

echo "===== SMOKE RUN B3: 3 epochs, batch size 2 ====="
python src/train_resnet34_unet_brr_b3_mbc_e200.py \
  --run-name smoke_b3_brr_rbsg_mbc_e3 \
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
  --mbc-weight 0.05 \
  --mbc-rb 3

echo "===== DONE SMOKE B3 ====="
EOF

chmod +x scripts/jobs/submitjob_b3_brr_rbsg_mbc_auglite_e200_constlr_bs6.sh
chmod +x scripts/jobs/submitjob_smoke_b3_brr_rbsg_mbc_e3.sh

echo "===== Shell syntax check ====="
bash -n scripts/jobs/submitjob_b3_brr_rbsg_mbc_auglite_e200_constlr_bs6.sh
bash -n scripts/jobs/submitjob_smoke_b3_brr_rbsg_mbc_e3.sh

echo "===== Python syntax check ====="
module load anaconda3/4.12.0 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
python -m py_compile src/train_resnet34_unet_brr_b3_mbc_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b3_mbc_e200.py
ls -lh scripts/jobs/submitjob_b3_brr_rbsg_mbc_auglite_e200_constlr_bs6.sh
ls -lh scripts/jobs/submitjob_smoke_b3_brr_rbsg_mbc_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch scripts/jobs/submitjob_smoke_b3_brr_rbsg_mbc_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b3_mbc_*.out"
echo
echo "3) If smoke passes, submit full B3:"
echo "   sbatch scripts/jobs/submitjob_b3_brr_rbsg_mbc_auglite_e200_constlr_bs6.sh"
