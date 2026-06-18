#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B2-rampwarmup ====="
cp -p "$SRC" "$DST"

echo "===== Patch B2-rampwarmup script ====="
python - <<'PY'
from pathlib import Path

p = Path("src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py")
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
    raise RuntimeError("Cannot find old B1 import block. Please check B2 script.")
s = s.replace(old_import, new_import)

insert_code = r'''

def get_ramp_aux_weights(
    epoch,
    warmup_epochs=30,
    boundary_max_weight=0.20,
    rim_max_weight=0.10,
):
    """
    Continuous ramp-up for boundary/rim auxiliary losses.

    epoch 1 -> small auxiliary weight
    epoch warmup_epochs and later -> max auxiliary weight
    """
    warmup_epochs = max(int(warmup_epochs), 1)
    ratio = min(1.0, float(epoch) / float(warmup_epochs))

    wb = float(boundary_max_weight) * ratio
    wr = float(rim_max_weight) * ratio
    return wb, wr


def compute_b2_rampwarmup_loss(
    outputs,
    masks,
    epoch,
    aux_warmup_epochs=30,
    boundary_max_weight=0.20,
    rim_max_weight=0.10,
):
    """
    B2-rampwarmup loss.

    Original B2 uses two-stage weights:
      epoch 1-20:    0.10 * L_boundary + 0.05 * L_rim
      epoch 21-end:  0.20 * L_boundary + 0.10 * L_rim

    This version uses continuous linear ramp-up:
      wb = boundary_max_weight * min(1, epoch / aux_warmup_epochs)
      wr = rim_max_weight      * min(1, epoch / aux_warmup_epochs)
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)
    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    wb, wr = get_ramp_aux_weights(
        epoch=epoch,
        warmup_epochs=aux_warmup_epochs,
        boundary_max_weight=boundary_max_weight,
        rim_max_weight=rim_max_weight,
    )

    total = loss_seg + wb * loss_boundary + wr * loss_rim

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "wb": wb,
        "wr": wr,
    }


def run_one_epoch_b2_rampwarmup(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    aux_warmup_epochs=30,
    boundary_max_weight=0.20,
    rim_max_weight=0.10,
):
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_wb = 0.0
    last_wr = 0.0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_b2_rampwarmup_loss(
                outputs,
                masks,
                epoch=epoch,
                aux_warmup_epochs=aux_warmup_epochs,
                boundary_max_weight=boundary_max_weight,
                rim_max_weight=rim_max_weight,
            )

            if train:
                loss.backward()
                optimizer.step()

        last_wb = loss_items["wb"]
        last_wr = loss_items["wr"]

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
        "wb": last_wb,
        "wr": last_wr,
    }
'''

marker = "\n\ndef main(args):"
if marker not in s:
    raise RuntimeError("Cannot find def main(args) marker.")
s = s.replace(marker, insert_code + marker)

s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B2-rampwarmup: B2 BRR-RBSG with continuous boundary-rim auxiliary weight ramp-up"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B2-rampwarmup keeps the B2 BRR-RBSG architecture and replaces the original two-stage boundary/rim auxiliary weights with continuous linear ramp-up during early training."',
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
        "aux_weight_schedule": "continuous linear ramp-up",
        "boundary_weight": "boundary_max_weight * min(1, epoch / aux_warmup_epochs)",
        "rim_weight": "rim_max_weight * min(1, epoch / aux_warmup_epochs)",
        "aux_warmup_epochs": args.aux_warmup_epochs,
        "boundary_max_weight": args.boundary_max_weight,
        "rim_max_weight": args.rim_max_weight
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B2-rampwarmup BRR-RBSG ResNet34-U-Net")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("aux_warmup_epochs =", args.aux_warmup_epochs)\n    print("boundary_max_weight =", args.boundary_max_weight)\n    print("rim_max_weight      =", args.rim_max_weight)\n',
)

repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b2_rampwarmup(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            aux_warmup_epochs=args.aux_warmup_epochs,
            boundary_max_weight=args.boundary_max_weight,
            rim_max_weight=args.rim_max_weight,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b2_rampwarmup(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            aux_warmup_epochs=args.aux_warmup_epochs,
            boundary_max_weight=args.boundary_max_weight,
            rim_max_weight=args.rim_max_weight,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b2_rampwarmup(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        aux_warmup_epochs=args.aux_warmup_epochs,
        boundary_max_weight=args.boundary_max_weight,
        rim_max_weight=args.rim_max_weight,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b2_rampwarmup(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        aux_warmup_epochs=args.aux_warmup_epochs,
        boundary_max_weight=args.boundary_max_weight,
        rim_max_weight=args.rim_max_weight,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b2_rampwarmup(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        aux_warmup_epochs=args.aux_warmup_epochs,
        boundary_max_weight=args.boundary_max_weight,
        rim_max_weight=args.rim_max_weight,
    )''',
}

for old, new in repls.items():
    if old not in s:
        raise RuntimeError(f"Cannot find call block: {old[:80]}")
    s = s.replace(old, new)

# Print current weights in epoch line.
s = s.replace(
'''            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
''',
'''            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"wb={train_metrics['wb']:.4f} wr={train_metrics['wr']:.4f}"
''',
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--aux-warmup-epochs", type=int, default=30)
    parser.add_argument("--boundary-max-weight", type=float, default=0.20)
    parser.add_argument("--rim-max-weight", type=float, default=0.10)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse insertion marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Create full training sbatch script ====="
cat > submitjob_b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b2_ramp30
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b2_rampwarmup30_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py

echo "===== RUN B2-rampwarmup30: BRR-RBSG + continuous aux weight ramp-up ====="
python src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py \
  --run-name b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6 \
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
  --aux-warmup-epochs 30 \
  --boundary-max-weight 0.20 \
  --rim-max-weight 0.10

echo "===== DONE B2-rampwarmup30 ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > submitjob_smoke_b2_rbsg_rampwarmup30_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b2rw
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b2_rampwarmup30_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py

echo "===== SMOKE RUN B2-rampwarmup30: 3 epochs ====="
python src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py \
  --run-name smoke_b2_rbsg_rampwarmup30_e3 \
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
  --aux-warmup-epochs 30 \
  --boundary-max-weight 0.20 \
  --rim-max-weight 0.10

echo "===== DONE SMOKE B2-rampwarmup30 ====="
EOF

chmod +x submitjob_b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6.sh
chmod +x submitjob_smoke_b2_rbsg_rampwarmup30_e3.sh

echo "===== Shell syntax check ====="
bash -n submitjob_b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6.sh
bash -n submitjob_smoke_b2_rbsg_rampwarmup30_e3.sh

echo "===== Python syntax check ====="
module load anaconda3/4.12.0 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
python -m py_compile src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b2_rbsg_rampwarmup_e200.py
ls -lh submitjob_b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6.sh
ls -lh submitjob_smoke_b2_rbsg_rampwarmup30_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch submitjob_smoke_b2_rbsg_rampwarmup30_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b2_rampwarmup30_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_b2_rbsg_rampwarmup30_auglite_e200_constlr_bs6.sh"
