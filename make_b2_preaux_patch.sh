#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b2_preaux_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B2-preaux ====="
cp -p "$SRC" "$DST"

echo "===== Patch B2-preaux script ====="
python - <<'PY'
from pathlib import Path

p = Path("src/train_resnet34_unet_brr_b2_preaux_e200.py")
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
    raise RuntimeError("Cannot find old B1 import block.")
s = s.replace(old_import, new_import)

old_return = '''        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
        }
'''
new_return = '''        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }
'''
if old_return not in s:
    raise RuntimeError("Cannot find return aux dict block.")
s = s.replace(old_return, new_return)

insert_code = r'''

def compute_b2_preaux_loss(
    outputs,
    masks,
    epoch,
    warmup_epochs=20,
    pre_boundary_weight=0.10,
    pre_rim_weight=0.05,
):
    """
    B2-preaux loss.

    It keeps original B2:
      L_seg + wb * L_boundary_final + wr * L_rim_final

    And adds direct supervision to the pre-gate hints:
      + wpb * L_boundary_pre + wpr * L_rim_pre

    rb=3, rr=9 are kept exactly the same as original B2/B1.
    lambda_pos=0.20, lambda_rim=0.10 are kept inside RBSGLite.
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]
    boundary_pre_logits = outputs["boundary_pre"]
    rim_pre_logits = outputs["rim_pre"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)

    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    loss_boundary_pre = bce_dice_loss(boundary_pre_logits, boundary_gt)
    loss_rim_pre = F.binary_cross_entropy_with_logits(rim_pre_logits, rim_gt)

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
        wpb = min(float(pre_boundary_weight), 0.05)
        wpr = min(float(pre_rim_weight), 0.025)
    else:
        wb = 0.20
        wr = 0.10
        wpb = float(pre_boundary_weight)
        wpr = float(pre_rim_weight)

    total = (
        loss_seg
        + wb * loss_boundary
        + wr * loss_rim
        + wpb * loss_boundary_pre
        + wpr * loss_rim_pre
    )

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "loss_boundary_pre": float(loss_boundary_pre.detach().cpu()),
        "loss_rim_pre": float(loss_rim_pre.detach().cpu()),
        "wb": wb,
        "wr": wr,
        "wpb": wpb,
        "wpr": wpr,
    }


def run_one_epoch_b2_preaux(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    pre_boundary_weight=0.10,
    pre_rim_weight=0.05,
):
    """
    Train/eval one epoch for B2-preaux.

    Metrics are computed only from final mask logits.
    Pre-boundary/pre-rim supervision only affects training loss.
    """
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_items = {
        "loss_boundary_pre": 0.0,
        "loss_rim_pre": 0.0,
        "wpb": 0.0,
        "wpr": 0.0,
    }

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_b2_preaux_loss(
                outputs,
                masks,
                epoch=epoch,
                pre_boundary_weight=pre_boundary_weight,
                pre_rim_weight=pre_rim_weight,
            )

            if train:
                loss.backward()
                optimizer.step()

        last_items = loss_items

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
        "loss_boundary_pre": last_items["loss_boundary_pre"],
        "loss_rim_pre": last_items["loss_rim_pre"],
        "wpb": last_items["wpb"],
        "wpr": last_items["wpr"],
    }
'''

marker = "\n\ndef main(args):"
if marker not in s:
    raise RuntimeError("Cannot find def main(args) marker.")
s = s.replace(marker, insert_code + marker)

s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B2-preaux: B2 BRR-RBSG with pre-boundary/pre-rim supervision"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B2-preaux keeps B2 architecture and lambda_pos=0.20/lambda_rim=0.10/rb=3/rr=9, and adds direct supervision to boundary_head_pre and rim_head_pre so that RBSG hints become more semantically reliable."',
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
        "boundary_final": "BCEWithLogits + Dice",
        "rim_final": "BCEWithLogits",
        "boundary_pre": "BCEWithLogits + Dice",
        "rim_pre": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim + 0.05*L_boundary_pre + 0.025*L_rim_pre",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim + pre_boundary_weight*L_boundary_pre + pre_rim_weight*L_rim_pre",
        "pre_boundary_weight": args.pre_boundary_weight,
        "pre_rim_weight": args.pre_rim_weight
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B2-preaux BRR-RBSG ResNet34-U-Net")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("pre_boundary_weight =", args.pre_boundary_weight)\n    print("pre_rim_weight      =", args.pre_rim_weight)\n',
)

repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b2_preaux(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            pre_boundary_weight=args.pre_boundary_weight,
            pre_rim_weight=args.pre_rim_weight,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b2_preaux(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            pre_boundary_weight=args.pre_boundary_weight,
            pre_rim_weight=args.pre_rim_weight,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b2_preaux(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        pre_boundary_weight=args.pre_boundary_weight,
        pre_rim_weight=args.pre_rim_weight,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b2_preaux(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        pre_boundary_weight=args.pre_boundary_weight,
        pre_rim_weight=args.pre_rim_weight,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b2_preaux(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        pre_boundary_weight=args.pre_boundary_weight,
        pre_rim_weight=args.pre_rim_weight,
    )''',
}

for old, new in repls.items():
    if old not in s:
        raise RuntimeError(f"Cannot find call block: {old[:80]}")
    s = s.replace(old, new)

s = s.replace(
'''            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
''',
'''            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"pre_bd={train_metrics['loss_boundary_pre']:.4f} pre_rim={train_metrics['loss_rim_pre']:.4f}"
''',
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b2_preaux_rbsg_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--pre-boundary-weight", type=float, default=0.10)
    parser.add_argument("--pre-rim-weight", type=float, default=0.05)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse insertion marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Create full training sbatch script ====="
cat > submitjob_b2_preaux_rbsg_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b2_preaux
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b2_preaux_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== RUN B2-preaux: BRR-RBSG + pre-boundary/pre-rim supervision ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name b2_preaux_rbsg_auglite_e200_constlr_bs6 \
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
  --pre-boundary-weight 0.10 \
  --pre-rim-weight 0.05

echo "===== DONE B2-preaux ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > submitjob_smoke_b2_preaux_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b2pa
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b2_preaux_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== SMOKE RUN B2-preaux: 3 epochs ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name smoke_b2_preaux_e3 \
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
  --pre-boundary-weight 0.10 \
  --pre-rim-weight 0.05

echo "===== DONE SMOKE B2-preaux ====="
EOF

chmod +x submitjob_b2_preaux_rbsg_auglite_e200_constlr_bs6.sh
chmod +x submitjob_smoke_b2_preaux_e3.sh

echo "===== Shell syntax check ====="
bash -n submitjob_b2_preaux_rbsg_auglite_e200_constlr_bs6.sh
bash -n submitjob_smoke_b2_preaux_e3.sh

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b2_preaux_e200.py
ls -lh submitjob_b2_preaux_rbsg_auglite_e200_constlr_bs6.sh
ls -lh submitjob_smoke_b2_preaux_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch submitjob_smoke_b2_preaux_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b2_preaux_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_b2_preaux_rbsg_auglite_e200_constlr_bs6.sh"
