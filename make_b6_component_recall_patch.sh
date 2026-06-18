#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b6_component_recall_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B6 ====="
cp -p "$SRC" "$DST"

echo "===== Patch B6 script: add Component-Recall Loss ====="
python - <<'PY'
from pathlib import Path

p = Path("src/train_resnet34_unet_brr_b6_component_recall_e200.py")
s = p.read_text(encoding="utf-8")

# Make sure needed imports exist.
if "import math" not in s:
    s = s.replace("import os\n", "import os\nimport math\n", 1) if "import os\n" in s else "import math\n" + s

if "import numpy as np" not in s:
    s = s.replace("import math\n", "import math\nimport numpy as np\n", 1)

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

def _connected_components_numpy(binary):
    """
    Fallback 8-connected component labeling for 2D binary mask.
    Used only if scipy/cv2 is unavailable.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue

            current += 1
            stack = [(y, x)]
            labels[y, x] = current

            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary[ny, nx] != 0 and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                stack.append((ny, nx))

    return labels, current


def connected_components_2d(binary):
    """
    Return label map and number of foreground components.
    Prefer scipy/cv2 if available; fallback to pure numpy labeling.
    """
    binary = binary.astype(np.uint8)

    try:
        from scipy import ndimage
        structure = np.ones((3, 3), dtype=np.int32)
        labels, n = ndimage.label(binary, structure=structure)
        return labels.astype(np.int32), int(n)
    except Exception:
        pass

    try:
        import cv2
        n, labels = cv2.connectedComponents(binary, connectivity=8)
        return labels.astype(np.int32), int(n - 1)
    except Exception:
        pass

    return _connected_components_numpy(binary)


def component_recall_loss(
    mask_logits,
    masks,
    topk_ratio=0.30,
    gamma=2.0,
    min_area=3,
    max_components=64,
    eps=1e-6,
):
    """
    Component-Recall Loss.

    Motivation:
      Dice can be dominated by large lesions/background.
      Missing one tiny caries component may not greatly change global Dice,
      but it matters clinically.

    For each GT connected component C_i:
      q_i = mean_topk(P_mask inside C_i)
      L_i = (1 - q_i)^gamma

    Smaller components get larger normalized weights:
      w_i = 1 / sqrt(area_i + eps)

    This loss uses GT mask only as supervision target. It is not model input.
    """
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    probs = torch.sigmoid(mask_logits)
    bsz = probs.shape[0]
    device = probs.device

    total = probs.new_tensor(0.0)
    valid_samples = 0

    for b in range(bsz):
        gt_np = (masks[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        labels, num = connected_components_2d(gt_np)

        if num <= 0:
            continue

        comp_losses = []
        comp_weights = []

        # Sort components by area, keep at most max_components to avoid rare overhead.
        areas = []
        for cid in range(1, num + 1):
            area = int((labels == cid).sum())
            if area >= int(min_area):
                areas.append((cid, area))

        if not areas:
            continue

        areas = sorted(areas, key=lambda x: x[1])[: int(max_components)]

        for cid, area in areas:
            comp_np = labels == cid
            comp_mask = torch.from_numpy(comp_np).to(device=device, dtype=torch.bool)

            vals = probs[b, 0][comp_mask]
            if vals.numel() == 0:
                continue

            k = max(1, int(math.ceil(float(topk_ratio) * vals.numel())))
            k = min(k, vals.numel())

            q = torch.topk(vals, k=k, largest=True).values.mean()
            comp_loss = torch.clamp(1.0 - q, min=0.0).pow(float(gamma))

            # Smaller component -> larger weight.
            comp_weight = 1.0 / math.sqrt(float(area) + float(eps))

            comp_losses.append(comp_loss)
            comp_weights.append(comp_weight)

        if not comp_losses:
            continue

        loss_vec = torch.stack(comp_losses)
        weight_vec = torch.tensor(comp_weights, device=device, dtype=loss_vec.dtype)

        sample_loss = (loss_vec * weight_vec).sum() / (weight_vec.sum() + eps)
        total = total + sample_loss
        valid_samples += 1

    if valid_samples == 0:
        return probs.new_tensor(0.0)

    return total / float(valid_samples)


def compute_b6_loss(
    outputs,
    masks,
    epoch,
    warmup_epochs=20,
    component_weight=0.15,
    component_topk_ratio=0.30,
    component_gamma=2.0,
    component_min_area=3,
    component_max_count=64,
):
    """
    B6 loss = B2 loss + Component-Recall Loss.

    B2:
      L_seg + wb * L_boundary + wr * L_rim

    B6:
      L_seg + wb * L_boundary + wr * L_rim + wc * L_component
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)
    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    loss_component = component_recall_loss(
        mask_logits,
        masks,
        topk_ratio=component_topk_ratio,
        gamma=component_gamma,
        min_area=component_min_area,
        max_components=component_max_count,
    )

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
        wc = min(float(component_weight), 0.05)
    else:
        wb = 0.20
        wr = 0.10
        wc = float(component_weight)

    total = loss_seg + wb * loss_boundary + wr * loss_rim + wc * loss_component

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "loss_component": float(loss_component.detach().cpu()),
        "wb": wb,
        "wr": wr,
        "wc": wc,
    }


def run_one_epoch_b6(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    component_weight=0.15,
    component_topk_ratio=0.30,
    component_gamma=2.0,
    component_min_area=3,
    component_max_count=64,
):
    """
    Train/eval one epoch for B6.

    Metrics are still computed only from final mask logits.
    Component-Recall Loss only affects loss optimization.
    """
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_wc = 0.0
    last_component_loss = 0.0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_b6_loss(
                outputs,
                masks,
                epoch=epoch,
                component_weight=component_weight,
                component_topk_ratio=component_topk_ratio,
                component_gamma=component_gamma,
                component_min_area=component_min_area,
                component_max_count=component_max_count,
            )

            if train:
                loss.backward()
                optimizer.step()

        last_wc = loss_items["wc"]
        last_component_loss = loss_items["loss_component"]

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
        "wc": last_wc,
        "loss_component": last_component_loss,
    }
'''

marker = "\n\ndef main(args):"
if marker not in s:
    raise RuntimeError("Cannot find def main(args) marker.")
s = s.replace(marker, insert_code + marker)

s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B6: B2 BRR-RBSG + Component-Recall Loss"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B6 keeps B2 BRR-RBSG architecture and adds Component-Recall Loss to reduce missed small caries components. No extra inference parameter is introduced."',
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
        "component_recall": "component-level recall loss using top-k mean probability inside each GT connected component",
        "component_weight": args.component_weight,
        "component_topk_ratio": args.component_topk_ratio,
        "component_gamma": args.component_gamma,
        "component_min_area": args.component_min_area,
        "component_max_count": args.component_max_count,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim + 0.05*L_component",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim + component_weight*L_component"
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B6 BRR-RBSG + Component-Recall Loss")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("component_weight      =", args.component_weight)\n    print("component_topk_ratio  =", args.component_topk_ratio)\n    print("component_gamma       =", args.component_gamma)\n    print("component_min_area    =", args.component_min_area)\n    print("component_max_count   =", args.component_max_count)\n',
)

repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b6(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            component_weight=args.component_weight,
            component_topk_ratio=args.component_topk_ratio,
            component_gamma=args.component_gamma,
            component_min_area=args.component_min_area,
            component_max_count=args.component_max_count,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b6(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            component_weight=args.component_weight,
            component_topk_ratio=args.component_topk_ratio,
            component_gamma=args.component_gamma,
            component_min_area=args.component_min_area,
            component_max_count=args.component_max_count,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b6(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        component_weight=args.component_weight,
        component_topk_ratio=args.component_topk_ratio,
        component_gamma=args.component_gamma,
        component_min_area=args.component_min_area,
        component_max_count=args.component_max_count,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b6(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        component_weight=args.component_weight,
        component_topk_ratio=args.component_topk_ratio,
        component_gamma=args.component_gamma,
        component_min_area=args.component_min_area,
        component_max_count=args.component_max_count,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b6(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        component_weight=args.component_weight,
        component_topk_ratio=args.component_topk_ratio,
        component_gamma=args.component_gamma,
        component_min_area=args.component_min_area,
        component_max_count=args.component_max_count,
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
            f"comp={train_metrics['loss_component']:.4f} wc={train_metrics['wc']:.4f}"
''',
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--component-weight", type=float, default=0.15)
    parser.add_argument("--component-topk-ratio", type=float, default=0.30)
    parser.add_argument("--component-gamma", type=float, default=2.0)
    parser.add_argument("--component-min-area", type=int, default=3)
    parser.add_argument("--component-max-count", type=int, default=64)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse insertion marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Create full training sbatch script ====="
cat > submitjob_b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b6_comp
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b6_component_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b6_component_recall_e200.py

echo "===== RUN B6: BRR-RBSG + Component-Recall Loss ====="
python src/train_resnet34_unet_brr_b6_component_recall_e200.py \
  --run-name b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6 \
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
  --component-weight 0.15 \
  --component-topk-ratio 0.30 \
  --component-gamma 2.0 \
  --component-min-area 3 \
  --component-max-count 64

echo "===== DONE B6 ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > submitjob_smoke_b6_brr_rbsg_component_recall_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b6
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b6_component_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b6_component_recall_e200.py

echo "===== SMOKE RUN B6: 3 epochs, batch size 2 ====="
python src/train_resnet34_unet_brr_b6_component_recall_e200.py \
  --run-name smoke_b6_brr_rbsg_component_recall_e3 \
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
  --component-weight 0.15 \
  --component-topk-ratio 0.30 \
  --component-gamma 2.0 \
  --component-min-area 3 \
  --component-max-count 64

echo "===== DONE SMOKE B6 ====="
EOF

chmod +x submitjob_b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6.sh
chmod +x submitjob_smoke_b6_brr_rbsg_component_recall_e3.sh

echo "===== Shell syntax check ====="
bash -n submitjob_b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6.sh
bash -n submitjob_smoke_b6_brr_rbsg_component_recall_e3.sh

echo "===== Python syntax check ====="
module load anaconda3/4.12.0 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
python -m py_compile src/train_resnet34_unet_brr_b6_component_recall_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b6_component_recall_e200.py
ls -lh submitjob_b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6.sh
ls -lh submitjob_smoke_b6_brr_rbsg_component_recall_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch submitjob_smoke_b6_brr_rbsg_component_recall_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b6_component_*.out"
echo
echo "3) If smoke passes, submit full B6:"
echo "   sbatch submitjob_b6_brr_rbsg_component_recall_auglite_e200_constlr_bs6.sh"
