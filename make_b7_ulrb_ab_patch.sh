#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

python - <<'PY'
from pathlib import Path
import re

SRC = Path("src/train_resnet34_unet_brr_b2_rbsg_e200.py")

variants = [
    {
        "tag": "ulrb_a",
        "dst": Path("src/train_resnet34_unet_brr_b7_ulrb_a_e200.py"),
        "class_name": "ResNet34UNetBRRB7ULRBA",
        "experiment": "B7A: B2 BRR-RBSG + ULRB-basic",
        "note": "B7A keeps original B2 RBSG-lite and adds an uncertainty-guided lesion recovery branch. ULRB uses refined decoder feature, shallow encoder detail, mask probability and uncertainty map to predict residual mask correction: final_logits = mask_logits + alpha * delta_logits.",
        "default_run": "b7a_b2_ulrb_basic_auglite_e200_constlr_bs6",
        "gated": False,
    },
    {
        "tag": "ulrb_b",
        "dst": Path("src/train_resnet34_unet_brr_b7_ulrb_b_e200.py"),
        "class_name": "ResNet34UNetBRRB7ULRBB",
        "experiment": "B7B: B2 BRR-RBSG + ULRB-gated",
        "note": "B7B keeps original B2 RBSG-lite and adds an uncertainty-guided lesion recovery branch. Compared with B7A, B7B uses a soft uncertainty gate: final_logits = mask_logits + alpha * (0.5 + 0.5U) * delta_logits.",
        "default_run": "b7b_b2_ulrb_gated_auglite_e200_constlr_bs6",
        "gated": True,
    },
]

base = SRC.read_text(encoding="utf-8")

old_import = '''from train_resnet34_unet_brr_b1_aux_e200 import (
    run_one_epoch_b1,
)
'''
new_import = '''from train_resnet34_unet_brr_b1_aux_e200 import (
    bce_dice_loss,
    mask_to_boundary_and_rim,
)
'''

if old_import not in base:
    raise RuntimeError("Cannot find old B1 import block in B2 script.")

for v in variants:
    s = base.replace(old_import, new_import)

    gated_literal = "True" if v["gated"] else "False"

    new_model_block = f'''
class ULRB(nn.Module):
    """
    Uncertainty-guided Lesion Recovery Branch.

    Inputs:
      - refined decoder feature from B2 RBSG-lite
      - shallow encoder feature for high-resolution local detail
      - mask probability P = sigmoid(mask_logits)
      - uncertainty map U = 4P(1-P)

    Output:
      - residual correction delta_logits
      - final_logits = mask_logits + alpha * delta_logits
        or gated version:
        final_logits = mask_logits + alpha * (0.5 + 0.5U) * delta_logits
    """
    def __init__(
        self,
        refined_channels,
        shallow_channels,
        detail_channels=16,
        hidden_channels=32,
        alpha=0.20,
        gated={gated_literal},
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.gated = bool(gated)

        self.refined_proj = nn.Sequential(
            nn.Conv2d(refined_channels, detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True),
        )

        self.shallow_proj = nn.Sequential(
            nn.Conv2d(shallow_channels, detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True),
        )

        in_ch = detail_channels + detail_channels + 1 + 1

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, refined_feature, shallow_feature, mask_logits):
        target_size = mask_logits.shape[2:]

        refined_detail = self.refined_proj(refined_feature)
        refined_detail = F.interpolate(
            refined_detail,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        shallow_detail = self.shallow_proj(shallow_feature)
        shallow_detail = F.interpolate(
            shallow_detail,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        prob = torch.sigmoid(mask_logits)
        uncertainty = 4.0 * prob * (1.0 - prob)

        x = torch.cat(
            [
                refined_detail,
                shallow_detail,
                prob,
                uncertainty,
            ],
            dim=1,
        )

        delta_logits = self.fuse(x)

        if self.gated:
            soft_gate = 0.5 + 0.5 * uncertainty
            final_logits = mask_logits + self.alpha * soft_gate * delta_logits
        else:
            final_logits = mask_logits + self.alpha * delta_logits

        return final_logits, delta_logits, uncertainty


class {v["class_name"]}(nn.Module):
    """
    B7 ULRB variant based on original B2.

    It keeps:
      - final-stage RBSG-lite
      - rb=3, rr=9
      - lambda_pos=0.20, lambda_rim=0.10
      - original boundary/rim auxiliary supervision

    It adds:
      - ULRB lesion recovery branch after B2 mask logits
      - segmentation loss is computed on final_logits
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        ulrb_alpha=0.20,
        ulrb_hidden=32,
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

        # SMP ResNet encoder usually returns:
        # features[0]: input-like, H
        # features[1]: H/2
        # features[2]: H/4  ← shallow detail used here
        try:
            shallow_ch = int(self.unet.encoder.out_channels[2])
        except Exception:
            shallow_ch = 64

        self.shallow_index = 2

        self.ulrb = ULRB(
            refined_channels=dec_ch,
            shallow_channels=shallow_ch,
            detail_channels=16,
            hidden_channels=int(ulrb_hidden),
            alpha=float(ulrb_alpha),
            gated={gated_literal},
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)

        mask_base_logits = self.mask_head(refined)

        if len(features) > self.shallow_index:
            shallow_feature = features[self.shallow_index]
        else:
            shallow_feature = features[1]

        final_mask_logits, delta_logits, uncertainty = self.ulrb(
            refined,
            shallow_feature,
            mask_base_logits,
        )

        if not return_aux:
            return final_mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        return {{
            "mask": final_mask_logits,
            "mask_base": mask_base_logits,
            "delta": delta_logits,
            "uncertainty": uncertainty,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }}


def compute_b7_ulrb_loss(outputs, masks, epoch, warmup_epochs=20):
    """
    B7 ULRB loss.

    The main segmentation loss is computed on final mask logits:
      outputs["mask"]

    Boundary/rim auxiliary supervision follows original B2:
      rb=3, rr=9
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)
    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
    else:
        wb = 0.20
        wr = 0.10

    total = loss_seg + wb * loss_boundary + wr * loss_rim

    with torch.no_grad():
        delta_abs = outputs["delta"].detach().abs().mean()
        uncertainty_mean = outputs["uncertainty"].detach().mean()

    return total, {{
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "delta_abs": float(delta_abs.cpu()),
        "uncertainty_mean": float(uncertainty_mean.cpu()),
        "wb": wb,
        "wr": wr,
    }}


def run_one_epoch_b7_ulrb(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
):
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_items = {{
        "delta_abs": 0.0,
        "uncertainty_mean": 0.0,
    }}

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_b7_ulrb_loss(
                outputs,
                masks,
                epoch=epoch,
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

    return {{
        "loss": total_loss / max(n_batches, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "delta_abs": last_items["delta_abs"],
        "uncertainty_mean": last_items["uncertainty_mean"],
    }}
'''

    pattern = r'class ResNet34UNetBRRB2RBSG\(nn\.Module\):.*?\n\ndef main\(args\):'
    if not re.search(pattern, s, flags=re.S):
        raise RuntimeError("Cannot locate original B2 model class block.")

    s = re.sub(pattern, new_model_block + "\n\ndef main(args):", s, flags=re.S)

    # Config updates.
    s = s.replace(
        'config["model_class"] = "ResNet34UNetBRRB2RBSG"',
        f'config["model_class"] = "{v["class_name"]}"',
    )
    s = s.replace(
        'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
        f'config["experiment"] = "{v["experiment"]}"',
    )
    s = s.replace(
        'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
        f'config["note"] = "{v["note"]}"',
    )

    old_rbsg_block = '''    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
'''
    new_rbsg_block = f'''    config["rbsg"] = {{
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }}
    config["ulrb"] = {{
        "enabled": True,
        "variant": "{v["tag"]}",
        "gated": {str(v["gated"])},
        "alpha": args.ulrb_alpha,
        "hidden_channels": args.ulrb_hidden,
        "inputs": ["refined decoder feature", "shallow encoder feature", "mask probability", "uncertainty map"],
        "uncertainty": "U = 4P(1-P)",
        "fusion": "residual mask correction"
    }}
'''
    if old_rbsg_block in s:
        s = s.replace(old_rbsg_block, new_rbsg_block)
    else:
        print(f"[WARN] rbsg config block not found for {v['tag']}")

    s = s.replace(
'''    model = ResNet34UNetBRRB2RBSG(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
''',
f'''    model = {v["class_name"]}(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        ulrb_alpha=args.ulrb_alpha,
        ulrb_hidden=args.ulrb_hidden,
    ).to(device)
'''
    )

    s = s.replace(
        'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
        f'print("Model ready: {v["experiment"]}")',
    )
    s = s.replace(
        '    print("lr              =", args.lr)\n',
        '    print("lr              =", args.lr)\n    print("ulrb_alpha      =", args.ulrb_alpha)\n    print("ulrb_hidden     =", args.ulrb_hidden)\n',
    )

    repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b7_ulrb(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b7_ulrb(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b7_ulrb(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b7_ulrb(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b7_ulrb(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )''',
    }

    for old, new in repls.items():
        if old not in s:
            raise RuntimeError(f"Cannot find call block in {v['tag']}: {old[:80]}")
        s = s.replace(old, new)

    s = s.replace(
'''            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
''',
'''            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"delta={train_metrics['delta_abs']:.4f} u={train_metrics['uncertainty_mean']:.4f}"
'''
    )

    s = s.replace(
        'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
        f'parser.add_argument("--run-name", type=str, default="{v["default_run"]}")',
    )

    arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
    arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--ulrb-alpha", type=float, default=0.20)
    parser.add_argument("--ulrb-hidden", type=int, default=32)
'''
    if arg_marker not in s:
        raise RuntimeError(f"Cannot find argparse marker in {v['tag']}")
    s = s.replace(arg_marker, arg_insert)

    v["dst"].write_text(s, encoding="utf-8")
    print(f"Patched: {v['dst']}")

PY

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_b7_ulrb_a_e200.py
python -m py_compile src/train_resnet34_unet_brr_b7_ulrb_b_e200.py

echo "===== Create smoke and full sbatch scripts ====="

cat > submitjob_smoke_b7_ulrb_a_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_uA
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b7_ulrb_a_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b7_ulrb_a_e200.py

echo "===== SMOKE RUN B7A ULRB-basic ====="
python src/train_resnet34_unet_brr_b7_ulrb_a_e200.py \
  --run-name smoke_b7a_ulrb_basic_e3 \
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
  --ulrb-alpha 0.20 \
  --ulrb-hidden 32

echo "===== DONE SMOKE B7A ====="
EOF

cat > submitjob_smoke_b7_ulrb_b_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_uB
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b7_ulrb_b_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b7_ulrb_b_e200.py

echo "===== SMOKE RUN B7B ULRB-gated ====="
python src/train_resnet34_unet_brr_b7_ulrb_b_e200.py \
  --run-name smoke_b7b_ulrb_gated_e3 \
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
  --ulrb-alpha 0.20 \
  --ulrb-hidden 32

echo "===== DONE SMOKE B7B ====="
EOF

cat > submitjob_b7_ulrb_a_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b7_ulrbA
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b7_ulrb_a_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b7_ulrb_a_e200.py

echo "===== RUN B7A: B2 + ULRB-basic ====="
python src/train_resnet34_unet_brr_b7_ulrb_a_e200.py \
  --run-name b7a_b2_ulrb_basic_auglite_e200_constlr_bs6 \
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
  --ulrb-alpha 0.20 \
  --ulrb-hidden 32

echo "===== DONE B7A ====="
EOF

cat > submitjob_b7_ulrb_b_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b7_ulrbB
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b7_ulrb_b_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b7_ulrb_b_e200.py

echo "===== RUN B7B: B2 + ULRB-gated ====="
python src/train_resnet34_unet_brr_b7_ulrb_b_e200.py \
  --run-name b7b_b2_ulrb_gated_auglite_e200_constlr_bs6 \
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
  --ulrb-alpha 0.20 \
  --ulrb-hidden 32

echo "===== DONE B7B ====="
EOF

chmod +x submitjob_smoke_b7_ulrb_a_e3.sh
chmod +x submitjob_smoke_b7_ulrb_b_e3.sh
chmod +x submitjob_b7_ulrb_a_auglite_e200_constlr_bs6.sh
chmod +x submitjob_b7_ulrb_b_auglite_e200_constlr_bs6.sh

echo "===== Shell syntax check ====="
bash -n submitjob_smoke_b7_ulrb_a_e3.sh
bash -n submitjob_smoke_b7_ulrb_b_e3.sh
bash -n submitjob_b7_ulrb_a_auglite_e200_constlr_bs6.sh
bash -n submitjob_b7_ulrb_b_auglite_e200_constlr_bs6.sh

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b7_ulrb_a_e200.py
ls -lh src/train_resnet34_unet_brr_b7_ulrb_b_e200.py
ls -lh submitjob_smoke_b7_ulrb_a_e3.sh
ls -lh submitjob_smoke_b7_ulrb_b_e3.sh
ls -lh submitjob_b7_ulrb_a_auglite_e200_constlr_bs6.sh
ls -lh submitjob_b7_ulrb_b_auglite_e200_constlr_bs6.sh

echo "===== Next step ====="
echo "1) Submit smoke tests:"
echo "   sbatch submitjob_smoke_b7_ulrb_a_e3.sh"
echo "   sbatch submitjob_smoke_b7_ulrb_b_e3.sh"
echo
echo "2) Watch smoke logs:"
echo "   tail -f runs/logs/smoke_b7_ulrb_a_*.out"
echo "   tail -f runs/logs/smoke_b7_ulrb_b_*.out"
echo
echo "3) If both smoke tests pass, submit full runs:"
echo "   sbatch submitjob_b7_ulrb_a_auglite_e200_constlr_bs6.sh"
echo "   sbatch submitjob_b7_ulrb_b_auglite_e200_constlr_bs6.sh"
