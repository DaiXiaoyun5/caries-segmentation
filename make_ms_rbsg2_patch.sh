#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_ms_rbsg2_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to MS-RBSG-2 ====="
cp -p "$SRC" "$DST"

echo "===== Patch MS-RBSG-2 script ====="
python - <<'PY'
from pathlib import Path
import re

p = Path("src/train_resnet34_unet_brr_ms_rbsg2_e200.py")
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

new_model_class = r'''
class ResNet34UNetBRRMSRBSG2(nn.Module):
    """
    MS-RBSG-2:
      - keeps original final RBSG-lite of B2
      - additionally applies RBSG-lite at decoder 1/8 and 1/4 stages
      - uses rb=3, rr=9, lambda_pos=0.20, lambda_rim=0.10
      - final inference output is still only mask logits
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
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

        decoder_channels = self._get_decoder_channels()
        if len(decoder_channels) < 3:
            # SMP default for resnet34 Unet decoder is usually [256,128,64,32,16]
            decoder_channels = [256, 128, 64, 32, dec_ch]

        # Decoder block index mapping in SMP Unet:
        # block 0: 1/16, block 1: 1/8, block 2: 1/4, block 3: 1/2, block 4: full
        self.ms_targets = {
            1: "1_8",
            2: "1_4",
        }

        ch_1_8 = decoder_channels[1]
        ch_1_4 = decoder_channels[2]

        self.ms_boundary_head_pre = nn.ModuleDict({
            "1_8": nn.Conv2d(ch_1_8, 1, kernel_size=1),
            "1_4": nn.Conv2d(ch_1_4, 1, kernel_size=1),
        })
        self.ms_rim_head_pre = nn.ModuleDict({
            "1_8": nn.Conv2d(ch_1_8, 1, kernel_size=1),
            "1_4": nn.Conv2d(ch_1_4, 1, kernel_size=1),
        })
        self.ms_rbsg = nn.ModuleDict({
            "1_8": RBSGLite(ch_1_8, lambda_pos=0.20, lambda_rim=0.10),
            "1_4": RBSGLite(ch_1_4, lambda_pos=0.20, lambda_rim=0.10),
        })

        # Original final B2 RBSG-lite.
        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def _get_decoder_channels(self):
        channels = []

        for block in self.unet.decoder.blocks:
            out_ch = None

            conv2 = getattr(block, "conv2", None)
            if conv2 is not None:
                for m in conv2.modules():
                    if isinstance(m, nn.Conv2d):
                        out_ch = m.out_channels

            if out_ch is None:
                conv1 = getattr(block, "conv1", None)
                if conv1 is not None:
                    for m in conv1.modules():
                        if isinstance(m, nn.Conv2d):
                            out_ch = m.out_channels

            if out_ch is None:
                for m in reversed(list(block.modules())):
                    if isinstance(m, nn.Conv2d):
                        out_ch = m.out_channels
                        break

            if out_ch is not None:
                channels.append(int(out_ch))

        return channels

    @staticmethod
    def _run_decoder_block(block, x, skip):
        """
        Compatible with common SMP Unet decoder block signatures.
        """
        try:
            return block(x, skip)
        except TypeError:
            pass

        if skip is not None:
            height, width = skip.shape[2], skip.shape[3]
        else:
            height, width = x.shape[2] * 2, x.shape[3] * 2

        try:
            return block(x, height, width, skip)
        except TypeError:
            pass

        try:
            return block(x, height=height, width=width, skip_connection=skip)
        except TypeError as e:
            raise RuntimeError(
                "Cannot run SMP decoder block. Please send the smoke log to ChatGPT."
            ) from e

    def _decode_with_ms_rbsg(self, features):
        """
        Manually runs SMP Unet decoder and inserts RBSG-lite at 1/8 and 1/4 stages.
        """
        decoder = self.unet.decoder

        # This follows SMP UnetDecoder.forward logic:
        # remove first skip with same spatial resolution, reverse features,
        # use deepest feature as head, then decode with skip connections.
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = decoder.center(head)
        ms_outputs = {}

        for i, decoder_block in enumerate(decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = self._run_decoder_block(decoder_block, x, skip)

            if i in self.ms_targets:
                key = self.ms_targets[i]

                b_ms = self.ms_boundary_head_pre[key](x)
                r_ms = self.ms_rim_head_pre[key](x)

                x = self.ms_rbsg[key](x, b_ms, r_ms)

                ms_outputs[f"boundary_{key}"] = b_ms
                ms_outputs[f"rim_{key}"] = r_ms

        return x, ms_outputs

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output, ms_outputs = self._decode_with_ms_rbsg(features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)

        mask_logits = self.mask_head(refined)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        outputs = {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }
        outputs.update(ms_outputs)
        return outputs


def resize_target_to_logits(target, logits):
    return F.interpolate(
        target.float(),
        size=logits.shape[2:],
        mode="nearest",
    )


def compute_ms_rbsg2_loss(
    outputs,
    masks,
    epoch,
    warmup_epochs=20,
    ms_boundary_weight=0.05,
    ms_rim_weight=0.025,
):
    """
    MS-RBSG-2 loss.

    Keeps original B2 final supervision:
      L_seg + wb * L_boundary + wr * L_rim

    Adds conservative multi-scale supervision at 1/8 and 1/4:
      + wmb * L_boundary_1_8 + wmr * L_rim_1_8
      + wmb * L_boundary_1_4 + wmr * L_rim_1_4

    rb=3, rr=9 are unchanged.
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
        wmb = min(float(ms_boundary_weight), 0.025)
        wmr = min(float(ms_rim_weight), 0.0125)
    else:
        wb = 0.20
        wr = 0.10
        wmb = float(ms_boundary_weight)
        wmr = float(ms_rim_weight)

    loss_ms_boundary = mask_logits.new_tensor(0.0)
    loss_ms_rim = mask_logits.new_tensor(0.0)
    n_ms = 0

    for key in ["1_8", "1_4"]:
        b_key = f"boundary_{key}"
        r_key = f"rim_{key}"

        if b_key not in outputs or r_key not in outputs:
            continue

        b_logits = outputs[b_key]
        r_logits = outputs[r_key]

        b_gt_ds = resize_target_to_logits(boundary_gt, b_logits)
        r_gt_ds = resize_target_to_logits(rim_gt, r_logits)

        loss_ms_boundary = loss_ms_boundary + bce_dice_loss(b_logits, b_gt_ds)
        loss_ms_rim = loss_ms_rim + F.binary_cross_entropy_with_logits(r_logits, r_gt_ds)
        n_ms += 1

    if n_ms > 0:
        loss_ms_boundary = loss_ms_boundary / n_ms
        loss_ms_rim = loss_ms_rim / n_ms

    total = (
        loss_seg
        + wb * loss_boundary
        + wr * loss_rim
        + wmb * loss_ms_boundary
        + wmr * loss_ms_rim
    )

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "loss_ms_boundary": float(loss_ms_boundary.detach().cpu()),
        "loss_ms_rim": float(loss_ms_rim.detach().cpu()),
        "wb": wb,
        "wr": wr,
        "wmb": wmb,
        "wmr": wmr,
    }


def run_one_epoch_ms_rbsg2(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    ms_boundary_weight=0.05,
    ms_rim_weight=0.025,
):
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_items = {
        "loss_ms_boundary": 0.0,
        "loss_ms_rim": 0.0,
        "wmb": 0.0,
        "wmr": 0.0,
    }

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_ms_rbsg2_loss(
                outputs,
                masks,
                epoch=epoch,
                ms_boundary_weight=ms_boundary_weight,
                ms_rim_weight=ms_rim_weight,
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
        "loss_ms_boundary": last_items["loss_ms_boundary"],
        "loss_ms_rim": last_items["loss_ms_rim"],
        "wmb": last_items["wmb"],
        "wmr": last_items["wmr"],
    }
'''

pattern = r'class ResNet34UNetBRRB2RBSG\(nn\.Module\):.*?\n\ndef main\(args\):'
m = re.search(pattern, s, flags=re.S)
if not m:
    raise RuntimeError("Cannot locate original B2 model class block.")
s = re.sub(pattern, new_model_class + "\n\ndef main(args):", s, flags=re.S)

s = s.replace(
    'config["model_class"] = "ResNet34UNetBRRB2RBSG"',
    'config["model_class"] = "ResNet34UNetBRRMSRBSG2"',
)
s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "MS-RBSG-2: B2 with RBSG-lite at decoder 1/8 and 1/4 stages"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "MS-RBSG-2 keeps final B2 RBSG-lite and additionally inserts RBSG-lite at decoder 1/8 and 1/4 stages. It uses rb=3, rr=9, lambda_pos=0.20, lambda_rim=0.10. Multi-scale auxiliary supervision is kept conservative to reduce overfitting risk."',
)

old_rbsg_block = '''    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
'''
new_rbsg_block = '''    config["rbsg"] = {
        "position": "final decoder feature + decoder 1/8 + decoder 1/4",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "multi_scale_targets": ["1/8", "1/4"],
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
'''
if old_rbsg_block in s:
    s = s.replace(old_rbsg_block, new_rbsg_block)

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
        "ms_boundary": "BCEWithLogits + Dice at decoder 1/8 and 1/4",
        "ms_rim": "BCEWithLogits at decoder 1/8 and 1/4",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim + 0.025*L_ms_boundary + 0.0125*L_ms_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim + ms_boundary_weight*L_ms_boundary + ms_rim_weight*L_ms_rim",
        "ms_boundary_weight": args.ms_boundary_weight,
        "ms_rim_weight": args.ms_rim_weight
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

s = s.replace(
'''    model = ResNet34UNetBRRB2RBSG(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
''',
'''    model = ResNet34UNetBRRMSRBSG2(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
'''
)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: MS-RBSG-2 ResNet34-U-Net")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("ms_boundary_weight =", args.ms_boundary_weight)\n    print("ms_rim_weight      =", args.ms_rim_weight)\n',
)

repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_ms_rbsg2(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            ms_boundary_weight=args.ms_boundary_weight,
            ms_rim_weight=args.ms_rim_weight,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_ms_rbsg2(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            ms_boundary_weight=args.ms_boundary_weight,
            ms_rim_weight=args.ms_rim_weight,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_ms_rbsg2(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        ms_boundary_weight=args.ms_boundary_weight,
        ms_rim_weight=args.ms_rim_weight,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_ms_rbsg2(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        ms_boundary_weight=args.ms_boundary_weight,
        ms_rim_weight=args.ms_rim_weight,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_ms_rbsg2(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        ms_boundary_weight=args.ms_boundary_weight,
        ms_rim_weight=args.ms_rim_weight,
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
            f"ms_bd={train_metrics['loss_ms_boundary']:.4f} ms_rim={train_metrics['loss_ms_rim']:.4f}"
''',
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="ms_rbsg2_1_8_1_4_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--ms-boundary-weight", type=float, default=0.05)
    parser.add_argument("--ms-rim-weight", type=float, default=0.025)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Create full training sbatch script ====="
cat > submitjob_ms_rbsg2_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J ms_rbsg2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/ms_rbsg2_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_ms_rbsg2_e200.py

echo "===== RUN MS-RBSG-2: RBSG at decoder 1/8 and 1/4 + final B2 RBSG ====="
python src/train_resnet34_unet_brr_ms_rbsg2_e200.py \
  --run-name ms_rbsg2_1_8_1_4_auglite_e200_constlr_bs6 \
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
  --ms-boundary-weight 0.05 \
  --ms-rim-weight 0.025

echo "===== DONE MS-RBSG-2 ====="
EOF

echo "===== Create smoke test sbatch script ====="
cat > submitjob_smoke_ms_rbsg2_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_msrg
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_ms_rbsg2_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_ms_rbsg2_e200.py

echo "===== SMOKE RUN MS-RBSG-2: 3 epochs ====="
python src/train_resnet34_unet_brr_ms_rbsg2_e200.py \
  --run-name smoke_ms_rbsg2_e3 \
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
  --ms-boundary-weight 0.05 \
  --ms-rim-weight 0.025

echo "===== DONE SMOKE MS-RBSG-2 ====="
EOF

chmod +x submitjob_ms_rbsg2_auglite_e200_constlr_bs6.sh
chmod +x submitjob_smoke_ms_rbsg2_e3.sh

echo "===== Shell syntax check ====="
bash -n submitjob_ms_rbsg2_auglite_e200_constlr_bs6.sh
bash -n submitjob_smoke_ms_rbsg2_e3.sh

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_ms_rbsg2_e200.py

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_ms_rbsg2_e200.py
ls -lh submitjob_ms_rbsg2_auglite_e200_constlr_bs6.sh
ls -lh submitjob_smoke_ms_rbsg2_e3.sh

echo "===== Next step ====="
echo "1) First submit smoke test:"
echo "   sbatch submitjob_smoke_ms_rbsg2_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_ms_rbsg2_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_ms_rbsg2_auglite_e200_constlr_bs6.sh"
