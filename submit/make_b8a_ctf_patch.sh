#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b8a_ctf_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B8A CTF ====="
cp -p "$SRC" "$DST"

echo "===== Patch B8A Coarse-to-Fine script ====="
python - <<'PY'
from pathlib import Path
import re

p = Path("src/train_resnet34_unet_brr_b8a_ctf_e200.py")
s = p.read_text(encoding="utf-8")

# Extra imports.
if "import math" not in s:
    if "import os\n" in s:
        s = s.replace("import os\n", "import os\nimport math\n", 1)
    else:
        s = "import math\n" + s

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
    raise RuntimeError("Cannot find old B1 import block.")
s = s.replace(old_import, new_import)

new_block = r'''
class LocalRefiner(nn.Module):
    """
    Lightweight local crop refiner.

    Input channels:
      image crop: 3
      coarse probability crop: 1
      uncertainty crop: 1

    It predicts a residual correction on local logits:
      local_logits = coarse_logits + alpha * delta_logits
    """
    def __init__(self, in_channels=5, hidden_channels=32, alpha=0.30):
        super().__init__()
        self.alpha = float(alpha)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, image_crop, coarse_logit_crop):
        coarse_prob = torch.sigmoid(coarse_logit_crop)
        uncertainty = 4.0 * coarse_prob * (1.0 - coarse_prob)

        x = torch.cat(
            [
                image_crop,
                coarse_prob,
                uncertainty,
            ],
            dim=1,
        )

        delta = self.net(x)
        local_logits = coarse_logit_crop + self.alpha * delta
        return local_logits, delta, uncertainty


class ResNet34UNetBRRB8ACTF(nn.Module):
    """
    B8A Coarse-to-Fine Lesion Refinement.

    Global branch:
      Original B2 = ResNet34-U-Net + final RBSG-lite + boundary/rim heads.

    Local branch:
      A lightweight crop-level LocalRefiner trained with GT ROI crops.
      During validation/testing, ROI crops are generated from coarse predicted components.

    It keeps:
      lambda_pos=0.20
      lambda_rim=0.10
      boundary rb=3
      rim rr=9
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        crop_size=256,
        local_alpha=0.30,
        local_hidden=32,
    ):
        super().__init__()

        self.crop_size = int(crop_size)

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

        self.local_refiner = LocalRefiner(
            in_channels=5,
            hidden_channels=int(local_hidden),
            alpha=float(local_alpha),
        )

    def forward_global(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)
        mask_logits = self.mask_head(refined)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }

    def forward(self, x, return_aux=False):
        return self.forward_global(x, return_aux=return_aux)


def _connected_components_numpy(binary):
    """
    Fallback 8-connected component labeling for 2D binary mask.
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


def square_bbox_from_mask(mask_np, margin=32, min_size=64):
    """
    Return square bbox y1,y2,x1,x2 from 2D binary mask.
    Fallback to center crop if no foreground.
    """
    h, w = mask_np.shape
    ys, xs = np.where(mask_np > 0)

    if len(ys) == 0:
        cy, cx = h // 2, w // 2
        size = max(int(min_size), min(h, w) // 2)
    else:
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2

        box_h = y2 - y1
        box_w = x2 - x1
        size = max(box_h, box_w) + 2 * int(margin)
        size = max(size, int(min_size))

    size = min(size, max(h, w))

    half = size // 2
    y1 = cy - half
    x1 = cx - half
    y2 = y1 + size
    x2 = x1 + size

    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w

    y1 = max(0, int(y1))
    x1 = max(0, int(x1))
    y2 = min(h, int(y2))
    x2 = min(w, int(x2))

    return y1, y2, x1, x2


def crop_resize_tensor(x, bbox, out_size, mode="bilinear"):
    """
    x: C,H,W tensor
    return: C,out_size,out_size tensor
    """
    y1, y2, x1, x2 = bbox
    crop = x[:, y1:y2, x1:x2].unsqueeze(0)

    if mode == "nearest":
        out = F.interpolate(crop, size=(out_size, out_size), mode="nearest")
    else:
        out = F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False)

    return out.squeeze(0)


def make_gt_local_crops(images, masks, coarse_logits, crop_size=256, margin=32):
    """
    Generate one GT-centered local crop per image for training local refiner.

    coarse_logits is detached before local refinement to keep global B2 stable.
    """
    bsz = images.shape[0]
    image_crops = []
    mask_crops = []
    coarse_crops = []

    for b in range(bsz):
        gt_np = (masks[b, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        bbox = square_bbox_from_mask(gt_np, margin=margin, min_size=64)

        img_crop = crop_resize_tensor(images[b], bbox, crop_size, mode="bilinear")
        mask_crop = crop_resize_tensor(masks[b], bbox, crop_size, mode="nearest")
        coarse_crop = crop_resize_tensor(coarse_logits[b].detach(), bbox, crop_size, mode="bilinear")

        image_crops.append(img_crop)
        mask_crops.append(mask_crop)
        coarse_crops.append(coarse_crop)

    return (
        torch.stack(image_crops, dim=0),
        torch.stack(mask_crops, dim=0),
        torch.stack(coarse_crops, dim=0),
    )


def find_candidate_bboxes_from_coarse(
    prob_np,
    threshold=0.30,
    margin=32,
    min_area=3,
    max_rois=5,
):
    """
    Generate candidate bboxes from coarse predicted probability.
    Lower threshold is used to include weak candidates.
    """
    binary = (prob_np > float(threshold)).astype(np.uint8)
    labels, n = connected_components_2d(binary)

    if n <= 0:
        return []

    comps = []
    for cid in range(1, n + 1):
        area = int((labels == cid).sum())
        if area < int(min_area):
            continue
        comps.append((cid, area))

    comps = sorted(comps, key=lambda x: x[1], reverse=True)[: int(max_rois)]

    bboxes = []
    for cid, area in comps:
        comp = labels == cid
        bbox = square_bbox_from_mask(comp.astype(np.uint8), margin=margin, min_size=64)
        bboxes.append(bbox)

    return bboxes


@torch.no_grad()
def refine_logits_with_predicted_rois(
    model,
    images,
    coarse_logits,
    crop_size=256,
    threshold=0.30,
    margin=32,
    min_area=3,
    max_rois=5,
):
    """
    Inference-style coarse-to-fine refinement.

    1. Generate candidate ROIs from coarse prediction.
    2. Run local_refiner on each ROI.
    3. Paste local refined logits back to full image.

    This uses only model prediction, not GT mask.
    """
    model_was_training = model.training
    model.eval()

    final_logits = coarse_logits.detach().clone()
    bsz, _, h, w = coarse_logits.shape

    for b in range(bsz):
        prob_np = torch.sigmoid(coarse_logits[b, 0]).detach().cpu().numpy()
        bboxes = find_candidate_bboxes_from_coarse(
            prob_np,
            threshold=threshold,
            margin=margin,
            min_area=min_area,
            max_rois=max_rois,
        )

        for bbox in bboxes:
            y1, y2, x1, x2 = bbox
            img_crop = crop_resize_tensor(images[b], bbox, crop_size, mode="bilinear").unsqueeze(0)
            coarse_crop = crop_resize_tensor(coarse_logits[b].detach(), bbox, crop_size, mode="bilinear").unsqueeze(0)

            local_logits, _, _ = model.local_refiner(img_crop, coarse_crop)

            local_full = F.interpolate(
                local_logits,
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )

            # Replace coarse logits inside candidate ROI with local refined logits.
            final_logits[b:b+1, :, y1:y2, x1:x2] = local_full

    if model_was_training:
        model.train()

    return final_logits


def compute_b8a_ctf_loss(
    model,
    outputs,
    images,
    masks,
    epoch,
    warmup_epochs=20,
    local_weight=0.30,
    crop_size=256,
    crop_margin=32,
):
    """
    B8A loss = original B2 full-image loss + local crop refinement loss.

    Full-image B2:
      L_seg(coarse_full) + wb*L_boundary + wr*L_rim

    Local refinement:
      local crop loss on GT-centered ROI crop.
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)
    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    image_crops, mask_crops, coarse_crops = make_gt_local_crops(
        images,
        masks,
        mask_logits,
        crop_size=crop_size,
        margin=crop_margin,
    )

    local_logits, local_delta, local_uncertainty = model.local_refiner(
        image_crops,
        coarse_crops,
    )

    loss_local = bce_dice_loss(local_logits, mask_crops)

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
        wl = min(float(local_weight), 0.10)
    else:
        wb = 0.20
        wr = 0.10
        wl = float(local_weight)

    total = loss_seg + wb * loss_boundary + wr * loss_rim + wl * loss_local

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "loss_local": float(loss_local.detach().cpu()),
        "local_delta_abs": float(local_delta.detach().abs().mean().cpu()),
        "local_uncertainty": float(local_uncertainty.detach().mean().cpu()),
        "wb": wb,
        "wr": wr,
        "wl": wl,
    }


def run_one_epoch_b8a_ctf(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    local_weight=0.30,
    crop_size=256,
    crop_margin=32,
    infer_threshold=0.30,
    infer_max_rois=5,
):
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_items = {
        "loss_local": 0.0,
        "local_delta_abs": 0.0,
        "local_uncertainty": 0.0,
        "wl": 0.0,
    }

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_b8a_ctf_loss(
                model,
                outputs,
                images,
                masks,
                epoch=epoch,
                local_weight=local_weight,
                crop_size=crop_size,
                crop_margin=crop_margin,
            )

            if train:
                loss.backward()
                optimizer.step()

        last_items = loss_items

        # Metrics use inference-style predicted ROI refinement, without GT.
        with torch.no_grad():
            coarse_logits = outputs["mask"].detach()
            refined_logits = refine_logits_with_predicted_rois(
                model,
                images,
                coarse_logits,
                crop_size=crop_size,
                threshold=infer_threshold,
                margin=crop_margin,
                min_area=3,
                max_rois=infer_max_rois,
            )

            probs = torch.sigmoid(refined_logits)
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
        "loss_local": last_items["loss_local"],
        "local_delta_abs": last_items["local_delta_abs"],
        "local_uncertainty": last_items["local_uncertainty"],
        "wl": last_items["wl"],
    }
'''

pattern = r'class ResNet34UNetBRRB2RBSG\(nn\.Module\):.*?\n\ndef main\(args\):'
if not re.search(pattern, s, flags=re.S):
    raise RuntimeError("Cannot locate original B2 model class block.")
s = re.sub(pattern, new_block + "\n\ndef main(args):", s, flags=re.S)

s = s.replace(
    'config["model_class"] = "ResNet34UNetBRRB2RBSG"',
    'config["model_class"] = "ResNet34UNetBRRB8ACTF"',
)
s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B8A: B2 BRR-RBSG + Coarse-to-Fine Lesion Refinement"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B8A keeps original B2 global branch and adds a coarse-to-fine local crop refiner. Training uses GT-centered ROI crops for local loss, while validation/test use predicted coarse components to generate ROIs and paste refined logits back to full image."',
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
        "main": "BCEWithLogits + Dice on global coarse mask",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "local_refinement": "BCEWithLogits + Dice on GT-centered ROI crop",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim + 0.10*L_local",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim + local_weight*L_local",
        "local_weight": args.local_weight,
        "crop_size": args.crop_size,
        "crop_margin": args.crop_margin,
        "infer_threshold": args.infer_threshold,
        "infer_max_rois": args.infer_max_rois
    }
'''
if old_loss_block not in s:
    raise RuntimeError("Cannot find original config loss block.")
s = s.replace(old_loss_block, new_loss_block)

old_rbsg_block = '''    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
'''
new_rbsg_block = '''    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
    config["coarse_to_fine"] = {
        "enabled": True,
        "global_branch": "original B2",
        "local_branch": "lightweight crop-level LocalRefiner",
        "train_roi": "GT mask bbox with margin",
        "eval_roi": "predicted coarse connected components",
        "local_input": ["image_crop", "coarse_prob_crop", "uncertainty_crop"],
        "local_output": "local_logits = coarse_logits + alpha * delta_logits"
    }
'''
if old_rbsg_block in s:
    s = s.replace(old_rbsg_block, new_rbsg_block)

s = s.replace(
'''    model = ResNet34UNetBRRB2RBSG(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
''',
'''    model = ResNet34UNetBRRB8ACTF(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        crop_size=args.crop_size,
        local_alpha=args.local_alpha,
        local_hidden=args.local_hidden,
    ).to(device)
'''
)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B8A Coarse-to-Fine BRR-RBSG ResNet34-U-Net")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("local_weight    =", args.local_weight)\n    print("crop_size       =", args.crop_size)\n    print("crop_margin     =", args.crop_margin)\n    print("local_alpha     =", args.local_alpha)\n    print("local_hidden    =", args.local_hidden)\n    print("infer_threshold =", args.infer_threshold)\n    print("infer_max_rois  =", args.infer_max_rois)\n',
)

repls = {
'''        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)''':
'''        train_metrics = run_one_epoch_b8a_ctf(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            local_weight=args.local_weight,
            crop_size=args.crop_size,
            crop_margin=args.crop_margin,
            infer_threshold=args.infer_threshold,
            infer_max_rois=args.infer_max_rois,
        )''',

'''        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)''':
'''        val_metrics = run_one_epoch_b8a_ctf(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            local_weight=args.local_weight,
            crop_size=args.crop_size,
            crop_margin=args.crop_margin,
            infer_threshold=args.infer_threshold,
            infer_max_rois=args.infer_max_rois,
        )''',

'''    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_train_metrics = run_one_epoch_b8a_ctf(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        local_weight=args.local_weight,
        crop_size=args.crop_size,
        crop_margin=args.crop_margin,
        infer_threshold=args.infer_threshold,
        infer_max_rois=args.infer_max_rois,
    )''',

'''    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    best_val_metrics = run_one_epoch_b8a_ctf(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        local_weight=args.local_weight,
        crop_size=args.crop_size,
        crop_margin=args.crop_margin,
        infer_threshold=args.infer_threshold,
        infer_max_rois=args.infer_max_rois,
    )''',

'''    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)''':
'''    test_metrics = run_one_epoch_b8a_ctf(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        local_weight=args.local_weight,
        crop_size=args.crop_size,
        crop_margin=args.crop_margin,
        infer_threshold=args.infer_threshold,
        infer_max_rois=args.infer_max_rois,
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
            f"local={train_metrics['loss_local']:.4f} wl={train_metrics['wl']:.3f}"
'''
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b8a_b2_ctf_local_refine_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-margin", type=int, default=32)
    parser.add_argument("--local-weight", type=float, default=0.30)
    parser.add_argument("--local-alpha", type=float, default=0.30)
    parser.add_argument("--local-hidden", type=int, default=32)
    parser.add_argument("--infer-threshold", type=float, default=0.30)
    parser.add_argument("--infer-max-rois", type=int, default=5)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_b8a_ctf_e200.py

echo "===== Create smoke script ====="
cat > submitjob_smoke_b8a_ctf_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b8a
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b8a_ctf_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b8a_ctf_e200.py

echo "===== SMOKE RUN B8A CTF: 3 epochs ====="
python src/train_resnet34_unet_brr_b8a_ctf_e200.py \
  --run-name smoke_b8a_ctf_e3 \
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
  --crop-size 256 \
  --crop-margin 32 \
  --local-weight 0.30 \
  --local-alpha 0.30 \
  --local-hidden 32 \
  --infer-threshold 0.30 \
  --infer-max-rois 5

echo "===== DONE SMOKE B8A CTF ====="
EOF

echo "===== Create full training script ====="
cat > submitjob_b8a_ctf_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b8a_ctf
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b8a_ctf_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b8a_ctf_e200.py

echo "===== RUN B8A: B2 + Coarse-to-Fine Local Lesion Refinement ====="
python src/train_resnet34_unet_brr_b8a_ctf_e200.py \
  --run-name b8a_b2_ctf_local_refine_auglite_e200_constlr_bs6 \
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
  --crop-size 256 \
  --crop-margin 32 \
  --local-weight 0.30 \
  --local-alpha 0.30 \
  --local-hidden 32 \
  --infer-threshold 0.30 \
  --infer-max-rois 5

echo "===== DONE B8A CTF ====="
EOF

chmod +x submitjob_smoke_b8a_ctf_e3.sh
chmod +x submitjob_b8a_ctf_auglite_e200_constlr_bs6.sh

echo "===== Shell syntax check ====="
bash -n submitjob_smoke_b8a_ctf_e3.sh
bash -n submitjob_b8a_ctf_auglite_e200_constlr_bs6.sh

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b8a_ctf_e200.py
ls -lh submitjob_smoke_b8a_ctf_e3.sh
ls -lh submitjob_b8a_ctf_auglite_e200_constlr_bs6.sh

echo "===== Next step ====="
echo "1) Submit smoke test:"
echo "   sbatch submitjob_smoke_b8a_ctf_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b8a_ctf_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_b8a_ctf_auglite_e200_constlr_bs6.sh"
