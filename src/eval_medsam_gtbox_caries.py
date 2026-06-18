
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")

def read_split(p):
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Path(line.split()[0]))
    return out

def read_polygons(label_path):
    polys = []
    if not label_path.exists():
        return polys
    raw = label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return polys
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = [float(x) for x in parts[1:]]
        if len(vals) % 2:
            vals = vals[:-1]
        if len(vals) < 6:
            continue
        pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        polys.append(np.clip(pts, 0.0, 1.0))
    return polys

def polygons_to_mask_and_boxes(polys, h, w, pad_ratio):
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
    for pts_norm in polys:
        pts = pts_norm.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        x1, y1 = float(pts[:,0].min()), float(pts[:,1].min())
        x2, y2 = float(pts[:,0].max()), float(pts[:,1].max())
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        x1 = max(0.0, x1 - bw * pad_ratio)
        y1 = max(0.0, y1 - bh * pad_ratio)
        x2 = min(float(w - 1), x2 + bw * pad_ratio)
        y2 = min(float(h - 1), y2 + bh * pad_ratio)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return mask, boxes

def preprocess(image_rgb, size=1024):
    img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
    return torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)

@torch.no_grad()
def predict_boxes(model, image_rgb, boxes, device, image_size=1024, thr=0.5):
    h, w = image_rgb.shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    if not boxes:
        return union
    img_t = preprocess(image_rgb, image_size).to(device)
    emb = model.image_encoder(img_t)
    scale = np.array([image_size/w, image_size/h, image_size/w, image_size/h], dtype=np.float32)
    for box in boxes:
        b = np.array(box, dtype=np.float32) * scale
        b_t = torch.as_tensor(b[None, :], dtype=torch.float, device=device)
        sparse, dense = model.prompt_encoder(points=None, boxes=b_t, masks=None)
        logits, _ = model.mask_decoder(
            image_embeddings=emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        prob = torch.sigmoid(logits)
        prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
        union = np.maximum(union, (prob[0,0].detach().cpu().numpy() > thr).astype(np.uint8))
    return union

def metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return {
        "dice": float((2*tp)/(2*tp+fp+fn+1e-9)),
        "iou": float(tp/(tp+fp+fn+1e-9)),
        "precision": float(tp/(tp+fp+1e-9)),
        "recall": float(tp/(tp+fn+1e-9)),
        "gt_area": int(g.sum()),
        "pred_area": int(p.sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }

def make_preview(image_rgb, gt, pred, out_path):
    h, w = gt.shape
    base = Image.fromarray(image_rgb).convert("RGB")
    def overlay(mask, color):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask.astype(bool)] = color
        return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, "RGBA")).convert("RGB")
    canvas = Image.new("RGB", (w*3, h), (255,255,255))
    canvas.paste(base, (0,0))
    canvas.paste(overlay(gt, [255,0,0,120]), (w,0))
    canvas.paste(overlay(pred, [0,255,0,120]), (2*w,0))
    if canvas.width > 1800:
        s = 1800 / canvas.width
        canvas = canvas.resize((int(canvas.width*s), int(canvas.height*s)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)

def main(args):
    repo = PROJECT_ROOT / args.medsam_repo
    sys.path.insert(0, str(repo))
    from segment_anything import sam_model_registry

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("device:", device)
    if device.type == "cpu":
        print("WARNING: CUDA unavailable; CPU inference will be slow.")

    ckpt = PROJECT_ROOT / args.checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model = sam_model_registry[args.model_type](checkpoint=str(ckpt))
    model.to(device).eval()

    imgs = read_split(PROJECT_ROOT / "data" / "splits" / f"{args.split}.txt")
    label_dir = PROJECT_ROOT / "data" / "unzip" / "labels"
    out_dir = PROJECT_ROOT / "runs_medsam" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, img_path in enumerate(imgs, 1):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        polys = read_polygons(label_dir / f"{img_path.stem}.txt")
        gt, boxes = polygons_to_mask_and_boxes(polys, h, w, args.box_pad_ratio)
        pred = predict_boxes(model, rgb, boxes, device, args.image_size, args.mask_threshold)
        row = metrics(pred, gt)
        row.update({"image_path": str(img_path), "num_prompts": len(boxes), "height": h, "width": w})
        rows.append(row)
        if args.save_preds:
            d = out_dir / "pred_masks"; d.mkdir(exist_ok=True)
            Image.fromarray((pred*255).astype(np.uint8)).save(d / f"{img_path.stem}_pred.png")
        if idx <= args.num_preview:
            make_preview(rgb, gt, pred, out_dir / "preview" / f"{idx:03d}_{img_path.stem}.jpg")
        if idx % 20 == 0 or idx == len(imgs):
            print(f"[{idx}/{len(imgs)}] dice={row['dice']:.4f}, prompts={len(boxes)}")

    keys = ["image_path","num_prompts","height","width","dice","iou","precision","recall","gt_area","pred_area","tp","fp","fn"]
    with open(out_dir / "per_case_metrics.csv", "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {}
    for m in ["dice","iou","precision","recall","gt_area","pred_area"]:
        vals = np.array([r[m] for r in rows], dtype=float)
        summary[f"mean_{m}"] = float(vals.mean())
        summary[f"median_{m}"] = float(np.median(vals))
        summary[f"std_{m}"] = float(vals.std())
    summary.update({
        "num_cases": len(rows),
        "split": args.split,
        "run_name": args.run_name,
        "model_type": args.model_type,
        "box_prompt": "ground_truth_polygon_bbox",
        "box_pad_ratio": args.box_pad_ratio,
        "mask_threshold": args.mask_threshold,
        "note": "GT-box/oracle prompt baseline, not a fully automatic method."
    })
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="external_models/medsam_vit_b.pth")
    ap.add_argument("--medsam-repo", default="external_models/MedSAM")
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--run-name", default="medsam_gtbox_test_pad005")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--box-pad-ratio", type=float, default=0.05)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--save-preds", type=int, default=0)
    ap.add_argument("--num-preview", type=int, default=20)
    main(ap.parse_args())
