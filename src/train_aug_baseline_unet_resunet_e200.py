
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path('/share/home/u2515283028/caries_project')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(split_path: Path):
    items = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Path(line.split()[0]))
    return items


def read_polygons(label_path: Path):
    polygons = []
    if not label_path.exists():
        return polygons
    raw = label_path.read_text(encoding='utf-8').strip()
    if not raw:
        return polygons
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        vals = [float(x) for x in parts[1:]]
        if len(vals) % 2 != 0:
            vals = vals[:-1]
        if len(vals) < 6:
            continue
        pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        polygons.append(np.clip(pts, 0.0, 1.0))
    return polygons


def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for pts_norm in polygons:
        pts = pts_norm.copy()
        pts[:, 0] *= width
        pts[:, 1] *= height
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


class CariXrayAugDataset(Dataset):
    def __init__(self, split, image_size=512, auglite=False):
        self.split = split
        self.image_size = image_size
        self.auglite = auglite
        self.image_paths = read_split(PROJECT_ROOT / 'data' / 'splits' / f'{split}.txt')
        self.label_dir = PROJECT_ROOT / 'data' / 'unzip' / 'labels'

    def __len__(self):
        return len(self.image_paths)

    def _load_image_mask(self, img_path):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'Failed to read image: {img_path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        polygons = read_polygons(self.label_dir / f'{img_path.stem}.txt')
        mask = polygons_to_mask(polygons, h, w)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return rgb.astype(np.float32) / 255.0, mask.astype(np.float32)

    def _apply_auglite(self, image, mask):
        h, w = mask.shape
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.80:
            angle = random.uniform(-10.0, 10.0)
            scale = random.uniform(0.90, 1.10)
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.05, 0.05) * h
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if random.random() < 0.40:
            alpha = random.uniform(0.90, 1.10)
            beta = random.uniform(-0.08, 0.08)
            image = image * alpha + beta
        if random.random() < 0.30:
            gamma = random.uniform(0.85, 1.15)
            image = np.power(np.clip(image, 0, 1), gamma)
        if random.random() < 0.15:
            image = image + np.random.normal(0, 0.02, image.shape).astype(np.float32)
        if random.random() < 0.10:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return np.clip(image, 0, 1).astype(np.float32), (mask > 0.5).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image, mask = self._load_image_mask(img_path)
        if self.auglite:
            image, mask = self._apply_auglite(image, mask)
        return torch.from_numpy(image.transpose(2, 0, 1)).float(), torch.from_numpy(mask[None]).float(), str(img_path)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class PlainUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); self.pools.append(nn.MaxPool2d(2)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.up_trans, self.up_convs = nn.ModuleList(), nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.up_trans.append(nn.ConvTranspose2d(ch, f, 2, 2)); self.up_convs.append(DoubleConv(f * 2, f)); ch = f
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.up_trans, self.up_convs, skips[::-1]):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = conv(torch.cat([skip, x], dim=1))
        return self.out_conv(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ClassicResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.enc_blocks, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc_blocks.append(ResidualConvBlock(ch, f)); self.pools.append(nn.MaxPool2d(2)); ch = f
        self.bridge = ResidualConvBlock(features[-1], features[-1] * 2)
        self.up_trans, self.dec_blocks = nn.ModuleList(), nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.up_trans.append(nn.ConvTranspose2d(ch, f, 2, 2)); self.dec_blocks.append(ResidualConvBlock(f * 2, f)); ch = f
        self.out_conv = nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bridge(x)
        for up, dec, skip in zip(self.up_trans, self.dec_blocks, skips[::-1]):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.out_conv(x)


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = torch.sum(probs * targets, (1, 2, 3))
    denom = torch.sum(probs, (1, 2, 3)) + torch.sum(targets, (1, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def loss_fn(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_from_logits(logits, targets)


def metric_arrays(logits, targets, eps=1e-7):
    preds = (torch.sigmoid(logits) > 0.5).float(); targets = targets.float()
    tp = torch.sum(preds * targets, (1, 2, 3)); fp = torch.sum(preds * (1-targets), (1, 2, 3)); fn = torch.sum((1-preds) * targets, (1, 2, 3))
    return {
        'dice': ((2*tp+eps)/(2*tp+fp+fn+eps)).detach().cpu().numpy(),
        'iou': ((tp+eps)/(tp+fp+fn+eps)).detach().cpu().numpy(),
        'precision': ((tp+eps)/(tp+fp+eps)).detach().cpu().numpy(),
        'recall': ((tp+eps)/(tp+fn+eps)).detach().cpu().numpy(),
    }


def aggregate(metric_lists):
    return {k: float(np.concatenate(v).mean()) for k, v in metric_lists.items()}


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    metric_lists = {k: [] for k in ['dice', 'iou', 'precision', 'recall']}; losses=[]
    for images, masks, _ in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.item()))
        ma = metric_arrays(logits, masks)
        for k in metric_lists: metric_lists[k].append(ma[k])
    out = aggregate(metric_lists); out['loss'] = float(np.mean(losses)); return out


@torch.no_grad()
def evaluate_per_case(model, loader, device):
    model.eval(); rows=[]; metric_lists = {k: [] for k in ['dice','iou','precision','recall']}; losses=[]
    for images, masks, paths in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(images); losses.append(float(loss_fn(logits, masks).item()))
        ma = metric_arrays(logits, masks)
        for k in metric_lists: metric_lists[k].append(ma[k])
        preds = (torch.sigmoid(logits) > 0.5).float()
        for i, p in enumerate(paths):
            pred_i, mask_i = preds[i], masks[i]
            tp = torch.sum(pred_i * mask_i).item(); fp = torch.sum(pred_i * (1-mask_i)).item(); fn = torch.sum((1-pred_i)*mask_i).item()
            rows.append({
                'image_path': p,
                'dice': float((2*tp+1e-7)/(2*tp+fp+fn+1e-7)),
                'iou': float((tp+1e-7)/(tp+fp+fn+1e-7)),
                'precision': float((tp+1e-7)/(tp+fp+1e-7)),
                'recall': float((tp+1e-7)/(tp+fn+1e-7)),
                'gt_area': int(mask_i.sum().item()), 'pred_area': int(pred_i.sum().item()),
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            })
    out = aggregate(metric_lists); out['loss'] = float(np.mean(losses)); return out, rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def make_model(name):
    if name == 'unet': return PlainUNet()
    if name == 'classic_resunet': return ClassicResUNet()
    raise ValueError(name)


def main(args):
    set_seed(args.seed)
    run_dir = PROJECT_ROOT / 'runs' / args.run_name; ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', {**vars(args), 'note': 'Train-only AugLite: mild affine, horizontal flip, brightness/contrast/gamma, light Gaussian noise and blur. Val/test are not augmented.'})
    print('='*80); print('Experiment:', args.run_name); print('Model:', args.model); print('='*80)
    train_ds = CariXrayAugDataset('train', args.image_size, True); val_ds = CariXrayAugDataset('val', args.image_size, False); test_ds = CariXrayAugDataset('test', args.image_size, False)
    print('Dataset ready'); print('train:', len(train_ds)); print('val  :', len(val_ds)); print('test :', len(test_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device =', device)
    model = make_model(args.model).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs+1):
        tr = run_epoch(model, train_loader, optimizer, device, True); va = run_epoch(model, val_loader, optimizer, device, False)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in tr.items()}, **{f'val_{k}': v for k, v in va.items()}}; history.append(row)
        if va['dice'] > best_val:
            best_val, best_epoch = va['dice'], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_dice': best_val, 'args': vars(args)}, ckpt_dir / 'best.pth')
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | train loss {tr['loss']:.4f} dice {tr['dice']:.4f} | val loss {va['loss']:.4f} dice {va['dice']:.4f} | best {best_val:.4f} @ {best_epoch}")
    write_csv(run_dir / 'history.csv', history)
    print('===== Load best checkpoint and evaluate test set =====')
    ckpt = torch.load(ckpt_dir / 'best.pth', map_location=device); model.load_state_dict(ckpt['model_state_dict'])
    test_summary, rows = evaluate_per_case(model, test_loader, device)
    result = {'run_name': args.run_name, 'model': args.model, 'best_epoch': int(best_epoch), 'best_val_dice': float(best_val), 'test_metrics': test_summary}
    write_json(run_dir / 'test_metrics.json', result); write_csv(run_dir / 'per_case_metrics.csv', rows)
    pcs = {}
    for key in ['dice','iou','precision','recall','gt_area','pred_area']:
        arr = np.array([r[key] for r in rows], dtype=float); pcs[f'mean_{key}'] = float(arr.mean()); pcs[f'median_{key}'] = float(np.median(arr)); pcs[f'std_{key}'] = float(arr.std())
    pcs['num_cases'] = len(rows); write_json(run_dir / 'per_case_metrics_summary.json', pcs)
    print('===== test_metrics.json ====='); print(json.dumps(result, indent=2, ensure_ascii=False))
    print('===== per_case_metrics_summary.json ====='); print(json.dumps(pcs, indent=2, ensure_ascii=False))
    print('Done. Run directory:', run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['unet','classic_resunet'])
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-every', type=int, default=10)
    main(parser.parse_args())
