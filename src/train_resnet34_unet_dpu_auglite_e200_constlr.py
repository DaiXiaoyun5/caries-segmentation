from pathlib import Path
import csv
import argparse
from math import gcd

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    run_one_epoch,
    save_prediction_panel,
)


class AugmentedCariesMaskDataset(torch.utils.data.Dataset):
    """
    Online augmentation dataset for CariXray segmentation.

    It reads the same train.txt format as CariesMaskDataset:
        image_path<TAB>mask_path

    Augmentation is applied only to the training set.
    Geometry transforms are synchronized for image and mask.
    Photometric transforms are applied only to image.
    """
    def __init__(self, split_file, target_size=(512, 512), profile="lite"):
        self.items = []
        self.target_size = target_size
        self.profile = profile.lower()

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, mask_path = line.split("\t")
                self.items.append((Path(img_path), Path(mask_path)))

        if self.profile not in {"lite", "strong", "none"}:
            raise ValueError(f"Unknown augmentation profile: {profile}")

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _affine_matrix_for_pil(width, height, angle_deg, scale, translate_x, translate_y):
        """
        PIL Image.transform with AFFINE expects inverse mapping:
        input_coord = M(output_coord).

        Forward transform:
            x_out = scale * R * (x_in - center) + center + translate
        This function returns inverse M.
        """
        angle = np.deg2rad(angle_deg)
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)

        cx = width * 0.5
        cy = height * 0.5
        tx = translate_x
        ty = translate_y

        a = cos_t / scale
        b = sin_t / scale
        d = -sin_t / scale
        e = cos_t / scale

        c = cx - a * (cx + tx) - b * (cy + ty)
        f = cy - d * (cx + tx) - e * (cy + ty)

        return (a, b, c, d, e, f)

    def _geometry_aug(self, image, mask):
        w, h = image.size

        if self.profile == "lite":
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.85, 1.20)
            max_translate = 0.05
            do_hflip = random.random() < 0.5
            do_vflip = False
        elif self.profile == "strong":
            angle = random.uniform(-30.0, 30.0)
            scale = random.uniform(0.70, 1.40)
            max_translate = 0.10
            do_hflip = random.random() < 0.5
            do_vflip = random.random() < 0.2
        else:
            return image, mask

        tx = random.uniform(-max_translate, max_translate) * w
        ty = random.uniform(-max_translate, max_translate) * h

        matrix = self._affine_matrix_for_pil(
            width=w,
            height=h,
            angle_deg=angle,
            scale=scale,
            translate_x=tx,
            translate_y=ty,
        )

        image = image.transform(
            image.size,
            Image.AFFINE,
            matrix,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        mask = mask.transform(
            mask.size,
            Image.AFFINE,
            matrix,
            resample=Image.NEAREST,
            fillcolor=0,
        )

        if do_hflip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if do_vflip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return image, mask

    def _photometric_aug(self, image):
        if self.profile == "none":
            return image

        if self.profile == "lite":
            brightness_range = (0.85, 1.15)
            contrast_range = (0.85, 1.15)
            gamma_range = (0.80, 1.25)
            noise_prob = 0.10
            noise_std_range = (0.005, 0.020)
            blur_prob = 0.10
            blur_radius_range = (0.2, 0.8)
        else:
            brightness_range = (0.80, 1.20)
            contrast_range = (0.80, 1.25)
            gamma_range = (0.70, 1.50)
            noise_prob = 0.15
            noise_std_range = (0.005, 0.030)
            blur_prob = 0.15
            blur_radius_range = (0.2, 1.2)

        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))

        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))

        if random.random() < 0.5:
            gamma = random.uniform(*gamma_range)
            arr = np.asarray(image).astype(np.float32) / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.power(arr, gamma)
            image = Image.fromarray((arr * 255.0).astype(np.uint8))

        if random.random() < noise_prob:
            arr = np.asarray(image).astype(np.float32) / 255.0
            std = random.uniform(*noise_std_range)
            noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
            image = Image.fromarray((arr * 255.0).astype(np.uint8))

        if random.random() < blur_prob:
            radius = random.uniform(*blur_radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Keep the same final input size as P2.
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)

        if self.profile != "none":
            image, mask = self._geometry_aug(image, mask)
            image = self._photometric_aug(image)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32)
        mask_np = (mask_np > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return image_tensor, mask_tensor, str(img_path), str(mask_path)


class CBSABlock(nn.Module):
    def __init__(self, e_lambda=1e-4, alpha=1.0, beta=0.1):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = alpha
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def simam_attention(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        mean = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mean).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / max(n, 1)
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return self.sigmoid(e_inv)

    def boundary_gradient(self, x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))
        g = gx + gy
        g = g.mean(dim=1, keepdim=True)
        g_min = g.amin(dim=(2, 3), keepdim=True)
        g_max = g.amax(dim=(2, 3), keepdim=True)
        return (g - g_min) / (g_max - g_min + 1e-6)

    def forward(self, x):
        simam = self.simam_attention(x)
        boundary = self.boundary_gradient(x)
        att = simam * (1.0 + self.alpha * boundary)
        return x + self.beta * x * att


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    if groups <= 1 or num_channels % groups != 0:
        return x
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class MultiKernelDepthwiseConv(nn.Module):
    """
    Extracted from official MK-UNet mkunet_network.py.
    """
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel is False:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    """
    Extracted from official MK-UNet mkunet_network.py with minimal dependency cleanup.
    """
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True,
                 add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = True if self.stride == 1 else False
        self.ex_c = int(self.in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )

        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            self.ex_c, self.kernel_sizes, self.stride,
            activation, dw_parallel=dw_parallel
        )

        if self.add is True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales

        # Official MK-UNet pconv2 uses pointwise Conv + BatchNorm.
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )

        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)

        if self.add is True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)

        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True,
                      add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
    convs = []
    xx = MultiKernelInvertedResidualBlock(
        in_c, out_c, s, expansion_factor=expansion_factor,
        dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
        activation=activation
    )
    convs.append(xx)
    if n > 1:
        for _ in range(1, n):
            xx = MultiKernelInvertedResidualBlock(
                out_c, out_c, 1, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
                activation=activation
            )
            convs.append(xx)
    return nn.Sequential(*convs)



class DPUBilinearConv(nn.Module):
    """
    Detail-Preserving Upsampling refinement block for final decoder features.

    Placement used in N1:
        image -> ResNet34 encoder -> SMP U-Net decoder -> DPU -> segmentation head -> logits

    The input and output shapes are identical, e.g. [B, 16, 512, 512].
    A bilinear down-up branch estimates smoothed features; the residual high-frequency
    component is refined with lightweight depthwise/pointwise convolutions.
    """
    def __init__(self, channels=16, gamma_init=0.1):
        super().__init__()
        self.local_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.detail_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        h, w = x.shape[-2:]

        # Bilinear down-up smoothing. The difference x - smooth is a lightweight
        # high-frequency cue on the reconstructed high-resolution feature map.
        if h >= 4 and w >= 4:
            smooth = F.interpolate(
                x,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            smooth = F.interpolate(smooth, size=(h, w), mode="bilinear", align_corners=False)
            detail = x - smooth
        else:
            detail = x

        local_feat = self.local_refine(x)
        detail_feat = self.detail_refine(detail)
        refined = self.fuse(torch.cat([local_feat, detail_feat], dim=1))
        return x + self.gamma * refined


class ResNet34UNetDPU(nn.Module):
    """
    N1: Plain ResNet34-U-Net + DPU-BilinearConv with AugLite training.

    This model keeps the strongest plain ResNet34-U-Net baseline unchanged and inserts
    a lightweight Detail-Preserving Upsampling refinement block between the SMP U-Net
    decoder and the segmentation head.

    Forward:
        image -> ResNet34 encoder -> SMP U-Net decoder -> DPU -> segmentation head -> mask logits
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        dpu_channels=16,
        dpu_gamma=0.1,
        # Keep ignored arguments for compatibility with config/eval scripts.
        cbsa_lambda=1e-4,
        cbsa_alpha=1.0,
        cbsa_beta=0.1,
        mk_expansion_factor=2,
        mk_kernel_sizes=(1, 3, 5),
        mk_depth=1,
        mk_add=True,
        mk_activation="relu6",
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        self.dpu = DPUBilinearConv(channels=dpu_channels, gamma_init=dpu_gamma)

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        decoder_output = self.dpu(decoder_output)
        masks = self.unet.segmentation_head(decoder_output)
        return masks


def main(args):
    set_seed(args.seed)

    project_root = Path("/share/home/u2515283028/caries_project")
    train_split = project_root / "data" / "splits" / "train.txt"
    val_split = project_root / "data" / "splits" / "val.txt"
    test_split = project_root / "data" / "splits" / "test.txt"

    run_dir = project_root / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "preds"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["model_class"] = "ResNet34UNetDPU"
    config["experiment"] = "N1-DPU: ResNet34-U-Net + DPU-BilinearConv with AugLite, 200 epochs, constant lr"
    config["note"] = "N1 adds a lightweight Detail-Preserving Upsampling refinement block after the SMP U-Net decoder and before the segmentation head. Other training settings are kept identical to the plain AugLite E200 ConstLR baseline."
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(train_split, target_size=(args.image_size, args.image_size), profile=args.augment_profile)
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("=" * 80)
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    kernel_sizes = tuple(int(v) for v in args.mk_kernel_sizes.split(","))

    model = ResNet34UNetDPU(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        dpu_channels=16,
        dpu_gamma=args.dpu_gamma,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        mk_expansion_factor=args.mk_expansion_factor,
        mk_kernel_sizes=kernel_sizes,
        mk_depth=args.mk_depth,
        mk_add=args.mk_add,
        mk_activation=args.mk_activation,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: N1 ResNet34-U-Net + DPU-BilinearConv + AugLite")
    print("run_name             =", args.run_name)
    print("encoder_name         =", args.encoder_name)
    print("encoder_weights      =", args.encoder_weights)
    print("dpu_gamma            =", args.dpu_gamma)
    print("cbsa_lambda          =", args.cbsa_lambda)
    print("cbsa_alpha           =", args.cbsa_alpha)
    print("cbsa_beta            =", args.cbsa_beta)
    print("mk_expansion_factor  =", args.mk_expansion_factor)
    print("mk_kernel_sizes      =", kernel_sizes)
    print("mk_depth             =", args.mk_depth)
    print("mk_add               =", args.mk_add)
    print("mk_activation        =", args.mk_activation)
    print("augment_profile      =", args.augment_profile)
    print("batch_size           =", args.batch_size)
    print("epochs               =", args.epochs)
    print("lr                   =", args.lr)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, deep_supervision=False, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, deep_supervision=False, train=False)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch(model, train_loader, optimizer=None, device=device, deep_supervision=False, train=False)
    best_val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, deep_supervision=False, train=False)
    test_metrics = run_one_epoch(model, test_loader, optimizer=None, device=device, deep_supervision=False, train=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    print("=" * 80)
    print("Best checkpoint summary")
    print(summary)
    print("=" * 80)

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="resnet34_unet_dpu_auglite_e200_constlr_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--augment-profile", type=str, default="lite", choices=["lite", "strong", "none"])
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--dpu-gamma", type=float, default=0.1)

    parser.add_argument("--cbsa-lambda", type=float, default=1e-4)
    parser.add_argument("--cbsa-alpha", type=float, default=1.0)
    parser.add_argument("--cbsa-beta", type=float, default=0.1)

    parser.add_argument("--mk-expansion-factor", type=int, default=2)
    parser.add_argument("--mk-kernel-sizes", type=str, default="1,3,5")
    parser.add_argument("--mk-depth", type=int, default=1)
    parser.add_argument("--mk-add", action="store_true", default=True)
    parser.add_argument("--mk-activation", type=str, default="relu6")

    args = parser.parse_args()
    main(args)
