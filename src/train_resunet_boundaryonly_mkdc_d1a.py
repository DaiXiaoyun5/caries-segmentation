from pathlib import Path
import csv
import argparse
from math import gcd

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


class BoundaryOnlyResidualCueBlock(nn.Module):
    """
    Lightweight feature-gradient boundary cue block.

    This is an ablation of CBSA:
    - SimAM attention is removed.
    - Only the feature-gradient boundary cue is retained.
    - Residual scaling is used to avoid overly disturbing decoder features.
    """
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

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
        boundary = self.boundary_gradient(x)
        return x + self.beta * x * boundary



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


class ResNet34UNetBoundaryOnlyMKDC(nn.Module):
    """
    D1a: ResNet34-U-Net + Boundary-only residual cue + MKDC/MKIR

    Fairness design:
    - SMP ResNet34-U-Net main body unchanged.
    - Original CBSA is replaced by a boundary-cue-only refinement block.
    - MKDC/MKIR block remains inserted as decoder-output refinement before segmentation head.
    - This experiment isolates whether the feature-gradient boundary cue itself is useful.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
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

        decoder_channels = self.unet.segmentation_head[0].in_channels

        # cbsa_lambda is kept in the argument list only for command compatibility.
        self.boundary_cue = BoundaryOnlyResidualCueBlock(
            alpha=cbsa_alpha,
            beta=cbsa_beta,
        )

        self.mkdc_refine = mk_irb_bottleneck(
            decoder_channels,
            decoder_channels,
            n=mk_depth,
            s=1,
            expansion_factor=mk_expansion_factor,
            dw_parallel=True,
            add=mk_add,
            kernel_sizes=list(mk_kernel_sizes),
            activation=mk_activation,
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        decoder_output = self.boundary_cue(decoder_output)
        decoder_output = self.mkdc_refine(decoder_output)

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
    config["model_class"] = "ResNet34UNetBoundaryOnlyMKDC"
    config["experiment"] = "D1a: ResNet34-U-Net + Boundary-only residual cue + MKDC/MKIR"
    config["note"] = "CBSA is replaced by boundary-only residual cue: boundary = boundary_gradient(x); out = x + beta*x*boundary. MKDC remains decoder-output refinement."
    save_json(config, run_dir / "config.json")

    train_ds = CariesMaskDataset(train_split, target_size=(args.image_size, args.image_size))
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

    model = ResNet34UNetBoundaryOnlyMKDC(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
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
    print("Model ready: D1a ResNet34-U-Net + Boundary-only residual cue + MKDC")
    print("run_name             =", args.run_name)
    print("encoder_name         =", args.encoder_name)
    print("encoder_weights      =", args.encoder_weights)
    print("cbsa_lambda          =", args.cbsa_lambda)
    print("cbsa_alpha           =", args.cbsa_alpha)
    print("cbsa_beta            =", args.cbsa_beta)
    print("mk_expansion_factor  =", args.mk_expansion_factor)
    print("mk_kernel_sizes      =", kernel_sizes)
    print("mk_depth             =", args.mk_depth)
    print("mk_add               =", args.mk_add)
    print("mk_activation        =", args.mk_activation)
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

    parser.add_argument("--run-name", type=str, default="resunet_boundaryonly_mkdc_d1a_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

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
