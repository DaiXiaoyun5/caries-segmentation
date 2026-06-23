#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b8b_context_e200.py"
DST="src/train_resnet34_unet_brr_b9_dcf_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  echo "Please make sure B8B context script exists:"
  echo "  src/train_resnet34_unet_brr_b8b_context_e200.py"
  exit 1
fi

echo "===== Copy B8B-context script to B9-DCF ====="
cp -p "$SRC" "$DST"

echo "===== Patch B9-DCF script ====="
python - <<'PY'
from pathlib import Path
import re

p = Path("src/train_resnet34_unet_brr_b9_dcf_e200.py")
s = p.read_text(encoding="utf-8")

new_block = r'''
class MultiScaleContextBottleneck(nn.Module):
    """
    Lightweight multi-scale context bottleneck.

    Same as B8B:
      - dilation=1 / 3 / 5 branches
      - global average pooling branch
      - residual fusion

    It adds semantic multi-scale context before the decoder.
    """
    def __init__(
        self,
        channels,
        branch_channels=128,
        dilations=(1, 3, 5),
        dropout=0.0,
        init_scale=0.10,
    ):
        super().__init__()

        channels = int(channels)
        branch_channels = int(branch_channels)
        self.dilations = tuple(int(d) for d in dilations)

        branches = []
        for d in self.dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        kernel_size=3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        fuse_layers = [
            nn.Conv2d(
                branch_channels * (len(self.dilations) + 1),
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        if float(dropout) > 0:
            fuse_layers.append(nn.Dropout2d(float(dropout)))

        self.fuse = nn.Sequential(*fuse_layers)

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        size = x.shape[2:]

        outs = [branch(x) for branch in self.branches]

        gp = self.global_branch(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        outs.append(gp)

        context = self.fuse(torch.cat(outs, dim=1))
        return x + self.scale * context


class DetailAwareCrossLayerDecoderFusion(nn.Module):
    """
    DCF: Detail-aware Cross-Layer Decoder Fusion.

    Inputs:
      - final refined decoder feature after RBSG-lite
      - decoder 1/4 feature: more spatial detail
      - decoder 1/8 feature: local semantic + shape context

    Unlike MS-RBSG-2, DCF does NOT add boundary/rim gates at multiple scales.
    It only fuses decoder features, so it should not directly strengthen
    rim suppression or make the model overly conservative.

    Output:
      F_out = F_refined + beta * DCF(F_refined, F_1/4, F_1/8)
    """
    def __init__(
        self,
        final_channels,
        ch_1_8,
        ch_1_4,
        dcf_channels=16,
        beta=0.20,
    ):
        super().__init__()

        final_channels = int(final_channels)
        ch_1_8 = int(ch_1_8)
        ch_1_4 = int(ch_1_4)
        dcf_channels = int(dcf_channels)

        self.beta = float(beta)

        self.final_proj = nn.Sequential(
            nn.Conv2d(final_channels, dcf_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dcf_channels),
            nn.ReLU(inplace=True),
        )

        self.proj_1_8 = nn.Sequential(
            nn.Conv2d(ch_1_8, dcf_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dcf_channels),
            nn.ReLU(inplace=True),
        )

        self.proj_1_4 = nn.Sequential(
            nn.Conv2d(ch_1_4, dcf_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dcf_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(dcf_channels * 3, dcf_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dcf_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(dcf_channels * 2, dcf_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dcf_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(dcf_channels * 2, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    def forward(self, final_refined, feat_1_8, feat_1_4):
        target_size = final_refined.shape[2:]

        f_final = self.final_proj(final_refined)

        f_1_8 = self.proj_1_8(feat_1_8)
        f_1_8 = F.interpolate(
            f_1_8,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        f_1_4 = self.proj_1_4(feat_1_4)
        f_1_4 = F.interpolate(
            f_1_4,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        residual = self.fuse(torch.cat([f_final, f_1_4, f_1_8], dim=1))
        return final_refined + self.beta * residual


class ResNet34UNetBRRB9DCF(nn.Module):
    """
    B9-DCF:
      B8B Context Bottleneck
      + final-stage RBSG-lite
      + Detail-aware Cross-Layer Decoder Fusion.

    Keeps:
      lambda_pos = 0.20
      lambda_rim = 0.10
      boundary rb = 3
      rim rr = 9
      original B2 boundary/rim auxiliary supervision
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        context_branch_channels=128,
        context_dropout=0.0,
        context_init_scale=0.10,
        dcf_channels=16,
        dcf_beta=0.20,
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

        try:
            bottleneck_ch = int(self.unet.encoder.out_channels[-1])
        except Exception:
            bottleneck_ch = 512

        self.context_bottleneck = MultiScaleContextBottleneck(
            channels=bottleneck_ch,
            branch_channels=int(context_branch_channels),
            dilations=(1, 3, 5),
            dropout=float(context_dropout),
            init_scale=float(context_init_scale),
        )

        decoder_channels = self._get_decoder_channels()
        if len(decoder_channels) < 3:
            # SMP default for resnet34 Unet decoder is usually [256, 128, 64, 32, 16]
            decoder_channels = [256, 128, 64, 32, dec_ch]

        ch_1_8 = decoder_channels[1]
        ch_1_4 = decoder_channels[2]

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.dcf = DetailAwareCrossLayerDecoderFusion(
            final_channels=dec_ch,
            ch_1_8=ch_1_8,
            ch_1_4=ch_1_4,
            dcf_channels=int(dcf_channels),
            beta=float(dcf_beta),
        )

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

    def _decode_collect(self, features):
        """
        Manually run SMP Unet decoder so that we can collect 1/8 and 1/4 decoder features.

        SMP Unet decoder block index mapping for ResNet34-like U-Net:
          block 0: 1/16
          block 1: 1/8
          block 2: 1/4
          block 3: 1/2
          block 4: full
        """
        decoder = self.unet.decoder

        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = decoder.center(head)
        collected = {}

        for i, decoder_block in enumerate(decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = self._run_decoder_block(decoder_block, x, skip)

            if i == 1:
                collected["1_8"] = x
            elif i == 2:
                collected["1_4"] = x

        return x, collected

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))

        # Same as B8B: add multi-scale context on the deepest encoder feature.
        features[-1] = self.context_bottleneck(features[-1])

        decoder_output, collected = self._decode_collect(features)

        if "1_8" not in collected or "1_4" not in collected:
            raise RuntimeError("DCF could not collect decoder 1/8 and 1/4 features.")

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)

        # New B9 DCF: fuse final refined feature with decoder 1/8 and 1/4 detail features.
        fused = self.dcf(
            refined,
            collected["1_8"],
            collected["1_4"],
        )

        mask_logits = self.mask_head(fused)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(fused)
        rim_logits = self.rim_head(fused)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }
'''

# Replace B8B classes block with B9 classes block.
pattern = r'class MultiScaleContextBottleneck\(nn\.Module\):.*?\n\ndef main\(args\):'
if not re.search(pattern, s, flags=re.S):
    raise RuntimeError("Cannot locate B8B model block.")
s = re.sub(pattern, new_block + "\n\ndef main(args):", s, flags=re.S)

s = s.replace(
    'config["model_class"] = "ResNet34UNetBRRB8BContext"',
    'config["model_class"] = "ResNet34UNetBRRB9DCF"',
)
s = s.replace(
    'config["experiment"] = "B8B-context: B2 BRR-RBSG + Multi-scale Context Bottleneck"',
    'config["experiment"] = "B9-DCF: B8B Context Bottleneck + Detail-aware Cross-Layer Decoder Fusion"',
)
s = s.replace(
    'config["note"] = "B8B-context keeps the original B2 RBSG-lite and boundary/rim auxiliary supervision, and adds a lightweight multi-scale context bottleneck on the deepest encoder feature before the decoder. This adds semantic multi-scale context without strengthening boundary/rim suppression."',
    'config["note"] = "B9-DCF keeps B8B multi-scale context bottleneck and final RBSG-lite, and adds a lightweight detail-aware cross-layer decoder fusion module. DCF fuses decoder 1/8, decoder 1/4 and final refined features without adding extra boundary/rim gates, aiming to improve small-lesion detail while avoiding over-conservative rim suppression."',
)

# Add DCF config after context config if present.
old_context_tail = '''        "fusion": "residual: x + learnable_scale * context(x)"
    }
'''
new_context_tail = '''        "fusion": "residual: x + learnable_scale * context(x)"
    }
    config["dcf"] = {
        "enabled": True,
        "position": "after final RBSG-lite before output heads",
        "features": ["decoder_1/8", "decoder_1/4", "final_refined"],
        "dcf_channels": args.dcf_channels,
        "dcf_beta": args.dcf_beta,
        "fusion": "F_out = F_refined + beta * DCF(F_refined, F_1/4, F_1/8)",
        "note": "DCF fuses multi-level decoder features without additional boundary/rim suppression."
    }
'''
if old_context_tail in s:
    s = s.replace(old_context_tail, new_context_tail)
else:
    print("[WARN] Context config tail not found; model patch still applied.")

s = s.replace(
'''    model = ResNet34UNetBRRB8BContext(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        context_branch_channels=args.context_branch_channels,
        context_dropout=args.context_dropout,
        context_init_scale=args.context_init_scale,
    ).to(device)
''',
'''    model = ResNet34UNetBRRB9DCF(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        context_branch_channels=args.context_branch_channels,
        context_dropout=args.context_dropout,
        context_init_scale=args.context_init_scale,
        dcf_channels=args.dcf_channels,
        dcf_beta=args.dcf_beta,
    ).to(device)
'''
)

s = s.replace(
    'print("Model ready: B8B-context BRR-RBSG ResNet34-U-Net")',
    'print("Model ready: B9-DCF BRR-RBSG Context + Decoder Fusion ResNet34-U-Net")',
)
s = s.replace(
    '    print("context_init_scale      =", args.context_init_scale)\n',
    '    print("context_init_scale      =", args.context_init_scale)\n    print("dcf_channels            =", args.dcf_channels)\n    print("dcf_beta                =", args.dcf_beta)\n',
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b8b_b2_context_bottleneck_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b9_dcf_b8b_context_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--context-branch-channels", type=int, default=128)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--context-init-scale", type=float, default=0.10)
'''
arg_insert = '''    parser.add_argument("--context-branch-channels", type=int, default=128)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--context-init-scale", type=float, default=0.10)

    parser.add_argument("--dcf-channels", type=int, default=16)
    parser.add_argument("--dcf-beta", type=float, default=0.20)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find B8B context argparse block.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_b9_dcf_e200.py

echo "===== Create smoke script ====="
cat > submitjob_smoke_b9_dcf_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b9
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b9_dcf_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b9_dcf_e200.py

echo "===== SMOKE RUN B9-DCF: 3 epochs ====="
python src/train_resnet34_unet_brr_b9_dcf_e200.py \
  --run-name smoke_b9_dcf_e3 \
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
  --context-branch-channels 128 \
  --context-dropout 0.0 \
  --context-init-scale 0.10 \
  --dcf-channels 16 \
  --dcf-beta 0.20

echo "===== DONE SMOKE B9-DCF ====="
EOF

echo "===== Create full training script ====="
cat > submitjob_b9_dcf_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b9_dcf
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b9_dcf_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b9_dcf_e200.py

echo "===== RUN B9-DCF: B8B Context + Detail-aware Cross-Layer Decoder Fusion ====="
python src/train_resnet34_unet_brr_b9_dcf_e200.py \
  --run-name b9_dcf_b8b_context_auglite_e200_constlr_bs6 \
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
  --context-branch-channels 128 \
  --context-dropout 0.0 \
  --context-init-scale 0.10 \
  --dcf-channels 16 \
  --dcf-beta 0.20

echo "===== DONE B9-DCF ====="
EOF

chmod +x submitjob_smoke_b9_dcf_e3.sh
chmod +x submitjob_b9_dcf_auglite_e200_constlr_bs6.sh

echo "===== Shell syntax check ====="
bash -n submitjob_smoke_b9_dcf_e3.sh
bash -n submitjob_b9_dcf_auglite_e200_constlr_bs6.sh

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b9_dcf_e200.py
ls -lh submitjob_smoke_b9_dcf_e3.sh
ls -lh submitjob_b9_dcf_auglite_e200_constlr_bs6.sh

echo "===== Next step ====="
echo "1) Submit smoke test:"
echo "   sbatch submitjob_smoke_b9_dcf_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b9_dcf_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_b9_dcf_auglite_e200_constlr_bs6.sh"
