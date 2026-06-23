#!/bin/bash
set -euo pipefail

cd /share/home/u2515283028/caries_project

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

SRC="src/train_resnet34_unet_brr_b2_rbsg_e200.py"
DST="src/train_resnet34_unet_brr_b8b_context_e200.py"

if [ ! -f "$SRC" ]; then
  echo "ERROR: cannot find $SRC"
  exit 1
fi

echo "===== Copy B2 script to B8B-context ====="
cp -p "$SRC" "$DST"

echo "===== Patch B8B-context script ====="
python - <<'PY'
from pathlib import Path
import re

p = Path("src/train_resnet34_unet_brr_b8b_context_e200.py")
s = p.read_text(encoding="utf-8")

new_block = r'''
class MultiScaleContextBottleneck(nn.Module):
    """
    Lightweight multi-scale context bottleneck.

    Position:
      deepest encoder feature before U-Net decoder.

    Motivation:
      B2 already handles boundary/rim at the final decoder feature.
      This block does NOT add more boundary/rim constraints.
      It adds multi-scale semantic context using dilated convolutions and
      a global pooling branch, then residual-fuses it back.

    Branches:
      3x3 conv dilation=1
      3x3 conv dilation=3
      3x3 conv dilation=5
      global average pooling branch

    Output:
      x + learnable_scale * context(x)
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

        # Start conservatively. This makes the model close to original B2 at initialization.
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        size = x.shape[2:]

        outs = [branch(x) for branch in self.branches]

        gp = self.global_branch(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        outs.append(gp)

        context = self.fuse(torch.cat(outs, dim=1))
        return x + self.scale * context


class ResNet34UNetBRRB8BContext(nn.Module):
    """
    B8B-context:
      Original B2 BRR-RBSG
      + Multi-scale Context Bottleneck between encoder and decoder.

    Keeps:
      lambda_pos=0.20
      lambda_rim=0.10
      boundary rb=3
      rim rr=9
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

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))

        # Apply multi-scale context only to the deepest encoder feature.
        features[-1] = self.context_bottleneck(features[-1])

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
'''

pattern = r'class ResNet34UNetBRRB2RBSG\(nn\.Module\):.*?\n\ndef main\(args\):'
if not re.search(pattern, s, flags=re.S):
    raise RuntimeError("Cannot locate original B2 model class block.")
s = re.sub(pattern, new_block + "\n\ndef main(args):", s, flags=re.S)

s = s.replace(
    'config["model_class"] = "ResNet34UNetBRRB2RBSG"',
    'config["model_class"] = "ResNet34UNetBRRB8BContext"',
)
s = s.replace(
    'config["experiment"] = "B2: B1 + RBSG-lite decoder feature gate"',
    'config["experiment"] = "B8B-context: B2 BRR-RBSG + Multi-scale Context Bottleneck"',
)
s = s.replace(
    'config["note"] = "B2 keeps B1 boundary/rim auxiliary supervision and adds RBSG-lite on final decoder feature. This avoids rewriting SMP decoder internals."',
    'config["note"] = "B8B-context keeps the original B2 RBSG-lite and boundary/rim auxiliary supervision, and adds a lightweight multi-scale context bottleneck on the deepest encoder feature before the decoder. This adds semantic multi-scale context without strengthening boundary/rim suppression."',
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
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
    config["context_bottleneck"] = {
        "enabled": True,
        "position": "deepest encoder feature before decoder",
        "branches": ["dilation=1", "dilation=3", "dilation=5", "global average pooling"],
        "branch_channels": args.context_branch_channels,
        "dropout": args.context_dropout,
        "init_scale": args.context_init_scale,
        "fusion": "residual: x + learnable_scale * context(x)"
    }
'''
if old_rbsg_block in s:
    s = s.replace(old_rbsg_block, new_rbsg_block)
else:
    print("[WARN] rbsg config block not found; model patch still applied.")

s = s.replace(
'''    model = ResNet34UNetBRRB2RBSG(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)
''',
'''    model = ResNet34UNetBRRB8BContext(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        context_branch_channels=args.context_branch_channels,
        context_dropout=args.context_dropout,
        context_init_scale=args.context_init_scale,
    ).to(device)
'''
)

s = s.replace(
    'print("Model ready: B2 BRR RBSG-lite ResNet34-U-Net")',
    'print("Model ready: B8B-context BRR-RBSG ResNet34-U-Net")',
)
s = s.replace(
    '    print("lr              =", args.lr)\n',
    '    print("lr              =", args.lr)\n    print("context_branch_channels =", args.context_branch_channels)\n    print("context_dropout         =", args.context_dropout)\n    print("context_init_scale      =", args.context_init_scale)\n',
)

s = s.replace(
    'parser.add_argument("--run-name", type=str, default="b2_brr_rbsg_auglite_e200_constlr_bs6")',
    'parser.add_argument("--run-name", type=str, default="b8b_b2_context_bottleneck_auglite_e200_constlr_bs6")',
)

arg_marker = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
'''
arg_insert = '''    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--context-branch-channels", type=int, default=128)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--context-init-scale", type=float, default=0.10)
'''
if arg_marker not in s:
    raise RuntimeError("Cannot find argparse insertion marker.")
s = s.replace(arg_marker, arg_insert)

p.write_text(s, encoding="utf-8")
print(f"Patched: {p}")
PY

echo "===== Python syntax check ====="
python -m py_compile src/train_resnet34_unet_brr_b8b_context_e200.py

echo "===== Create smoke script ====="
cat > submitjob_smoke_b8b_context_e3.sh <<'EOF'
#!/bin/bash
#SBATCH -J smoke_b8b
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b8b_context_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b8b_context_e200.py

echo "===== SMOKE RUN B8B-context: 3 epochs ====="
python src/train_resnet34_unet_brr_b8b_context_e200.py \
  --run-name smoke_b8b_context_e3 \
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
  --context-init-scale 0.10

echo "===== DONE SMOKE B8B-context ====="
EOF

echo "===== Create full training script ====="
cat > submitjob_b8b_context_auglite_e200_constlr_bs6.sh <<'EOF'
#!/bin/bash
#SBATCH -J b8b_ctx
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b8b_context_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b8b_context_e200.py

echo "===== RUN B8B: B2 + Multi-scale Context Bottleneck ====="
python src/train_resnet34_unet_brr_b8b_context_e200.py \
  --run-name b8b_b2_context_bottleneck_auglite_e200_constlr_bs6 \
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
  --context-init-scale 0.10

echo "===== DONE B8B-context ====="
EOF

chmod +x submitjob_smoke_b8b_context_e3.sh
chmod +x submitjob_b8b_context_auglite_e200_constlr_bs6.sh

echo "===== Shell syntax check ====="
bash -n submitjob_smoke_b8b_context_e3.sh
bash -n submitjob_b8b_context_auglite_e200_constlr_bs6.sh

echo "===== Created files ====="
ls -lh src/train_resnet34_unet_brr_b8b_context_e200.py
ls -lh submitjob_smoke_b8b_context_e3.sh
ls -lh submitjob_b8b_context_auglite_e200_constlr_bs6.sh

echo "===== Next step ====="
echo "1) Submit smoke test:"
echo "   sbatch submitjob_smoke_b8b_context_e3.sh"
echo
echo "2) Watch smoke log:"
echo "   tail -f runs/logs/smoke_b8b_context_*.out"
echo
echo "3) If smoke passes, submit full run:"
echo "   sbatch submitjob_b8b_context_auglite_e200_constlr_bs6.sh"
