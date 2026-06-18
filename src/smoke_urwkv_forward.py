import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/u2515283028/caries_project")
URWKV_ROOT = ROOT / "third_party" / "U-RWKV-main"

sys.path.insert(0, str(URWKV_ROOT))

print("===== U-RWKV smoke forward test =====")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run this script on a GPU node via sbatch.")

from models.v_enc_256.model import v_enc_256_fffse_dec_fusion_rwkv_with2x4

device = torch.device("cuda")
model = v_enc_256_fffse_dec_fusion_rwkv_with2x4().to(device)
model.eval()

x = torch.randn(1, 3, 256, 256, device=device)

with torch.no_grad():
    y = model(x)

print("input shape:", tuple(x.shape))

if isinstance(y, (list, tuple)):
    print("output type: list/tuple, length =", len(y))
    for i, yy in enumerate(y):
        if hasattr(yy, "shape"):
            print(f"output[{i}] shape:", tuple(yy.shape))
        else:
            print(f"output[{i}] type:", type(yy))
else:
    print("output shape:", tuple(y.shape))

print("===== smoke forward OK =====")
