import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/u2515283028/caries_project")
RWKV_UNET_ROOT = ROOT / "third_party" / "RWKV-UNet-main"
sys.path.insert(0, str(RWKV_UNET_ROOT))

print("===== RWKV-UNet smoke forward test =====")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run this script on a GPU node via sbatch.")

from rwkv_unet import RWKV_UNet_T

device = torch.device("cuda")

# Tiny 版，二分类输出，不加载预训练权重
model = RWKV_UNet_T(
    num_classes=1,
    img_size=256,
    pretrained_path=None,
).to(device)

model.eval()

# RWKV_UNet_T 版本不 repeat channel，因此这里用 3 通道输入
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
