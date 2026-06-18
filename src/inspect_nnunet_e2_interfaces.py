import inspect
import json
from pathlib import Path

import torch
import nnunetv2

print("===== nnU-Net v2 path =====")
print("nnunetv2:", nnunetv2.__file__)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

print("\n===== nnUNetTrainer methods =====")
for name in [
    "__init__",
    "_build_loss",
    "train_step",
    "validation_step",
    "perform_actual_validation",
    "initialize",
    "build_network_architecture",
]:
    if hasattr(nnUNetTrainer, name):
        obj = getattr(nnUNetTrainer, name)
        print(f"\n--- {name} signature ---")
        try:
            print(inspect.signature(obj))
        except Exception as e:
            print("signature failed:", repr(e))
        print(f"--- {name} source head ---")
        try:
            src = inspect.getsource(obj)
            print("\n".join(src.splitlines()[:80]))
        except Exception as e:
            print("source failed:", repr(e))
    else:
        print(f"\n--- {name}: MISSING ---")

print("\n===== plans info =====")
plans_path = Path("/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed/Dataset701_CariXray/nnUNetPlans.json")
dataset_json_path = Path("/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw/Dataset701_CariXray/dataset.json")

print("plans exists:", plans_path.exists(), plans_path)
print("dataset.json exists:", dataset_json_path.exists(), dataset_json_path)

if plans_path.exists():
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    print("plans keys:", list(plans.keys()))
    print("configurations keys:", list(plans.get("configurations", {}).keys()))
    if "2d" in plans.get("configurations", {}):
        c = plans["configurations"]["2d"]
        print("2d config keys:", list(c.keys()))
        print("network_arch_class_name:", c.get("network_arch_class_name"))
        print("network_arch_init_kwargs keys:", list(c.get("network_arch_init_kwargs", {}).keys()))
        print("patch_size:", c.get("patch_size"))
        print("batch_size:", c.get("batch_size"))

print("\n===== try instantiate trainer and inspect network =====")
try:
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_500epochs import nnUNetTrainer_500epochs

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    plans["continue_training"] = False
    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))

    trainer = nnUNetTrainer_500epochs(
        plans=plans,
        configuration="2d",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cpu"),
    )

    print("trainer instantiated:", type(trainer))

    # initialize may build network and loss. It should not train.
    trainer.initialize()
    net = trainer.network

    print("\nnetwork type:", type(net))
    print("network:", net.__class__)
    print("has encoder:", hasattr(net, "encoder"))
    print("has decoder:", hasattr(net, "decoder"))

    if hasattr(net, "decoder"):
        dec = net.decoder
        print("decoder type:", type(dec))
        for attr in ["stages", "transpconvs", "seg_layers", "deep_supervision"]:
            print(f"decoder has {attr}:", hasattr(dec, attr))
            if hasattr(dec, attr):
                val = getattr(dec, attr)
                if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                    print(f"  len({attr}):", len(val))
                else:
                    print(f"  value({attr}):", val)

        if hasattr(dec, "seg_layers"):
            print("\nseg_layers:")
            for i, layer in enumerate(dec.seg_layers):
                print(f"  [{i}] {layer}")

    print("\n===== forward shape smoke on CPU =====")
    # use planned patch size if possible
    patch_size = plans["configurations"]["2d"].get("patch_size", [512, 512])
    h, w = int(patch_size[0]), int(patch_size[1])
    x = torch.randn(1, 1, h, w)
    net.eval()
    with torch.no_grad():
        y = net(x)

    print("input shape:", tuple(x.shape))
    if isinstance(y, (list, tuple)):
        print("output is list/tuple len:", len(y))
        for i, yy in enumerate(y):
            print(f"  output[{i}] shape:", tuple(yy.shape))
    else:
        print("output shape:", tuple(y.shape))

except Exception as e:
    print("INSPECT_FAILED:", repr(e))
    import traceback
    traceback.print_exc()
