import torch
import segmentation_models_pytorch as smp


def shape_of(obj):
    if torch.is_tensor(obj):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [shape_of(x) for x in obj]
    if isinstance(obj, dict):
        return {k: shape_of(v) for k, v in obj.items()}
    return str(type(obj))


def main():
    print("=" * 100)
    print("Inspect SMP U-Net decoder structure")
    print("=" * 100)

    try:
        print("segmentation_models_pytorch version:", smp.__version__)
    except Exception:
        print("segmentation_models_pytorch version: unknown")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    model.eval()

    print("\n" + "=" * 100)
    print("Model decoder object")
    print("=" * 100)
    print(model.decoder)

    print("\n" + "=" * 100)
    print("Encoder out_channels")
    print("=" * 100)
    print(model.encoder.out_channels)

    print("\n" + "=" * 100)
    print("Segmentation head")
    print("=" * 100)
    print(model.segmentation_head)

    print("\n" + "=" * 100)
    print("Decoder blocks")
    print("=" * 100)
    if hasattr(model.decoder, "blocks"):
        for i, block in enumerate(model.decoder.blocks):
            print(f"\n--- decoder.blocks[{i}] ---")
            print(block)
    else:
        print("WARNING: model.decoder has no attribute 'blocks'.")

    hook_outputs = {}

    def make_hook(name):
        def hook(module, inputs, output):
            hook_outputs[name] = {
                "input_shapes": shape_of(inputs),
                "output_shape": shape_of(output),
                "module": module.__class__.__name__,
            }
        return hook

    handles = []
    if hasattr(model.decoder, "blocks"):
        for i, block in enumerate(model.decoder.blocks):
            handles.append(block.register_forward_hook(make_hook(f"decoder.blocks[{i}]")))

    x = torch.randn(1, 3, 512, 512, device=device)

    with torch.no_grad():
        features = model.encoder(x)

        print("\n" + "=" * 100)
        print("Encoder feature shapes")
        print("=" * 100)
        for i, f in enumerate(features):
            print(f"features[{i}]: {tuple(f.shape)}")

        decoder_output = model.decoder(*features)

        print("\n" + "=" * 100)
        print("Decoder output shape")
        print("=" * 100)
        print(tuple(decoder_output.shape))

        masks = model.segmentation_head(decoder_output)

        print("\n" + "=" * 100)
        print("Final mask logits shape")
        print("=" * 100)
        print(tuple(masks.shape))

    print("\n" + "=" * 100)
    print("Forward hook outputs of decoder blocks")
    print("=" * 100)
    for name, info in hook_outputs.items():
        print(f"\n{name}")
        print("module       :", info["module"])
        print("input_shapes :", info["input_shapes"])
        print("output_shape :", info["output_shape"])

    for h in handles:
        h.remove()

    print("\n" + "=" * 100)
    print("Inspection finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()
