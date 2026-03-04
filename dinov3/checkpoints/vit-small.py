"""
Convert the official PyTorch ViT-S/16 DINOv3 checkpoint to MLX format.

Download from:
    https://dl.fbaipublicfiles.com/dinov3/dinov3-vits16-pretrain-lvd1689m.pth
and update `path` below if needed.
"""

import mlx.core as mx
import mlx.nn as nn
import torch
from mlxim.io import read_rgb
from mlxim.transform import ImageNetTransform
from dinov3.models import vit_small


path = "/Users/macbookprom1/Documents/AI/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def main() -> None:
    # Load PyTorch checkpoint on CPU
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Infer number of storage tokens from checkpoint, if present.
    n_storage_tokens = 0
    if "storage_tokens" in state:
        n_storage_tokens = state["storage_tokens"].shape[1]
    print("n_storage_tokens", n_storage_tokens)
    # Build the MLX ViT-S/16 model with matching storage tokens.
    model = vit_small(
        patch_size=16,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )

    # Convert PyTorch tensors to MLX arrays.
    weights = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            # MLX does not accept torch.bfloat16 tensors directly; convert via float32.
            if v.dtype == torch.bfloat16:
                arr = mx.array(v.float().numpy(), dtype=mx.bfloat16)
            else:
                arr = mx.array(v.numpy())
            weights.append((k, arr))

    # Load weights into the MLX model.
    model.load_weights(weights, strict=True)

    # Save MLX weights as .npz (supported by MLX).
    model.save_weights("dinov3/checkpoints/vit-small.npz")


def load_mlx_weights(model: nn.Module, path: str) -> None:
    model.load_weights(path)
    return model

if __name__ == "__main__":
    main()
    model = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    model = load_mlx_weights(model, "dinov3/checkpoints/vit-small.npz")
    image_path = "image.jpg"
    image = read_rgb(image_path)
    transform = ImageNetTransform(224)
    image = transform(image)
    print(image.shape)
    print(model(image).shape)