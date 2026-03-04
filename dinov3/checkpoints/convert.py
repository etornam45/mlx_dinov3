"""
Convert the official PyTorch ViT-S/16 DINOv3 checkpoint to MLX format.
Download from:
    https://dl.fbaipublicfiles.com/dinov3/dinov3-vits16-pretrain-lvd1689m.pth
and update CHECKPOINT_PATH below if needed.
"""

from dinov3.models import DinoVisionTransformer
import mlx.core as mx
import mlx.nn as nn
import torch
from PIL import Image
from dinov3.models import vit_small
from transformers import AutoModel, AutoProcessor

CHECKPOINT_PATH = "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
IMAGE_PATH = "image.jpg"
MLX_WEIGHTS_PATH = "dinov3/checkpoints/model/vit-small.safetensors"
HF_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

def torch_to_mlx(v: torch.Tensor, key: str = "") -> mx.array:
    """Convert a PyTorch tensor to an MLX array, handling bfloat16 and Conv2d weights."""
    if v.dtype == torch.bfloat16:
        arr = mx.array(v.float().numpy(), dtype=mx.bfloat16)
    else:
        arr = mx.array(v.numpy())

    # MLX Conv2d expects OHWI; PyTorch stores as OIHW
    if key == "patch_embed.proj.weight":
        arr = arr.transpose(0, 2, 3, 1)

    return arr


def load_pytorch_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    return ckpt


def build_and_save_mlx_model(state: dict, out_path: str) -> None:
    n_storage_tokens = 0
    if "storage_tokens" in state:
        n_storage_tokens = state["storage_tokens"].shape[1]
    print(f"n_storage_tokens: {n_storage_tokens}")

    model = vit_small(
        patch_size=16,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )

    weights = [
        (k, torch_to_mlx(v, k)) for k, v in state.items() if isinstance(v, torch.Tensor)
    ]

    model.load_weights(weights, strict=True)
    model.save_weights(out_path)
    print(f"Saved MLX weights to {out_path}")


def load_mlx_model(path: str) -> DinoVisionTransformer:
    model = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    model.load_weights(path)
    return model



    
if __name__ == "__main__":
    state = load_pytorch_checkpoint(CHECKPOINT_PATH)
    build_and_save_mlx_model(state, MLX_WEIGHTS_PATH)

    
    mlx_model = load_mlx_model(MLX_WEIGHTS_PATH)

    hf_model = AutoModel.from_pretrained(HF_MODEL_NAME)
    processor = AutoProcessor.from_pretrained(HF_MODEL_NAME)

    image = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    pixel_values_pt = inputs["pixel_values"]                        # PT format [B, C, H, W]
    pixel_values_mlx = mx.array(
        pixel_values_pt.permute(0, 2, 3, 1).numpy()                # MLX format [B, H, W, C]
    )
    print(f"Input shape (MLX): {pixel_values_mlx.shape}")

    mlx_out = mlx_model(pixel_values_mlx, is_training = True)
    mx.eval(mlx_out)
    print(f"MLX CLS shape: {mlx_out["x_norm_clstoken"].shape}")
    mlx_cls = mlx_out["x_norm_clstoken"]
    with torch.no_grad():
        hf_out = hf_model(**inputs)
    pt_cls = mx.array(hf_out.last_hidden_state[:, 0, :].numpy())  # (1, 384)
    print(f"HF  CLS shape: {pt_cls.shape}")

    diff = mx.abs(mlx_cls - pt_cls)
    print(f"\nMax difference (CLS):  {mx.max(diff).item():.6f}")
    print(f"Mean difference (CLS): {mx.mean(diff).item():.6f}")

    dot      = mx.sum(mlx_cls * pt_cls)
    norm_mlx = mx.sqrt(mx.sum(mlx_cls ** 2))
    norm_pt  = mx.sqrt(mx.sum(pt_cls  ** 2))
    print(f"Cosine similarity:     {(dot / (norm_mlx * norm_pt)).item():.6f}  (1.0 = identical)")

    print("\nMLX first 5:", mlx_cls[0, :5].tolist())
    print("HF  first 5:", pt_cls[0,  :5].tolist())

    # remove CLS and Storage/register tokens
    pt_patch_tokens = hf_out.last_hidden_state[:, 1+4:, :]
    mlx_patch_tokens = mlx_out["x_norm_patchtokens"]
    diff = mx.abs(mlx_patch_tokens - pt_patch_tokens)
    print(f"\nMax difference (patch tokens):  {mx.max(diff).item():.6f}")
    print(f"Mean difference (patch tokens): {mx.mean(diff).item():.6f}")

    dot      = mx.sum(mlx_patch_tokens * pt_patch_tokens)
    norm_mlx = mx.sqrt(mx.sum(mlx_patch_tokens ** 2))
    norm_pt  = mx.sqrt(mx.sum(mx.array(pt_patch_tokens.numpy()) ** 2))
    print(f"Cosine similarity:     {(dot / (norm_mlx * norm_pt)).item():.6f}  (1.0 = identical)")

    print("\nMLX first 5:", mlx_patch_tokens[0, :1, :5].tolist())
    print("HF  first 5:", pt_patch_tokens[0,  :1, :5].tolist())
