import math
from typing import Callable, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        # Input is assumed to be NCHW (B, C, H, W) to match the original DINOv3 code.
        # MLX Conv2d expects NHWC, so convert layout before and after convolution.
        B, C, H, W = x.shape

        # Optional sanity checks on divisibility by patch size could be added here.

        # NCHW -> NHWC
        x = mx.swapaxes(x, 1, 3)  # (B, W, H, C)
        x = mx.swapaxes(x, 1, 2)  # (B, H, W, C)

        # Convolution in NHWC
        x = self.proj(x)  # (B, H_out, W_out, embed_dim)
        B, H_out, W_out, C_out = x.shape

        # Flatten to (B, N, D) where N = H_out * W_out
        x = x.reshape(B, H_out * W_out, C_out)
        x = self.norm(x)

        if not self.flatten_embedding:
            # Return spatial map (B, H_out, W_out, D) in NHWC, which many MLX models prefer.
            x = x.reshape(B, H_out, W_out, self.embed_dim)

        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        # Reinitialize Conv2d weights with a uniform distribution similar to the
        # original DINOv3 init, using MLX random APIs.
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        limit = math.sqrt(k)
        self.proj.weight = mx.random.uniform(
            low=-limit,
            high=limit,
            shape=self.proj.weight.shape,
            dtype=self.proj.weight.dtype,
        )
        if "bias" in self.proj:
            self.proj.bias = mx.random.uniform(
                low=-limit,
                high=limit,
                shape=self.proj.bias.shape,
                dtype=self.proj.bias.dtype,
            )

if __name__ == "__main__":
	patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
	x = mx.random.uniform(shape=(1, 3, 224, 224))
	print(patch_embed(x).shape)