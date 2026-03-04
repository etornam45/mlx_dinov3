from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale


def _randperm(n: int) -> mx.array:
    """
    MLX does not provide randperm, so emulate it by sorting random keys.
    Returns a random permutation of [0, n).
    """
    keys = mx.random.uniform(shape=(n,))
    return mx.argsort(keys)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(
        rope: tuple[mx.array, mx.array] | None, indices: mx.array
    ) -> tuple[mx.array, mx.array] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            return sin, cos

    def _forward(self, x: mx.array, rope=None) -> mx.array:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """

        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = _randperm(b)[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = x.at[indices_1].add(self.ls1(residual_1) * residual_scale_factor)

            indices_2 = _randperm(b)[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = x_attn.at[indices_2].add(
                self.ls2(residual_2) * residual_scale_factor
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[mx.array], rope_list=None) -> List[mx.array]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [
            max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list
        ]
        residual_scale_factors = [
            b / sample_subset_size
            for b, sample_subset_size in zip(b_list, sample_subset_sizes)
        ]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                _randperm(b)[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [
                x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)
            ]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1)
                    for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                x.at[indices_1].add(self.ls1(residual_1) * residual_scale_factor)
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                _randperm(b)[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [
                x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)
            ]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                x_attn.at[indices_2].add(self.ls2(residual_2) * residual_scale_factor)
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def __call__(
        self,
        x_or_x_list: mx.array | List[mx.array],
        rope_or_rope_list: mx.array | List[mx.array] | None = None,
    ) -> mx.array | List[mx.array]:
        if isinstance(x_or_x_list, mx.array):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = (
            LayerScale(dim, init_values=ls_init_value)
            if ls_init_value
            else nn.Identity()
        )
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(
            dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob
        )

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = (
            LayerScale(dim, init_values=ls_init_value)
            if ls_init_value
            else nn.Identity()
        )

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        """
        Keep MLX's default parameter initialization; this method is
        retained only for API compatibility with the original code.
        """
        _ = init_attn_std
        _ = init_proj_std
        _ = init_fc_std
        _ = factor

    def __call__(
        self,
        x: mx.array,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


if __name__ == "__main__":
    block = SelfAttentionBlock(dim=1024, num_heads=16)
    x = mx.random.uniform(shape=(1, 50000, 1024))
    print(block(x).shape)

    block = CausalSelfAttentionBlock(dim=1024, num_heads=16)
    x = mx.random.uniform(shape=(1, 200, 1024))
    print(block(x).shape)