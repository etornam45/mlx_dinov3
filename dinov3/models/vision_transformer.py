import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from dinov3.layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from dinov3.utils import named_apply

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-5),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

# Map string names to MLX dtypes.
dtype_dict: Dict[str, mx.Dtype] = {
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    """
    Initialization helper adapted for MLX modules.
    """
    if isinstance(module, nn.Linear):
        # Truncated normal is approximated with a plain normal here.
        std = 0.02
        module.weight = mx.random.normal(
            shape=module.weight.shape,
            dtype=module.weight.dtype,
            loc=0.0,
            scale=std,
        )
        if "bias" in module:
            module.bias = mx.zeros_like(module.bias)

    if isinstance(module, nn.LayerNorm):
        # MLX LayerNorm exposes reset_parameters
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if isinstance(module, LayerScale):
        module.reset_parameters()

    if isinstance(module, PatchEmbed):
        module.reset_parameters()

    if isinstance(module, RMSNorm):
        # For RMSNorm we simply reset the scale to ones.
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # CLS, storage, and mask tokens are plain MLX arrays treated as parameters.
        self.cls_token = mx.zeros((1, 1, embed_dim), dtype=mx.float32)
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = mx.zeros(
                (1, n_storage_tokens, embed_dim), dtype=mx.float32
            )
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(
            f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new"
        )
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(
            f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new"
        )
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = blocks_list

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = mx.zeros((1, embed_dim), dtype=mx.float32)

    def init_weights(self):
        # Reinitialize RoPE periods and token parameters.
        self.rope_embed._init_weights()

        std = 0.02
        self.cls_token = mx.random.normal(
            shape=self.cls_token.shape,
            dtype=self.cls_token.dtype,
            loc=0.0,
            scale=std,
        )
        if self.n_storage_tokens > 0 and self.storage_tokens is not None:
            self.storage_tokens = mx.random.normal(
                shape=self.storage_tokens.shape,
                dtype=self.storage_tokens.dtype,
                loc=0.0,
                scale=std,
            )
        self.mask_token = mx.zeros_like(self.mask_token)

        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: mx.array, masks: Optional[mx.array] = None
    ) -> Tuple[mx.array, Tuple[int, int]]:
        x = self.patch_embed(x)  # [B, H, W, D]
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)  # [B, HW, D]

        if masks is not None:
            # masks: [B, HW] -> [B, HW, 1]
            cond = mx.expand_dims(masks.astype(bool), axis=-1)
            mask_token = self.mask_token.astype(x.dtype)  # [1, D]
            mask_token = mx.expand_dims(mask_token, axis=0)  # [1, 1, D]
            x = mx.where(cond, mask_token, x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token

        if self.n_storage_tokens > 0 and self.storage_tokens is not None:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = mx.zeros(
                (1, 0, cls_token.shape[-1]), dtype=cls_token.dtype
            )

        # Expand CLS and storage tokens across the batch and concatenate.
        cls_tokens = mx.repeat(cls_token, repeats=B, axis=0)
        storage_tokens_exp = mx.repeat(storage_tokens, repeats=B, axis=0)

        x = mx.concat(
            [
                cls_tokens,
                storage_tokens_exp,
                x,
            ],
            axis=1,
        )

        return x, (H, W)

    def forward_features_list(
        self, x_list: List[mx.array], masks_list: List[Optional[mx.array]]
    ) -> List[Dict[str, mx.array]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output: List[Dict[str, mx.array]] = []
        for idx, (x_val, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(
                        x_val[:, : self.n_storage_tokens + 1]
                    )
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(
                        x_val[:, : self.n_storage_tokens + 1]
                    )
                else:
                    x_norm_cls_reg = self.norm(x_val[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x_val[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x_val)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x_val,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self,
        x: mx.array | List[mx.array],
        masks: Optional[mx.array | List[Optional[mx.array]]] = None,
    ) -> Union[Dict[str, mx.array], List[Dict[str, mx.array]]]:
        if isinstance(x, mx.array):
            return self.forward_features_list([x], [masks])[0]
        else:
            if masks is None:
                masks = [None] * len(x)
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self, x: mx.array, n: int = 1
    ) -> List[mx.array]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: mx.array,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[mx.array, Tuple[mx.array, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(
                        mx.concat((x_norm_cls_reg, x_norm_patch), axis=1)
                    )
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                mx.transpose(
                    out.reshape(B, h // self.patch_size, w // self.patch_size, -1),
                    (0, 3, 1, 2),
                )
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def __call__(
        self, *args, is_training: bool = False, **kwargs
    ) -> Union[List[Dict[str, mx.array]], mx.array]:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    model = vit_small(patch_size=16)
    model.init_weights()
    x = mx.random.uniform(shape=(1, 3, 224, 224))
    print(model(x, is_training=True))
    print(model)
