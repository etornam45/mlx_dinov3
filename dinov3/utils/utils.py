import logging
from math import prod
from typing import Callable, List, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("dinov3")


def cat_keep_shapes(x_list: List[mx.array]) -> Tuple[mx.array, List[Tuple[int, ...]], List[int]]:
		"""
		Flatten all leading dimensions of each tensor into a single token dimension,
		keeping the last (feature) dimension intact, and concatenate along tokens.

		Args:
				x_list: List of tensors with arbitrary leading dims and a last feature dim.

		Returns:
				flattened: Concatenated tensor of shape (sum_i tokens_i, C).
				shapes: Original shapes of each tensor.
				num_tokens: Number of tokens (product of all dims except last) per tensor.
		"""
		shapes: List[Tuple[int, ...]] = [x.shape for x in x_list]
		num_tokens: List[int] = [int(prod(shape[:-1])) for shape in shapes]

		flattened_pieces = [
				x.reshape(-1, x.shape[-1]) for x in x_list
		]
		flattened = mx.concat(flattened_pieces, axis=0)
		return flattened, shapes, num_tokens


def uncat_with_shapes(
		flattened: mx.array,
		shapes: List[Tuple[int, ...]],
		num_tokens: List[int],
) -> List[mx.array]:
		"""
		Inverse of `cat_keep_shapes`.

		Args:
				flattened: Tensor of shape (sum_i tokens_i, C_flat).
				shapes: Original shapes returned by `cat_keep_shapes`.
				num_tokens: Token counts per original tensor.

		Returns:
				List of tensors, each reshaped back to its original leading
				dimensions with the (possibly updated) feature dimension.
		"""
		outputs: List[mx.array] = []
		start = 0
		feature_dim = flattened.shape[-1]

		for shape, n_tokens in zip(shapes, num_tokens):
				end = start + n_tokens
				slice_i = flattened[start:end]

				target_shape = shape[:-1] + (feature_dim,)
				outputs.append(slice_i.reshape(target_shape))

				start = end

		return outputs

def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """
    MLX-friendly version of timm.named_apply.

    Iterates over all submodules (and optionally the root) using
    :meth:`mlx.nn.Module.named_modules` and applies ``fn(module=..., name=...)``.
    """
    # Collect all modules with their hierarchical names.
    for mod_name, mod in module.named_modules():
        if mod_name == "" and not include_root:
            continue
        fn(module=mod, name=mod_name)
    return module