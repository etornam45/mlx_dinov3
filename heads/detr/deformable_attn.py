import numpy as np
import mlx.nn as nn
import mlx.core as mx
from heads.detr.ffn import FFN


def bilinear_sample(feat_grid: mx.array, points: mx.array) -> mx.array:
    """
    Differentiable bilinear sampling.

    Args:
        feat_grid:  (B, H, W, C)
        points:     (B, Q, K, 2)  —  (x, y) normalised to [0, 1]
    Returns:
        sampled:    (B, Q, K, C)
    """
    B, H, W, C = feat_grid.shape
    _, Q, K, _ = points.shape

    x = points[..., 0] * (W - 1)   # (B, Q, K)
    y = points[..., 1] * (H - 1)

    x0 = mx.floor(x).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x1 = mx.clip(x0 + 1, 0, W - 1)
    y1 = mx.clip(y0 + 1, 0, H - 1)
    x0 = mx.clip(x0,     0, W - 1)
    y0 = mx.clip(y0,     0, H - 1)

    # Bilinear weights
    wx = (x - x0.astype(mx.float32))[..., None]   # (B, Q, K, 1)
    wy = (y - y0.astype(mx.float32))[..., None]

    # Flatten spatial dims for indexing: (B, H*W, C)
    feat_flat = feat_grid.reshape(B, H * W, C)
    b_idx = mx.broadcast_to(mx.arange(B)[:, None, None], (B, Q, K))

    f00 = feat_flat[b_idx, y0 * W + x0]   # (B, Q, K, C)
    f01 = feat_flat[b_idx, y0 * W + x1]
    f10 = feat_flat[b_idx, y1 * W + x0]
    f11 = feat_flat[b_idx, y1 * W + x1]

    return (1 - wy) * ((1 - wx) * f00 + wx * f01) \
         +      wy  * ((1 - wx) * f10 + wx * f11)


class DeformableCrossAttention(nn.Module):
    """
    Single-scale deformable cross-attention (Zhu et al., 2020).

    Each query predicts n_points (x, y) offsets per head around a
    reference point.  Features are sampled via bilinear interpolation
    and aggregated with softmax attention weights.

    Complexity: O(Q · n_heads · n_points)  vs  O(Q · H · W)  for dense MHA.
    """

    def __init__(self, d_model: int, n_heads: int, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_points  = n_points
        self.head_dim  = d_model // n_heads

        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj        = nn.Linear(d_model, d_model)
        self.out_proj          = nn.Linear(d_model, d_model)

        self._init_offsets()

    def _init_offsets(self):
        """
        Initialise offset bias so the n_points form a uniform ring
        around the reference point at the start of training.
        (Replicates the paper's init — critical for stable early training.)
        """
        thetas   = np.arange(self.n_heads) * (2 * np.pi / self.n_heads)
        grid     = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)  # (nH, 2)
        grid    /= np.abs(grid).max(-1, keepdims=True)
        grid     = np.tile(grid[:, None, :], (1, self.n_points, 1))     # (nH, nP, 2)
        for j in range(self.n_points):
            grid[:, j, :] *= (j + 1)   # space points at increasing radii
        self.sampling_offsets.bias = mx.array(grid.reshape(-1).astype(np.float32))

    def __call__(
        self,
        query:            mx.array,   # (B, Q, d_model)
        memory:           mx.array,   # (B, H*W, d_model)
        reference_points: mx.array,   # (B, Q, 2)  x,y ∈ [0, 1]
        h: int,
        w: int,
    ) -> mx.array:
        B, Q, _ = query.shape

        value = self.value_proj(memory)                                  # (B, HW, d_model)
        value = value.reshape(B, h, w, self.n_heads, self.head_dim)     # (B, H, W, nH, hd)

        offsets = self.sampling_offsets(query)                           # (B, Q, nH*nP*2)
        offsets = offsets.reshape(B, Q, self.n_heads, self.n_points, 2)
        offsets = offsets / mx.array([w, h], dtype=mx.float32)

        ref        = reference_points[:, :, None, None, :]              # (B, Q, 1, 1, 2)
        sample_pts = mx.clip(ref + offsets, 0.0, 1.0)                  # (B, Q, nH, nP, 2)

        attn = self.attention_weights(query)                             # (B, Q, nH*nP)
        attn = mx.softmax(attn.reshape(B, Q, self.n_heads, self.n_points), axis=-1)

        head_out = []
        for hd in range(self.n_heads):
            v_h      = value[:, :, :, hd, :]           # (B, H, W, head_dim)
            pts_h    = sample_pts[:, :, hd, :, :]      # (B, Q, n_points, 2)
            sampled  = bilinear_sample(v_h, pts_h)     # (B, Q, n_points, head_dim)
            w_h      = attn[:, :, hd, :, None]         # (B, Q, n_points, 1)
            head_out.append(mx.sum(sampled * w_h, axis=2))   # (B, Q, head_dim)

        out = mx.concatenate(head_out, axis=-1)         # (B, Q, d_model)
        return self.out_proj(out)


class DeformableDecoderLayer(nn.Module):
    """
    Drop-in replacement for DecoderLayer.
    Reference points are predicted per-layer from the current query state.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        dim_ff:   int,
        n_points: int   = 4,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.self_attn  = nn.MultiHeadAttention(d_model, n_heads)
        self.cross_attn = DeformableCrossAttention(d_model, n_heads, n_points)
        self.ffn        = FFN(d_model, dim_ff, dropout=dropout, n_layers=1)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        # Each query predicts its own reference point
        self.ref_point_head = nn.Linear(d_model, 2)

    def __call__(
        self,
        tgt:      mx.array,         # (B, Q, d_model)
        memory:   mx.array,         # (B, H*W, d_model)
        h: int,
        w: int,
        tgt_mask=None,
    ) -> mx.array:
        # Self-attention
        x   = self.norm1(tgt)
        tgt = tgt + self.dropout(self.self_attn(x, x, x, mask=tgt_mask)[0])

        # Reference points from current query state
        ref_pts = nn.sigmoid(self.ref_point_head(tgt))       # (B, Q, 2)

        # Deformable cross-attention
        x   = self.norm2(tgt)
        tgt = tgt + self.dropout(self.cross_attn(x, memory, ref_pts, h, w))

        # FFN
        tgt = tgt + self.dropout(self.ffn(self.norm3(tgt)))
        return tgt


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)

    d_model = 384
    n_heads = 6
    n_points = 4
    dim_ff = 1024
    B = 2
    Q = 100
    H = W = 14

    decoder = [
        DeformableDecoderLayer(d_model, n_heads, dim_ff, n_points),
        DeformableDecoderLayer(d_model, n_heads, dim_ff, n_points),
        DeformableDecoderLayer(d_model, n_heads, dim_ff, n_points),
    ]

    query = mx.random.normal(shape=(B, Q, d_model))
    memory = mx.random.normal(shape=(B, H * W, d_model))
    reference_points = mx.random.uniform(shape=(B, Q, 2))
    for l in decoder:
        query = l(query, memory, H, W)
    out = query
    print("Output shape:", out.shape)  # (B, Q, d_model)

    # total = sum(p.size for _, p in nn.utils.tree_flatten(decoder.parameters()))
    # print(f"Total parameters: {total / 1e6:.1f}M")