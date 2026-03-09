# Use dinov3 vit-small as backbone
# for DETR (Decoder) detection model

import mlx.nn as nn
import mlx.core as mx
from typing import Dict
from heads.detr.ffn import FFN
from heads.detr.deformable_attn import DeformableDecoderLayer


def build_2d_sincos_pos_embed(h, w, d_model):
    y, x = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    assert d_model % 4 == 0
    omega = mx.arange(d_model // 4) / (d_model // 4)
    omega = 1.0 / (10000**omega)
    y_enc = y.flatten()[:, None] * omega[None]
    x_enc = x.flatten()[:, None] * omega[None]
    pos = mx.concatenate(
        [mx.sin(x_enc), mx.cos(x_enc), mx.sin(y_enc), mx.cos(y_enc)], axis=1
    )  # (H*W, d_model)
    return pos[None]  # (1, N, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, n_heads)
        self.cross_attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model, dim_ff, dropout=dropout, n_layers=0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, tgt, memory, tgt_mask=None, memory_mask=None) -> mx.array:
        x = self.norm1(tgt)
        tgt2 = self.dropout(self.self_attn(x, x, x, mask=tgt_mask)[0])
        tgt = tgt + tgt2

        x = self.norm2(tgt)
        tgt2 = self.dropout(self.cross_attn(x, memory, memory, mask=memory_mask)[0])
        tgt = tgt + tgt2

        x = self.norm3(tgt)
        tgt2 = self.dropout(self.ffn(x))
        tgt = tgt + tgt2
        return tgt


class DETRTDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_ff, n_points=4, dropout=0.1):
        super().__init__()
        self.layers = [
            DeformableDecoderLayer(d_model, n_heads, dim_ff, n_points, dropout)
            for _ in range(num_layers)
        ]

    def __call__(self, tgt, memory, h, w, tgt_mask=None):
        intermediates = []
        for layer in self.layers:
            tgt = layer(tgt, memory, h, w, tgt_mask)
            intermediates.append(tgt)
        return intermediates  # List of (B, Q, d_model), one per layer


class DETR(nn.Module):
    def __init__(
        self,
        d_model=384,
        n_heads=6,
        num_layers=6,
        dim_ff=1024,
        n_classes=20,
        n_queries=50,
        dropout=0.1,
        img_size=224,
        patch_size=16,
        n_points=4,
    ):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.decoder = DETRTDecoder(
            d_model, n_heads, num_layers, dim_ff, n_points, dropout
        )
        self.bbox_pred = FFN(d_model, d_model, 4, n_layers=2, dropout=dropout)
        self.class_pred = FFN(d_model, d_model, n_classes, n_layers=2, dropout=dropout)

        h = w = img_size // patch_size
        self._pos = build_2d_sincos_pos_embed(h, w, d_model)

        self.n_queries = n_queries
        self.d_model = d_model

    def __call__(self, img_embed, tgt_mask=None, memory_mask=None) -> Dict:
        B, N, _ = img_embed.shape
        h = w = int(N**0.5)
        img_embed = img_embed + self._pos

        query_ids = mx.arange(self.n_queries)
        tgt = mx.broadcast_to(
            self.query_embed(query_ids)[None],
            (B, self.n_queries, self.d_model),
        )

        all_hs = self.decoder(tgt, img_embed, h, w, tgt_mask)

        final_hs = all_hs[-1]
        out = {
            "logits": self.class_pred(final_hs),
            "boxes": nn.sigmoid(self.bbox_pred(final_hs)),
        }

        out["aux"] = [
            {
                "logits": self.class_pred(hs),
                "boxes": nn.sigmoid(self.bbox_pred(hs)),
            }
            for hs in all_hs[:-1]
        ]

        return out


def build_detr(
    d_model: int = 384,
    n_heads: int = 6,
    num_layers: int = 6,
    dim_ff: int = 1024,
    n_classes: int = 20,
    n_queries: int = 100,
    n_points: int = 4,
    dropout: int = 0.1,
) -> DETR:
    return DETR(
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_ff=dim_ff,
        n_classes=n_classes,
        n_queries=n_queries,
        dropout=dropout,
        n_points=n_points,
    )


if __name__ == "__main__":
    import time
    mx.set_default_device(mx.gpu)

    model = build_detr(num_layers=4, n_classes=91, dropout=0.2, n_points=10)
    # print(model)
    N = (224 // 16) ** 2
    print(f"n_patches: {N}")
    img_embed = mx.random.normal((1, N, 384))  # [B, N, d_model]
    start = time.time()
    out = model(img_embed)
    mx.eval(out["logits"], out["boxes"])
    print(time.time() - start)
    print("logits :", out["logits"].shape)  # (1, 100, 81)
    print("boxes  :", out["boxes"].shape)  # (1, 100, 4)

    total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"Total parameters: {total / 1e6:.1f}M")
