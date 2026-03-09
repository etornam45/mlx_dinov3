from tqdm import tqdm
import mlx.nn as nn
import mlx.core as mx

import mlx.optimizers as optim
from dinov3.models import vit_small
from heads.detr.transformer import DETR
from heads.detr.dataset import make_stream
from heads.detr.matcher import HungarianLoss
from heads.detr.transformer import build_detr

mx.set_default_device(mx.gpu)

dset, dset_ln = make_stream(
    "coco/images/val2017",
    "coco/annotations/instances_val2017.json",
    img_size=224,
    batch_size=32,
    shuffle=True,
)


dinov3_small = vit_small(
    patch_size=16,
    n_storage_tokens=4,
    layerscale_init=1e-5,
    mask_k_bias=True,
)
dinov3_small.load_weights("dinov3/checkpoints/model/vit-small.safetensors")
dinov3_small.freeze()
total = sum(p.size for _, p in nn.utils.tree_flatten(dinov3_small.parameters()))
print(f"Total parameters: {total / 1e6:.1f}M")

detr_decoder = build_detr(
    d_model=384,
    num_layers=3,
    n_classes=92,
    n_points=4,
)
out_path = "dinov3/checkpoints/model/detr_decoder.safetensors"
detr_decoder.load_weights(out_path)


total = sum(p.size for _, p in nn.utils.tree_flatten(detr_decoder.parameters()))
print(f"Total Decoder parameters: {total / 1e6:.1f}M")

optimizer = optim.AdamW(learning_rate=1e-5, weight_decay=0.01)
lf = HungarianLoss(num_classes=91)


def loss_fn(model: DETR, img_embed, target):
    out = model(img_embed)
    loss, stats = lf(out, target)
    return loss


loss_and_grad_fn = nn.value_and_grad(detr_decoder, loss_fn)


for i in range(50):
    dset.reset()
    total_loss = 0
    prog_bar = tqdm(dset, desc="Training", unit="batch", total=dset_ln / 32)
    for batch in prog_bar:
        image = mx.array(batch["image"])
        image = image.transpose(0, 3, 1, 2)
        patches = dinov3_small(image, masks=None, is_training=True)[
            "x_norm_patchtokens"
        ]

        loss, grad = loss_and_grad_fn(detr_decoder, patches, batch)
        optimizer.update(detr_decoder, grad)

        mx.eval(loss, detr_decoder.parameters(), grad, optimizer.state)
        # print(f"Loss: {loss.item()}")
        prog_bar.set_postfix(loss=f"{loss.item():.4f}")
        total_loss += loss.item()
    prog_bar.close()
    print(f"Epoch {i}, Loss: {total_loss / (dset_ln / 32)}")
    detr_decoder.save_weights(out_path)
