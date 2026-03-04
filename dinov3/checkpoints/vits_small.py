from PIL import Image
from dinov3.models import vit_small
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


model = vit_small(
    patch_size=16,
    n_storage_tokens=4,
    layerscale_init=1e-5,
    mask_k_bias=True,
)

model.load_weights("dinov3/checkpoints/model/vit-small.safetensors")
image = Image.open("image.jpg")

image = mx.array(np.array(image.resize((224, 224))))
image = mx.expand_dims(image, axis=0)
image = image.astype(mx.float32)

output = model(image, is_training=True)

x_norm_clstoken, x_storage_tokens, x_norm_patchtokens, x_prenorm = (
    output["x_norm_clstoken"],
    output["x_storage_tokens"],
    output["x_norm_patchtokens"],
    output["x_prenorm"],
)


# Plot image (left) and attention map (right)
image_np = np.array(image.squeeze(axis=0)) / 255.0
patch_tokens = x_norm_patchtokens[0]  # [196, 384]
# Average across embedding dimension and reshape to grid
heatmap = mx.mean(patch_tokens, axis=-1).reshape(14, 14)
heatmap_np = np.array(heatmap)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis("off")

im = axes[1].imshow(heatmap_np, cmap="viridis")
axes[1].set_title("Patch Token Heatmap (Mean)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()
