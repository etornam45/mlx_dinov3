from dinov3.models import vit_small

model = vit_small(
    patch_size=16,
    n_storage_tokens=4,
    layerscale_init=1e-5,
    mask_k_bias=True,
)

model.load_weights("dinov3/checkpoints/vit-small.safetensors")

print(model)
