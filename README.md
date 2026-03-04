# MLX DINOv3

A high-performance, MLX-native implementation of **DINOv3** (the latest iteration of self-supervised ViT models from Meta) optimized for Apple Silicon.

This repository provides an implementation of the DinoVisionTransformer architecture, including modern features like Rotary Position Embeddings (RoPE), SwiGLU Feed-Forward Networks, and LayerScale, all built using the [MLX](https://github.com/ml-explore/mlx) framework.

## Features

- **MLX-Native**: Built from the ground up for Apple Silicon using MLX.
- **Multiple Architectures**: Supports various ViT scales:
  - `vit_small` (384 embed dim, 12 layers, 6 heads)
  - `vit_base` (768 embed dim, 12 layers, 12 heads)
  - `vit_large` (1024 embed dim, 24 layers, 16 heads)
  - `vit_giant2` (1536 embed dim, 40 layers, 24 heads)
  - `vit_7b` (4096 embed dim, 40 layers, 32 heads)
- **Advanced Components**:
  - **RoPE**: Integrated Rotary Position Embeddings for 2D images.
  - **SwiGLU**: Efficient SwiGLU FFN implementation.
  - **LayerScale**: For improved training stability in deep transformers.
  - **Registers/Storage Tokens**: Full support for additional storage tokens.
- **Weight Conversion**: Includes scripts to convert official PyTorch/HuggingFace checkpoints to MLX `.safetensors`.

## Installation

Ensure you have Python 3.12+ and the necessary dependencies installed:

```bash
pip install mlx torch transformers pillow
```

## Usage

### 1. Convert Weights

To use the model, you first need to convert an official checkpoint to MLX format. Use the provided conversion script:

```bash
python dinov3/checkpoints/convert.py
```

*Note: You may need to update `CHECKPOINT_PATH` in the script to point to your downloaded `.pth` file or it will attempt to download/locate the default ViT-S/16 checkpoint.*

### 2. Running Inference

Loading and running the model in MLX is straightforward:

```python
import mlx.core as mx
from dinov3.models import vit_small

# Initialize model
model = vit_small(patch_size=16, n_storage_tokens=4)

# Load converted weights
model.load_weights("path/to/vit-small.safetensors")

# Forward pass
image = mx.random.uniform(shape=(1, 224, 224, 3))  # MLX uses NHWC
outputs = model(image, is_training=False)  # Returns CLS token
print(outputs.shape)
```

## Repository Structure

- `dinov3/models/`: Core ViT architecture implementations.
- `dinov3/layers/`: Custom MLX layers (RoPE, SwiGLU, Attention, etc.).
- `dinov3/checkpoints/`: Weight conversion and utility scripts.
- `main.py`: Entry point for testing.

## Acknowledgments

This implementation is based on the official [DINOv3](https://github.com/facebookresearch/dinov3) research by Meta AI. Special thanks to the MLX team for providing the framework that makes this possible.