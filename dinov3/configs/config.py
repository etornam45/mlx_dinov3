import yaml
from dataclasses import dataclass
from typing import Any, Dict
from dinov3.models import (
    vit_small,
    vit_base,
    vit_large,
    vit_so400m,
    vit_huge2,
    vit_giant2,
    vit_7b,
)


@dataclass
class DinoConfig:
    model_type: str = "vit_small"
    patch_size: int = 16
    img_size: int = 224
    in_chans: int = 3
    layerscale_init: float = 1e-5
    n_storage_tokens: int = 4
    mask_k_bias: bool = True
    pos_embed_rope_base: float = 100.0

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        # Filter out model_type as it's not a parameter of the ViT models themselves
        return {k: v for k, v in self.__dict__.items() if k != "model_type"}

    def build_model(self):
        model_fns = {
            "vit_small": vit_small,
            "vit_base": vit_base,
            "vit_large": vit_large,
            "vit_so400m": vit_so400m,
            "vit_huge2": vit_huge2,
            "vit_giant2": vit_giant2,
            "vit_7b": vit_7b,
        }

        if self.model_type not in model_fns:
            raise ValueError(f"Unknown model type: {self.model_type}")

        kwargs = self.to_dict()
        return model_fns[self.model_type](**kwargs)
