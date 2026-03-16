"""
src/models/model_factory.py
Central registry for creating and loading models.
"""

from typing import Optional
import torch

from src.models.vit_model import FloodViT, build_vit
from src.models.cnn_model import FloodCNN, build_cnn
from src.utils.config_loader import load_config

cfg = load_config()
NUM_CLASSES = cfg["data"]["num_classes"]

SUPPORTED_MODELS = {
    "vit": build_vit,
    "cnn": build_cnn,
}


def get_model(
    model_type: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """
    Build a model by type string.

    Args:
        model_type: One of 'vit', 'cnn'.
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        checkpoint_path: Optional checkpoint to load.
        device: Target device.

    Returns:
        Model instance.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: '{model_type}'. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )

    builder = SUPPORTED_MODELS[model_type]
    return builder(
        num_classes=num_classes,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def get_model_summary(model: torch.nn.Module) -> dict:
    """Return parameter count summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_pct": 100 * trainable / total,
    }
