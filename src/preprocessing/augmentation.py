"""
src/preprocessing/augmentation.py
Augmentation pipelines for training and validation/test phases.
Uses both torchvision transforms and Albumentations.
"""

from typing import Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms

from src.utils.config_loader import load_config

cfg = load_config()
aug_cfg = cfg["augmentation"]
IMG_SIZE = cfg["data"]["image_size"]
MEAN = aug_cfg["normalize_mean"]
STD = aug_cfg["normalize_std"]


# ---------------------------------------------------------------------------
# Albumentations-based transforms (used during training)
# ---------------------------------------------------------------------------
class AlbumentationsWrapper:
    """Wrapper to use Albumentations transforms with PIL images in PyTorch."""

    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return augmented["image"]


def get_train_transforms(size: Optional[int] = None) -> AlbumentationsWrapper:
    """
    Training augmentation pipeline.

    Includes:
    - Geometric augmentations (flip, rotate, elastic)
    - Color / radiometric augmentations
    - Normalization + ToTensor
    """
    s = size or IMG_SIZE

    aug_list = [
        A.Resize(s, s),
        A.HorizontalFlip(p=0.5) if aug_cfg["horizontal_flip"] else A.NoOp(),
        A.VerticalFlip(p=0.3) if aug_cfg["vertical_flip"] else A.NoOp(),
        A.Rotate(
            limit=aug_cfg["rotation_limit"],
            border_mode=0,
            p=0.5,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=15,
                    p=1.0,
                ),
                A.CLAHE(clip_limit=3.0, p=1.0),
            ],
            p=0.6,
        )
        if aug_cfg["brightness_contrast"]
        else A.NoOp(),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3)
        if aug_cfg["gaussian_noise"]
        else A.NoOp(),
        A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, p=0.2)
        if aug_cfg["elastic_transform"]
        else A.NoOp(),
        A.CoarseDropout(
            max_holes=8,
            max_height=s // 16,
            max_width=s // 16,
            fill_value=0,
            p=0.2,
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]

    return AlbumentationsWrapper(A.Compose(aug_list))


def get_val_transforms(size: Optional[int] = None) -> transforms.Compose:
    """
    Validation / test / inference transforms (no augmentation).

    Returns a standard torchvision Compose pipeline.
    """
    s = size or IMG_SIZE
    return transforms.Compose(
        [
            transforms.Resize((s, s)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse normalization to visualize a tensor as an image.

    Args:
        tensor: Shape (3, H, W) or (1, 3, H, W).

    Returns:
        NumPy uint8 array (H, W, 3).
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
