"""
src/utils/generate_demo_data.py
Generate synthetic satellite-like images for testing the pipeline
without requiring a real dataset download.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from src.utils.config_loader import load_config

cfg = load_config()
CLASS_NAMES = cfg["data"]["class_names"]
IMG_SIZE = cfg["data"]["image_size"]
RAW_DIR = Path(cfg["data"]["raw_dir"])


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def generate_non_flooded(size: int) -> np.ndarray:
    """Green vegetation + brown soil — healthy land."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Base green
    img[:, :] = [34, 139, 34]
    # Add texture
    noise = np.random.randint(-30, 30, (size, size, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    # Urban patches
    n_patches = random.randint(2, 6)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(n_patches):
        x, y = random.randint(0, size-30), random.randint(0, size-30)
        w, h = random.randint(10, 40), random.randint(10, 40)
        color = (
            random.randint(150, 200),
            random.randint(140, 180),
            random.randint(130, 160),
        )
        draw.rectangle([x, y, x+w, y+h], fill=color)
    return np.array(pil)


def generate_flooded(size: int) -> np.ndarray:
    """Dark blue water covering most of the scene."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [20, 40, 120]
    noise = np.random.randint(-20, 20, (size, size, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    # Submerged structure remnants
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(random.randint(1, 4)):
        x, y = random.randint(0, size-20), random.randint(0, size-20)
        draw.rectangle([x, y, x+15, y+15], fill=(60, 70, 80))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=1))
    return np.array(pil)


def generate_high_risk(size: int) -> np.ndarray:
    """Saturated soil, partial flooding, muted colors."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Lower half flooded
    img[:size//2, :] = [50, 100, 150]
    img[size//2:, :] = [80, 100, 60]
    noise = np.random.randint(-25, 25, (size, size, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=0.5))
    return np.array(pil)


def generate_medium_risk(size: int) -> np.ndarray:
    """Mix of green and wet areas."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [80, 120, 80]
    noise = np.random.randint(-30, 30, (size, size, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # Water patches
    for _ in range(random.randint(3, 8)):
        x, y = random.randint(0, size-40), random.randint(0, size-40)
        w, h = random.randint(15, 50), random.randint(10, 35)
        draw.ellipse([x, y, x+w, y+h], fill=(40, 80, 140, 200))
    return np.array(pil)


def generate_low_risk(size: int) -> np.ndarray:
    """Mostly green with a few small water features."""
    img = generate_non_flooded(size)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # Small water body
    x, y = random.randint(30, size-60), random.randint(30, size-60)
    draw.ellipse([x, y, x+30, y+20], fill=(60, 100, 170))
    return np.array(pil)


GENERATORS = {
    "Non-Flooded": generate_non_flooded,
    "Low Risk": generate_low_risk,
    "Medium Risk": generate_medium_risk,
    "High Risk": generate_high_risk,
    "Flooded": generate_flooded,
}


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_demo_data(
    n_per_class: int = 100,
    size: int = IMG_SIZE,
    output_dir: Path = RAW_DIR,
):
    """
    Generate synthetic flood-risk satellite images for testing.

    Args:
        n_per_class: Number of images per class.
        size: Image dimensions (square).
        output_dir: Destination root directory.
    """
    output_dir = Path(output_dir)
    print(f"\nGenerating synthetic demo dataset → {output_dir}")
    print(f"  {n_per_class} images × {len(CLASS_NAMES)} classes = "
          f"{n_per_class * len(CLASS_NAMES)} total")

    for class_name, generator in GENERATORS.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n_per_class), desc=f"  {class_name:<18}", ncols=70):
            img_array = generator(size)
            pil_img = Image.fromarray(img_array)
            pil_img.save(class_dir / f"{class_name.replace(' ', '_')}_{i:04d}.jpg")

    print("\nDemo data generation complete.")
    print(f"Run: python src/preprocessing/dataset.py to split into train/val/test")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic flood demo data")
    parser.add_argument("--n", type=int, default=100, help="Images per class")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--output", type=str, default=str(RAW_DIR))
    args = parser.parse_args()

    generate_demo_data(n_per_class=args.n, size=args.size, output_dir=Path(args.output))
