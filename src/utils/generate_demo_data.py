"""
src/utils/generate_demo_data.py
Generate synthetic satellite-like images for training the flood risk pipeline.
Produces visually diverse images with realistic color palettes, textures,
and feature distributions to enable meaningful model learning.
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
# Helper utilities
# ---------------------------------------------------------------------------

def _random_color(base, spread=30):
    """Generate a random color around a base RGB tuple."""
    return tuple(max(0, min(255, c + random.randint(-spread, spread))) for c in base)


def _add_perlin_noise(img_np, intensity=25):
    """Add multi-scale noise for realistic texture."""
    h, w, c = img_np.shape
    result = img_np.astype(np.float32)
    # Multi-scale noise
    for scale in [1, 2, 4, 8]:
        sh, sw = max(1, h // scale), max(1, w // scale)
        noise = np.random.randn(sh, sw, c) * (intensity / scale)
        noise_img = Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8))
        noise_up = np.array(noise_img.resize((w, h), Image.BILINEAR)).astype(np.float32) - 128
        result += noise_up
    return np.clip(result, 0, 255).astype(np.uint8)


def _gradient_background(size, color_top, color_bot):
    """Create a vertical gradient background."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        t = y / max(size - 1, 1)
        for c in range(3):
            img[y, :, c] = color_top[c] * (1 - t) + color_bot[c] * t
    return img.astype(np.uint8)


def _add_random_shapes(pil_img, shapes, colors, count_range=(3, 10)):
    """Add random geometric shapes to image."""
    draw = ImageDraw.Draw(pil_img)
    size = pil_img.size[0]
    for _ in range(random.randint(*count_range)):
        shape = random.choice(shapes)
        color = _random_color(random.choice(colors))
        x = random.randint(0, size - 10)
        y = random.randint(0, size - 10)
        w = random.randint(5, size // 4)
        h = random.randint(5, size // 4)
        if shape == "rect":
            draw.rectangle([x, y, x + w, y + h], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x, y, x + w, y + h], fill=color)
    return pil_img


# ---------------------------------------------------------------------------
# Non-Flooded: dry land, green vegetation, urban areas, arid terrain
# ---------------------------------------------------------------------------

def generate_non_flooded(size: int) -> np.ndarray:
    """Generate diverse non-flooded terrain: vegetation, urban, arid, farmland."""
    variant = random.choice(["vegetation", "urban", "arid", "farmland", "forest"])

    if variant == "vegetation":
        base_top = _random_color((45, 140, 40), 20)
        base_bot = _random_color((70, 160, 50), 25)
        img = _gradient_background(size, base_top, base_bot)
        img = _add_perlin_noise(img, intensity=30)
        pil = Image.fromarray(img)
        # Field line patterns
        draw = ImageDraw.Draw(pil)
        for _ in range(random.randint(2, 6)):
            y = random.randint(0, size)
            c = _random_color((55, 130, 35), 15)
            draw.line([(0, y), (size, y + random.randint(-20, 20))], fill=c, width=random.randint(1, 4))
        # Occasional brown patches (soil)
        _add_random_shapes(pil, ["rect", "ellipse"],
                           [(140, 120, 80), (160, 140, 90), (130, 110, 70)],
                           count_range=(1, 4))
        return np.array(pil)

    elif variant == "urban":
        base = _random_color((150, 150, 150), 20)
        img = np.full((size, size, 3), base, dtype=np.uint8)
        img = _add_perlin_noise(img, intensity=15)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Building-like rectangles
        for _ in range(random.randint(6, 20)):
            x, y = random.randint(0, size - 15), random.randint(0, size - 15)
            w, h = random.randint(8, 40), random.randint(8, 40)
            c = _random_color(random.choice([
                (180, 170, 160), (200, 195, 190), (160, 160, 170),
                (140, 135, 130), (190, 180, 170)
            ]), 10)
            draw.rectangle([x, y, x + w, y + h], fill=c, outline=_random_color((100, 100, 100), 15))
        # Roads
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, size)
            draw.line([(0, y), (size, y + random.randint(-10, 10))],
                      fill=_random_color((80, 80, 85), 10), width=random.randint(3, 8))
        # Add small green parks
        _add_random_shapes(pil, ["ellipse"], [(50, 130, 50), (60, 140, 55)], count_range=(0, 3))
        return np.array(pil)

    elif variant == "arid":
        base_top = _random_color((190, 170, 130), 20)
        base_bot = _random_color((170, 150, 110), 20)
        img = _gradient_background(size, base_top, base_bot)
        img = _add_perlin_noise(img, intensity=20)
        pil = Image.fromarray(img)
        # Sandy/rocky patches
        _add_random_shapes(pil, ["rect", "ellipse"],
                           [(200, 180, 140), (180, 160, 120), (160, 145, 110)],
                           count_range=(2, 6))
        # Sparse scrub vegetation
        _add_random_shapes(pil, ["ellipse"],
                           [(100, 130, 70), (90, 120, 60)],
                           count_range=(0, 4))
        return np.array(pil)

    elif variant == "farmland":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Alternating crop/soil rows
        row_h = random.randint(8, 25)
        colors_a = _random_color((50, 140, 45), 15)
        colors_b = _random_color((140, 120, 80), 15)
        for y in range(0, size, row_h * 2):
            img[y:y + row_h, :] = colors_a
            img[y + row_h:y + row_h * 2, :] = colors_b
        img = _add_perlin_noise(img, intensity=20)
        return img

    else:  # forest
        base = _random_color((25, 100, 25), 15)
        img = np.full((size, size, 3), base, dtype=np.uint8)
        img = _add_perlin_noise(img, intensity=35)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Tree-like canopy circles
        for _ in range(random.randint(15, 40)):
            x, y = random.randint(0, size - 5), random.randint(0, size - 5)
            r = random.randint(4, 18)
            c = _random_color(random.choice([
                (20, 90, 20), (30, 110, 30), (40, 120, 35), (15, 80, 15)
            ]), 12)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
        return np.array(pil)


# ---------------------------------------------------------------------------
# Flooded: extensive water coverage, submerged features
# ---------------------------------------------------------------------------

def generate_flooded(size: int) -> np.ndarray:
    """Generate flooded terrain: extensive water, submerged structures."""
    variant = random.choice(["deep_water", "muddy_flood", "urban_flood"])

    if variant == "deep_water":
        base_top = _random_color((15, 35, 110), 15)
        base_bot = _random_color((25, 50, 130), 15)
        img = _gradient_background(size, base_top, base_bot)
        img = _add_perlin_noise(img, intensity=18)
        pil = Image.fromarray(img)
        # Specular highlights (water reflections)
        draw = ImageDraw.Draw(pil)
        for _ in range(random.randint(3, 10)):
            x, y = random.randint(0, size), random.randint(0, size)
            draw.line([(x, y), (x + random.randint(10, 50), y + random.randint(-5, 5))],
                      fill=_random_color((80, 120, 180), 20), width=1)
        # Submerged structure remnants
        for _ in range(random.randint(0, 3)):
            x, y = random.randint(0, size - 20), random.randint(0, size - 20)
            draw.rectangle([x, y, x + random.randint(6, 18), y + random.randint(6, 18)],
                           fill=_random_color((55, 65, 75), 10))
        return np.array(pil)

    elif variant == "muddy_flood":
        base_top = _random_color((90, 70, 45), 15)
        base_bot = _random_color((70, 55, 35), 15)
        img = _gradient_background(size, base_top, base_bot)
        img = _add_perlin_noise(img, intensity=25)
        pil = Image.fromarray(img)
        # Debris
        _add_random_shapes(pil, ["rect", "ellipse"],
                           [(60, 50, 30), (80, 60, 40), (100, 80, 55)],
                           count_range=(2, 8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
        return np.array(pil)

    else:  # urban_flood
        # Water base
        base = _random_color((30, 55, 120), 15)
        img = np.full((size, size, 3), base, dtype=np.uint8)
        img = _add_perlin_noise(img, intensity=20)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Partially submerged buildings (tops visible)
        for _ in range(random.randint(3, 10)):
            x, y = random.randint(0, size - 20), random.randint(0, size - 20)
            w, h = random.randint(8, 30), random.randint(8, 30)
            c = _random_color((120, 115, 110), 15)
            draw.rectangle([x, y, x + w, y + h], fill=c, outline=_random_color((90, 85, 80), 10))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.7))
        return np.array(pil)


# ---------------------------------------------------------------------------
# High Risk: significant water, saturated soil
# ---------------------------------------------------------------------------

def generate_high_risk(size: int) -> np.ndarray:
    """Generate high risk terrain: large water areas, saturated soil."""
    variant = random.choice(["half_flooded", "waterlogged", "river_overflow"])

    if variant == "half_flooded":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        split = random.randint(size // 3, 2 * size // 3)
        # Water portion
        img[:split, :] = _random_color((35, 60, 130), 15)
        # Saturated land
        img[split:, :] = _random_color((70, 90, 55), 15)
        img = _add_perlin_noise(img, intensity=22)
        pil = Image.fromarray(img)
        # Transition zone (muddy)
        draw = ImageDraw.Draw(pil)
        for x in range(0, size, 3):
            y = split + random.randint(-15, 15)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                         fill=_random_color((80, 80, 60), 15))
        return np.array(pil)

    elif variant == "waterlogged":
        base = _random_color((60, 85, 55), 15)
        img = np.full((size, size, 3), base, dtype=np.uint8)
        img = _add_perlin_noise(img, intensity=25)
        pil = Image.fromarray(img)
        # Large water patches
        _add_random_shapes(pil, ["ellipse"],
                           [(30, 55, 120), (40, 65, 130), (35, 60, 115)],
                           count_range=(4, 12))
        # Muddy areas
        _add_random_shapes(pil, ["ellipse"],
                           [(85, 70, 45), (95, 75, 50)],
                           count_range=(2, 5))
        return np.array(pil)

    else:  # river_overflow
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Land background
        img[:, :] = _random_color((70, 100, 55), 15)
        img = _add_perlin_noise(img, intensity=20)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Wide river/overflow
        river_y = random.randint(size // 4, 3 * size // 4)
        river_w = random.randint(size // 3, size // 2)
        for x in range(size):
            y_off = int(15 * np.sin(x / 25.0))
            draw.rectangle([x, river_y + y_off - river_w // 2,
                            x + 1, river_y + y_off + river_w // 2],
                           fill=_random_color((30, 55, 115), 12))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.5))
        return np.array(pil)


# ---------------------------------------------------------------------------
# Medium Risk: partial water, stressed vegetation
# ---------------------------------------------------------------------------

def generate_medium_risk(size: int) -> np.ndarray:
    """Generate medium risk terrain: scattered water pools, stressed vegetation."""
    variant = random.choice(["puddles", "damp_fields", "edge_water"])

    if variant == "puddles":
        base = _random_color((75, 115, 70), 15)
        img = np.full((size, size, 3), base, dtype=np.uint8)
        img = _add_perlin_noise(img, intensity=25)
        pil = Image.fromarray(img)
        # Medium water puddles
        _add_random_shapes(pil, ["ellipse"],
                           [(40, 75, 135), (45, 80, 140), (50, 85, 130)],
                           count_range=(4, 10))
        # Stressed yellow-green vegetation
        _add_random_shapes(pil, ["ellipse"],
                           [(120, 140, 60), (130, 145, 55)],
                           count_range=(2, 6))
        return np.array(pil)

    elif variant == "damp_fields":
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Alternating damp/dry patches
        patch_size = random.randint(20, 50)
        for y in range(0, size, patch_size):
            for x in range(0, size, patch_size):
                if random.random() < 0.35:
                    color = _random_color((50, 75, 60), 15)  # Damp/dark
                else:
                    color = _random_color((80, 120, 70), 15)  # Dry
                img[y:y + patch_size, x:x + patch_size] = color
        img = _add_perlin_noise(img, intensity=20)
        pil = Image.fromarray(img)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.2))
        return np.array(pil)

    else:  # edge_water
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Mostly green with water along one edge
        img[:, :] = _random_color((65, 120, 60), 15)
        edge = random.choice(["left", "right", "top", "bottom"])
        water_w = random.randint(size // 5, size // 3)
        water_color = _random_color((35, 65, 125), 15)
        if edge == "left":
            img[:, :water_w] = water_color
        elif edge == "right":
            img[:, size - water_w:] = water_color
        elif edge == "top":
            img[:water_w, :] = water_color
        else:
            img[size - water_w:, :] = water_color
        img = _add_perlin_noise(img, intensity=22)
        pil = Image.fromarray(img)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.8))
        return np.array(pil)


# ---------------------------------------------------------------------------
# Low Risk: mostly dry with minor features
# ---------------------------------------------------------------------------

def generate_low_risk(size: int) -> np.ndarray:
    """Generate low risk terrain: mostly vegetation with small water bodies."""
    variant = random.choice(["small_pond", "ditch", "damp_patch"])

    if variant == "small_pond":
        img = generate_non_flooded(size)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # One small water body
        x, y = random.randint(30, size - 60), random.randint(30, size - 60)
        w = random.randint(15, 35)
        h = random.randint(10, 25)
        draw.ellipse([x, y, x + w, y + h], fill=_random_color((50, 90, 150), 15))
        return np.array(pil)

    elif variant == "ditch":
        img = generate_non_flooded(size)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Narrow water channel
        y_start = random.randint(size // 4, 3 * size // 4)
        for x in range(size):
            y_off = int(8 * np.sin(x / 20.0))
            draw.rectangle([x, y_start + y_off - 3, x + 1, y_start + y_off + 3],
                           fill=_random_color((45, 80, 140), 12))
        return np.array(pil)

    else:  # damp_patch
        img = generate_non_flooded(size)
        pil = Image.fromarray(img)
        # Small darker (damp) patch
        _add_random_shapes(pil, ["ellipse"],
                           [(50, 80, 55), (55, 85, 60)],
                           count_range=(1, 3))
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
    n_per_class: int = 200,
    size: int = IMG_SIZE,
    output_dir: Path = RAW_DIR,
):
    """
    Generate synthetic flood-risk satellite images for training.

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
    parser.add_argument("--n", type=int, default=200, help="Images per class")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--output", type=str, default=str(RAW_DIR))
    args = parser.parse_args()

    generate_demo_data(n_per_class=args.n, size=args.size, output_dir=Path(args.output))
