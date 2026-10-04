"""
src/preprocessing/dataset.py
Dataset classes, data loading, and preparation pipeline for flood risk mapping.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.preprocessing.augmentation import get_train_transforms, get_val_transforms
from src.utils.config_loader import load_config

cfg = load_config()


# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = cfg["data"]["class_names"]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for name, i in CLASS_TO_IDX.items()}

CLASS_COLORS: List[Tuple[int, int, int]] = [
    tuple(c) for c in cfg["data"]["class_colors"]
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FloodDataset(Dataset):

    def __init__(
        self,
        root: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.samples: List[Tuple[Path, int]] = []

        self._load_samples()

    def _load_samples(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

        if not self.root.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.root}\n"
                f"Run 'python src/utils/generate_demo_data.py' or "
                f"'prepare_dataset()' before building dataloaders."
            )

        missing, empty = [], []

        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root / class_name

            if not class_dir.exists():
                missing.append(class_name)
                continue

            found = 0
            for fpath in sorted(class_dir.iterdir()):
                if fpath.suffix.lower() in valid_exts:
                    self.samples.append((fpath, idx))
                    found += 1

            if found == 0:
                empty.append(class_name)

        # A missing class directory used to be skipped silently, which let a
        # binary Flooded/Non-Flooded layout train as a 1-class dataset and report
        # meaningless accuracy. Class names are also case/spacing sensitive
        # ("Non-Flooded" vs "non_flooded"), so show what was expected.
        problems = []
        if missing:
            problems.append(
                f"  missing directories : {', '.join(missing)}\n"
                f"    (expected under {self.root})"
            )
        if empty:
            problems.append(
                f"  empty directories   : {', '.join(empty)}\n"
                f"    (found no .jpg/.jpeg/.png/.tif/.tiff files)"
            )

        if problems:
            raise ValueError(
                f"Dataset split '{self.root.name}' is incomplete "
                f"({len(self.samples)} images loaded):\n"
                + "\n".join(problems)
                + f"\n\nConfigured classes ({len(self.class_to_idx)}): "
                + ", ".join(self.class_to_idx.keys())
                + "\nDirectory names must match these exactly. FloodNet ships a "
                "binary layout (Flooded/Non-Flooded) — map those folders onto the "
                "five risk tiers before training, or set "
                "data.class_names/data.num_classes in config.yaml."
            )

        if not self.samples:
            raise ValueError(
                f"No images found under {self.root}. "
                f"Expected: {', '.join(self.class_to_idx.keys())}"
            )

        labels = {idx for _, idx in self.samples}
        if len(labels) < 2:
            raise ValueError(
                f"Only {len(labels)} distinct class present in {self.root}. "
                f"A single-class dataset cannot train a classifier - each split "
                f"needs at least 2 classes to compute a meaningful loss."
            )

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        counts = np.zeros(len(self.class_to_idx))
        for _, label in self.samples:
            counts[label] += 1

        counts = np.where(counts == 0, 1, counts)

        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)

        return torch.FloatTensor(weights)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_dataset(raw_dir: str, processed_dir: str) -> None:

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    train_ratio = cfg["data"]["train_ratio"]
    val_ratio = cfg["data"]["val_ratio"]

    print(f"Preparing dataset: {raw_dir} → {processed_dir}")

    for class_dir in sorted(raw_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        images = list(class_dir.glob("*.[jpJP][pnPN][gG]")) + list(
            class_dir.glob("*.tif")
        )

        random.shuffle(images)

        n = len(images)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_imgs in splits.items():

            dest = proc_path / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)

            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

        print(f"  {class_name}: {n_train} train | {n_val} val | {n-n_train-n_val} test")

    print("Dataset preparation complete.")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    processed_dir: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:

    bs = batch_size or cfg["training"]["batch_size"]

    # Honour the caller's num_workers, defaulting to config. multiprocessing
    # dataloaders require an if __name__ == "__main__" guard, which every entry
    # point here already has; set to 0 to serialise.
    nw = cfg["training"]["num_workers"] if num_workers is None else num_workers
    nw = max(0, int(nw))

    proc = Path(processed_dir)

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    datasets = {
        "train": FloodDataset(proc / "train", transform=train_transform),
        "val": FloodDataset(proc / "val", transform=val_transform),
        "test": FloodDataset(proc / "test", transform=val_transform),
    }

    train_sampler = None

    if use_weighted_sampler and len(datasets["train"]) > 0:

        class_weights = datasets["train"].get_class_weights()

        sample_weights = [
            class_weights[label].item()
            for _, label in datasets["train"].samples
        ]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(datasets["train"]),
            replacement=True,
        )

    loaders = {

        "train": DataLoader(
            datasets["train"],
            batch_size=bs,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=nw,
            pin_memory=False,
            drop_last=True,
        ),

        "val": DataLoader(
            datasets["val"],
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=False,
        ),

        "test": DataLoader(
            datasets["test"],
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=False,
        ),
    }

    print(f"\nDataLoaders created:")

    for split, loader in loaders.items():

        n = len(loader.dataset)

        print(f"  {split:5s}: {n:5d} samples  ({len(loader)} batches)")

    return loaders


# ---------------------------------------------------------------------------
# Single image loader (for inference)
# ---------------------------------------------------------------------------
def load_single_image(
    image_path: str,
    image_size: Optional[int] = None,
) -> torch.Tensor:

    size = image_size or cfg["data"]["image_size"]

    transform = get_val_transforms(size)

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)

    return tensor.unsqueeze(0)