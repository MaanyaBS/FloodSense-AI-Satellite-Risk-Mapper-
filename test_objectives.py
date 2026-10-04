"""
test_objectives.py
Verification tests for the project objectives.

Focuses on the behaviours the objectives claim, rather than on model accuracy:
  O2/O6  reproducible, seeded, correctly-parameterised data loading
  O5     five-class contract enforced, not silently degraded
  O6     num_workers actually honoured

Run:  python test_objectives.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.dataset import FloodDataset, get_dataloaders
from src.utils.config_loader import load_config, set_seed

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))


def make_images(root, class_name, n=4):
    """Write n tiny valid PNGs into root/class_name."""
    from PIL import Image
    import numpy as np

    d = Path(root) / class_name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = (np.random.rand(24, 24, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(d / f"{class_name.replace(' ', '_')}_{i}.png")
    return d


print("\n" + "=" * 62)
print("  FloodSense - objective verification tests")
print("=" * 62)

cfg = load_config()
CLASSES = cfg["data"]["class_names"]


# ---------------------------------------------------------------------------
print("\nO5: five-class contract")
# ---------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp())

try:
    # Only one of five classes present -> must raise, not silently load.
    make_images(tmp / "train", CLASSES[0])
    try:
        FloodDataset(tmp / "train")
        check("single-class split is rejected", False, "no error raised")
    except ValueError as e:
        check("single-class split is rejected", True, str(e).split("\n")[0])

    # All five classes, but one left empty -> must raise and name it.
    for c in CLASSES[:-1]:
        make_images(tmp / "train2", c)
    (tmp / "train2" / CLASSES[-1]).mkdir(parents=True, exist_ok=True)
    try:
        FloodDataset(tmp / "train2")
        check("empty class dir is rejected", False, "no error raised")
    except ValueError as e:
        named = CLASSES[-1] in str(e)
        check("empty class dir is rejected", named, f"names {CLASSES[-1]!r}")

    # Case sensitivity: wrong folder name must be caught.
    make_images(tmp / "train3", "non_flooded")  # wrong: should be "Non-Flooded"
    try:
        FloodDataset(tmp / "train3")
        check("case-mismatched folder is caught", False, "no error raised")
    except ValueError:
        check("case-mismatched folder is caught", True)

    # All five populated -> loads cleanly, 5 distinct labels.
    for c in CLASSES:
        make_images(tmp / "good", c, n=4)
    ds = FloodDataset(tmp / "good")
    check(
        "valid 5-class split loads",
        len({lab for _, lab in ds.samples}) == 5,
        f"{len(ds)} images, {len({l for _, l in ds.samples})} classes",
    )

    # Missing split directory -> clear error, not an empty loader.
    try:
        FloodDataset(tmp / "does_not_exist")
        check("missing split dir raises", False, "no error raised")
    except FileNotFoundError as e:
        check("missing split dir raises", True, str(e).split("\n")[0])

except Exception as e:  # noqa: BLE001
    check("O5 suite ran", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
print("\nO6: reproducibility and honoured parameters")
# ---------------------------------------------------------------------------

# Seed actually varies the data order (was previously never called anywhere).
set_seed(42)
a = FloodDataset(tmp / "good")
first_a = [str(p) for p, _ in a.samples]
set_seed(42)
b = FloodDataset(tmp / "good")
first_b = [str(p) for p, _ in b.samples]
check("seed 42 reproduces identical order", first_a == first_b)

set_seed(7)
c = FloodDataset(tmp / "good")
first_c = [str(p) for p, _ in c.samples]
check("a different seed changes the order", first_a != first_c)

# get_dataloaders expects a prepared processed_dir containing train/ val/ test/
# subdirectories, one per class. Build that layout explicitly.
proc = tmp / "processed"
for split in ("train", "val", "test"):
    for c in CLASSES:
        make_images(proc / split, c, n=6)

try:
    loaders = get_dataloaders(proc, batch_size=2, num_workers=0)
    check(
        "num_workers=0 honoured",
        loaders["train"].num_workers == 0,
        f"train loader reports {loaders['train'].num_workers}",
    )
    check("train loader yields batches", len(loaders["train"]) > 0,
          f"{len(loaders['train'])} batches")
    imgs, labels = next(iter(loaders["train"]))
    check(
        "batch tensors well-formed",
        imgs.ndim == 4 and labels.ndim == 1,
        f"images {tuple(imgs.shape)}, labels {tuple(labels.shape)}",
    )
    check(
        "image tensor matches configured size",
        imgs.shape[-1] == cfg["data"]["image_size"]
        and imgs.shape[-2] == cfg["data"]["image_size"],
        f"{tuple(imgs.shape)} vs image_size={cfg['data']['image_size']}",
    )
except Exception as e:  # noqa: BLE001
    check("get_dataloaders honours num_workers", False, f"{type(e).__name__}: {e}")

# Default should come from config, not a hardcoded 0.
try:
    d = get_dataloaders(proc, batch_size=2)
    check(
        "default num_workers comes from config",
        d["train"].num_workers == cfg["training"]["num_workers"],
        f"config says {cfg['training']['num_workers']}, "
        f"loader uses {d['train'].num_workers}",
    )
except Exception as e:  # noqa: BLE001
    check("default num_workers comes from config", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
print("\nConfig contract")
# ---------------------------------------------------------------------------
check("num_classes matches class_names length",
      cfg["data"]["num_classes"] == len(cfg["data"]["class_names"]),
      f"{cfg['data']['num_classes']} vs {len(cfg['data']['class_names'])}")
check("class_colors length matches num_classes",
      len(cfg["data"]["class_colors"]) == cfg["data"]["num_classes"])
check("split ratios sum to 1.0",
      abs(cfg["data"]["train_ratio"] + cfg["data"]["val_ratio"]
          + cfg["data"]["test_ratio"] - 1.0) < 1e-6,
      f"{cfg['data']['train_ratio']}+{cfg['data']['val_ratio']}"
      f"+{cfg['data']['test_ratio']}")
check("project.seed is declared", "seed" in cfg["project"],
      f"seed={cfg['project'].get('seed')}")

shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 62)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  Failed:")
    for f in FAIL:
        print(f"    - {f}")
print("=" * 62)
sys.exit(1 if FAIL else 0)