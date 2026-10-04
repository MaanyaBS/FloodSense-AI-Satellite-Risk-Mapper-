"""
setup_floodnet.py
─────────────────────────────────────────────────────────────
Downloads FloodNet dataset from Kaggle, organizes it into the
5 flood-risk classes expected by the training pipeline, then
prepares train/val/test splits and starts training.

Usage:
    python setup_floodnet.py
    python setup_floodnet.py --skip_download   # if already downloaded
    python setup_floodnet.py --epochs 20       # custom epoch count
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# ── Root of project ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.console import enable_utf8_console

enable_utf8_console()

RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
ZIP_NAME = "floodnet-dataset.zip"
ZIP_PATH = ROOT / ZIP_NAME

# FloodNet labels → our 5 risk classes
LABEL_MAP = {
    # FloodNet has two top-level folders: Flooded / Non-Flooded
    "Flooded":     "Flooded",
    "Non-Flooded": "Non-Flooded",
    # Sub-categories we infer from FloodNet v1 structure
    "flooded":     "Flooded",
    "non-flooded": "Non-Flooded",
}


def check_kaggle_credentials():
    """Verify kaggle.json exists and Kaggle CLI is usable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n" + "=" * 60)
        print("  Kaggle credentials NOT found!")
        print("=" * 60)
        print("""
To download FloodNet you need a free Kaggle account and API key.

Steps:
  1. Go to  https://www.kaggle.com  and sign in / create account.
  2. Click your profile picture (top-right) → Settings.
  3. Scroll to 'API' section → click 'Create New Token'.
  4. A file called  kaggle.json  downloads to your computer.
  5. Move it to:  C:\\Users\\<YourName>\\.kaggle\\kaggle.json
     (Create the  .kaggle  folder if it doesn't exist.)
  6. Run this script again:  python setup_floodnet.py

Alternatively, you can download the dataset manually:
  → https://www.kaggle.com/datasets/kmader/floodnet-dataset
  Then extract into  data/raw/  following the structure:
  data/raw/
    Flooded/     (flood images)
    Non-Flooded/ (safe images)
  Then run:  python setup_floodnet.py --skip_download
""")
        sys.exit(1)
    print("✅  Kaggle credentials found.")


def download_floodnet():
    """Download via Kaggle CLI."""
    print("\n📥  Downloading FloodNet dataset (~2–4 GB) ...")
    ret = os.system(
        f"kaggle datasets download -d kmader/floodnet-dataset -p \"{ROOT}\" --unzip"
    )
    if ret != 0:
        print("❌  Download failed. Check your Kaggle credentials and network.")
        sys.exit(1)
    print("✅  Download complete.")


def organize_raw(source: Path):
    """
    Scan `source` recursively for image files, map them into our 5-class
    raw directory structure, creating synthetic Medium/Low/High Risk splits
    from the Non-Flooded and Flooded groups.
    """
    print(f"\n📂  Organising dataset from: {source}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    classes = ["Non-Flooded", "Low Risk", "Medium Risk", "High Risk", "Flooded"]
    for cls in classes:
        (RAW_DIR / cls).mkdir(exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    flooded_imgs     = []
    non_flooded_imgs = []

    for fpath in source.rglob("*"):
        if fpath.suffix.lower() not in img_exts:
            continue
        parent = fpath.parent.name.lower()
        if "flooded" in parent and "non" not in parent:
            flooded_imgs.append(fpath)
        elif "non" in parent or "non-flooded" in parent:
            non_flooded_imgs.append(fpath)

    # If flat structure (no sub-folders), put all in Flooded/Non-Flooded
    if not flooded_imgs and not non_flooded_imgs:
        print("  ⚠️  Could not auto-detect labels. Treating all images as Non-Flooded.")
        for fpath in source.rglob("*"):
            if fpath.suffix.lower() in img_exts:
                non_flooded_imgs.append(fpath)

    # ── Map onto the configured class set ───────────────────────────────────
    #
    # FloodNet provides only two ground-truth labels: Flooded / Non-Flooded.
    # It carries NO risk-severity information, so the intermediate tiers
    # (Low / Medium / High Risk) cannot be derived from it.
    #
    # This function previously shuffled each binary class and re-split it
    # across the five tiers, which produced a model that learned to
    # separate random noise rather than flood severity - every accuracy
    # figure derived from it was meaningless.
    #
    # We now map each FloodNet image to exactly one class, using the
    # class_names defined in config.yaml. If that config requests five
    # tiers we abort with instructions rather than invent labels.
    import yaml

    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["data"]["class_names"]

    if len(class_names) != 2:
        print(
            "\n" + "=" * 60
        )
        print("  CANNOT BUILD 5-CLASS LABELS FROM FloodNet")
        print("=" * 60)
        print(
            "\n  FloodNet is labelled binary (Flooded / Non-Flooded). It has no\n"
            "  risk-severity ground truth, so 'Low Risk', 'Medium Risk' and\n"
            "  'High Risk' cannot be derived from it without fabricating labels.\n\n"
            "  Your config.yaml currently requests:\n"
            f"    num_classes : {cfg['data']['num_classes']}\n"
            f"    class_names : {', '.join(class_names)}\n\n"
            "  Pick one:\n"
            "   1. Honest binary run - edit config.yaml to:\n"
            "        num_classes: 2\n"
            "        class_names: ['Non-Flooded', 'Flooded']\n"
            "        class_colors: [[0,200,0], [139,0,0]]\n"
            "      Then re-run this script.\n\n"
            "   2. Real 5-tier labels - use SEN12-FLOOD, or generate risk tiers\n"
            "      from DEM/topographic data, then point class_names at those.\n\n"
            "  Nothing was copied. Aborting."
        )
        return None

    # Binary config: two classes, mapped 1:1 from FloodNet ground truth.
    mapping = {
        class_names[0]: non_flooded_imgs,
        class_names[1]: flooded_imgs,
    }

    total = 0
    for cls_name, imgs in mapping.items():
        if not imgs:
            continue
        dest = RAW_DIR / cls_name
        for img in imgs:
            shutil.copy2(img, dest / img.name)
        print(f"  {cls_name:<15}: {len(imgs):5d} images")
        total += len(imgs)

    print(f"\n  Total: {total} images across {len(mapping)} classes "
          f"(1:1 from FloodNet ground truth, no synthetic tiers).")
    return total


BINARY_BLOCK = '''  num_classes: 2
  class_names:
    - "Non-Flooded"
    - "Flooded"
  class_colors:        # RGB for visualization overlays
    - [0, 200, 0]      # Green - Non-Flooded
    - [139, 0, 0]      # Dark Red - Flooded
'''


def apply_binary_config():
    """
    Rewrite config.yaml to a 2-class setup for a ground-truth binary run.

    FloodNet cannot supply risk-severity labels, so a binary run is the
    only defensible way to train on it. This edits the real config in
    place, preserving comments and key order, and backs up the original.
    """
    config_path = ROOT / "config.yaml"
    backup_path = ROOT / "config.yaml.bak"

    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"\n[backup] original config saved to {backup_path.name}")

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)

    out, i = [], 0
    while i < len(lines):
        line = lines[i]

        if line.strip() == "num_classes: 5":
            out.append(BINARY_BLOCK)
            i += 1
            # Skip the old class_names list and class_colors block, up to train_ratio.
            while i < len(lines) and not lines[i].lstrip().startswith("train_ratio:"):
                i += 1
            continue

        # Alert thresholds keyed to the 5-tier scheme are meaningless for 2 classes.
        if line.strip().startswith("risk_thresholds:"):
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                i += 1
            out.append("  risk_thresholds:\n")
            out.append("    flooded: 0.50      # binary: p(flooded) at/above this is an alert\n")
            continue

        out.append(line)
        i += 1

    config_path.write_text("".join(out), encoding="utf-8")

    # Validate the result actually loads and is self-consistent.
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg["data"]["class_names"]
    colors = cfg["data"]["class_colors"]

    if cfg["data"]["num_classes"] != 2 or len(names) != 2 or len(colors) != 2:
        print("\n[error] Failed to rewrite config.yaml to a valid 2-class setup.")
        print(f"    num_classes={cfg['data']['num_classes']} "
              f"class_names={len(names)} class_colors={len(colors)}")
        print(f"    Restoring from {backup_path.name}")
        shutil.copy2(backup_path, config_path)
        return False

    print(f"[ok] config.yaml set to 2 classes: {', '.join(names)}")
    return True


def find_extracted_dir():
    """Find the directory that Kaggle created after extracting."""
    for p in ROOT.iterdir():
        if p.is_dir() and "flood" in p.name.lower():
            return p
    # fallback: look inside data/
    data_dir = ROOT / "data"
    if data_dir.exists():
        for p in data_dir.iterdir():
            if p.is_dir() and "flood" in p.name.lower():
                return p
    return ROOT  # scan root itself


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download (data already in data/raw/)")
    parser.add_argument("--skip_organize", action="store_true",
                        help="Skip organize step")
    parser.add_argument("--epochs_cnn", type=int, default=25,
                        help="Epochs to train CNN (default: 25)")
    parser.add_argument("--epochs_vit", type=int, default=30,
                        help="Epochs to train ViT (default: 30)")
    parser.add_argument("--cnn_only", action="store_true")
    parser.add_argument("--vit_only", action="store_true")
    parser.add_argument(
        "--binary", action="store_true",
        help="Allow binary FloodNet run; rewrites config.yaml to 2 classes "
             "(Non-Flooded / Flooded) so labels stay ground-truth derived"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FloodNet Full Training Pipeline")
    print("=" * 60)

    # ── Download ──────────────────────────────────────────────────────────
    if args.binary:
        apply_binary_config()

    if not args.skip_download:
        check_kaggle_credentials()
        download_floodnet()

        # Organise raw directory
        if not args.skip_organize:
            extracted = find_extracted_dir()
            organize_raw(extracted)
    else:
        # Check data/raw already has images
        total = sum(1 for p in RAW_DIR.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"\n✅  Skipping download — found {total} images in data/raw/")

    # ── Prepare splits ────────────────────────────────────────────────────
    print("\n📊  Preparing train/val/test splits ...")
    from src.preprocessing.dataset import prepare_dataset

    try:
        prepare_dataset(str(RAW_DIR), str(PROC_DIR))
    except ValueError as e:
        # Refuse to train on a split that does not match config.
        print(f"\n❌  Dataset does not match config.yaml:\n\n{e}")
        return 1

    # ── Train CNN first (faster) ──────────────────────────────────────────
    ret_code = 0

    if not args.vit_only:
        print(f"\n🚀  Training CNN (EfficientNet-B3) for {args.epochs_cnn} epochs ...")
        ret = os.system(
            f"python train.py --model cnn --epochs {args.epochs_cnn} --batch_size 32"
        )
        if ret != 0:
            print("❌  CNN training failed.")
            sys.exit(1)
        print("✅  CNN training complete — checkpoint saved.")

    # ── Train ViT ─────────────────────────────────────────────────────────
    if not args.cnn_only:
        print(f"\n🚀  Training ViT (ViT-B/16) for {args.epochs_vit} epochs ...")
        ret = os.system(
            f"python train.py --model vit --epochs {args.epochs_vit} --batch_size 16"
        )
        if ret != 0:
            print("❌  ViT training failed.")
            sys.exit(1)
        print("✅  ViT training complete — checkpoint saved.")

    print("\n" + "=" * 60)
    print("  ✅  All training complete!")
    print("  Launch the dashboard:  streamlit run dashboard/app.py")
    print("=" * 60)

    return ret_code


if __name__ == "__main__":
    sys.exit(main() or 0)


if __name__ == "__main__":
    main()
