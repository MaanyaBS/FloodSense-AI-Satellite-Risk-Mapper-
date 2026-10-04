"""
test_label_integrity.py
Guards against fabricating risk-severity labels from binary datasets.

setup_floodnet.py previously shuffled FloodNet's two real classes and
re-split them across five tiers, so 'Low Risk' and 'Non-Flooded' were the
same images and 'Medium/High/Flooded' were the same images. These tests
pin the corrected behaviour.

Run:  python test_label_integrity.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


print("\n" + "=" * 62)
print("  Label integrity tests")
print("=" * 62)

import yaml

CFG = ROOT / "config.yaml"


# ---------------------------------------------------------------------------
print("\nLabel provenance")
# ---------------------------------------------------------------------------
src = (ROOT / "setup_floodnet.py").read_text(encoding="utf-8")

# The fabrication looked like a random re-split of each binary class.
fabrication_markers = [
    "low_risk_imgs   = non_flooded_imgs[:split]",
    "safe_imgs       = non_flooded_imgs[split:]",
    "medium_imgs = flooded_imgs[:n // 3]",
    "high_imgs   = flooded_imgs[n // 3: 2 * n // 3]",
    "flood_imgs  = flooded_imgs[2 * n // 3:]",
]
for m in fabrication_markers:
    check(f"removed: {m.strip()[:46]}", m not in src)

# No random shuffle feeding class assignment should remain in organize_raw.
organize = src.split("def organize_raw")[1].split("def apply_binary_config")[0]
check("organize_raw does not shuffle into classes",
      "random.shuffle" not in organize,
      "no random.shuffle in organize_raw" if "random.shuffle" not in organize
      else "still shuffling")

check("organize_raw maps 1:1 from ground truth",
      "class_names[0]: non_flooded_imgs" in organize
      and "class_names[1]: flooded_imgs" in organize)

check("refuses to invent 5-tier labels",
      "CANNOT BUILD 5-CLASS LABELS FROM FloodNet" in organize)

check("reports no synthetic tiers",
      "no synthetic tiers" in organize)


# ---------------------------------------------------------------------------
print("\nBinary config rewrite (--binary)")
# ---------------------------------------------------------------------------
work = Path(tempfile.mkdtemp())
shutil.copy2(ROOT / "setup_floodnet.py", work / "setup_floodnet.py")
shutil.copy2(CFG, work / "config.yaml")
(work / "config.yaml.bak").write_bytes(CFG.read_bytes())

probe = work / "probe.py"
probe.write_text(
    "import sys, yaml\n"
    f"sys.path.insert(0, r'{work}')\n"
    "from setup_floodnet import apply_binary_config\n"
    "ok = apply_binary_config()\n"
    "print('RESULT', ok)\n",
    encoding="utf-8",
)

env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
res = subprocess.run([sys.executable, str(probe)], capture_output=True,
                     text=True, cwd=work, timeout=180, env=env)
check("apply_binary_config runs", res.returncode == 0,
      res.stderr.strip().splitlines()[-1] if res.returncode else "")

cfg = yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))
check("num_classes becomes 2", cfg["data"]["num_classes"] == 2,
      f"got {cfg['data']['num_classes']}")
check("class_names becomes 2 real labels",
      cfg["data"]["class_names"] == ["Non-Flooded", "Flooded"],
      str(cfg["data"]["class_names"]))
check("class_colors count matches", len(cfg["data"]["class_colors"]) == 2)
check("config still parses and keeps unrelated keys",
      "training" in cfg and "models" in cfg and "image_size" in cfg["data"],
      f"image_size={cfg['data']['image_size']}")
check("5-tier risk_thresholds replaced",
      "medium" not in cfg["prediction"]["risk_thresholds"],
      str(cfg["prediction"]["risk_thresholds"]))
check("backup of original config kept", (work / "config.yaml.bak").exists())

# Idempotency: running twice must not corrupt the file.
probe.write_text(
    "import sys\n"
    f"sys.path.insert(0, r'{work}')\n"
    "from setup_floodnet import apply_binary_config\n"
    "print('RESULT', apply_binary_config())\n",
    encoding="utf-8",
)
res2 = subprocess.run([sys.executable, str(probe)], capture_output=True,
                      text=True, cwd=work, timeout=180, env=env)
cfg2 = yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))
check("rewrite is idempotent",
      res2.returncode == 0 and cfg2["data"]["num_classes"] == 2
      and cfg2["data"]["class_names"] == ["Non-Flooded", "Flooded"],
      f"second run: {cfg2['data']['class_names']}")

shutil.rmtree(work, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\nLive config unchanged by tests")
# ---------------------------------------------------------------------------
live = yaml.safe_load(CFG.read_text(encoding="utf-8"))
check("repo config.yaml untouched",
      live["data"]["num_classes"] == 5,
      f"still {live['data']['num_classes']} classes "
      f"({', '.join(live['data']['class_names'])})")

print("\n" + "=" * 62)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"    - {f}")
print("=" * 62)
sys.exit(1 if FAIL else 0)