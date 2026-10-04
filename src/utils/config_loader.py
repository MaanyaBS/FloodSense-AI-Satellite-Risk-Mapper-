"""
src/utils/config_loader.py
Load and cache the central YAML configuration file.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@lru_cache(maxsize=1)
def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load config.yaml from project root.

    The function is LRU-cached so it only reads the file once per session.

    Args:
        config_path: Override path (default: project_root/config.yaml).

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        # Walk up from this file's location to find config.yaml
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
        if config_path is None:
            raise FileNotFoundError("config.yaml not found in project tree.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int = None, deterministic: bool = True) -> int:
    """
    Seed every RNG that affects a training run.

    config.yaml declares project.seed but nothing previously read it, so runs
    were not reproducible. Call this before building models and dataloaders:

        from src.utils.config_loader import load_config, set_seed
        cfg = load_config()
        set_seed(cfg["project"]["seed"])

    Args:
        seed: Seed value. Defaults to project.seed from config.
        deterministic: Force cuDNN into deterministic mode (slower, but
            bit-identical across runs on the same hardware).

    Returns:
        The seed that was applied.
    """
    import random

    import numpy as np

    if seed is None:
        seed = load_config()["project"].get("seed", 42)

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
