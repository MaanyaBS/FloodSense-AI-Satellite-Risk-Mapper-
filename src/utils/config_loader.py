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
