"""Utility functions for RocketMind: robust path resolution, seeding, and metadata handling.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def get_rocketmind_dir() -> Path:
    """Return the absolute path to the Rocketmind project directory.
    
    Traverses upward from this file until it locates the directory containing
    'src/rocketmind' or 'results'.
    """
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "src" / "rocketmind").exists() or (parent / "results").exists():
            return parent
    return Path(__file__).resolve().parents[2]


def get_workspace_root() -> Path:
    """Return the outer workspace root containing 'models/' and '.venv/'."""
    rocketmind_dir = get_rocketmind_dir()
    if (rocketmind_dir.parent / "models").exists():
        return rocketmind_dir.parent
    return rocketmind_dir


def get_model_path(model_filename: str) -> Path:
    """Locate a model checkpoint across known search directories.
    
    Checks both D:\\PORJESCTS\\models and D:\\PORJESCTS\\Rocketmind\\models.
    """
    name = model_filename if model_filename.endswith(".zip") else f"{model_filename}.zip"
    
    # 1. Search in outer workspace models directory
    ws_model = get_workspace_root() / "models" / name
    if ws_model.exists():
        return ws_model
        
    # 2. Search in Rocketmind models directory
    rm_model = get_rocketmind_dir() / "models" / name
    if rm_model.exists():
        return rm_model

    # Default fallback to outer models directory
    return ws_model


def get_results_dir() -> Path:
    """Return the path to the results directory, creating it if needed."""
    results_dir = get_rocketmind_dir() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def set_seed(seed: int) -> None:
    """Set random seeds for python random, numpy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON with indentation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file into a dictionary."""
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)

