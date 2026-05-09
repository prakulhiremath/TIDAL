"""
utils/config.py
---------------
Configuration management for TIDAL experiments.

Loads YAML configs via OmegaConf, applies CLI overrides, validates required
fields, and resolves runtime values (device, class weights, paths).

Usage
-----
    cfg = load_config("configs/default.yaml", overrides=["model.name=lstm", "training.lr=5e-4"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load a YAML config and apply dot-notation CLI overrides.

    Parameters
    ----------
    config_path:
        Path to the base YAML config file.
    overrides:
        List of "key=value" strings, e.g. ["model.name=lstm", "training.lr=1e-3"].

    Returns
    -------
    DictConfig
        Merged, validated, runtime-resolved config.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg: DictConfig = OmegaConf.load(path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    _validate(cfg)
    _resolve_runtime(cfg)
    return cfg


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Persist a resolved config to disk (for experiment reproducibility)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)


def get_device(cfg: DictConfig) -> torch.device:
    """Resolve the compute device from config.

    "auto" selects CUDA > MPS > CPU in that priority order.
    """
    spec = cfg.experiment.get("device", "auto")
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def get_output_dir(cfg: DictConfig, subdir: Optional[str] = None) -> Path:
    """Return (and create) the experiment output directory."""
    base = Path(cfg.experiment.output_dir) / cfg.experiment.name
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = [
    "experiment.name",
    "experiment.seed",
    "data.dataset",
    "data.horizons",
    "data.seq_len",
    "model.name",
    "model.input_dim",
    "model.num_classes",
    "training.epochs",
    "training.lr",
    "training.batch_size",
]


def _validate(cfg: DictConfig) -> None:
    """Check that all required keys are present and have sensible values."""
    for key in _REQUIRED_KEYS:
        parts = key.split(".")
        node: Any = cfg
        for part in parts:
            if not hasattr(node, part) and part not in node:
                raise ValueError(f"Required config key missing: {key}")
            node = node[part]

    fracs = (
        cfg.data.train_frac
        + cfg.data.val_frac
        + cfg.data.test_frac
    )
    if not abs(fracs - 1.0) < 1e-6:
        raise ValueError(
            f"data.train_frac + val_frac + test_frac must sum to 1.0, got {fracs:.4f}"
        )

    if cfg.model.num_classes not in (2, 3):
        raise ValueError("model.num_classes must be 2 (binary) or 3 (three-regime).")

    if cfg.data.seq_len < 10:
        raise ValueError("data.seq_len must be >= 10.")


def _resolve_runtime(cfg: DictConfig) -> None:
    """Fill in runtime-computed values that cannot be expressed in YAML."""
    # Ensure output directories exist
    for key in ("output_dir", "checkpoint_dir", "log_dir"):
        val = cfg.experiment.get(key)
        if val:
            Path(val).mkdir(parents=True, exist_ok=True)

    # Disable AMP on non-CUDA devices
    device_str = cfg.experiment.get("device", "auto")
    if device_str == "auto":
        has_cuda = torch.cuda.is_available()
    else:
        has_cuda = device_str.startswith("cuda")

    if not has_cuda and cfg.training.get("mixed_precision", False):
        OmegaConf.update(cfg, "training.mixed_precision", False, merge=True)
