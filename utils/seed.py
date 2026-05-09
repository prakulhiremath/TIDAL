"""
utils/seed.py
─────────────
Reproducibility utilities for TIDAL experiments.
Sets all relevant random seeds across Python, NumPy, and PyTorch.
"""

import os
import random
import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for full reproducibility across all frameworks.

    Args:
        seed: Integer seed value.
        deterministic: If True, enforce deterministic CUDA operations
                       (may slow down training slightly).
    """
    logger.info(f"Setting global seed to {seed}")

    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Deterministic ops (trades speed for reproducibility)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic flag
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except AttributeError:
            pass  # Older PyTorch versions


def get_rng_state() -> dict:
    """
    Capture current RNG state for checkpointing.

    Returns:
        Dictionary containing all RNG states.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """
    Restore RNG state from a checkpoint.

    Args:
        state: Dictionary from get_rng_state().
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    logger.debug("RNG state restored from checkpoint.")
