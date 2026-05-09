"""
training/callbacks.py
─────────────────────
Training callbacks for TIDAL experiments.

Provides:
  - EarlyStopping       : monitors a metric and stops training when it plateaus
  - ModelCheckpoint     : saves best and/or last model weights
  - LRSchedulerCallback : wraps PyTorch schedulers with warmup support
  - MetricTracker       : accumulates per-epoch metric history
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric has not improved for `patience` epochs.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    monitor:
        Metric key to watch (e.g. ``"val_auroc"``).
    mode:
        ``"max"`` if higher is better (AUROC, F1), ``"min"`` if lower is better (loss).
    min_delta:
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_auroc",
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience   = patience
        self.monitor    = monitor
        self.mode       = mode
        self.min_delta  = min_delta

        self._best      = -math.inf if mode == "max" else math.inf
        self._counter   = 0
        self.should_stop = False
        self.best_epoch  = 0

    def step(self, metrics: Dict[str, float], epoch: int) -> bool:
        """Update state; return True if training should stop."""
        value = metrics.get(self.monitor)
        if value is None:
            logger.warning(f"EarlyStopping: '{self.monitor}' not in metrics dict. Skipping.")
            return False

        improved = (
            (value > self._best + self.min_delta)
            if self.mode == "max"
            else (value < self._best - self.min_delta)
        )

        if improved:
            self._best      = value
            self._counter   = 0
            self.best_epoch = epoch
        else:
            self._counter += 1
            logger.info(
                f"EarlyStopping [{self.monitor}={value:.4f}] "
                f"no improvement for {self._counter}/{self.patience} epochs."
            )

        if self._counter >= self.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            self.should_stop = True

        return self.should_stop

    @property
    def best_value(self) -> float:
        return self._best

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best": self._best,
            "counter": self._counter,
            "best_epoch": self.best_epoch,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._best       = state["best"]
        self._counter    = state["counter"]
        self.best_epoch  = state["best_epoch"]
        self.should_stop = state["should_stop"]


# ---------------------------------------------------------------------------
# Model Checkpoint
# ---------------------------------------------------------------------------

class ModelCheckpoint:
    """Save model weights to disk based on metric improvement.

    Parameters
    ----------
    checkpoint_dir:
        Directory to save ``.pt`` checkpoint files.
    monitor:
        Metric key to watch.
    mode:
        ``"max"`` or ``"min"``.
    save_best:
        If True, overwrite ``best_model.pt`` when metric improves.
    save_last:
        If True, always save ``last_model.pt`` after each epoch.
    save_every_n_epochs:
        If > 0, save a timestamped checkpoint every N epochs.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_auroc",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
        save_every_n_epochs: int = 0,
    ) -> None:
        self.checkpoint_dir     = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor            = monitor
        self.mode               = mode
        self.save_best          = save_best
        self.save_last          = save_last
        self.save_every_n_epochs = save_every_n_epochs

        self._best = -math.inf if mode == "max" else math.inf
        self.best_path: Optional[Path] = None

    def step(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Evaluate metrics and save checkpoints as needed.

        Returns
        -------
        Path of the saved best checkpoint, or None if not saved this epoch.
        """
        value    = metrics.get(self.monitor, None)
        saved_as = None

        payload: Dict[str, Any] = {
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "metrics":      metrics,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if extra:
            payload.update(extra)

        # Always save last
        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            torch.save(payload, last_path)

        # Save best
        if self.save_best and value is not None:
            improved = (
                (value > self._best) if self.mode == "max" else (value < self._best)
            )
            if improved:
                self._best    = value
                best_path     = self.checkpoint_dir / "best_model.pt"
                torch.save(payload, best_path)
                self.best_path = best_path
                saved_as       = best_path
                logger.info(
                    f"Checkpoint saved: {best_path} "
                    f"[{self.monitor}={value:.4f}]"
                )

        # Periodic saves
        if self.save_every_n_epochs > 0 and (epoch % self.save_every_n_epochs == 0):
            periodic_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save(payload, periodic_path)

        return saved_as

    def load_best(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """Load best checkpoint weights into ``model`` in-place."""
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError("No best checkpoint found.")
        payload = torch.load(self.best_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        logger.info(f"Loaded best checkpoint from {self.best_path}")
        return payload

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        model: nn.Module,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Generic checkpoint loader."""
        payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        logger.info(f"Loaded checkpoint: epoch={payload.get('epoch', '?')}")
        return payload


# ---------------------------------------------------------------------------
# LR Scheduler Callback
# ---------------------------------------------------------------------------

class LRSchedulerCallback:
    """Thin wrapper around a PyTorch LR scheduler with optional linear warmup.

    Parameters
    ----------
    optimizer:
        The optimizer whose LR will be managed.
    scheduler_name:
        ``"cosine"`` | ``"step"`` | ``"plateau"`` | ``"none"``.
    total_epochs:
        Total training epochs (needed for cosine annealing).
    warmup_epochs:
        Number of epochs for linear LR warm-up (ramps from lr/100 → lr).
    cfg_training:
        OmegaConf training sub-config for scheduler hyper-parameters.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str = "cosine",
        total_epochs: int = 50,
        warmup_epochs: int = 3,
        cfg_training: Optional[Any] = None,
    ) -> None:
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self._base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self._scheduler     = self._build(scheduler_name, total_epochs, cfg_training)

    def _build(
        self,
        name: str,
        total_epochs: int,
        cfg: Optional[Any],
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_epochs - self.warmup_epochs, 1),
                eta_min=1e-6,
            )
        elif name == "step":
            step_size = getattr(cfg, "lr_step_size", 10) if cfg else 10
            gamma     = getattr(cfg, "lr_step_gamma", 0.5) if cfg else 0.5
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif name == "plateau":
            patience = getattr(cfg, "lr_plateau_patience", 5) if cfg else 5
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=patience, factor=0.5
            )
        elif name == "none":
            return None
        else:
            logger.warning(f"Unknown scheduler '{name}', using none.")
            return None

    def step(self, epoch: int, val_metric: Optional[float] = None) -> float:
        """Advance the scheduler by one epoch and return current LR.

        During warmup, linearly scales LR from lr/100 → base lr.
        """
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * max(scale, 0.01)
        else:
            if self._scheduler is not None:
                if isinstance(
                    self._scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if val_metric is not None:
                        self._scheduler.step(val_metric)
                else:
                    self._scheduler.step()

        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Accumulate and retrieve per-epoch training metrics.

    Parameters
    ----------
    keys:
        List of metric names to track.
    """

    def __init__(self, keys: Optional[List[str]] = None) -> None:
        self._history: Dict[str, List[float]] = {}
        if keys:
            for k in keys:
                self._history[k] = []

    def update(self, metrics: Dict[str, float]) -> None:
        """Append one epoch of metrics."""
        for k, v in metrics.items():
            if k not in self._history:
                self._history[k] = []
            self._history[k].append(float(v))

    def get(self, key: str) -> List[float]:
        return self._history.get(key, [])

    def latest(self, key: str) -> Optional[float]:
        vals = self._history.get(key, [])
        return vals[-1] if vals else None

    def best(self, key: str, mode: str = "max") -> Optional[float]:
        vals = self._history.get(key, [])
        if not vals:
            return None
        return max(vals) if mode == "max" else min(vals)

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    def to_dict(self) -> Dict[str, List[float]]:
        return self.history
