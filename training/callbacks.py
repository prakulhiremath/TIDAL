"""
training/callbacks.py
──────────────────────
Training callbacks for TIDAL experiments.

Implements:
    - EarlyStopping: Halt training when validation metric plateaus
    - ModelCheckpoint: Save best model weights during training
    - LRSchedulerCallback: Adaptive learning rate management
    - MetricHistory: Track and export metric trajectories
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.

    Monitors a validation metric and halts training when
    no improvement is seen for `patience` consecutive epochs.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: 'max' for metrics like AUROC, 'min' for loss.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current epoch validation metric.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(f"EarlyStopping: counter={self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset callback state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Save the best model checkpoint during training.

    Monitors a metric and saves model weights whenever improvement
    is detected. Keeps only the single best checkpoint by default.

    Args:
        dirpath: Directory to save checkpoints.
        filename: Checkpoint filename (without extension).
        monitor: Metric name to monitor.
        mode: 'max' or 'min'.
        save_last: Also save last epoch checkpoint.
    """

    def __init__(
        self,
        dirpath: str = "results/checkpoints",
        filename: str = "best_model",
        monitor: str = "val_auroc",
        mode: str = "max",
        save_last: bool = False,
    ):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last

        self.best_score: Optional[float] = None
        self.best_model_path: Optional[str] = None

    def __call__(self, model: nn.Module, score: float, epoch: int) -> bool:
        """
        Check if model improved and save if so.

        Args:
            model: PyTorch model to save.
            score: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if model was saved (improvement detected).
        """
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )

        if improved:
            self.best_score = score
            save_path = self.dirpath / f"{self.filename}.pt"
            torch.save(model.state_dict(), save_path)
            self.best_model_path = str(save_path)
            logger.info(f"✓ New best model saved: {save_path} ({self.monitor}={score:.4f})")
            return True

        if self.save_last:
            last_path = self.dirpath / f"{self.filename}_last.pt"
            torch.save(model.state_dict(), last_path)

        return False


class LRSchedulerCallback:
    """
    Wrapper for learning rate schedulers with logging.

    Args:
        scheduler: PyTorch LR scheduler instance.
        monitor: Metric to pass to ReduceLROnPlateau (if applicable).
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: str = "val_loss",
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self._is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    def step(self, metric: Optional[float] = None) -> None:
        """
        Advance the scheduler by one step.

        Args:
            metric: Current metric value (required for ReduceLROnPlateau).
        """
        if self._is_plateau and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

        current_lr = self.scheduler.optimizer.param_groups[0]["lr"]
        logger.debug(f"LR updated to {current_lr:.2e}")


class MetricHistory:
    """
    Track and export metric trajectories across training.

    Supports logging per-epoch metrics and exporting to CSV/JSON.

    Usage:
        history = MetricHistory()
        history.update(epoch=1, train_loss=0.5, val_auroc=0.72)
        history.to_csv("results/training_history.csv")
    """

    def __init__(self):
        self._records = []
        self._columns = set()

    def update(self, epoch: int, **metrics) -> None:
        """
        Record metrics for one epoch.

        Args:
            epoch: Current epoch number.
            **metrics: Metric name-value pairs.
        """
        record = {"epoch": epoch, **metrics}
        self._records.append(record)
        self._columns.update(metrics.keys())

    def to_dataframe(self):
        """Return history as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._records)

    def to_csv(self, path: str) -> None:
        """Save history to CSV file."""
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Training history saved to {path}")

    def get_best_epoch(self, metric: str = "val_auroc", mode: str = "max") -> dict:
        """
        Return the record for the best epoch on a given metric.

        Args:
            metric: Metric name to optimize.
            mode: 'max' or 'min'.

        Returns:
            Record dictionary for best epoch.
        """
        if not self._records:
            return {}
        func = max if mode == "max" else min
        return func(self._records, key=lambda r: r.get(metric, 0))

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"MetricHistory({len(self)} epochs, metrics={list(self._columns)})"
