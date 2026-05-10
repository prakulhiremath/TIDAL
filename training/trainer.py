"""
training/trainer.py
────────────────────
Universal training loop for all TIDAL models.

Supports:
    - Deep learning models (TIDAL, LSTM, Transformer, SSM)
    - Sklearn-compatible baselines (Logistic Regression, XGBoost)
    - Multi-horizon label training
    - Early stopping with best model checkpointing
    - TensorBoard logging
    - Gradient clipping
    - Learning rate scheduling
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from loguru import logger

from training.losses import MultiHorizonLoss
from training.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
from evaluation.metrics import compute_binary_metrics


class TIDALTrainer:
    """
    Universal trainer for TIDAL and baseline deep learning models.

    Handles the full training lifecycle:
        - Epoch loop with train/val passes
        - Multi-horizon label unpacking
        - Metric tracking and logging
        - Early stopping and checkpointing
        - Final model evaluation

    Usage:
        trainer = TIDALTrainer(model, optimizer, loss_fn, cfg)
        results = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        cfg: dict,
        device: Optional[torch.device] = None,
        experiment_name: str = "tidal",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train.
            optimizer: Configured optimizer.
            loss_fn: Loss function (MultiHorizonLoss).
            cfg: Full experiment config dict.
            device: Compute device. Auto-detected if None.
            experiment_name: Name for logging and checkpointing.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.experiment_name = experiment_name

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        # Paths
        log_cfg = cfg.get("logging", {})
        self.checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "results/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_cfg.get("log_dir", "logs")) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if log_cfg.get("tensorboard", True):
            self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training config
        train_cfg = cfg.get("training", {})
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.grad_clip = train_cfg.get("gradient_clip", 1.0)
        self.log_every = log_cfg.get("log_every_n_steps", 50)

        # Horizons
        self.horizons = cfg.get("labeling", {}).get("horizons", [10, 30, 60])

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 15),
            mode="max",  # Maximize AUROC
        )
        self.checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"{experiment_name}_best",
            monitor="val_auroc",
            mode="max",
        )

        # Scheduler
        scheduler_name = train_cfg.get("scheduler", "cosine")
        self.scheduler = self._build_scheduler(scheduler_name)

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_auroc": [], "val_f1": [],
        }

        logger.info(f"Trainer initialized: device={self.device}, model={type(model).__name__}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary of training results and final metrics.
        """
        logger.info(f"Starting training: max_epochs={self.max_epochs}")
        best_val_auroc = 0.0
        global_step = 0

        for epoch in range(1, self.max_epochs + 1):
            t_start = time.time()

            # ── Train epoch ─────────────────────────────────────────────────
            train_loss = self._train_epoch(train_loader, epoch, global_step)
            self.history["train_loss"].append(train_loss)

            # ── Validation ──────────────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auroc"].append(val_metrics.get("auroc_h10", 0.0))
            self.history["val_f1"].append(val_metrics.get("f1_h10", 0.0))

            epoch_time = time.time() - t_start
            val_auroc = val_metrics.get("auroc_h10", 0.0)

            logger.info(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_auroc={val_auroc:.4f} | "
                f"val_f1={val_metrics.get('f1_h10', 0):.4f} | "
                f"time={epoch_time:.1f}s"
            )

            # ── TensorBoard logging ─────────────────────────────────────────
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                for k, v in val_metrics.items():
                    if k != "loss":
                        self.writer.add_scalar(f"Val/{k}", v, epoch)

            # ── Checkpoint ──────────────────────────────────────────────────
            self.checkpoint(self.model, val_auroc, epoch)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc

            # ── LR Scheduler ────────────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_auroc)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.writer:
                    self.writer.add_scalar("LR", current_lr, epoch)

            # ── Early stopping ──────────────────────────────────────────────
            if self.early_stopping(val_auroc):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # ── Load best model and return ──────────────────────────────────────
        best_path = self.checkpoint.best_model_path
        if best_path and Path(best_path).exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            logger.info(f"Loaded best model from {best_path}")

        if self.writer:
            self.writer.close()

        results = {
            "best_val_auroc": best_val_auroc,
            "history": self.history,
            "best_model_path": str(best_path) if best_path else None,
            "epochs_trained": len(self.history["train_loss"]),
        }
        self._save_history(results)
        return results

    def _train_epoch(
        self, loader: DataLoader, epoch: int, global_step: int
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (sequences, label_dict) in enumerate(loader):
            sequences = sequences.to(self.device)

            # Build multi-horizon target tensor (B, n_horizons)
            targets = self._build_target_tensor(label_dict)

            # Forward
            self.optimizer.zero_grad()
            output = self.model(sequences)
            logits = output["logits"]

            loss = self.loss_fn(logits, targets)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % self.log_every == 0:
                logger.debug(
                    f"  Epoch {epoch} | step {batch_idx+1}/{len(loader)} | "
                    f"loss={loss.item():.4f}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation and compute full metrics."""
        self.model.eval()
        all_logits, all_targets = [], []
        total_loss = 0.0
        n_batches = 0

        for sequences, label_dict in loader:
            sequences = sequences.to(self.device)
            targets = self._build_target_tensor(label_dict)

            output = self.model(sequences)
            logits = output["logits"]
            loss = self.loss_fn(logits, targets)

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = 1 / (1 + np.exp(-all_logits))

        metrics = {"loss": total_loss / max(n_batches, 1)}

        for i, h in enumerate(self.horizons):
            h_metrics = compute_binary_metrics(
                all_targets[:, i], all_probs[:, i], threshold=0.5
            )
            for k, v in h_metrics.items():
                metrics[f"{k}_h{h}"] = v

        return metrics

    def _build_target_tensor(
        self, label_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Stack per-horizon labels into a (B, n_horizons) tensor.

        Args:
            label_dict: Dict mapping horizon key → (B,) tensor.

        Returns:
            (B, n_horizons) float tensor.
        """
        horizon_tensors = []
        for h in self.horizons:
            key = f"instability_h{h}"
            if key in label_dict:
                horizon_tensors.append(label_dict[key].float())
            else:
                # Fallback to zeros if horizon not available
                sample = next(iter(label_dict.values()))
                horizon_tensors.append(torch.zeros_like(sample).float())

        return torch.stack(horizon_tensors, dim=1).to(self.device)

    def _build_scheduler(
        self, name: str
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build LR scheduler from config name."""
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        elif name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )
        return None

    def _save_history(self, results: dict) -> None:
        """Save training history to JSON."""
        history_path = self.log_dir / "training_history.json"
        serializable = {
            k: v if not isinstance(v, list) else [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            for k, v in results.items()
            if k != "history"
        }
        serializable["history"] = {
            k: [float(x) for x in v]
            for k, v in results["history"].items()
        }
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Training history saved to {history_path}")


class SklearnTrainer:
    """
    Trainer wrapper for sklearn-compatible baseline models
    (Logistic Regression, XGBoost).

    Usage:
        trainer = SklearnTrainer(model, cfg)
        results = trainer.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(self, model: Any, cfg: dict, model_name: str = "baseline"):
        self.model = model
        self.cfg = cfg
        self.model_name = model_name
        self.horizons = cfg.get("labeling", {}).get("horizons", [10, 30, 60])

        ckpt_dir = cfg.get("logging", {}).get("checkpoint_dir", "results/checkpoints")
        self.checkpoint_dir = Path(ckpt_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train sklearn model and evaluate on validation set.

        Args:
            X_train: Flattened training sequences (N, window*features).
            y_train: Binary instability labels (N,).
            X_val: Validation sequences.
            y_val: Validation labels.

        Returns:
            Results dictionary with metrics.
        """
        logger.info(f"Training {self.model_name}: X_train={X_train.shape}")
        t0 = time.time()

        self.model.fit(X_train, y_train)

        elapsed = time.time() - t0
        logger.info(f"Training complete in {elapsed:.1f}s")

        # Predict
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_val)[:, 1]
        else:
            probs = self.model.predict(X_val).astype(float)

        metrics = compute_binary_metrics(y_val, probs, threshold=0.5)
        logger.info(f"Val metrics: AUROC={metrics.get('auroc', 0):.4f}, F1={metrics.get('f1', 0):.4f}")

        # Save model
        import joblib
        model_path = self.checkpoint_dir / f"{self.model_name}_best.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        return {
            "model_name": self.model_name,
            "val_metrics": metrics,
            "training_time": elapsed,
            "model_path": str(model_path),
        }


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    Build optimizer from config.

    Args:
        model: PyTorch model.
        cfg: Training config dict.

    Returns:
        Configured optimizer.
    """
    train_cfg = cfg.get("training", {})
    name = train_cfg.get("optimizer", "adam")
    lr = train_cfg.get("learning_rate", 0.001)
    wd = train_cfg.get("weight_decay", 0.0001)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
