"""
training/trainer.py
────────────────────
Config-driven training engine for all TIDAL model architectures.

Features
--------
- Multi-horizon cross-entropy loss with focal weighting
- Mixed-precision (AMP) training on CUDA, auto-disabled elsewhere
- Gradient clipping
- LR scheduling with linear warm-up
- Early stopping
- ModelCheckpoint (best + last)
- TensorBoard logging (optional)
- Per-epoch metric tracking (AUROC, AUPRC, loss, F1)
- Full reproducibility via seeded DataLoaders
- Config-driven: all hyper-params come from OmegaConf DictConfig

Tensor conventions (unchanged from existing codebase)
------------------------------------------------------
    x       : (B, T, F)         — feature sequences
    y       : (B, H)            — int64 regime labels, H = num horizons
    logits  : (B, H, C)         — raw model output
    probs   : (B, H, C)         — softmax over class dim
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.callbacks import (
    EarlyStopping,
    LRSchedulerCallback,
    MetricTracker,
    ModelCheckpoint,
)
from utils.config import get_device, get_output_dir
from utils.logger import get_logger
from utils.seed import set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss builder (multi-class, multi-horizon, compatible with model output)
# ---------------------------------------------------------------------------

class MultiHorizonCELoss(nn.Module):
    """Weighted cross-entropy loss summed across prediction horizons.

    Works with model output shape ``(B, H, C)`` and label shape ``(B, H)``.

    Parameters
    ----------
    num_horizons:
        Number of prediction horizons (H).
    horizon_weights:
        Per-horizon scalar weights. Defaults to the config value.
    class_weights:
        Tensor of shape ``(C,)`` for class imbalance handling.
    focal_gamma:
        Focal modulation (0 = standard CE, >0 = focal).
    """

    def __init__(
        self,
        num_horizons: int,
        horizon_weights: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_horizons  = num_horizons
        self.focal_gamma   = focal_gamma
        self.register_buffer(
            "horizon_weights",
            horizon_weights if horizon_weights is not None
            else torch.ones(num_horizons) / num_horizons,
        )
        self.register_buffer(
            "class_weights",
            class_weights,  # may be None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-horizon loss.

        Parameters
        ----------
        logits  : (B, H, C)
        targets : (B, H)   int64

        Returns
        -------
        Scalar loss.
        """
        total = torch.tensor(0.0, device=logits.device)
        for h in range(self.num_horizons):
            l_h = logits[:, h, :]   # (B, C)
            t_h = targets[:, h]     # (B,)

            ce = F.cross_entropy(
                l_h, t_h,
                weight=self.class_weights,
                reduction="none",
            )  # (B,)

            if self.focal_gamma > 0.0:
                p_t = F.softmax(l_h, dim=-1).gather(1, t_h.unsqueeze(1)).squeeze(1)
                focal_w = (1.0 - p_t) ** self.focal_gamma
                ce = focal_w * ce

            total = total + self.horizon_weights[h] * ce.mean()

        return total


def _build_loss(cfg: DictConfig, class_weights: Optional[torch.Tensor] = None) -> MultiHorizonCELoss:
    """Instantiate loss from config."""
    lc         = cfg.training.loss
    n_horizons = len(cfg.data.horizons)
    hw_list    = list(cfg.training.get("horizon_weights", [1.0 / n_horizons] * n_horizons))
    hw         = torch.tensor(hw_list, dtype=torch.float32)
    focal_gamma = float(lc.get("focal_gamma", 0.0)) if lc.get("type", "ce") == "focal" else 0.0

    return MultiHorizonCELoss(
        num_horizons   = n_horizons,
        horizon_weights = hw,
        class_weights  = class_weights,
        focal_gamma    = focal_gamma,
    )


def _build_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.lower()
    lr   = float(cfg.training.lr)
    wd   = float(cfg.training.weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# Per-batch metric helpers
# ---------------------------------------------------------------------------

def _compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Accuracy and per-class counts for a single batch (cheap, no sklearn)."""
    preds   = logits.argmax(dim=-1)  # (B, H)
    correct = (preds == targets).float().mean().item()
    return {"acc": correct}


# ---------------------------------------------------------------------------
# Epoch-level AUROC / AUPRC via sklearn (deferred import)
# ---------------------------------------------------------------------------

def _epoch_metrics(
    all_probs:   torch.Tensor,  # (N, H, C)
    all_targets: torch.Tensor,  # (N, H)
    num_classes: int = 3,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, macro-F1 over the full epoch."""
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    import numpy as np

    probs   = all_probs.cpu().numpy()    # (N, H, C)
    targets = all_targets.cpu().numpy()  # (N, H)
    N, H, C = probs.shape

    auroc_list, auprc_list, f1_list = [], [], []

    for h in range(H):
        t_h = targets[:, h]
        p_h = probs[:, h, :]

        # AUROC (OVR, macro average across classes)
        try:
            if len(np.unique(t_h)) > 1:
                auroc = roc_auc_score(t_h, p_h, multi_class="ovr", average="macro")
            else:
                auroc = float("nan")
        except Exception:
            auroc = float("nan")

        # AUPRC averaged across classes OVR
        auprc_per_class = []
        for c in range(C):
            binary_t = (t_h == c).astype(int)
            if binary_t.sum() > 0:
                auprc_per_class.append(average_precision_score(binary_t, p_h[:, c]))
        auprc = float(np.mean(auprc_per_class)) if auprc_per_class else float("nan")

        preds = p_h.argmax(axis=-1)
        f1    = f1_score(t_h, preds, average="macro", zero_division=0)

        auroc_list.append(auroc)
        auprc_list.append(auprc)
        f1_list.append(f1)

    def _safe_mean(lst):
        valid = [v for v in lst if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    return {
        "auroc":     _safe_mean(auroc_list),
        "auprc":     _safe_mean(auprc_list),
        "macro_f1":  _safe_mean(f1_list),
        **{f"auroc_h{h}": auroc_list[h] for h in range(H)},
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training engine for TIDAL models.

    Parameters
    ----------
    model:
        Any model with ``forward(x) -> (B, H, C)`` interface.
    cfg:
        Full OmegaConf experiment config.
    class_weights:
        Optional float32 tensor of shape ``(C,)`` for loss weighting.

    Usage
    -----
    ::

        trainer = Trainer(model, cfg)
        history = trainer.fit(train_loader, val_loader)
        results = trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model:         nn.Module,
        cfg:           DictConfig,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.cfg          = cfg
        self.device       = get_device(cfg)
        self.model        = model.to(self.device)
        self.num_horizons = len(cfg.data.horizons)
        self.num_classes  = cfg.model.num_classes

        # Move class weights to device
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        # Loss
        self.criterion = _build_loss(cfg, class_weights).to(self.device)

        # Optimizer
        self.optimizer = _build_optimizer(cfg, model)

        # LR scheduler
        self.lr_scheduler = LRSchedulerCallback(
            optimizer       = self.optimizer,
            scheduler_name  = cfg.training.get("lr_scheduler", "cosine"),
            total_epochs    = cfg.training.epochs,
            warmup_epochs   = cfg.training.get("lr_warmup_epochs", 3),
            cfg_training    = cfg.training,
        )

        # AMP scaler (CUDA only)
        self.use_amp = (
            cfg.training.get("mixed_precision", False)
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler() if self.use_amp else None

        # Callbacks
        es_cfg = cfg.training.early_stopping
        self.early_stopping = EarlyStopping(
            patience  = es_cfg.patience,
            monitor   = es_cfg.monitor,
            mode      = es_cfg.mode,
            min_delta = es_cfg.min_delta,
        )

        ck_cfg = cfg.training.checkpointing
        ckpt_dir = Path(cfg.experiment.checkpoint_dir) / cfg.experiment.name
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir      = ckpt_dir,
            monitor             = es_cfg.monitor,
            mode                = es_cfg.mode,
            save_best           = ck_cfg.save_best,
            save_last           = ck_cfg.save_last,
            save_every_n_epochs = ck_cfg.save_every_n_epochs,
        )

        self.metric_tracker = MetricTracker()

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if cfg.logging.get("tensorboard", False):
            log_dir = Path(cfg.experiment.log_dir) / cfg.experiment.name
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

        self._gradient_clip = float(cfg.training.get("gradient_clip", 1.0))
        self._log_every_n   = cfg.logging.get("log_every_n_steps", 50)

        log.info(
            f"Trainer ready | device={self.device} | AMP={self.use_amp} | "
            f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        resume_from:  Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader, val_loader:
            PyTorch DataLoaders returning ``(x, y)`` batches.
        resume_from:
            Optional path to a checkpoint to resume from.

        Returns
        -------
        Dict containing ``history`` (MetricTracker dict) and ``best_epoch``.
        """
        start_epoch = 0
        if resume_from is not None:
            payload     = ModelCheckpoint.load_checkpoint(
                resume_from, self.model, self.device, self.optimizer
            )
            start_epoch = payload.get("epoch", 0) + 1
            log.info(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, self.cfg.training.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics   = self._eval_epoch(val_loader, prefix="val")

            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}":   v for k, v in val_metrics.items()}}

            lr = self.lr_scheduler.step(epoch, val_metric=val_metrics.get("auroc"))
            metrics["lr"] = lr

            self.metric_tracker.update(metrics)

            # Logging
            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch:03d}/{self.cfg.training.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_auroc={val_metrics.get('auroc', 0):.4f} | "
                f"val_auprc={val_metrics.get('auprc', 0):.4f} | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            if self.writer:
                for k, v in metrics.items():
                    self.writer.add_scalar(k, v, epoch)

            # Checkpoint
            self.checkpoint.step(
                model=self.model, metrics=metrics, epoch=epoch,
                optimizer=self.optimizer,
                extra={"early_stopping": self.early_stopping.state_dict()},
            )

            # Early stopping (uses prefixed key "val_auroc" etc.)
            es_monitor = self.cfg.training.early_stopping.monitor  # e.g. "val_auroc"
            if self.early_stopping.step(metrics, epoch):
                break

        if self.writer:
            self.writer.close()

        # Save full metric history
        self._save_history()

        return {
            "history":    self.metric_tracker.to_dict(),
            "best_epoch": self.early_stopping.best_epoch,
            "best_value": self.early_stopping.best_value,
        }

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for step, (x, y) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)  # (B, T, F)
            y = y.to(self.device, non_blocking=True)   # (B, H)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(x)      # (B, H, C)
                loss   = self.criterion(logits, y)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self._gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self._gradient_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if (step + 1) % self._log_every_n == 0:
                log.debug(f"  step {step+1}/{len(loader)} | loss={loss.item():.4f}")

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader, prefix: str = "val"
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss   = 0.0
        n_batches    = 0
        all_probs    = []
        all_targets  = []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)            # (B, H, C)
            loss   = self.criterion(logits, y)
            total_loss += loss.item()
            n_batches  += 1

            probs = F.softmax(logits, dim=-1)  # (B, H, C)
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

        all_probs   = torch.cat(all_probs,   dim=0)  # (N, H, C)
        all_targets = torch.cat(all_targets, dim=0)  # (N, H)

        epoch_m = _epoch_metrics(all_probs, all_targets, self.num_classes)
        epoch_m["loss"] = total_loss / max(n_batches, 1)
        return epoch_m

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        load_best:   bool = True,
    ) -> Dict[str, Any]:
        """Full test-set evaluation.

        Parameters
        ----------
        test_loader:
            DataLoader for the held-out test split.
        load_best:
            If True, load the best checkpoint before evaluating.

        Returns
        -------
        Dict with metrics, per-sample predictions, and probabilities.
        """
        if load_best and self.checkpoint.best_path is not None:
            self.checkpoint.load_best(self.model, self.device)

        self.model.eval()
        all_probs   = []
        all_targets = []
        all_scores  = []

        for x, y in test_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            probs  = F.softmax(logits, dim=-1)
            # Instability score: P(Trans) + P(Unstable)
            score  = probs[:, :, 1:].sum(dim=-1)  # (B, H)

            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())
            all_scores.append(score.cpu())

        all_probs   = torch.cat(all_probs,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_scores  = torch.cat(all_scores,  dim=0)

        metrics = _epoch_metrics(all_probs, all_targets, self.num_classes)
        log.info(
            f"Test results | auroc={metrics['auroc']:.4f} | "
            f"auprc={metrics['auprc']:.4f} | f1={metrics['macro_f1']:.4f}"
        )

        # Save results JSON
        out_dir = get_output_dir(self.cfg)
        results_path = out_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump({k: round(v, 6) for k, v in metrics.items()}, f, indent=2)
        log.info(f"Test results saved → {results_path}")

        return {
            "metrics":     metrics,
            "probs":       all_probs,
            "targets":     all_targets,
            "scores":      all_scores,
            "results_path": str(results_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        out_dir = get_output_dir(self.cfg)
        hist_path = out_dir / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.metric_tracker.to_dict(), f, indent=2)
        log.info(f"Training history saved → {hist_path}")
