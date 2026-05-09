"""
training/losses.py
───────────────────
Custom loss functions for financial instability detection.

Instability labels are heavily imbalanced (typically 10-20% positive),
requiring loss functions that down-weight easy negatives and focus
training on hard examples and rare instability events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.

    Reduces the relative loss for well-classified examples,
    focusing training on hard, misclassified examples.

    Reference:
        Lin et al. (2017) "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    Args:
        gamma: Focusing parameter (0 = cross-entropy, 2 = default focal).
        alpha: Weighting factor for positive class (None = no weighting).
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw predictions (B,) or (B, H) — not sigmoid-ed.
            targets: Binary targets same shape as logits.

        Returns:
            Scalar loss.
        """
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy loss with per-sample or per-class weighting.

    Args:
        pos_weight: Weight for positive class. Higher values penalize false negatives more.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, pos_weight: float = 5.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor(self.pos_weight, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction=self.reduction
        )


class MultiHorizonLoss(nn.Module):
    """
    Combined loss across multiple prediction horizons.

    Applies a base loss function to each horizon's predictions
    with optional horizon-specific weighting.

    Near horizons are typically harder and more actionable — they
    receive higher weight by default.

    Args:
        base_loss: Loss module to apply per horizon.
        n_horizons: Number of prediction horizons.
        horizon_weights: Per-horizon loss weights. None = uniform.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        n_horizons: int = 3,
        horizon_weights: Optional[list] = None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.n_horizons = n_horizons

        if horizon_weights is None:
            # Linearly decrease weight with horizon distance
            # Shorter horizons (more actionable) weighted higher
            weights = [1.0 / (i + 1) for i in range(n_horizons)]
            total = sum(weights)
            horizon_weights = [w / total for w in weights]

        self.register_buffer(
            "horizon_weights",
            torch.tensor(horizon_weights, dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted sum of per-horizon losses.

        Args:
            logits: (batch, n_horizons)
            targets: (batch, n_horizons)

        Returns:
            Scalar combined loss.
        """
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        for h in range(self.n_horizons):
            h_loss = self.base_loss(logits[:, h], targets[:, h])
            total_loss = total_loss + self.horizon_weights[h] * h_loss
        return total_loss


def build_loss(cfg: dict, pos_class_ratio: Optional[float] = None) -> nn.Module:
    """
    Build loss function from configuration.

    Args:
        cfg: Loss config dict with keys: name, focal_gamma, focal_alpha, etc.
        pos_class_ratio: Fraction of positive samples (used for auto weighting).

    Returns:
        Configured loss module.
    """
    name = cfg.get("name", "focal")
    n_horizons = cfg.get("n_horizons", 3)

    # Compute class weights if requested
    pos_weight = 1.0
    if cfg.get("class_weight_strategy") == "auto" and pos_class_ratio is not None:
        neg_ratio = 1.0 - pos_class_ratio
        pos_weight = neg_ratio / max(pos_class_ratio, 1e-6)
        pos_weight = min(pos_weight, 20.0)  # Cap to avoid instability

    if name == "focal":
        base = FocalLoss(
            gamma=cfg.get("focal_gamma", 2.0),
            alpha=cfg.get("focal_alpha", 0.25),
        )
    elif name == "weighted_bce":
        base = WeightedBCELoss(pos_weight=pos_weight)
    elif name == "bce":
        base = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")

    return MultiHorizonLoss(base_loss=base, n_horizons=n_horizons)
