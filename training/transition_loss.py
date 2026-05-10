"""
training/transition_loss.py
────────────────────────────
Transition-aware loss function for TIDAL.

Scientific motivation:
    Standard binary losses treat all misclassifications equally.
    However, for instability surveillance, the MOST CRITICAL mistakes are:

    1. Missing a TRANSITIONAL regime (false negative at transition)
       — This means the system fails to detect the early warning signal.

    2. Misclassifying TRANSITIONAL as STABLE (the most common failure mode)
       — The system gives an "all clear" during a latent stress build-up.

This loss function adds a transition-sensitive penalty that amplifies
the gradient signal at regime boundaries, teaching the model to be
especially sensitive to the transitional state.

Mathematical formulation:
    L_transition = L_base + λ · Σ_t [ w_t · BCE(ŷ_t, y_t) ]

Where:
    w_t = transition_weight if t is near a regime transition else 1.0
    λ   = transition penalty coefficient

The transition neighborhood is defined as a window of ±k steps
around each regime boundary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from loguru import logger

from training.losses import FocalLoss, MultiHorizonLoss


class TransitionAwareLoss(nn.Module):
    """
    Transition-sensitive loss function for instability detection.

    Amplifies the loss at and around regime transition boundaries,
    forcing the model to pay special attention to the transitional state.

    The transition mask can be:
        1. Provided externally from regime labels (preferred)
        2. Estimated from prediction confidence (self-supervised)

    Args:
        base_loss: Base loss function to augment.
        transition_weight: Multiplier for loss at transition boundaries.
        transition_neighborhood: Steps around boundary to upweight.
        n_horizons: Number of prediction horizons.
        lambda_transition: Overall weight of transition penalty.
    """

    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        transition_weight: float = 3.0,
        transition_neighborhood: int = 5,
        n_horizons: int = 3,
        lambda_transition: float = 0.5,
    ):
        super().__init__()
        self.base_loss = base_loss or FocalLoss(gamma=2.0, alpha=0.25)
        self.transition_weight = transition_weight
        self.transition_neighborhood = transition_neighborhood
        self.n_horizons = n_horizons
        self.lambda_transition = lambda_transition

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        transition_mask: Optional[torch.Tensor] = None,
        regime_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute transition-aware loss.

        Args:
            logits: Raw predictions (B, n_horizons).
            targets: Binary targets (B, n_horizons).
            transition_mask: Boolean mask (B,) — True at transition steps.
                             If None, uses self-supervised estimation.
            regime_probs: Regime probabilities from RegimeModule (B, n_regimes).
                          Used for additional regime consistency term.

        Returns:
            Scalar combined loss.
        """
        # ── Base multi-horizon loss ─────────────────────────────────────────
        base_loss_value = torch.tensor(0.0, device=logits.device, requires_grad=True)
        for h in range(self.n_horizons):
            base_loss_value = base_loss_value + self.base_loss(logits[:, h], targets[:, h])
        base_loss_value = base_loss_value / self.n_horizons

        # ── Transition-sensitive penalty ────────────────────────────────────
        if transition_mask is not None:
            t_loss = self._transition_penalty(logits, targets, transition_mask)
        else:
            # Self-supervised: detect transitions from label changes
            t_loss = self._self_supervised_transition_penalty(logits, targets)

        # ── Regime consistency term (if regime probs available) ─────────────
        regime_loss = torch.tensor(0.0, device=logits.device)
        if regime_probs is not None:
            regime_loss = self._regime_entropy_regularizer(regime_probs)

        total = (
            base_loss_value
            + self.lambda_transition * t_loss
            + 0.01 * regime_loss
        )
        return total

    def _transition_penalty(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        transition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply amplified BCE loss at transition steps.

        Args:
            logits: (B, n_horizons)
            targets: (B, n_horizons)
            transition_mask: (B,) boolean

        Returns:
            Scalar transition loss.
        """
        if transition_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # Expand mask to match logits shape
        mask = transition_mask.float().unsqueeze(1).expand_as(logits)

        # Per-sample BCE (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Weight by transition mask
        weighted = bce * (1.0 + (self.transition_weight - 1.0) * mask)
        return weighted.mean()

    def _self_supervised_transition_penalty(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate transition penalty without explicit regime labels.

        Identifies transitions by looking at abrupt changes in the target
        label distribution within a batch (ordered by time).

        Args:
            logits: (B, n_horizons)
            targets: (B, n_horizons)

        Returns:
            Scalar transition penalty.
        """
        # Detect label-change boundaries in batch (assume temporal ordering)
        if targets.shape[0] < 2:
            return torch.tensor(0.0, device=logits.device)

        label_changes = (targets[1:] - targets[:-1]).abs().sum(dim=1) > 0
        # Pad to full batch size
        transition_mask = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=targets.device),
            label_changes
        ])

        # Expand neighborhood
        if self.transition_neighborhood > 0:
            transition_mask = self._expand_mask(transition_mask, self.transition_neighborhood)

        return self._transition_penalty(logits, targets, transition_mask)

    def _expand_mask(
        self, mask: torch.Tensor, neighborhood: int
    ) -> torch.Tensor:
        """
        Expand boolean mask by ±neighborhood steps using 1D max pooling.

        Args:
            mask: (B,) boolean tensor.
            neighborhood: Number of steps to expand in each direction.

        Returns:
            Expanded boolean tensor (B,).
        """
        mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, B)
        kernel = 2 * neighborhood + 1
        pooled = F.max_pool1d(
            mask_f, kernel_size=kernel, stride=1,
            padding=neighborhood
        )
        return pooled.squeeze(0).squeeze(0) > 0.5

    def _regime_entropy_regularizer(
        self, regime_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy regularizer on regime distribution.

        Encourages the model to commit to a regime (low entropy)
        rather than being uniformly uncertain.

        Args:
            regime_probs: (B, n_regimes) soft regime distribution.

        Returns:
            Mean entropy across batch (to minimize).
        """
        # Entropy: H(p) = -Σ p log p
        log_probs = torch.log(regime_probs + 1e-8)
        entropy = -(regime_probs * log_probs).sum(dim=-1)
        return entropy.mean()


def build_transition_aware_loss(
    cfg: dict,
    pos_class_ratio: Optional[float] = None,
) -> TransitionAwareLoss:
    """
    Build TransitionAwareLoss from config.

    Args:
        cfg: Loss config dict.
        pos_class_ratio: Fraction of positive samples.

    Returns:
        Configured TransitionAwareLoss.
    """
    from training.losses import FocalLoss

    base = FocalLoss(
        gamma=cfg.get("focal_gamma", 2.0),
        alpha=cfg.get("focal_alpha", 0.25),
    )

    return TransitionAwareLoss(
        base_loss=base,
        transition_weight=cfg.get("transition_weight", 3.0),
        transition_neighborhood=cfg.get("transition_neighborhood", 5),
        n_horizons=cfg.get("n_horizons", 3),
        lambda_transition=cfg.get("lambda_transition", 0.5),
    )
