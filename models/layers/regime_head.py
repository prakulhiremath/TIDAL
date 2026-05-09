"""
models/layers/regime_head.py
─────────────────────────────
Instability prediction head with latent regime representation.

Components:
    - RegimeModule: Infers latent market regime (Stable/Transitional/Unstable)
    - InstabilityHead: Multi-horizon binary prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class RegimeModule(nn.Module):
    """
    Latent regime inference module.

    Given the temporal encoding, this module infers a soft distribution
    over market regimes and produces a regime-conditioned representation.

    Regimes:
        0: Stable       — normal low-volatility market
        1: Transitional — stress accumulating, not yet unstable
        2: Unstable     — active instability episode

    This module is NOT directly supervised — it learns through backpropagation
    from the instability prediction task. The regime distribution emerges as
    a latent variable that the model finds useful.
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 3,
        regime_dim: int = 64,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Dimension of temporal encoder output.
            n_regimes: Number of latent market regimes.
            regime_dim: Regime embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_regimes = n_regimes
        self.regime_dim = regime_dim

        # Regime classifier (soft assignment)
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, regime_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regime_dim, n_regimes),
        )

        # Regime prototype embeddings (learned)
        self.regime_prototypes = nn.Embedding(n_regimes, regime_dim)

        # Stress accumulation gate
        self.stress_gate = nn.Sequential(
            nn.Linear(input_dim, regime_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(regime_dim)
        self.output_dim = regime_dim

    def forward(
        self, encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infer regime distribution and produce regime-conditioned representation.

        Args:
            encoded: Temporal encoding (batch, input_dim).

        Returns:
            Tuple of:
                - regime_repr: Regime-conditioned representation (batch, regime_dim)
                - regime_probs: Soft regime probabilities (batch, n_regimes)
        """
        # Soft regime assignment
        logits = self.regime_classifier(encoded)          # (B, n_regimes)
        regime_probs = F.softmax(logits, dim=-1)          # (B, n_regimes)

        # Weighted combination of regime prototypes
        # regime_prototypes: (n_regimes, regime_dim)
        proto = self.regime_prototypes.weight              # (n_regimes, regime_dim)
        regime_repr = torch.matmul(regime_probs, proto)   # (B, regime_dim)

        # Stress accumulation modulation
        gate = self.stress_gate(encoded)                  # (B, regime_dim)
        regime_repr = self.norm(regime_repr * gate)       # (B, regime_dim)

        return regime_repr, regime_probs


class InstabilityHead(nn.Module):
    """
    Multi-horizon instability prediction head.

    Predicts binary instability probability for each configured horizon.
    Takes the concatenation of temporal encoding and regime representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_horizons: int = 3,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Concatenated encoder + regime dimension.
            hidden_dim: MLP hidden dimension.
            n_horizons: Number of prediction horizons.
            dropout: Dropout rate.
        """
        super().__init__()
        self.n_horizons = n_horizons

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # Per-horizon prediction heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict instability probability per horizon.

        Args:
            x: Combined representation (batch, input_dim).

        Returns:
            Predictions tensor (batch, n_horizons) — raw logits.
        """
        shared = self.shared(x)
        predictions = torch.cat(
            [head(shared) for head in self.horizon_heads], dim=-1
        )  # (B, n_horizons)
        return predictions
