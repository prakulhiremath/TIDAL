"""
models/layers/regime_head.py
-----------------------------
Multi-horizon instability prediction head for TIDAL.

The regime head consumes the temporal encoder's output and independently
predicts the probability of each instability regime at multiple prediction
horizons (e.g. 10, 30, 60 steps ahead).

Architecture
------------
For each horizon H, a separate lightweight MLP maps the pooled encoder
representation to class logits. The three classes are:
    0: Stable
    1: Transitional  (the scientific novelty — early warning region)
    2: Unstable

Pooling strategy: The encoder produces per-step representations (B, T, D).
We apply adaptive temporal pooling that combines:
    - Last hidden state h_T     (captures current regime)
    - Mean pooling h_mean       (captures average trajectory)
    - Max pooling h_max         (captures worst-case microstructure)

This gives a richer aggregate than using h_T alone, which can miss sustained
stress that decays slightly before the window end.

Input  : (B, T, hidden_dim)  — from TemporalEncoder
Output : list of (B, num_classes) tensors, one per horizon
         or stacked: (B, num_horizons, num_classes)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonHead(nn.Module):
    """Single-horizon classification head.

    Parameters
    ----------
    hidden_dim:  Input dimension from encoder.
    num_classes: Number of regime classes (3).
    pooled_dim:  Dimension of the pooled representation (3 * hidden_dim by default).
    dropout:     Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 3,
        pooled_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        from typing import Optional
        super().__init__()
        pooled_dim = pooled_dim or (3 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: (B, pooled_dim) → logits: (B, num_classes)"""
        return self.mlp(pooled)


class RegimeHead(nn.Module):
    """Multi-horizon instability regime prediction head.

    Parameters
    ----------
    hidden_dim:   Encoder output dimension.
    num_horizons: Number of prediction horizons (e.g. 3 for [10, 30, 60]).
    num_classes:  Number of regime classes (default 3).
    dropout:      Dropout probability.
    share_mlp:    If True, all horizons share MLP weights (reduces parameters
                  but may degrade longer-horizon accuracy).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_horizons: int = 3,
        num_classes: int = 3,
        dropout: float = 0.1,
        share_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_horizons = num_horizons
        self.num_classes  = num_classes
        pooled_dim = 3 * hidden_dim  # last + mean + max

        if share_mlp:
            # Shared MLP + horizon embedding to distinguish horizons
            self.horizon_embed = nn.Embedding(num_horizons, hidden_dim)
            _head = HorizonHead(hidden_dim, num_classes, pooled_dim + hidden_dim, dropout)
            self.heads = nn.ModuleList([_head] * num_horizons)
            self._share_mlp = True
        else:
            self.heads = nn.ModuleList([
                HorizonHead(hidden_dim, num_classes, pooled_dim, dropout)
                for _ in range(num_horizons)
            ])
            self._share_mlp = False

        # Temporal attention pooling weights (learned)
        self.temporal_attn = nn.Linear(hidden_dim, 1)

    def _pool(self, encoded: torch.Tensor) -> torch.Tensor:
        """Adaptive temporal pooling: concatenate last + mean + attn-weighted.

        Parameters
        ----------
        encoded : (B, T, hidden_dim)

        Returns
        -------
        pooled : (B, 3 * hidden_dim)
        """
        # Last hidden state
        h_last = encoded[:, -1, :]                    # (B, D)

        # Attention-weighted mean (replaces simple mean for richer aggregation)
        attn_w = torch.softmax(self.temporal_attn(encoded), dim=1)  # (B, T, 1)
        h_mean = (encoded * attn_w).sum(dim=1)        # (B, D)

        # Max pooling across time (captures worst-case stress signal)
        h_max, _ = encoded.max(dim=1)                 # (B, D)

        return torch.cat([h_last, h_mean, h_max], dim=-1)  # (B, 3D)

    def forward(
        self,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        encoded : (B, T, hidden_dim)

        Returns
        -------
        logits : (B, num_horizons, num_classes)
        """
        pooled = self._pool(encoded)  # (B, 3D)

        horizon_logits = []
        for h_idx, head in enumerate(self.heads):
            if self._share_mlp:
                h_emb = self.horizon_embed(
                    torch.tensor(h_idx, device=encoded.device)
                ).unsqueeze(0).expand(pooled.size(0), -1)  # (B, D)
                inp = torch.cat([pooled, h_emb], dim=-1)
            else:
                inp = pooled
            horizon_logits.append(head(inp))              # (B, num_classes)

        return torch.stack(horizon_logits, dim=1)         # (B, H, num_classes)
