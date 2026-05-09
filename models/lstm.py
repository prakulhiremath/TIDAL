"""
models/lstm.py
--------------
Bidirectional LSTM baseline for TIDAL benchmark comparisons.

This baseline serves as the recurrent reference point: it captures sequential
dependencies but lacks explicit regime-transition modelling, structured state
spaces, and the gated SSM+attention fusion of TIDAL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class LSTMModel(nn.Module):
    """Bidirectional stacked LSTM for multi-horizon instability classification.

    Parameters
    ----------
    cfg : OmegaConf config with ``model.lstm`` and ``model`` sections.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mc = cfg.model
        lc = mc.lstm

        self.num_horizons = len(cfg.data.horizons)
        self.num_classes  = mc.num_classes
        self.horizons     = list(cfg.data.horizons)
        self.bidirectional = lc.get("bidirectional", True)
        directions = 2 if self.bidirectional else 1

        self.input_proj = nn.Linear(mc.input_dim, mc.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=mc.hidden_dim,
            hidden_size=mc.hidden_dim,
            num_layers=mc.num_layers,
            batch_first=True,
            dropout=mc.dropout if mc.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        lstm_out_dim = mc.hidden_dim * directions
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lstm_out_dim, mc.hidden_dim),
                nn.GELU(),
                nn.Dropout(mc.dropout),
                nn.Linear(mc.hidden_dim, mc.num_classes),
            )
            for _ in range(self.num_horizons)
        ])

        self.norm = nn.LayerNorm(lstm_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → logits: (B, H, C)"""
        h = self.input_proj(x)                 # (B, T, hidden)
        out, _ = self.lstm(h)                  # (B, T, hidden*dirs)
        last = self.norm(out[:, -1, :])        # (B, hidden*dirs)
        logits = torch.stack([head(last) for head in self.heads], dim=1)
        return logits                           # (B, H, C)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predict_instability_score(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return probs[:, :, 1:].sum(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
