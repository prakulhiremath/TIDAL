"""
models/ssm.py
--------------
State Space Model (S4-lite) baseline for TIDAL benchmark comparisons.

This is a pure-PyTorch implementation of a diagonal structured state space
model, usable without CUDA or mamba-ssm. It captures long-range temporal
dependencies more efficiently than RNNs and serves as the SSM reference
baseline against which TIDAL's hybrid architecture is compared.

The full Mamba/S4 implementation (mamba-ssm) can be enabled per-config
(requires CUDA); this file falls back to the diagonal SSM from temporal_encoder.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.layers.temporal_encoder import DiagonalSSM


class SSMModel(nn.Module):
    """Stacked diagonal SSM baseline for multi-horizon instability classification.

    Parameters
    ----------
    cfg : OmegaConf config.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mc = cfg.model
        sc = mc.ssm

        self.num_horizons = len(cfg.data.horizons)
        self.num_classes  = mc.num_classes
        self.horizons     = list(cfg.data.horizons)

        self.input_proj = nn.Linear(mc.input_dim, mc.hidden_dim)

        # Stack of SSM layers — fully causal (recurrent forward pass)
        self.ssm_layers = nn.ModuleList([
            DiagonalSSM(
                input_dim=mc.hidden_dim,
                state_dim=sc.get("state_dim", 64),
                output_dim=mc.hidden_dim,
                dropout=mc.dropout,
            )
            for _ in range(mc.num_layers)
        ])

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mc.hidden_dim, mc.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(mc.dropout),
                nn.Linear(mc.hidden_dim // 2, mc.num_classes),
            )
            for _ in range(self.num_horizons)
        ])

        self.norm = nn.LayerNorm(mc.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → logits: (B, H, C)"""
        h = self.input_proj(x)  # (B, T, hidden)
        for layer in self.ssm_layers:
            h = layer(h)
        last = self.norm(h[:, -1, :])
        logits = torch.stack([head(last) for head in self.heads], dim=1)
        return logits  # (B, H, C)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predict_instability_score(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return probs[:, :, 1:].sum(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
