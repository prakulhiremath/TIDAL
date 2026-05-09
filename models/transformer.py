"""
models/transformer.py
----------------------
Causal Transformer baseline for TIDAL benchmark comparisons.

Uses causal (autoregressive) masking so no future microstructure information
leaks into the prediction at each step. Serves as the pure-attention reference
point against TIDAL's hybrid SSM+attention design.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class TransformerModel(nn.Module):
    """Causal Transformer encoder for multi-horizon instability classification.

    Parameters
    ----------
    cfg : OmegaConf config with ``model.transformer`` and ``model`` sections.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mc = cfg.model
        tc = mc.transformer

        self.num_horizons = len(cfg.data.horizons)
        self.num_classes  = mc.num_classes
        self.horizons     = list(cfg.data.horizons)

        self.input_proj = nn.Linear(mc.input_dim, mc.hidden_dim)

        # Sinusoidal position encoding (shared with TemporalEncoder)
        max_len = tc.get("max_seq_len", 512)
        pe = torch.zeros(max_len, mc.hidden_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, mc.hidden_dim, 2).float()
            * (-math.log(10000.0) / mc.hidden_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mc.hidden_dim,
            nhead=mc.num_heads,
            dim_feedforward=tc.get("ff_dim", mc.hidden_dim * 2),
            dropout=mc.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=mc.num_layers)

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
        self.dropout = nn.Dropout(mc.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → logits: (B, H, C)"""
        B, T, _ = x.shape
        h = self.dropout(self.input_proj(x) + self.pe[:, :T, :])
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
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
