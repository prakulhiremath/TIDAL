"""
models/tidal.py
----------------
TIDAL: Temporal AI for Early Detection of Latent Financial Market Instability.

Model Architecture
------------------
TIDAL is a hybrid temporal deep learning model designed for proactive financial
instability surveillance. It combines structured state-space modelling (for
capturing persistent regime dynamics) with causal multi-head self-attention
(for detecting rapid microstructure transitions) through a learned gated fusion.

Pipeline:
    Input (B, T, F)
      → TemporalEncoder  [SSM branch || Attention branch → gated fusion]
      → RegimeHead       [per-horizon pooling + MLP classifiers]
      → Logits (B, H, C) where H=horizons, C=classes

The model predicts the probability of each instability regime {Stable,
Transitional, Unstable} at each of the configured prediction horizons.

Scientific framing
------------------
TIDAL does not predict prices or generate trading signals. The output is
a multi-horizon instability probability surface, designed to support
proactive financial surveillance systems. The Transitional class is the
core novelty: it captures the latent accumulation of microstructure stress
before visible volatility onset.

Usage
-----
    from models.tidal import TIDALModel
    from utils.config import load_config

    cfg = load_config("configs/default.yaml")
    model = TIDALModel(cfg)
    logits = model(x)   # x: (B, T, F), logits: (B, H, C)
    probs = model.predict_proba(x)  # (B, H, C)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.layers.temporal_encoder import TemporalEncoder
from models.layers.regime_head import RegimeHead


class TIDALModel(nn.Module):
    """Main TIDAL instability surveillance model.

    Parameters
    ----------
    cfg:
        Full OmegaConf experiment config. Reads ``model`` and ``data`` sections.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mc = cfg.model
        tc = mc.tidal

        self.num_horizons = len(cfg.data.horizons)
        self.num_classes  = mc.num_classes
        self.horizons     = list(cfg.data.horizons)

        # ── Temporal encoder ──────────────────────────────────────────────
        self.encoder = TemporalEncoder(
            input_dim    = mc.input_dim,
            hidden_dim   = mc.hidden_dim,
            ssm_state_dim= tc.ssm_dim,
            ssm_layers   = tc.ssm_layers,
            attn_layers  = tc.attn_layers,
            num_heads    = mc.num_heads,
            dropout      = mc.dropout,
        )

        # ── Multi-horizon regime head ─────────────────────────────────────
        self.head = RegimeHead(
            hidden_dim   = mc.hidden_dim,
            num_horizons = self.num_horizons,
            num_classes  = mc.num_classes,
            dropout      = mc.dropout,
            share_mlp    = False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise linear layers with Xavier uniform, biases to zero."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, T, F) — normalised feature sequences

        Returns
        -------
        logits : (B, num_horizons, num_classes)
        """
        encoded = self.encoder(x)     # (B, T, hidden_dim)
        logits  = self.head(encoded)  # (B, H, C)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities.

        Returns
        -------
        probs : (B, num_horizons, num_classes) in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_instability_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar instability score in [0, 1] per horizon.

        Computed as P(Transitional) + P(Unstable) — the probability that
        the market is *not* in a stable regime. This is the primary output
        for surveillance dashboards.

        Returns
        -------
        score : (B, num_horizons)
        """
        probs = self.predict_proba(x)
        return probs[:, :, 1:].sum(dim=-1)  # P(trans) + P(unstable)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"TIDALModel("
            f"horizons={self.horizons}, "
            f"classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
