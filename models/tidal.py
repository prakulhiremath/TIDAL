"""
models/tidal.py
────────────────
TIDAL: Temporal AI for Early Detection of Latent Financial Market Instability

Main model architecture combining:
    1. TemporalEncoder  — CNN + BiGRU + self-attention for sequence encoding
    2. RegimeModule     — Latent market regime inference
    3. InstabilityHead  — Multi-horizon binary instability prediction

The model is designed for proactive market surveillance — detecting
latent instability transitions before they manifest as visible disruption.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.layers.temporal_encoder import TemporalEncoder
from models.layers.regime_head import RegimeModule, InstabilityHead


class TIDALModel(nn.Module):
    """
    TIDAL: Temporal AI for Latent Instability Detection.

    Architecture:
        Input sequence (B, T, F)
            │
            ▼
        TemporalEncoder (CNN → BiGRU → Self-Attention)
            │
            ├──→ Sequence representation (B, T, H)
            └──→ Last hidden state (B, H)
                        │
                        ▼
                  RegimeModule
                  (Latent regime distribution + stress gate)
                        │
                        ▼
                  InstabilityHead
                  (Multi-horizon binary prediction)
                        │
                        ▼
                  Logits (B, n_horizons)

    The model learns to:
    1. Encode temporal microstructure patterns
    2. Infer hidden market regime states
    3. Predict future instability at multiple horizons

    Usage:
        model = TIDALModel(input_dim=40, n_horizons=3)
        output = model(sequences)
        logits = output['logits']              # (B, n_horizons)
        regime_probs = output['regime_probs']  # (B, n_regimes)
    """

    def __init__(
        self,
        input_dim: int,
        # Temporal encoder params
        cnn_channels: list = [64, 128],
        cnn_kernel_size: int = 3,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
        attention_heads: int = 4,
        # Regime module params
        n_regimes: int = 3,
        regime_dim: int = 64,
        # Instability head params
        head_hidden_dim: int = 64,
        n_horizons: int = 3,
        head_dropout: float = 0.3,
        # General
        dropout: float = 0.2,
    ):
        """
        Initialize TIDAL model.

        Args:
            input_dim: Feature dimension of each LOB snapshot.
            cnn_channels: Conv channel sizes for temporal encoder.
            cnn_kernel_size: Convolutional kernel size.
            rnn_hidden: GRU hidden state size.
            rnn_layers: Number of stacked GRU layers.
            rnn_dropout: GRU inter-layer dropout.
            bidirectional: Use bidirectional GRU.
            use_attention: Apply self-attention on GRU output.
            attention_heads: Number of attention heads.
            n_regimes: Number of latent market regimes.
            regime_dim: Regime embedding dimension.
            head_hidden_dim: InstabilityHead MLP hidden size.
            n_horizons: Number of prediction horizons.
            head_dropout: Dropout in prediction head.
            dropout: General dropout rate.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_horizons = n_horizons
        self.n_regimes = n_regimes

        # ── Temporal Encoder ────────────────────────────────────────────────
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            cnn_channels=cnn_channels,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            attention_heads=attention_heads,
            dropout=dropout,
        )
        enc_dim = self.temporal_encoder.output_dim

        # ── Regime Module ───────────────────────────────────────────────────
        self.regime_module = RegimeModule(
            input_dim=enc_dim,
            n_regimes=n_regimes,
            regime_dim=regime_dim,
            dropout=dropout,
        )

        # ── Instability Head ────────────────────────────────────────────────
        head_input_dim = enc_dim + regime_dim
        self.instability_head = InstabilityHead(
            input_dim=head_input_dim,
            hidden_dim=head_hidden_dim,
            n_horizons=n_horizons,
            dropout=head_dropout,
        )

        self._init_weights()

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input sequence tensor (batch, seq_len, input_dim).
            return_intermediates: If True, include encoder outputs in result.

        Returns:
            Dictionary containing:
                - 'logits': Raw predictions (batch, n_horizons)
                - 'probs': Sigmoid probabilities (batch, n_horizons)
                - 'regime_probs': Regime distribution (batch, n_regimes)
                - 'regime_repr': Regime embedding (batch, regime_dim) [if return_intermediates]
                - 'sequence_encoding': Full sequence output (batch, T, H) [if return_intermediates]
        """
        # ── Temporal encoding ───────────────────────────────────────────────
        sequence_enc, last_hidden = self.temporal_encoder(x, return_sequence=True)

        # Use the last time step's encoding for regime inference
        # (Captures accumulated temporal context)
        context = sequence_enc[:, -1, :]             # (B, H)

        # ── Regime inference ────────────────────────────────────────────────
        regime_repr, regime_probs = self.regime_module(context)  # (B, R), (B, n_reg)

        # ── Instability prediction ──────────────────────────────────────────
        combined = torch.cat([context, regime_repr], dim=-1)     # (B, H+R)
        logits = self.instability_head(combined)                 # (B, n_horizons)
        probs = torch.sigmoid(logits)

        output = {
            "logits": logits,
            "probs": probs,
            "regime_probs": regime_probs,
        }

        if return_intermediates:
            output["sequence_encoding"] = sequence_enc
            output["regime_repr"] = regime_repr
            output["last_hidden"] = last_hidden

        return output

    def predict_instability(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method: predict instability labels.

        Args:
            x: Input sequence (batch, seq_len, input_dim).
            threshold: Decision threshold for binary prediction.

        Returns:
            Dict with 'probs', 'labels', 'regime_probs'.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        output["labels"] = (output["probs"] > threshold).long()
        return output

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract attention weights for visualization.

        Args:
            x: Input sequence (batch, seq_len, input_dim).

        Returns:
            Attention weight tensor or None if attention disabled.
        """
        if not self.temporal_encoder.use_attention:
            return None

        self.eval()
        with torch.no_grad():
            # Access internal attention module
            x_cnn = x.permute(0, 2, 1)
            x_cnn = self.temporal_encoder.cnn(x_cnn)
            x_cnn = x_cnn.permute(0, 2, 1)

            gru_out, _ = self.temporal_encoder.gru(x_cnn)
            gru_out = self.temporal_encoder.rnn_norm(gru_out)

            _, attn_weights = self.temporal_encoder.attention(
                gru_out, gru_out, gru_out
            )
        return attn_weights  # (B, T, T)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        """Initialize weights with sensible defaults."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def __repr__(self) -> str:
        return (
            f"TIDALModel(\n"
            f"  input_dim={self.input_dim},\n"
            f"  n_horizons={self.n_horizons},\n"
            f"  n_regimes={self.n_regimes},\n"
            f"  encoder={self.temporal_encoder.output_dim}d,\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )


def build_tidal_from_config(cfg: dict, input_dim: int) -> "TIDALModel":
    """
    Instantiate TIDALModel from a config dictionary.

    Args:
        cfg: Model configuration dict (from YAML).
        input_dim: Feature dimension (set at runtime from data).

    Returns:
        Instantiated TIDALModel.
    """
    enc_cfg = cfg.get("temporal_encoder", {})
    reg_cfg = cfg.get("regime_module", {})
    head_cfg = cfg.get("instability_head", {})

    return TIDALModel(
        input_dim=input_dim,
        cnn_channels=enc_cfg.get("cnn_channels", [64, 128]),
        cnn_kernel_size=enc_cfg.get("cnn_kernel_size", 3),
        rnn_hidden=enc_cfg.get("rnn_hidden", 128),
        rnn_layers=enc_cfg.get("rnn_layers", 2),
        rnn_dropout=enc_cfg.get("rnn_dropout", 0.2),
        bidirectional=enc_cfg.get("bidirectional", True),
        use_attention=reg_cfg.get("use_attention", True),
        attention_heads=reg_cfg.get("attention_heads", 4),
        n_regimes=reg_cfg.get("n_regimes", 3),
        regime_dim=reg_cfg.get("regime_dim", 64),
        head_hidden_dim=head_cfg.get("hidden_dim", 64),
        n_horizons=head_cfg.get("n_horizons", 3),
        head_dropout=head_cfg.get("dropout", 0.3),
    )
