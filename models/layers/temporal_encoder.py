"""
models/layers/temporal_encoder.py
───────────────────────────────────
Temporal encoder layer shared across TIDAL and baseline models.

Architecture:
    - 1D convolutional feature extraction (local temporal patterns)
    - Bidirectional GRU for sequential context
    - Optional multi-head self-attention
    - Layer normalization throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalConvBlock(nn.Module):
    """
    1D convolutional block for local temporal feature extraction.

    Applies multiple Conv1D layers with residual connections to
    capture short-range temporal dependencies in LOB sequences.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if channels change
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class TemporalEncoder(nn.Module):
    """
    Full temporal encoder: CNN → GRU → optional attention.

    Processes a sequence of LOB feature snapshots and produces
    a rich temporal embedding capturing both local and global patterns.

    Args:
        input_dim: Feature dimension per time step.
        cnn_channels: List of channel sizes for CNN layers.
        rnn_hidden: Hidden size for GRU.
        rnn_layers: Number of GRU layers.
        rnn_dropout: GRU inter-layer dropout.
        bidirectional: Use bidirectional GRU.
        use_attention: Apply self-attention on top of GRU output.
        attention_heads: Number of attention heads.
        dropout: General dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: list = [64, 128],
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        rnn_out_dim = rnn_hidden * (2 if bidirectional else 1)

        # ── CNN stack ──────────────────────────────────────────────────────
        cnn_layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(cnn_channels):
            cnn_layers.append(
                TemporalConvBlock(in_ch, out_ch, kernel_size=3, dilation=1, dropout=dropout)
            )
            # Add dilated conv for broader receptive field
            cnn_layers.append(
                TemporalConvBlock(out_ch, out_ch, kernel_size=3, dilation=2, dropout=dropout)
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_dim = in_ch

        # ── GRU ────────────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.rnn_norm = nn.LayerNorm(rnn_out_dim)

        # ── Self-attention ──────────────────────────────────────────────────
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_out_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(rnn_out_dim)

        self.output_dim = rnn_out_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_sequence: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a sequence of LOB snapshots.

        Args:
            x: Input tensor (batch, seq_len, input_dim).
            return_sequence: If True, return full sequence; else last step.

        Returns:
            Tuple of:
                - encoded: (batch, seq_len, output_dim) or (batch, output_dim)
                - last_hidden: (batch, output_dim) — final hidden state
        """
        batch, seq_len, _ = x.shape

        # ── CNN: operate on (batch, channels, seq_len) ──────────────────────
        x_cnn = x.permute(0, 2, 1)           # (B, F, T)
        x_cnn = self.cnn(x_cnn)              # (B, C, T)
        x_cnn = x_cnn.permute(0, 2, 1)       # (B, T, C)

        # ── GRU ────────────────────────────────────────────────────────────
        gru_out, hidden = self.gru(x_cnn)    # (B, T, H*dirs)
        gru_out = self.rnn_norm(gru_out)
        gru_out = self.dropout(gru_out)

        # Concatenate final forward/backward hidden states
        if self.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]

        # ── Self-attention ──────────────────────────────────────────────────
        if self.use_attention:
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            gru_out = self.attn_norm(gru_out + attn_out)

        if return_sequence:
            return gru_out, last_hidden
        else:
            return gru_out[:, -1, :], last_hidden
