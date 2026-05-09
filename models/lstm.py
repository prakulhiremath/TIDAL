"""
models/lstm.py
───────────────
LSTM baseline model for financial instability detection.

A standard bidirectional LSTM with multi-horizon output heads,
used as a sequential deep learning baseline against TIDAL.
"""

import torch
import torch.nn as nn
from typing import Dict


class LSTMBaseline(nn.Module):
    """
    Bidirectional LSTM baseline for instability prediction.

    Architecture:
        - Projection layer (input_dim → hidden_size)
        - Stacked BiLSTM
        - FC head per prediction horizon

    Usage:
        model = LSTMBaseline(input_dim=40, hidden_size=128, n_horizons=3)
        logits = model(x)['logits']  # (B, n_horizons)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        fc_hidden: int = 64,
        n_horizons: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_size)
        dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * dirs

        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            Dict with 'logits' and 'probs'.
        """
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.norm(lstm_out[:, -1, :])
        context = self.dropout(context)
        logits = self.classifier(context)
        return {"logits": logits, "probs": torch.sigmoid(logits)}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
