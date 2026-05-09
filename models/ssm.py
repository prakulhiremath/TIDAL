"""
models/ssm.py
──────────────
State Space Model (SSM) baseline for financial instability detection.

Implements a simplified structured state space sequence model inspired by S4/Mamba.
This is a self-contained PyTorch implementation that does NOT require
the mamba-ssm CUDA package — enabling CPU/GPU compatibility.

Reference:
    Gu et al. (2021) "Efficiently Modeling Long Sequences with Structured State Spaces"
    Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class S4Block(nn.Module):
    """
    Simplified S4-inspired block using diagonal state space approximation.

    Approximates the structured state space convolution using a learnable
    diagonal A matrix, enabling efficient training without CUDA kernels.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal A (log parameterization for stability)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B_batch, T, D = x.shape
        residual = x

        # Diagonal SSM via recurrence (simplified)
        A = -torch.exp(self.A_log)                   # (d_model, d_state)
        h = torch.zeros(B_batch, D, self.d_state, device=x.device)
        outputs = []

        for t in range(T):
            u = x[:, t, :]                            # (B, d_model)
            # State update: h_t = A * h_{t-1} + B * u_t
            A_disc = torch.exp(A * 0.01)             # discrete approx
            Bu = torch.einsum("bd,ds->bds", u, self.B)  # (B, D, d_state)
            h = A_disc.unsqueeze(0) * h + Bu
            # Output: y_t = C * h_t + D * u_t
            y = torch.einsum("bds,ds->bd", h, self.C) + self.D * u
            outputs.append(y.unsqueeze(1))

        ssm_out = torch.cat(outputs, dim=1)           # (B, T, D)

        # Residual + FF
        out = self.norm(residual + ssm_out)
        out = out + self.ff(out)
        return self.dropout(out)


class SSMBaseline(nn.Module):
    """
    State Space Model baseline for instability detection.

    Stacks multiple S4-inspired blocks to model long-range LOB dynamics
    with sub-quadratic complexity in sequence length.

    Usage:
        model = SSMBaseline(input_dim=40, d_model=128, num_layers=3, n_horizons=3)
        output = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        fc_hidden: int = 64,
        n_horizons: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList([
            S4Block(d_model, d_state, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
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
        for block in self.blocks:
            x = block(x)
        context = self.norm(x[:, -1, :])
        logits = self.classifier(context)
        return {"logits": logits, "probs": torch.sigmoid(logits)}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
