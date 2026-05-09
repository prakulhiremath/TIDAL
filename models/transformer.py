"""
models/layers/temporal_encoder.py
-----------------------------------
Shared temporal encoder for TIDAL.

Architecture
------------
The TIDAL temporal encoder is a two-branch architecture designed to capture
complementary aspects of microstructure dynamics:

1. SSM branch (S4-lite / diagonal state space)
   - Captures long-range temporal dependencies efficiently
   - Suited to slowly evolving regime states and persistence signals
   - Pure-Python implementation; no CUDA-only dependencies

2. Causal Self-Attention branch
   - Captures complex short-range intra-window interactions
   - Multi-head, position-encoded, strictly causal (no future leakage)

3. Gated fusion
   - Learned sigmoid gate combines SSM and attention representations
   - Allows the model to route signal according to regime dynamics

Input  : (B, T, input_dim)
Output : (B, T, hidden_dim)

where B=batch, T=seq_len, input_dim=feature_dim, hidden_dim=encoder output dim.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# S4-lite: diagonal state space model (pure PyTorch, no CUDA deps)
# ---------------------------------------------------------------------------

class DiagonalSSM(nn.Module):
    """Simplified diagonal structured state space model (S4-lite).

    Recurrence:
        h_t = A·h_{t-1} + B·x_t
        y_t = C·h_t + D·x_t

    A is parametrized as -exp(log_A) (negative real diagonal) for stability.
    This is a simplified version; see Gu et al. (2022) for the full S4.

    Parameters
    ----------
    input_dim:  D_in  — input feature dimension
    state_dim:  N     — SSM state dimension
    output_dim: D_out — output dimension
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim

        # Diagonal A: kept stable via negative exponential
        self.log_A = nn.Parameter(torch.zeros(state_dim))
        # Input and output projection matrices
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, output_dim, bias=False)
        self.D = nn.Linear(input_dim, output_dim, bias=True)   # skip connection

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def _get_A(self) -> torch.Tensor:
        """Stable diagonal A: values are in (0, 1) after discretisation."""
        return torch.exp(-torch.exp(self.log_A))  # shape: (N,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        y : (B, T, output_dim)
        """
        B, T, _ = x.shape
        A = self._get_A()           # (N,)
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]               # (B, input_dim)
            h = A * h + self.B(x_t)        # (B, N)  diagonal multiply
            y_t = self.C(h) + self.D(x_t) # (B, output_dim)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)    # (B, T, output_dim)
        return self.layer_norm(self.dropout(y))


# ---------------------------------------------------------------------------
# Causal positional encoding
# ---------------------------------------------------------------------------

class CausalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (causal — no future position leakage)."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Causal self-attention block
# ---------------------------------------------------------------------------

class CausalSelfAttentionBlock(nn.Module):
    """Single causal multi-head self-attention block with residual + LN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) — applies causal mask internally."""
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Shared temporal encoder
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Two-branch temporal encoder: SSM + causal attention with gated fusion.

    Parameters
    ----------
    input_dim:   Input feature dimension (e.g. 40 for FI-2010 raw, or more with engineered features).
    hidden_dim:  Encoder output dimension.
    ssm_state_dim: State dimension for the SSM branch.
    ssm_layers:  Number of stacked SSM layers.
    attn_layers: Number of stacked causal self-attention layers.
    num_heads:   Attention heads per layer.
    dropout:     Dropout probability applied throughout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ssm_state_dim: int = 64,
        ssm_layers: int = 2,
        attn_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        half = hidden_dim // 2

        # Input projection (shared)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── SSM branch ────────────────────────────────────────────────────
        self.ssm_branch = nn.Sequential(
            *[DiagonalSSM(hidden_dim, ssm_state_dim, half, dropout=dropout)
              for _ in range(ssm_layers)]
        )

        # ── Causal attention branch ───────────────────────────────────────
        self.pos_enc = CausalPositionalEncoding(hidden_dim, dropout=dropout)
        self.attn_branch = nn.ModuleList([
            CausalSelfAttentionBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 2,
                dropout=dropout,
            )
            for _ in range(attn_layers)
        ])
        self.attn_proj = nn.Linear(hidden_dim, half)

        # ── Gated fusion ─────────────────────────────────────────────────
        # Gate: sigmoid(W·[ssm_out; attn_out]) ∈ (0,1) per hidden dim
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)

        Returns
        -------
        encoded : (B, T, hidden_dim)
        """
        # Shared input projection
        h = self.input_proj(x)  # (B, T, hidden_dim)

        # SSM branch
        ssm_out = self.ssm_branch(h)           # (B, T, half)

        # Causal attention branch
        attn_h = self.pos_enc(h)
        for layer in self.attn_branch:
            attn_h = layer(attn_h)
        attn_out = self.attn_proj(attn_h)      # (B, T, half)

        # Gated fusion
        combined = torch.cat([ssm_out, attn_out], dim=-1)  # (B, T, hidden_dim)
        gate = self.gate(combined)
        fused = gate * combined
        return self.out_norm(self.dropout(fused))           # (B, T, hidden_dim)
