"""
visualizations/attention_maps.py
──────────────────────────────────
Attention weight visualization for TIDAL's temporal encoder.

Generates heatmaps showing which time steps the model attends to
when predicting instability — a key explainability component.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_attention_heatmaps(
    attention_weights: np.ndarray,
    n_samples: int = 4,
    feature_names: Optional[list] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot attention weight heatmaps for multiple sequence examples.

    Args:
        attention_weights: Attention array (B, T, T) or (B, heads, T, T).
        n_samples: Number of example sequences to plot.
        feature_names: Optional time-step labels.
        save_path: Path to save figure.

    Returns:
        Figure object.
    """
    # Handle multi-head: average over heads
    if attention_weights.ndim == 4:
        attention_weights = attention_weights.mean(axis=1)  # (B, T, T)

    n_samples = min(n_samples, len(attention_weights))
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        attn = attention_weights[i]
        sns.heatmap(
            attn,
            ax=ax,
            cmap="Blues",
            xticklabels=False,
            yticklabels=False,
            cbar=(i == n_samples - 1),
            vmin=0, vmax=attn.max(),
        )
        ax.set_title(f"Example {i+1}", fontsize=10)
        ax.set_xlabel("Key Time Step", fontsize=9)
        if i == 0:
            ax.set_ylabel("Query Time Step", fontsize=9)

    fig.suptitle("TIDAL Temporal Self-Attention Weights", fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Attention maps saved to {save_path}")

    return fig


def plot_feature_attention_alignment(
    attention_weights: np.ndarray,
    instability_index: np.ndarray,
    regimes: np.ndarray,
    save_path: Optional[str] = None,
    max_steps: int = 500,
) -> plt.Figure:
    """
    Show how attention patterns align with instability periods.

    Plots mean attention weight (diagonal) alongside the instability index.

    Args:
        attention_weights: (B, T, T) attention weights.
        instability_index: I_t array (T,).
        regimes: Regime labels (T,).
        save_path: Output path.
        max_steps: Max time steps to show.

    Returns:
        Figure object.
    """
    T = min(max_steps, len(instability_index))
    t = np.arange(T)

    # Use mean of diagonal attention as "self-attention score"
    if attention_weights.ndim == 4:
        attn_diag = np.diagonal(attention_weights.mean(axis=1), axis1=1, axis2=2).mean(0)
    else:
        attn_diag = np.diagonal(attention_weights, axis1=1, axis2=2).mean(0)

    attn_diag = attn_diag[:T]
    # Normalize to [0,1]
    attn_norm = (attn_diag - attn_diag.min()) / (attn_diag.max() - attn_diag.min() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"hspace": 0.05})

    ax1.plot(t, instability_index[:T], color="#7B1FA2", linewidth=1.2, label="$I_t$")
    ax1.axhline(0.40, color="#FF9800", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.axhline(0.65, color="#F44336", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Instability Index")
    ax1.legend(loc="upper right")

    ax2.plot(t, attn_norm, color="#1565C0", linewidth=1.0, alpha=0.85, label="Mean Self-Attention")
    ax2.fill_between(t, 0, attn_norm, alpha=0.2, color="#1565C0")
    ax2.set_ylabel("Attention Intensity")
    ax2.set_xlabel("Time Step")
    ax2.legend(loc="upper right")

    fig.suptitle("TIDAL Attention Intensity vs. Instability Index", fontweight="bold")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
