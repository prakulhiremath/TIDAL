"""
visualizations/regime_plots.py
────────────────────────────────
Publication-quality regime transition visualization for TIDAL.

Generates:
    1. Stable → Transitional → Unstable timeline plots
    2. Instability index trajectory with regime shading
    3. Regime transition heatmaps
    4. Component contribution plots (V_t, S_t, L_t, O_t)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Publication style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Regime color palette (colorblind-friendly)
REGIME_COLORS = {
    0: "#2196F3",   # Stable → blue
    1: "#FF9800",   # Transitional → orange
    2: "#F44336",   # Unstable → red
}
REGIME_NAMES = {0: "Stable", 1: "Transitional", 2: "Unstable"}
REGIME_ALPHA = {0: 0.15, 1: 0.25, 2: 0.30}


def plot_regime_timeline(
    regimes: np.ndarray,
    instability_index: Optional[np.ndarray] = None,
    pred_probs: Optional[np.ndarray] = None,
    mid_price: Optional[np.ndarray] = None,
    title: str = "TIDAL: Regime Transition Timeline",
    save_path: Optional[str] = None,
    max_steps: int = 2000,
) -> plt.Figure:
    """
    Plot the full Stable → Transitional → Unstable timeline.

    Creates a multi-panel figure showing:
        Panel 1: Mid price with regime background shading
        Panel 2: Instability index I_t with threshold lines
        Panel 3: Model prediction confidence

    Args:
        regimes: Regime labels {0,1,2} (T,).
        instability_index: I_t values ∈ [0,1] (T,) — optional.
        pred_probs: Model prediction probabilities (T,) — optional.
        mid_price: Price series for price panel (T,) — optional.
        title: Figure title.
        save_path: Path to save figure. None = display only.
        max_steps: Maximum time steps to plot.

    Returns:
        matplotlib Figure object.
    """
    T = min(len(regimes), max_steps)
    regimes = regimes[:T]
    t = np.arange(T)

    n_panels = 1
    if instability_index is not None:
        n_panels += 1
    if pred_probs is not None:
        n_panels += 1
    if mid_price is not None:
        n_panels += 1

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 3 * n_panels),
        sharex=True, gridspec_kw={"hspace": 0.08}
    )
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # ── Panel: Mid price ────────────────────────────────────────────────────
    if mid_price is not None:
        ax = axes[ax_idx]
        price = mid_price[:T]
        ax.plot(t, price, color="black", linewidth=0.8, alpha=0.9, label="Mid Price")
        _shade_regimes(ax, regimes, T)
        ax.set_ylabel("Mid Price")
        ax.set_title(title, fontweight="bold", pad=10)
        _add_regime_legend(ax)
        ax_idx += 1

    # ── Panel: Instability index ─────────────────────────────────────────────
    if instability_index is not None:
        ax = axes[ax_idx]
        idx = instability_index[:T]
        ax.plot(t, idx, color="#9C27B0", linewidth=1.2, label="$I_t$ (Instability Index)", zorder=3)
        ax.axhline(0.40, color=REGIME_COLORS[1], linestyle="--", linewidth=1.0,
                   alpha=0.8, label="Transitional threshold")
        ax.axhline(0.65, color=REGIME_COLORS[2], linestyle="--", linewidth=1.0,
                   alpha=0.8, label="Unstable threshold")
        ax.fill_between(t, 0, idx, alpha=0.15, color="#9C27B0")
        _shade_regimes(ax, regimes, T)
        ax.set_ylabel("$I_t$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax_idx += 1

    # ── Panel: Model prediction confidence ──────────────────────────────────
    if pred_probs is not None:
        ax = axes[ax_idx]
        probs = pred_probs[:T]
        ax.plot(t, probs, color="#4CAF50", linewidth=1.0, label="TIDAL Instability Score", zorder=3)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7, label="Decision threshold")
        ax.fill_between(t, 0, probs, where=probs > 0.5, alpha=0.25, color=REGIME_COLORS[2], label="Alert region")
        _shade_regimes(ax, regimes, T)
        ax.set_ylabel("Prediction\nConfidence")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax_idx += 1

    # ── Regime bar (always last) ─────────────────────────────────────────────
    ax = axes[-1]
    regime_cmap = LinearSegmentedColormap.from_list(
        "regime", [REGIME_COLORS[0], REGIME_COLORS[1], REGIME_COLORS[2]]
    )
    ax.imshow(
        regimes[:T].reshape(1, -1),
        aspect="auto", cmap=regime_cmap, vmin=0, vmax=2,
        extent=[0, T, 0, 1], interpolation="nearest"
    )
    ax.set_yticks([])
    ax.set_ylabel("Regime", rotation=0, ha="right", va="center")
    ax.set_xlabel("Time Step")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Regime timeline saved to {save_path}")

    return fig


def plot_instability_components(
    V_t: np.ndarray,
    S_t: np.ndarray,
    L_t: np.ndarray,
    O_t: np.ndarray,
    I_t: np.ndarray,
    regimes: np.ndarray,
    save_path: Optional[str] = None,
    max_steps: int = 1500,
) -> plt.Figure:
    """
    Plot individual instability index components (V, S, L, O) and composite I_t.

    Args:
        V_t: Volatility component.
        S_t: Spread component.
        L_t: Liquidity component.
        O_t: Order imbalance component.
        I_t: Composite instability index.
        regimes: Regime labels.
        save_path: Output path.
        max_steps: Max time steps to display.

    Returns:
        Figure object.
    """
    T = min(len(I_t), max_steps)
    t = np.arange(T)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, gridspec_kw={"hspace": 0.08})

    components = [
        (V_t[:T], "Volatility $V_t$", "#E53935"),
        (S_t[:T], "Spread Stress $S_t$", "#FB8C00"),
        (L_t[:T], "Liquidity $L_t$", "#43A047"),
        (O_t[:T], "Imbalance $O_t$", "#1E88E5"),
    ]

    for ax, (data, label, color) in zip(axes[:4], components):
        ax.plot(t, data, color=color, linewidth=0.9, alpha=0.9)
        ax.fill_between(t, 0, data, alpha=0.18, color=color)
        _shade_regimes(ax, regimes[:T], T)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(-0.02, 1.05)

    # Composite I_t
    ax = axes[4]
    ax.plot(t, I_t[:T], color="#7B1FA2", linewidth=1.4, label="$I_t$ Composite", zorder=5)
    ax.axhline(0.40, color=REGIME_COLORS[1], linestyle="--", linewidth=1.0, alpha=0.8, label="Trans. threshold")
    ax.axhline(0.65, color=REGIME_COLORS[2], linestyle="--", linewidth=1.0, alpha=0.8, label="Unstable threshold")
    _shade_regimes(ax, regimes[:T], T)
    ax.set_ylabel("$I_t$ (Composite)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time Step")
    ax.legend(loc="upper right", fontsize=9)

    axes[0].set_title(
        "Instability Index Components: $I_t = \\alpha V_t + \\beta S_t + \\gamma L_t + \\delta O_t$",
        fontweight="bold", pad=10
    )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Component plot saved to {save_path}")

    return fig


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot regime transition probability heatmap.

    Args:
        transition_matrix: 3×3 transition probability matrix.
        save_path: Output path.

    Returns:
        Figure object.
    """
    labels = ["Stable", "Transitional", "Unstable"]
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Transition Probability"},
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Next Regime", fontsize=12)
    ax.set_ylabel("Current Regime", fontsize=12)
    ax.set_title("Empirical Regime Transition Probabilities", fontweight="bold")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Transition heatmap saved to {save_path}")

    return fig


# ── Internal helpers ──────────────────────────────────────────────────────────

def _shade_regimes(ax: plt.Axes, regimes: np.ndarray, T: int) -> None:
    """Shade background by regime type."""
    i = 0
    while i < T:
        r = regimes[i]
        j = i
        while j < T and regimes[j] == r:
            j += 1
        ax.axvspan(i, j, alpha=REGIME_ALPHA[r], color=REGIME_COLORS[r], linewidth=0)
        i = j


def _add_regime_legend(ax: plt.Axes) -> None:
    """Add regime color legend patches."""
    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5, label=REGIME_NAMES[r])
        for r in [0, 1, 2]
    ]
    ax.legend(handles=patches, loc="upper right", framealpha=0.9, fontsize=9)
