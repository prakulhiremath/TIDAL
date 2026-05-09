from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Publication-style rcParams
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

REGIME_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}  # Blue, Orange, Red
REGIME_NAMES  = {0: "Stable", 1: "Transitional", 2: "Unstable"}
MODEL_COLORS  = {
    "TIDAL":       "#1A237E",
    "LSTM":        "#00796B",
    "Transformer": "#E65100",
    "SSM":         "#6A1B9A",
}


# ---------------------------------------------------------------------------
# Regime timeline
# ---------------------------------------------------------------------------

def plot_regime_timeline(
    instability_index: np.ndarray,     # (N,)
    true_labels:       np.ndarray,     # (N,)
    pred_labels:       Optional[np.ndarray] = None,  # (N,)
    instability_scores: Optional[np.ndarray] = None, # (N,) P(Trans)+P(Unstable)
    time_axis:         Optional[np.ndarray] = None,
    title: str = "Instability Surveillance Timeline",
    save_path: Optional[str | Path] = None,
    figsize: Tuple = (14, 7),
) -> plt.Figure:
    """Plot instability index and regime labels over time.

    Parameters
    ----------
    instability_index:
        Composite instability index I_t in [0, 1].
    true_labels:
        Ground-truth regime labels {0, 1, 2}.
    pred_labels:
        Model-predicted regime labels (optional, overlaid).
    instability_scores:
        Model instability score = P(Trans)+P(Unstable) (optional, overlaid).
    time_axis:
        X-axis values. Defaults to step indices.
    """
    N = len(instability_index)
    t = time_axis if time_axis is not None else np.arange(N)

    n_rows = 2 + (pred_labels is not None) + (instability_scores is not None)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3] + [1] * (n_rows - 1)})
    if n_rows == 1:
        axes = [axes]

    ax_idx = 0

    # --- Instability index ---
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(t, instability_index, color="#333333", lw=0.8, label="Instability Index $I_t$")

    # Shade regimes
    for c, color in REGIME_COLORS.items():
        mask = true_labels == c
        if mask.any():
            ax.fill_between(t, 0, 1, where=mask, alpha=0.15, color=color, label=REGIME_NAMES[c])

    if instability_scores is not None:
        ax.plot(t, instability_scores, color="#E91E63", lw=0.7, alpha=0.7,
                label="Model Score", linestyle="--")

    ax.set_ylabel("Index / Score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.set_title(title)

    # --- True regime labels ---
    ax = axes[ax_idx]; ax_idx += 1
    _regime_strip(ax, t, true_labels, "Ground Truth")

    # --- Predicted regime labels ---
    if pred_labels is not None:
        ax = axes[ax_idx]; ax_idx += 1
        _regime_strip(ax, t, pred_labels, "Predicted")

    # --- Instability score standalone ---
    if instability_scores is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(t, instability_scores, color="#E91E63", lw=0.8)
        ax.axhline(0.5, color="gray", lw=0.6, linestyle=":")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig


def _regime_strip(ax: plt.Axes, t: np.ndarray, labels: np.ndarray, ylabel: str):
    """Draw a colour-coded regime strip (used internally)."""
    for c, color in REGIME_COLORS.items():
        mask = labels == c
        if mask.any():
            ax.fill_between(t, 0, 1, where=mask, color=color, alpha=0.9)
    patches = [mpatches.Patch(color=REGIME_COLORS[c], label=REGIME_NAMES[c]) for c in range(3)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=3)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Regime Classification — Confusion Matrix",
    save_path: Optional[str | Path] = None,
    figsize: Tuple = (6, 5),
    cmap: str = "Blues",
) -> plt.Figure:
    """Heatmap confusion matrix for three-regime classification."""
    if class_names is None:
        class_names = ["Stable", "Transitional", "Unstable"]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Predicted Regime")
    ax.set_ylabel("True Regime")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Transition heatmap
# ---------------------------------------------------------------------------

def plot_regime_transition_heatmap(
    transition_matrix: np.ndarray,   # (3, 3) row-normalised
    title: str = "Empirical Regime Transition Probabilities",
    save_path: Optional[str | Path] = None,
    figsize: Tuple = (5, 4),
) -> plt.Figure:
    """Heatmap of the empirical regime-to-regime transition matrix."""
    class_names = ["Stable", "Transitional", "Unstable"]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        transition_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=[f"→{n}" for n in class_names],
        yticklabels=class_names,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_xlabel("Next Regime")
    ax.set_ylabel("Current Regime")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    labels: np.ndarray,
    title: str = "Regime Class Distribution",
    save_path: Optional[str | Path] = None,
    figsize: Tuple = (5, 4),
) -> plt.Figure:
    """Bar chart of label frequencies for each regime class."""
    valid = labels[labels >= 0].flatten()
    classes = [0, 1, 2]
    counts  = [(valid == c).sum() for c in classes]
    total   = len(valid)
    fracs   = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        [REGIME_NAMES[c] for c in classes],
        fracs,
        color=[REGIME_COLORS[c] for c in classes],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, frac, count in zip(bars, fracs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{frac:.1f}%\n(n={count:,})",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Proportion (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(fracs) * 1.25)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig


# ---------------------------------------------------------------------------
# Instability score distribution
# ---------------------------------------------------------------------------

def plot_instability_score_distribution(
    scores:  np.ndarray,   # (N,) instability score
    labels:  np.ndarray,   # (N,) true regime labels
    threshold: float = 0.5,
    title: str = "Instability Score Distribution by Regime",
    save_path: Optional[str | Path] = None,
    figsize: Tuple = (7, 4),
) -> plt.Figure:
    """KDE plot of instability score distributions per true regime class."""
    fig, ax = plt.subplots(figsize=figsize)

    for c in [0, 1, 2]:
        mask = labels == c
        if mask.sum() < 10:
            continue
        s_c = scores[mask]
        ax.hist(s_c, bins=40, density=True, alpha=0.45,
                color=REGIME_COLORS[c], label=REGIME_NAMES[c])

    ax.axvline(threshold, color="black", lw=1.2, linestyle="--", label=f"θ={threshold}")
    ax.set_xlabel("Instability Score P(Trans) + P(Unstable)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig
