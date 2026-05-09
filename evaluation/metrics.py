"""
evaluation/metrics.py
──────────────────────
Comprehensive evaluation metrics for financial instability detection.

All metrics are designed for the three-regime surveillance framing:
    0 = Stable  |  1 = Transitional  |  2 = Unstable

Key metrics
-----------
- AUROC / AUPRC          : discrimination power (OVR macro)
- Transitional recall    : ability to detect the early-warning regime
- Lead time              : how many steps *before* instability onset a warning fires
- False alarm rate       : proportion of warnings during stable periods
- Calibration (ECE)      : reliability of predicted probabilities
- Bootstrap CIs          : 95% confidence intervals via stratified resampling

Tensor conventions
------------------
    probs   : (N, H, C)  — softmax probabilities
    targets : (N, H)     — int64 regime labels {0,1,2}
    scores  : (N, H)     — instability score = P(Trans)+P(Unstable)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def compute_auroc(
    probs: np.ndarray,   # (N, C)
    targets: np.ndarray, # (N,)
    multi_class: str = "ovr",
) -> float:
    """Macro-averaged AUROC for multi-class regime classification."""
    if len(np.unique(targets)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(targets, probs, multi_class=multi_class, average="macro"))
    except Exception:
        return float("nan")


def compute_auprc(
    probs: np.ndarray,   # (N, C)
    targets: np.ndarray, # (N,)
    num_classes: int = 3,
) -> float:
    """Macro-averaged AUPRC (OVR, one class at a time)."""
    aps = []
    for c in range(num_classes):
        binary_t = (targets == c).astype(int)
        if binary_t.sum() == 0:
            continue
        aps.append(average_precision_score(binary_t, probs[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def compute_f1(
    preds: np.ndarray,   # (N,) predicted class
    targets: np.ndarray, # (N,)
    average: str = "macro",
) -> float:
    return float(f1_score(targets, preds, average=average, zero_division=0))


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm = cm.astype(float) / row_sums
    return cm


# ---------------------------------------------------------------------------
# Instability-specific metrics
# ---------------------------------------------------------------------------

def compute_transitional_recall(
    preds: np.ndarray,
    targets: np.ndarray,
    transitional_class: int = 1,
) -> float:
    """Recall for the Transitional regime (the early-warning class).

    High transitional recall means the model reliably identifies the
    latent accumulation phase before observable instability onset.
    """
    mask = targets == transitional_class
    if mask.sum() == 0:
        return float("nan")
    return float((preds[mask] == transitional_class).mean())


def compute_instability_recall(
    preds: np.ndarray,
    targets: np.ndarray,
    unstable_class: int = 2,
) -> float:
    """Recall for the Unstable regime."""
    mask = targets == unstable_class
    if mask.sum() == 0:
        return float("nan")
    return float((preds[mask] == unstable_class).mean())


def compute_false_alarm_rate(
    scores: np.ndarray,   # (N,) instability score in [0, 1]
    targets: np.ndarray,  # (N,) int labels
    threshold: float = 0.5,
    stable_class: int = 0,
) -> float:
    """False alarm rate: proportion of alarms fired during stable periods.

    FA = |{t : score_t ≥ threshold, target_t = 0}| / |{t : target_t = 0}|
    """
    stable_mask = targets == stable_class
    if stable_mask.sum() == 0:
        return float("nan")
    alarms_in_stable = (scores[stable_mask] >= threshold).sum()
    return float(alarms_in_stable / stable_mask.sum())


def compute_lead_time(
    scores: np.ndarray,       # (N,) instability score (temporal sequence)
    targets: np.ndarray,      # (N,) int labels (temporal sequence)
    threshold: float = 0.5,
    lead_time_window: int = 60,
    unstable_class: int = 2,
) -> Dict[str, float]:
    """Compute early-warning lead time statistics.

    For each instability onset event, finds how many steps *before* onset
    the surveillance score first crossed the alert threshold.

    Parameters
    ----------
    scores:
        Temporal sequence of instability scores (P(Trans)+P(Unstable)).
    targets:
        Temporal sequence of ground-truth labels.
    threshold:
        Alert threshold for the instability score.
    lead_time_window:
        Max steps to look back for a pre-onset warning.
    unstable_class:
        Class index for the Unstable regime.

    Returns
    -------
    Dict with keys: ``mean_lead_time``, ``median_lead_time``, ``detected_fraction``.
    """
    # Find instability onset events: first step of each unstable episode
    is_unstable = (targets == unstable_class).astype(int)
    onsets = []
    in_event = False
    for t in range(len(is_unstable)):
        if is_unstable[t] == 1 and not in_event:
            onsets.append(t)
            in_event = True
        elif is_unstable[t] == 0:
            in_event = False

    if not onsets:
        return {"mean_lead_time": float("nan"),
                "median_lead_time": float("nan"),
                "detected_fraction": float("nan")}

    lead_times = []
    detected = 0

    for onset in onsets:
        start = max(0, onset - lead_time_window)
        window_scores = scores[start:onset]
        alarm_steps   = np.where(window_scores >= threshold)[0]

        if len(alarm_steps) > 0:
            first_alarm   = alarm_steps[0]
            lead_t        = (onset - start) - first_alarm
            lead_times.append(lead_t)
            detected += 1

    detected_frac = detected / len(onsets)
    if not lead_times:
        return {"mean_lead_time": 0.0,
                "median_lead_time": 0.0,
                "detected_fraction": float(detected_frac)}

    return {
        "mean_lead_time":   float(np.mean(lead_times)),
        "median_lead_time": float(np.median(lead_times)),
        "detected_fraction": float(detected_frac),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_ece(
    probs: np.ndarray,    # (N, C)
    targets: np.ndarray,  # (N,)
    n_bins: int = 15,
    num_classes: int = 3,
) -> float:
    """Expected Calibration Error (macro-averaged across classes, OVR).

    Lower ECE = better-calibrated probabilities.
    """
    eces = []
    for c in range(num_classes):
        binary_t = (targets == c).astype(float)
        p_c = probs[:, c]
        try:
            prob_true, prob_pred = calibration_curve(
                binary_t, p_c, n_bins=n_bins, strategy="quantile"
            )
            ece_c = float(np.abs(prob_true - prob_pred).mean())
        except Exception:
            ece_c = float("nan")
        eces.append(ece_c)
    valid = [v for v in eces if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_calibration_curve(
    probs: np.ndarray,    # (N, C)
    targets: np.ndarray,  # (N,)
    class_idx: int = 2,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fraction_positive, mean_predicted_prob) for one class."""
    binary_t = (targets == class_idx).astype(float)
    p_c = probs[:, class_idx]
    try:
        prob_true, prob_pred = calibration_curve(
            binary_t, p_c, n_bins=n_bins, strategy="quantile"
        )
    except Exception:
        prob_true, prob_pred = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return prob_true, prob_pred


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metric(
    metric_fn,
    *args,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for any scalar metric function.

    Parameters
    ----------
    metric_fn:
        Callable that accepts the same positional args and returns a float.
    *args:
        Arrays to resample (must all have the same length on axis 0).
    n_bootstrap:
        Number of bootstrap replicates.
    confidence:
        CI level (0.95 = 95%).
    seed:
        Random seed.

    Returns
    -------
    Dict with ``estimate``, ``lower``, ``upper``, ``std``.
    """
    rng = np.random.default_rng(seed)
    N   = len(args[0])
    estimates = []

    for _ in range(n_bootstrap):
        idx   = rng.integers(0, N, size=N)
        resampled = [a[idx] for a in args]
        try:
            val = metric_fn(*resampled)
            if not np.isnan(val):
                estimates.append(val)
        except Exception:
            pass

    if not estimates:
        return {"estimate": float("nan"), "lower": float("nan"),
                "upper": float("nan"), "std": float("nan")}

    estimates = np.array(estimates)
    alpha = 1.0 - confidence
    return {
        "estimate": float(np.mean(estimates)),
        "lower":    float(np.percentile(estimates, 100 * alpha / 2)),
        "upper":    float(np.percentile(estimates, 100 * (1 - alpha / 2))),
        "std":      float(np.std(estimates)),
    }


# ---------------------------------------------------------------------------
# Aggregated evaluation report
# ---------------------------------------------------------------------------

def full_evaluation_report(
    probs:      np.ndarray,   # (N, H, C)
    targets:    np.ndarray,   # (N, H)
    scores:     np.ndarray,   # (N, H) instability scores
    horizons:   List[int],
    threshold:  float = 0.5,
    lead_time_window: int = 60,
    n_bootstrap: int = 1000,
    num_classes: int = 3,
) -> Dict[str, Dict]:
    """Compute the full suite of evaluation metrics for each prediction horizon.

    Parameters
    ----------
    probs:    Softmax probabilities (N, H, C).
    targets:  Ground-truth labels (N, H).
    scores:   Instability score = P(Trans)+P(Unstable), shape (N, H).
    horizons: List of horizon step values (e.g. [10, 30, 60]).
    threshold, lead_time_window, n_bootstrap, num_classes: see individual functions.

    Returns
    -------
    Dict keyed by ``f"h{horizon}"`` with per-horizon metric dicts.
    """
    H = len(horizons)
    report: Dict[str, Dict] = {}

    for h_idx, horizon in enumerate(horizons):
        p_h = probs[:, h_idx, :]     # (N, C)
        t_h = targets[:, h_idx]      # (N,)
        s_h = scores[:, h_idx]       # (N,)
        pred_h = p_h.argmax(axis=-1) # (N,)

        auroc = compute_auroc(p_h, t_h)
        auprc = compute_auprc(p_h, t_h, num_classes)
        f1    = compute_f1(pred_h, t_h)
        ece   = compute_ece(p_h, t_h, num_classes=num_classes)

        trans_recall    = compute_transitional_recall(pred_h, t_h)
        unstable_recall = compute_instability_recall(pred_h, t_h)
        far             = compute_false_alarm_rate(s_h, t_h, threshold)
        lead            = compute_lead_time(s_h, t_h, threshold, lead_time_window)
        cm              = compute_confusion_matrix(pred_h, t_h, num_classes)

        # Bootstrap CI for AUROC
        auroc_ci = bootstrap_metric(
            lambda p, t: compute_auroc(p, t),
            p_h, t_h,
            n_bootstrap=n_bootstrap,
        )

        report[f"h{horizon}"] = {
            "auroc":              auroc,
            "auprc":              auprc,
            "macro_f1":           f1,
            "ece":                ece,
            "transitional_recall": trans_recall,
            "unstable_recall":    unstable_recall,
            "false_alarm_rate":   far,
            "lead_time":          lead,
            "auroc_ci":           auroc_ci,
            "confusion_matrix":   cm.tolist(),
        }

    return report
