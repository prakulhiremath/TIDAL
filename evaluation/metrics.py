"""
evaluation/metrics.py
──────────────────────
Comprehensive evaluation metrics for TIDAL instability detection.

Beyond standard classification metrics, implements:
    - Instability lead time (how early does detection occur?)
    - Early warning horizon
    - False alarm rate
    - Transition sensitivity (performance specifically at regime changes)
    - Detection latency (delay from instability onset to detection)

These metrics are critical for evaluating proactive surveillance systems,
where WHEN you detect is as important as WHETHER you detect.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from typing import Dict, List, Optional, Tuple
from loguru import logger


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels (N,).
        y_prob: Predicted probabilities (N,).
        threshold: Decision threshold for binary predictions.
        prefix: String prefix for metric keys.

    Returns:
        Dictionary of metric name → value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    p = prefix + "_" if prefix else ""

    # ── Standard classification metrics ────────────────────────────────────
    try:
        metrics[f"{p}auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics[f"{p}auroc"] = 0.0

    try:
        metrics[f"{p}auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics[f"{p}auprc"] = 0.0

    metrics[f"{p}f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics[f"{p}precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics[f"{p}recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics[f"{p}accuracy"] = float((y_true == y_pred).mean())

    # ── Confusion matrix derived ────────────────────────────────────────────
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        n_neg = tn + fp
        metrics[f"{p}false_alarm_rate"] = float(fp / max(n_neg, 1))
        metrics[f"{p}specificity"] = float(tn / max(n_neg, 1))
        metrics[f"{p}tp"] = int(tp)
        metrics[f"{p}fp"] = int(fp)
        metrics[f"{p}tn"] = int(tn)
        metrics[f"{p}fn"] = int(fn)
    else:
        metrics[f"{p}false_alarm_rate"] = 0.0
        metrics[f"{p}specificity"] = 1.0

    # ── Optimal threshold ───────────────────────────────────────────────────
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob)
    metrics[f"{p}optimal_threshold"] = float(opt_thresh)
    metrics[f"{p}optimal_f1"] = float(opt_f1)

    return metrics


def compute_early_warning_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    instability_episodes: List[Dict],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute early warning system performance metrics.

    Focuses on HOW EARLY the system detects upcoming instability.

    Args:
        y_true: Binary instability labels (T,).
        y_prob: Predicted probabilities (T,).
        instability_episodes: List of episode dicts from InstabilityIndex.
        threshold: Detection threshold.

    Returns:
        Dict of early warning metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    T = len(y_true)
    metrics = {}

    if not instability_episodes:
        return {"lead_time_mean": 0.0, "lead_time_std": 0.0, "detection_rate": 0.0}

    lead_times = []
    detected = []

    for episode in instability_episodes:
        start = episode["start"]
        # Look back for earliest detection before episode start
        lookback = min(start, 100)
        window = y_pred[max(0, start - lookback): start]

        if len(window) > 0 and window.any():
            # Lead time = steps before episode where prediction was positive
            first_detection = len(window) - 1 - np.argmax(window[::-1] > 0)
            lead_time = lookback - first_detection
            lead_times.append(lead_time)
            detected.append(True)
        else:
            detected.append(False)

    metrics["lead_time_mean"] = float(np.mean(lead_times)) if lead_times else 0.0
    metrics["lead_time_std"] = float(np.std(lead_times)) if lead_times else 0.0
    metrics["lead_time_max"] = float(np.max(lead_times)) if lead_times else 0.0
    metrics["detection_rate"] = float(np.mean(detected))

    # Detection latency: how long AFTER onset before detection
    latencies = []
    for episode in instability_episodes:
        start = episode["start"]
        end = episode["end"]
        episode_preds = y_pred[start:end]
        if episode_preds.any():
            latency = np.argmax(episode_preds)
            latencies.append(latency)

    metrics["detection_latency_mean"] = float(np.mean(latencies)) if latencies else float(end - start)
    metrics["detection_latency_std"] = float(np.std(latencies)) if latencies else 0.0

    # False alarm rate in stable periods
    stable_mask = y_true == 0
    if stable_mask.sum() > 0:
        false_alarms = y_pred[stable_mask].sum()
        metrics["false_alarm_rate_stable"] = float(false_alarms / stable_mask.sum())
    else:
        metrics["false_alarm_rate_stable"] = 0.0

    return metrics


def compute_transition_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    regimes: np.ndarray,
    threshold: float = 0.5,
    neighborhood: int = 10,
) -> Dict[str, float]:
    """
    Evaluate detection performance specifically at regime transitions.

    Transition sensitivity measures how well the model detects
    instability near regime change boundaries — the key scientific claim
    of TIDAL.

    Args:
        y_true: Binary instability labels (T,).
        y_prob: Predicted probabilities (T,).
        regimes: Regime labels {0,1,2} (T,).
        threshold: Decision threshold.
        neighborhood: Steps around transition to analyze.

    Returns:
        Dict of transition-specific metrics.
    """
    T = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    # Find all transition points
    transitions = np.zeros(T, dtype=bool)
    transitions[1:] = regimes[1:] != regimes[:-1]
    transition_indices = np.where(transitions)[0]

    if len(transition_indices) == 0:
        return {"transition_sensitivity": 0.0, "transition_auroc": 0.0}

    # Build transition neighborhood mask
    trans_mask = np.zeros(T, dtype=bool)
    for idx in transition_indices:
        lo = max(0, idx - neighborhood)
        hi = min(T, idx + neighborhood + 1)
        trans_mask[lo:hi] = True

    # Metrics within transition neighborhoods
    y_true_trans = y_true[trans_mask]
    y_prob_trans = y_prob[trans_mask]
    y_pred_trans = y_pred[trans_mask]

    metrics = {}
    metrics["n_transition_steps"] = int(trans_mask.sum())
    metrics["n_transitions"] = int(len(transition_indices))

    if len(y_true_trans) > 0 and len(np.unique(y_true_trans)) > 1:
        metrics["transition_auroc"] = float(roc_auc_score(y_true_trans, y_prob_trans))
        metrics["transition_recall"] = float(recall_score(y_true_trans, y_pred_trans, zero_division=0))
        metrics["transition_precision"] = float(precision_score(y_true_trans, y_pred_trans, zero_division=0))
        metrics["transition_f1"] = float(f1_score(y_true_trans, y_pred_trans, zero_division=0))
    else:
        metrics["transition_auroc"] = 0.0
        metrics["transition_recall"] = 0.0
        metrics["transition_precision"] = 0.0
        metrics["transition_f1"] = 0.0

    # Performance specifically on Stable→Transitional→Unstable sequences
    stable_to_trans = []
    for i in range(1, T):
        if regimes[i - 1] == 0 and regimes[i] == 1:
            stable_to_trans.append(i)

    metrics["n_stable_to_transitional"] = len(stable_to_trans)

    # Detection probability in transitional periods preceding instability
    trans_period_detected = []
    for t in transition_indices:
        if t > 0 and regimes[t - 1] == 1 and regimes[t] == 2:
            # Stable → Transitional → Unstable: look back in transitional period
            lookback_start = max(0, t - 20)
            window_preds = y_pred[lookback_start:t]
            trans_period_detected.append(window_preds.any())

    if trans_period_detected:
        metrics["transitional_detection_rate"] = float(np.mean(trans_period_detected))
    else:
        metrics["transitional_detection_rate"] = 0.0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Stable", "Transitional", "Unstable"],
) -> Dict[str, float]:
    """
    Compute multi-class classification metrics for regime prediction.

    Args:
        y_true: True regime labels {0,1,2} (T,).
        y_pred: Predicted regime labels {0,1,2} (T,).
        class_names: Human-readable class names.

    Returns:
        Dict of multi-class metrics.
    """
    metrics = {}

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        if i < len(f1_per_class):
            metrics[f"f1_{name.lower()}"] = float(f1_per_class[i])

    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["accuracy"] = float((y_true == y_pred).mean())

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximizes a given metric.

    Args:
        y_true: Binary ground truth (N,).
        y_prob: Predicted probabilities (N,).
        metric: Metric to maximize: 'f1', 'recall', 'precision'.

    Returns:
        Tuple of (optimal_threshold, optimal_metric_value).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_thresh = 0.5
    best_score = 0.0

    for i, thresh in enumerate(thresholds):
        p, r = precisions[i], recalls[i]
        if metric == "f1":
            score = 2 * p * r / (p + r + 1e-8)
        elif metric == "recall":
            score = r
        elif metric == "precision":
            score = p
        else:
            score = 2 * p * r / (p + r + 1e-8)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    metrics_to_show: Optional[List[str]] = None,
) -> str:
    """
    Format a results dictionary as a publication-ready LaTeX table.

    Args:
        results: Dict of model_name → metrics dict.
        metrics_to_show: List of metric keys to include.

    Returns:
        LaTeX table string.
    """
    if metrics_to_show is None:
        metrics_to_show = ["auroc", "f1", "precision", "recall", "false_alarm_rate"]

    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for m in metrics_to_show:
            val = metrics.get(m, metrics.get(f"_{m}", 0.0))
            row[m.upper()] = f"{val:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # LaTeX output
    latex = df.to_latex(
        bold_rows=True,
        caption="TIDAL vs. Baseline Instability Detection Performance",
        label="tab:main_results",
    )
    return latex
