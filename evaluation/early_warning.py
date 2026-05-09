"""
evaluation/early_warning.py
────────────────────────────
Early-warning system evaluation for TIDAL instability surveillance.

Evaluates the surveillance system as a *threshold-based alarm system*:
    - At each time step, the instability score s_t = P(Trans) + P(Unstable)
    - An alarm fires when s_t ≥ threshold θ
    - A true positive = alarm fires within `lead_time_window` steps before onset

This framing is distinct from simple classification accuracy because it
rewards early detection and penalises late or missed alarms.

Key outputs
-----------
- Lead-time distribution (how far in advance alarms fire)
- Precision-recall tradeoff across thresholds
- Hit rate / miss rate / false alarm rate at the operating threshold
- Alarm persistence analysis (sustained vs sporadic alarms)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_onset_events(
    targets: np.ndarray,    # (N,) int labels
    unstable_class: int = 2,
    min_gap: int = 10,
) -> List[int]:
    """Find indices of instability onset events.

    An onset is the first step of each new Unstable episode. Episodes
    separated by fewer than `min_gap` stable steps are merged.

    Parameters
    ----------
    targets:
        Temporal sequence of ground-truth regime labels.
    unstable_class:
        Class index for the Unstable regime.
    min_gap:
        Minimum gap (in steps) between two separate episodes.

    Returns
    -------
    List of onset time indices.
    """
    is_unstable = (targets == unstable_class).astype(int)
    onsets: List[int] = []
    in_event = False
    last_end  = -min_gap - 1

    for t in range(len(is_unstable)):
        if is_unstable[t] == 1:
            if not in_event:
                if t - last_end >= min_gap:
                    onsets.append(t)
                in_event = True
        else:
            if in_event:
                last_end = t
            in_event = False

    return onsets


def detect_alarm_events(
    scores: np.ndarray,    # (N,) instability scores
    threshold: float = 0.5,
    min_gap: int = 5,
) -> List[int]:
    """Find indices of alarm *onset* (first step of each alarm episode).

    Parameters
    ----------
    scores:
        Temporal sequence of instability scores.
    threshold:
        Alert threshold.
    min_gap:
        Merge alarm events separated by fewer than this many steps.

    Returns
    -------
    List of alarm onset time indices.
    """
    is_alarm = (scores >= threshold).astype(int)
    alarms: List[int] = []
    in_alarm = False
    last_end = -min_gap - 1

    for t in range(len(is_alarm)):
        if is_alarm[t] == 1:
            if not in_alarm:
                if t - last_end >= min_gap:
                    alarms.append(t)
                in_alarm = True
        else:
            if in_alarm:
                last_end = t
            in_alarm = False

    return alarms


# ---------------------------------------------------------------------------
# Lead-time analysis
# ---------------------------------------------------------------------------

def lead_time_analysis(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    lead_time_window: int = 60,
    unstable_class: int = 2,
    min_gap: int = 10,
) -> Dict:
    """Full lead-time analysis for a single horizon.

    Parameters
    ----------
    scores:
        (N,) instability score sequence.
    targets:
        (N,) ground-truth label sequence.
    threshold:
        Alert threshold.
    lead_time_window:
        Maximum steps before onset to credit as early warning.
    unstable_class:
        Unstable class index.
    min_gap:
        Minimum episode separation (steps).

    Returns
    -------
    Dict with detailed lead-time statistics.
    """
    onsets = detect_onset_events(targets, unstable_class, min_gap)
    if not onsets:
        return {
            "n_events": 0,
            "detected": 0,
            "missed": 0,
            "detection_rate": float("nan"),
            "lead_times": [],
            "mean_lead_time": float("nan"),
            "median_lead_time": float("nan"),
            "std_lead_time": float("nan"),
            "lead_time_hist": [],
        }

    lead_times = []
    missed     = 0

    for onset in onsets:
        window_start = max(0, onset - lead_time_window)
        window_scores = scores[window_start:onset]
        alarm_steps = np.where(window_scores >= threshold)[0]

        if len(alarm_steps) > 0:
            first_alarm = alarm_steps[0]
            lead_t = len(window_scores) - first_alarm
            lead_times.append(int(lead_t))
        else:
            missed += 1

    detected = len(onsets) - missed
    detection_rate = detected / len(onsets)

    lead_arr = np.array(lead_times) if lead_times else np.array([0])

    # Histogram bins for distribution
    bins = np.arange(0, lead_time_window + 10, 5)
    hist, _ = np.histogram(lead_arr, bins=bins)

    return {
        "n_events":       len(onsets),
        "detected":       detected,
        "missed":         missed,
        "detection_rate": float(detection_rate),
        "lead_times":     lead_times,
        "mean_lead_time":   float(np.mean(lead_arr)) if lead_times else 0.0,
        "median_lead_time": float(np.median(lead_arr)) if lead_times else 0.0,
        "std_lead_time":    float(np.std(lead_arr)) if lead_times else 0.0,
        "lead_time_hist": hist.tolist(),
        "lead_time_bins": bins.tolist(),
    }


# ---------------------------------------------------------------------------
# Precision-recall at various thresholds
# ---------------------------------------------------------------------------

def alarm_precision_recall_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    lead_time_window: int = 60,
    unstable_class: int = 2,
    min_gap: int = 10,
) -> Dict:
    """Event-level precision-recall curve across alarm thresholds.

    Unlike sample-level AUPRC, this evaluates *alarm events*:
        - Precision = detected_events / alarm_events
        - Recall    = detected_events / total_onset_events

    Parameters
    ----------
    scores:
        (N,) instability score sequence.
    targets:
        (N,) label sequence.
    thresholds:
        Array of threshold values to sweep. Defaults to 20 values in [0.1, 0.9].
    lead_time_window, unstable_class, min_gap:
        Passed to detect_onset_events / detect_alarm_events.

    Returns
    -------
    Dict with ``precision``, ``recall``, ``thresholds``, ``f1`` arrays.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 20)

    onsets  = detect_onset_events(targets, unstable_class, min_gap)
    n_onset = len(onsets)

    precisions, recalls, f1s = [], [], []

    for θ in thresholds:
        alarms  = detect_alarm_events(scores, θ, min_gap)
        n_alarm = len(alarms)

        if n_alarm == 0:
            precisions.append(1.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        # Match: each alarm event credited if within lead_time_window before an onset
        matched_onsets = set()
        tp = 0
        for alarm_t in alarms:
            for onset_t in onsets:
                if 0 < (onset_t - alarm_t) <= lead_time_window:
                    if onset_t not in matched_onsets:
                        tp += 1
                        matched_onsets.add(onset_t)
                        break

        precision = tp / n_alarm if n_alarm > 0 else 0.0
        recall    = tp / n_onset if n_onset > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall + 1e-8))

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "thresholds": thresholds.tolist(),
        "precision":  precisions,
        "recall":     recalls,
        "f1":         f1s,
        "n_onsets":   n_onset,
    }


# ---------------------------------------------------------------------------
# Alarm persistence analysis
# ---------------------------------------------------------------------------

def alarm_persistence_analysis(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    unstable_class: int = 2,
) -> Dict:
    """Analyse the temporal structure of alarm episodes.

    Returns statistics on alarm episode durations and whether they
    tend to cluster before genuine instability events.
    """
    is_alarm    = scores >= threshold
    is_unstable = targets == unstable_class

    # Episode durations
    durations = []
    current   = 0
    for a in is_alarm:
        if a:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    if current > 0:
        durations.append(current)

    dur_arr = np.array(durations) if durations else np.array([0])

    # Fraction of alarm steps co-occurring with non-stable
    alarm_steps  = is_alarm.sum()
    correct_alarm = (is_alarm & (targets > 0)).sum()
    precision    = float(correct_alarm / alarm_steps) if alarm_steps > 0 else float("nan")

    return {
        "n_alarm_episodes":    len(durations),
        "mean_episode_length": float(np.mean(dur_arr)),
        "max_episode_length":  int(np.max(dur_arr)),
        "alarm_precision":     precision,
        "alarm_rate":          float(is_alarm.mean()),
    }


# ---------------------------------------------------------------------------
# Multi-horizon early-warning summary
# ---------------------------------------------------------------------------

def multi_horizon_early_warning(
    scores:    np.ndarray,   # (N, H)
    targets:   np.ndarray,   # (N, H)
    horizons:  List[int],
    threshold: float = 0.5,
    lead_time_window: int = 60,
    unstable_class: int = 2,
) -> Dict:
    """Run full early-warning analysis for each prediction horizon.

    Returns
    -------
    Dict keyed by ``f"h{horizon}"`` with per-horizon result dicts.
    """
    results = {}
    for h_idx, horizon in enumerate(horizons):
        s_h = scores[:, h_idx]
        t_h = targets[:, h_idx]
        key = f"h{horizon}"

        lt  = lead_time_analysis(s_h, t_h, threshold, lead_time_window, unstable_class)
        prc = alarm_precision_recall_curve(s_h, t_h,
                                            lead_time_window=lead_time_window,
                                            unstable_class=unstable_class)
        alm = alarm_persistence_analysis(s_h, t_h, threshold, unstable_class)

        results[key] = {"lead_time": lt, "precision_recall": prc, "persistence": alm}

    return results
