"""
evaluation/transition_analysis.py
───────────────────────────────────
Regime transition detection and analysis for TIDAL.

Evaluates how well the model detects *transitions* between regimes,
with special focus on the Stable → Transitional → Unstable pathway.

The central scientific claim is that instability emerges through detectable
transitions. This module quantifies whether TIDAL captures those transitions
before observable disruption (i.e., while still in the Transitional regime).

Key analyses
------------
- Transition matrix (empirical vs predicted)
- Transition recall per pair (Stable→Trans, Trans→Unstable, Stable→Unstable)
- Transition boundary detection accuracy (how close to the true boundary)
- Latent trajectory analysis (sequence of predicted probabilities at transitions)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Regime sequence utilities
# ---------------------------------------------------------------------------

def find_transitions(
    labels: np.ndarray,  # (N,) int regime labels
    from_class: Optional[int] = None,
    to_class:   Optional[int] = None,
) -> List[int]:
    """Find time indices where a regime transition occurs.

    A transition at index t means labels[t-1] != labels[t].

    Parameters
    ----------
    labels:
        Temporal sequence of integer regime labels.
    from_class:
        If specified, only return transitions FROM this class.
    to_class:
        If specified, only return transitions TO this class.

    Returns
    -------
    List of transition time indices (the step t where the new regime begins).
    """
    transitions = []
    for t in range(1, len(labels)):
        prev, curr = int(labels[t - 1]), int(labels[t])
        if prev == curr:
            continue
        if from_class is not None and prev != from_class:
            continue
        if to_class is not None and curr != to_class:
            continue
        transitions.append(t)
    return transitions


def compute_transition_matrix(
    labels: np.ndarray,
    num_classes: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """Empirical regime transition matrix.

    Entry (i, j) = P(next_regime = j | current_regime = i).

    Parameters
    ----------
    labels:
        Temporal label sequence.
    num_classes:
        Number of regime classes.
    normalize:
        If True, normalize rows to sum to 1.

    Returns
    -------
    (num_classes, num_classes) transition matrix.
    """
    mat = np.zeros((num_classes, num_classes), dtype=float)
    for t in range(1, len(labels)):
        i, j = int(labels[t - 1]), int(labels[t])
        if 0 <= i < num_classes and 0 <= j < num_classes:
            mat[i, j] += 1.0

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True).clip(min=1e-8)
        mat = mat / row_sums

    return mat


# ---------------------------------------------------------------------------
# Transition detection metrics
# ---------------------------------------------------------------------------

def transition_detection_recall(
    pred_labels: np.ndarray,   # (N,) predicted class
    true_labels: np.ndarray,   # (N,) ground-truth class
    margin: int = 10,
    from_class: Optional[int] = None,
    to_class:   Optional[int] = None,
) -> float:
    """Fraction of true transitions detected within ±margin steps.

    A transition at time t is *detected* if the predicted labels
    also show a transition of the same type within [t-margin, t+margin].

    Parameters
    ----------
    pred_labels:
        Predicted regime sequence.
    true_labels:
        Ground-truth regime sequence.
    margin:
        Tolerance window (steps) for accepting a detection.
    from_class, to_class:
        Optional filters for specific transition pairs.

    Returns
    -------
    Detection recall in [0, 1].
    """
    true_trans = find_transitions(true_labels, from_class, to_class)
    if not true_trans:
        return float("nan")

    pred_trans = set(find_transitions(pred_labels, from_class, to_class))

    detected = 0
    for t in true_trans:
        # Check if any predicted transition falls within the margin window
        for offset in range(-margin, margin + 1):
            if (t + offset) in pred_trans:
                detected += 1
                break

    return float(detected / len(true_trans))


def boundary_proximity_error(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    from_class: Optional[int] = None,
    to_class:   Optional[int] = None,
    max_search: int = 30,
) -> Dict[str, float]:
    """Proximity of predicted transitions to true transition boundaries.

    For each true transition, finds the nearest predicted transition
    and records the distance. Small errors indicate the model detects
    transitions at almost the right time.

    Returns
    -------
    Dict with ``mean_error``, ``median_error``, ``within_5_frac``.
    """
    true_trans = find_transitions(true_labels, from_class, to_class)
    pred_trans = find_transitions(pred_labels, from_class, to_class)

    if not true_trans:
        return {"mean_error": float("nan"), "median_error": float("nan"),
                "within_5_frac": float("nan")}

    pred_set = np.array(pred_trans) if pred_trans else np.array([-9999])
    errors   = []

    for t in true_trans:
        if len(pred_set) > 0:
            dist = np.abs(pred_set - t).min()
            if dist <= max_search:
                errors.append(int(dist))

    if not errors:
        return {"mean_error": float(max_search), "median_error": float(max_search),
                "within_5_frac": 0.0}

    err_arr = np.array(errors)
    return {
        "mean_error":    float(np.mean(err_arr)),
        "median_error":  float(np.median(err_arr)),
        "within_5_frac": float((err_arr <= 5).mean()),
    }


# ---------------------------------------------------------------------------
# Latent trajectory analysis at transition points
# ---------------------------------------------------------------------------

def transition_probability_trajectory(
    probs: np.ndarray,     # (N, C) softmax probabilities
    true_labels: np.ndarray,  # (N,)
    from_class: int = 0,
    to_class:   int = 2,
    window:     int = 20,
) -> Dict:
    """Extract model probability trajectories around transition events.

    For each transition from `from_class` → `to_class`, extracts the
    probability time-series in the window [-window, +window] around the
    transition boundary.

    Returns
    -------
    Dict with ``mean_trajectory`` (2*window+1, C) and individual trajectories.
    """
    transitions = find_transitions(true_labels, from_class, to_class)
    C = probs.shape[1]
    N = len(probs)

    trajectories = []
    for t in transitions:
        start = t - window
        end   = t + window + 1
        if start < 0 or end > N:
            continue
        traj = probs[start:end, :]  # (2*window+1, C)
        trajectories.append(traj)

    if not trajectories:
        empty = np.full((2 * window + 1, C), fill_value=float("nan"))
        return {"mean_trajectory": empty, "n_events": 0}

    traj_stack   = np.stack(trajectories, axis=0)  # (E, 2*window+1, C)
    mean_traj    = np.nanmean(traj_stack, axis=0)   # (2*window+1, C)
    std_traj     = np.nanstd(traj_stack, axis=0)

    time_axis = np.arange(-window, window + 1)

    return {
        "mean_trajectory": mean_traj,
        "std_trajectory":  std_traj,
        "time_axis":       time_axis,
        "n_events":        len(trajectories),
        "class_names":     ["Stable", "Transitional", "Unstable"],
    }


# ---------------------------------------------------------------------------
# Full transition analysis report
# ---------------------------------------------------------------------------

def full_transition_report(
    probs:      np.ndarray,   # (N, H, C)
    pred_labels: np.ndarray,  # (N, H) predicted class
    true_labels: np.ndarray,  # (N, H) ground-truth class
    horizons:   List[int],
    margin:     int = 10,
    num_classes: int = 3,
) -> Dict:
    """Comprehensive transition analysis for all horizons.

    Returns
    -------
    Dict keyed by ``f"h{horizon}"`` with per-horizon transition analyses.
    """
    results = {}

    for h_idx, horizon in enumerate(horizons):
        p_h    = probs[:, h_idx, :]       # (N, C)
        pred_h = pred_labels[:, h_idx]    # (N,)
        true_h = true_labels[:, h_idx]    # (N,)

        key = f"h{horizon}"

        # Transition matrices
        true_mat = compute_transition_matrix(true_h, num_classes)
        pred_mat = compute_transition_matrix(pred_h, num_classes)

        # Per-pair detection recall
        pairs = [(0, 1), (1, 2), (0, 2)]  # Stable→Trans, Trans→Unstable, Stable→Unstable
        pair_recall = {}
        pair_error  = {}
        for (fc, tc) in pairs:
            pair_name = f"{fc}_to_{tc}"
            pair_recall[pair_name] = transition_detection_recall(
                pred_h, true_h, margin, fc, tc
            )
            pair_error[pair_name] = boundary_proximity_error(pred_h, true_h, fc, tc)

        # Overall transition recall
        overall_recall = transition_detection_recall(pred_h, true_h, margin)

        # Probability trajectories around Stable → Transitional boundary
        traj_s2t = transition_probability_trajectory(p_h, true_h, from_class=0, to_class=1)
        traj_t2u = transition_probability_trajectory(p_h, true_h, from_class=1, to_class=2)

        results[key] = {
            "true_transition_matrix":  true_mat.tolist(),
            "pred_transition_matrix":  pred_mat.tolist(),
            "overall_transition_recall": overall_recall,
            "per_pair_recall":         pair_recall,
            "per_pair_boundary_error": pair_error,
            "trajectory_stable_to_transitional": {
                "mean": traj_s2t["mean_trajectory"].tolist() if traj_s2t["n_events"] > 0 else [],
                "n_events": traj_s2t["n_events"],
            },
            "trajectory_transitional_to_unstable": {
                "mean": traj_t2u["mean_trajectory"].tolist() if traj_t2u["n_events"] > 0 else [],
                "n_events": traj_t2u["n_events"],
            },
        }

    return results
