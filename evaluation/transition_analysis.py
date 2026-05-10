"""
evaluation/transition_analysis.py
───────────────────────────────────
Regime transition analysis for TIDAL.

This module is central to the paper's scientific contribution:
quantifying how well TIDAL detects the TRANSITIONAL regime
before full instability manifests.

Key analyses:
    1. Transition detection timing
    2. Stable → Transitional → Unstable pathway analysis
    3. Regime dwell time statistics
    4. Hidden stress accumulation trajectory
    5. Transition confusion matrix
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
from loguru import logger


class RegimeTransitionAnalyzer:
    """
    Analyzes regime transition dynamics and detection performance.

    The core scientific claim of TIDAL is that instability passes through
    a detectable TRANSITIONAL state before becoming observable.
    This analyzer quantifies how well models capture this dynamic.

    Usage:
        analyzer = RegimeTransitionAnalyzer()
        stats = analyzer.analyze(true_regimes, pred_probs, instability_index)
    """

    REGIME_NAMES = {0: "Stable", 1: "Transitional", 2: "Unstable"}

    def __init__(self, neighborhood: int = 15):
        """
        Args:
            neighborhood: Steps around transitions to analyze (±steps).
        """
        self.neighborhood = neighborhood

    def analyze(
        self,
        true_regimes: np.ndarray,
        pred_probs: np.ndarray,
        instability_index: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Full transition analysis pipeline.

        Args:
            true_regimes: Ground truth regimes {0,1,2} of shape (T,).
            pred_probs: Model predictions (probabilities) of shape (T,) or (T, n_horizons).
            instability_index: Formal I_t values ∈ [0,1] of shape (T,).

        Returns:
            Comprehensive analysis dictionary.
        """
        T = len(true_regimes)
        # Use first horizon if multi-horizon
        if pred_probs.ndim > 1:
            pred_probs = pred_probs[:, 0]

        analysis = {}

        # ── Regime statistics ────────────────────────────────────────────────
        analysis["regime_statistics"] = self._compute_regime_statistics(true_regimes)

        # ── Transition matrix ────────────────────────────────────────────────
        analysis["transition_matrix"] = self._compute_transition_matrix(true_regimes)

        # ── Transition detection analysis ────────────────────────────────────
        analysis["transition_detection"] = self._analyze_transition_detection(
            true_regimes, pred_probs
        )

        # ── Pathway analysis ────────────────────────────────────────────────
        analysis["pathway_analysis"] = self._analyze_pathways(true_regimes)

        # ── Instability index analysis ───────────────────────────────────────
        if instability_index is not None:
            analysis["index_analysis"] = self._analyze_index_trajectory(
                true_regimes, instability_index
            )

        # ── Prediction confidence at transitions ─────────────────────────────
        analysis["confidence_at_transitions"] = self._analyze_prediction_confidence(
            true_regimes, pred_probs
        )

        # Print summary
        self._log_summary(analysis)
        return analysis

    def _compute_regime_statistics(self, regimes: np.ndarray) -> Dict:
        """Compute basic regime occupancy and duration statistics."""
        T = len(regimes)
        stats = {}

        for r in [0, 1, 2]:
            name = self.REGIME_NAMES[r]
            count = (regimes == r).sum()
            stats[name] = {
                "count": int(count),
                "fraction": float(count / T),
                "mean_duration": 0.0,
                "max_duration": 0,
                "n_episodes": 0,
            }

            # Episode statistics
            durations = []
            i = 0
            while i < T:
                if regimes[i] == r:
                    j = i
                    while j < T and regimes[j] == r:
                        j += 1
                    durations.append(j - i)
                    i = j
                else:
                    i += 1

            if durations:
                stats[name]["mean_duration"] = float(np.mean(durations))
                stats[name]["max_duration"] = int(np.max(durations))
                stats[name]["n_episodes"] = len(durations)
                stats[name]["duration_std"] = float(np.std(durations))

        return stats

    def _compute_transition_matrix(self, regimes: np.ndarray) -> Dict:
        """
        Compute empirical regime transition probability matrix.

        Returns:
            3×3 transition count matrix (as nested dict).
        """
        T = len(regimes)
        counts = np.zeros((3, 3), dtype=int)

        for t in range(T - 1):
            from_r = regimes[t]
            to_r = regimes[t + 1]
            counts[from_r, to_r] += 1

        # Normalize rows to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)

        result = {}
        for i in range(3):
            from_name = self.REGIME_NAMES[i]
            result[from_name] = {}
            for j in range(3):
                to_name = self.REGIME_NAMES[j]
                result[from_name][to_name] = {
                    "count": int(counts[i, j]),
                    "probability": float(probs[i, j]),
                }

        return result

    def _analyze_transition_detection(
        self,
        regimes: np.ndarray,
        pred_probs: np.ndarray,
    ) -> Dict:
        """
        Analyze how well the model signals upcoming transitions.

        For each Stable→Transitional→Unstable pathway, measures the
        model's confidence score trajectory leading up to the transition.
        """
        T = len(regimes)
        transitions = np.where(np.diff(regimes) != 0)[0] + 1

        pre_transition_scores = []      # Model scores before transition
        post_transition_scores = []     # Model scores after transition
        pre_unstable_scores = []        # Scores in transitional period before unstable

        for idx in transitions:
            prev_regime = regimes[idx - 1]
            curr_regime = regimes[idx]

            # Window before transition
            lo = max(0, idx - self.neighborhood)
            pre_scores = pred_probs[lo:idx]
            post_scores = pred_probs[idx:min(T, idx + self.neighborhood)]

            pre_transition_scores.extend(pre_scores.tolist())
            post_transition_scores.extend(post_scores.tolist())

            # Specifically: transitional → unstable
            if prev_regime == 1 and curr_regime == 2:
                trans_start = idx - 1
                # Find start of transitional episode
                while trans_start > 0 and regimes[trans_start - 1] == 1:
                    trans_start -= 1
                pre_unstable_scores.extend(pred_probs[trans_start:idx].tolist())

        return {
            "mean_score_pre_transition": float(np.mean(pre_transition_scores)) if pre_transition_scores else 0.0,
            "mean_score_post_transition": float(np.mean(post_transition_scores)) if post_transition_scores else 0.0,
            "mean_score_in_transitional_before_unstable": float(np.mean(pre_unstable_scores)) if pre_unstable_scores else 0.0,
            "n_transitions": len(transitions),
            "detection_lift": float(
                (np.mean(pre_unstable_scores) - np.mean(pre_transition_scores))
                if pre_unstable_scores and pre_transition_scores else 0.0
            ),
        }

    def _analyze_pathways(self, regimes: np.ndarray) -> Dict:
        """
        Analyze regime transition pathways.

        Specifically tracks the key scientific question:
            What fraction of Unstable episodes were preceded by Transitional?
        """
        T = len(regimes)
        transitions = []
        i = 0
        while i < T - 1:
            if regimes[i] != regimes[i + 1]:
                transitions.append((i, regimes[i], regimes[i + 1]))
            i += 1

        # Count pathway types
        pathway_counts = Counter()
        for _, from_r, to_r in transitions:
            key = f"{self.REGIME_NAMES[from_r]}→{self.REGIME_NAMES[to_r]}"
            pathway_counts[key] = pathway_counts.get(key, 0) + 1

        # How many Unstable episodes were preceded by Transitional?
        unstable_starts = [i for i in range(1, T) if regimes[i] == 2 and regimes[i-1] != 2]
        preceded_by_trans = sum(
            1 for t in unstable_starts if regimes[t - 1] == 1
        )
        preceded_by_stable = sum(
            1 for t in unstable_starts if regimes[t - 1] == 0
        )

        return {
            "pathway_counts": dict(pathway_counts),
            "n_unstable_episodes": len(unstable_starts),
            "preceded_by_transitional": preceded_by_trans,
            "preceded_by_stable_directly": preceded_by_stable,
            "transitional_gateway_fraction": float(preceded_by_trans / max(len(unstable_starts), 1)),
        }

    def _analyze_index_trajectory(
        self,
        regimes: np.ndarray,
        index: np.ndarray,
    ) -> Dict:
        """
        Analyze instability index trajectory within each regime.

        Args:
            regimes: Regime labels (T,).
            index: I_t values (T,).

        Returns:
            Per-regime index statistics and transition build-up analysis.
        """
        stats_per_regime = {}
        for r in [0, 1, 2]:
            name = self.REGIME_NAMES[r]
            mask = regimes == r
            if mask.sum() > 0:
                vals = index[mask]
                stats_per_regime[name] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "q25": float(np.percentile(vals, 25)),
                    "q75": float(np.percentile(vals, 75)),
                }
            else:
                stats_per_regime[name] = {"mean": 0.0}

        # Build-up trajectory: average index in N steps before Unstable onset
        T = len(regimes)
        unstable_starts = [i for i in range(1, T) if regimes[i] == 2 and regimes[i-1] != 2]
        buildup_traces = []
        window = min(30, self.neighborhood * 2)

        for t in unstable_starts:
            lo = max(0, t - window)
            trace = index[lo:t]
            if len(trace) == window:
                buildup_traces.append(trace)

        buildup_analysis = {}
        if buildup_traces:
            traces_arr = np.array(buildup_traces)
            buildup_analysis["mean_trajectory"] = traces_arr.mean(axis=0).tolist()
            buildup_analysis["std_trajectory"] = traces_arr.std(axis=0).tolist()
            buildup_analysis["n_episodes"] = len(buildup_traces)

        return {
            "per_regime": stats_per_regime,
            "buildup_before_unstable": buildup_analysis,
        }

    def _analyze_prediction_confidence(
        self,
        regimes: np.ndarray,
        pred_probs: np.ndarray,
    ) -> Dict:
        """Compute mean prediction confidence per regime type."""
        confidence = {}
        for r in [0, 1, 2]:
            name = self.REGIME_NAMES[r]
            mask = regimes == r
            if mask.sum() > 0:
                confidence[name] = {
                    "mean_prob": float(pred_probs[mask].mean()),
                    "std_prob": float(pred_probs[mask].std()),
                }
        return confidence

    def compute_transition_confusion_matrix(
        self,
        true_regimes: np.ndarray,
        pred_regimes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute 3×3 confusion matrix for regime classification.

        Args:
            true_regimes: True regime labels {0,1,2}.
            pred_regimes: Predicted regime labels {0,1,2}.

        Returns:
            (3, 3) confusion matrix.
        """
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_regimes, pred_regimes, labels=[0, 1, 2])
        return cm

    def _log_summary(self, analysis: Dict) -> None:
        """Log a summary of key findings."""
        pathway = analysis.get("pathway_analysis", {})
        logger.info("Transition Analysis Summary:")
        logger.info(
            f"  Transitional gateway fraction: "
            f"{pathway.get('transitional_gateway_fraction', 0):.3f} "
            f"({pathway.get('preceded_by_transitional', 0)}/{pathway.get('n_unstable_episodes', 0)} episodes)"
        )

        det = analysis.get("transition_detection", {})
        logger.info(
            f"  Mean model score in Transitional→Unstable: "
            f"{det.get('mean_score_in_transitional_before_unstable', 0):.4f}"
        )
        logger.info(
            f"  Detection lift at transitions: "
            f"{det.get('detection_lift', 0):.4f}"
        )
