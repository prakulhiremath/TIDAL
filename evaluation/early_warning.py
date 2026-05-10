"""
evaluation/early_warning.py
────────────────────────────
Early warning system evaluation for TIDAL.

This module analyzes the temporal relationship between model predictions
and ground-truth instability events, quantifying:

    - How early can TIDAL detect upcoming instability?
    - What is the optimal detection threshold for surveillance?
    - How does lead time vary across different instability types?
    - What is the precision-recall trade-off for early warning?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from evaluation.metrics import compute_binary_metrics, compute_early_warning_metrics


class EarlyWarningEvaluator:
    """
    Evaluates TIDAL as an early warning system for financial instability.

    Goes beyond point-in-time classification to evaluate:
        1. Lead time distribution
        2. Horizon-specific performance curves
        3. Operating point analysis
        4. Regime transition detection timing

    Usage:
        evaluator = EarlyWarningEvaluator(horizons=[10, 30, 60])
        report = evaluator.evaluate(probs, regimes, episodes)
        evaluator.save_report(report, "results/tables/early_warning.csv")
    """

    def __init__(
        self,
        horizons: List[int] = [10, 30, 60],
        threshold_range: Tuple[float, float, int] = (0.1, 0.9, 50),
        neighborhood: int = 10,
    ):
        """
        Args:
            horizons: Prediction horizons to evaluate.
            threshold_range: (min, max, n_steps) for threshold sweep.
            neighborhood: Steps around transitions for sensitivity analysis.
        """
        self.horizons = horizons
        self.threshold_range = threshold_range
        self.neighborhood = neighborhood

    def evaluate(
        self,
        probs_per_horizon: Dict[str, np.ndarray],
        y_true_per_horizon: Dict[str, np.ndarray],
        regimes: np.ndarray,
        instability_episodes: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Full early warning evaluation across all horizons.

        Args:
            probs_per_horizon: Dict mapping 'h10', 'h30', 'h60' → prob array (T,).
            y_true_per_horizon: Dict mapping same keys → binary label array (T,).
            regimes: Regime labels {0,1,2} of shape (T,).
            instability_episodes: List of episode dicts (from InstabilityIndex).

        Returns:
            Comprehensive evaluation report dictionary.
        """
        report = {
            "per_horizon": {},
            "threshold_sweep": {},
            "early_warning": {},
            "transition_sensitivity": {},
        }

        for key, probs in probs_per_horizon.items():
            h = int(key.replace("h", ""))
            y_true_key = f"instability_h{h}" if f"instability_h{h}" in y_true_per_horizon else key
            y_true = y_true_per_horizon.get(y_true_key, y_true_per_horizon.get(key))

            if y_true is None:
                continue

            # Standard metrics
            base_metrics = compute_binary_metrics(y_true, probs, threshold=0.5)
            report["per_horizon"][f"h{h}"] = base_metrics

            # Threshold sweep
            report["threshold_sweep"][f"h{h}"] = self._threshold_sweep(y_true, probs)

            # Early warning metrics
            if instability_episodes:
                ew_metrics = compute_early_warning_metrics(
                    y_true, probs, instability_episodes, threshold=0.5
                )
                report["early_warning"][f"h{h}"] = ew_metrics

        # Transition sensitivity (use shortest horizon as proxy)
        if "h10" in probs_per_horizon:
            probs_h10 = probs_per_horizon["h10"]
            y_true_h10 = y_true_per_horizon.get("instability_h10", y_true_per_horizon.get("h10"))
            if y_true_h10 is not None:
                from evaluation.metrics import compute_transition_sensitivity
                report["transition_sensitivity"] = compute_transition_sensitivity(
                    y_true_h10, probs_h10, regimes,
                    threshold=0.5, neighborhood=self.neighborhood
                )

        # Summary statistics
        report["summary"] = self._compute_summary(report)

        logger.info("Early warning evaluation complete")
        logger.info(f"  Lead time (h10): {report['early_warning'].get('h10', {}).get('lead_time_mean', 0):.1f} steps")
        logger.info(f"  Detection rate: {report['early_warning'].get('h10', {}).get('detection_rate', 0):.3f}")
        logger.info(f"  Transition sensitivity: {report['transition_sensitivity'].get('transition_auroc', 0):.4f}")

        return report

    def _threshold_sweep(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Compute metrics across a range of thresholds for ROC analysis.

        Args:
            y_true: Binary labels (T,).
            y_prob: Predicted probabilities (T,).

        Returns:
            List of metric dicts, one per threshold.
        """
        min_t, max_t, n_steps = self.threshold_range
        thresholds = np.linspace(min_t, max_t, n_steps)
        results = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())

            results.append({
                "threshold": float(thresh),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "false_alarm_rate": float(fp / max(tn + fp, 1)),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })

        return results

    def _compute_summary(self, report: Dict) -> Dict[str, float]:
        """Extract key summary statistics from the full report."""
        summary = {}

        # Best AUROC across horizons
        aurocs = [
            report["per_horizon"].get(f"h{h}", {}).get("auroc", 0)
            for h in self.horizons
        ]
        summary["best_auroc"] = float(max(aurocs)) if aurocs else 0.0
        summary["mean_auroc"] = float(np.mean(aurocs)) if aurocs else 0.0

        # Early warning lead time
        ew_h10 = report["early_warning"].get("h10", {})
        summary["lead_time_mean"] = float(ew_h10.get("lead_time_mean", 0))
        summary["detection_rate"] = float(ew_h10.get("detection_rate", 0))
        summary["false_alarm_rate"] = float(ew_h10.get("false_alarm_rate_stable", 0))

        # Transition sensitivity
        trans = report.get("transition_sensitivity", {})
        summary["transition_auroc"] = float(trans.get("transition_auroc", 0))
        summary["transitional_detection_rate"] = float(trans.get("transitional_detection_rate", 0))

        return summary

    def save_report(self, report: Dict, output_dir: str) -> None:
        """
        Save evaluation report as CSV and JSON files.

        Args:
            report: Report dictionary from evaluate().
            output_dir: Directory to save outputs.
        """
        import json
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Per-horizon metrics CSV
        horizon_rows = []
        for h_key, metrics in report["per_horizon"].items():
            row = {"horizon": h_key, **metrics}
            horizon_rows.append(row)
        if horizon_rows:
            pd.DataFrame(horizon_rows).to_csv(
                output_path / "per_horizon_metrics.csv", index=False
            )

        # Early warning CSV
        ew_rows = []
        for h_key, ew in report["early_warning"].items():
            row = {"horizon": h_key, **ew}
            ew_rows.append(row)
        if ew_rows:
            pd.DataFrame(ew_rows).to_csv(
                output_path / "early_warning_metrics.csv", index=False
            )

        # Summary JSON
        with open(output_path / "summary.json", "w") as f:
            json.dump(report.get("summary", {}), f, indent=2)

        # Transition sensitivity CSV
        if report.get("transition_sensitivity"):
            pd.DataFrame([report["transition_sensitivity"]]).to_csv(
                output_path / "transition_sensitivity.csv", index=False
            )

        logger.info(f"Early warning report saved to {output_path}")
