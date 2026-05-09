"""
evaluation/benchmark_compare.py
─────────────────────────────────
Statistical benchmark comparison between TIDAL and baseline models.

Provides:
- Side-by-side metric tables (AUROC, AUPRC, F1, ECE, lead time)
- Pairwise statistical significance testing (Wilcoxon signed-rank)
- Bootstrap-based confidence intervals for all comparisons
- Relative improvement percentage tables
- Publication-ready summary tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Metric table construction
# ---------------------------------------------------------------------------

MODEL_ORDER = ["TIDAL", "LSTM", "Transformer", "SSM"]

METRIC_DISPLAY = {
    "auroc":               "AUROC ↑",
    "auprc":               "AUPRC ↑",
    "macro_f1":            "Macro-F1 ↑",
    "ece":                 "ECE ↓",
    "transitional_recall": "Trans. Recall ↑",
    "false_alarm_rate":    "FAR ↓",
    "mean_lead_time":      "Lead Time (steps) ↑",
    "detection_rate":      "Detection Rate ↑",
}


def build_comparison_table(
    results: Dict[str, Dict],
    horizon: str = "h10",
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Build a model × metric comparison table for a given horizon.

    Parameters
    ----------
    results:
        Dict mapping model name → evaluation report (as returned by
        ``full_evaluation_report``).
    horizon:
        Horizon key, e.g. ``"h10"`` or ``"h30"``.
    metrics:
        Metric keys to include. Defaults to all METRIC_DISPLAY keys.

    Returns
    -------
    Dict: {metric_name: {model_name: value}}.
    """
    if metrics is None:
        metrics = list(METRIC_DISPLAY.keys())

    table: Dict[str, Dict[str, float]] = {m: {} for m in metrics}

    for model_name, report in results.items():
        h_report = report.get(horizon, {})
        for metric in metrics:
            # Lead time is nested
            if metric in ("mean_lead_time", "detection_rate"):
                lt = h_report.get("lead_time", {})
                val = lt.get(metric, float("nan"))
            else:
                val = h_report.get(metric, float("nan"))
            table[metric][model_name] = val

    return table


def format_comparison_table(
    table: Dict[str, Dict[str, float]],
    bold_best: bool = True,
) -> str:
    """Format a comparison table as a markdown/ASCII string.

    Parameters
    ----------
    table:
        Output from ``build_comparison_table``.
    bold_best:
        If True, mark the best model per metric with *.

    Returns
    -------
    Formatted string table.
    """
    models = sorted({m for row in table.values() for m in row})
    header = f"{'Metric':<28}" + "".join(f"{m:>14}" for m in models)
    lines  = [header, "-" * len(header)]

    for metric, row in table.items():
        display = METRIC_DISPLAY.get(metric, metric)
        higher_is_better = not display.endswith("↓")
        vals = [row.get(m, float("nan")) for m in models]

        if bold_best and not all(np.isnan(v) for v in vals):
            valid_vals = [v for v in vals if not np.isnan(v)]
            best_val = max(valid_vals) if higher_is_better else min(valid_vals)
        else:
            best_val = None

        row_str = f"{display:<28}"
        for m, v in zip(models, vals):
            cell = f"{v:.4f}" if not np.isnan(v) else "  N/A"
            if bold_best and best_val is not None and not np.isnan(v):
                if abs(v - best_val) < 1e-8:
                    cell = f"*{cell}"
            row_str += f"{cell:>14}"
        lines.append(row_str)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: np.ndarray,   # (N,) metric values (bootstrap replicates or per-sample)
    scores_b: np.ndarray,   # (N,)
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Wilcoxon signed-rank test between two paired score arrays.

    Parameters
    ----------
    scores_a, scores_b:
        Paired score arrays (same length N).
    alternative:
        ``"two-sided"`` | ``"greater"`` | ``"less"``.

    Returns
    -------
    Dict with ``statistic``, ``p_value``, ``significant`` (at α=0.05).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length.")

    diff = scores_a - scores_b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    except Exception:
        stat, p = float("nan"), float("nan")

    return {
        "statistic":   float(stat),
        "p_value":     float(p),
        "significant": bool(p < 0.05) if not np.isnan(p) else False,
    }


def pairwise_significance_matrix(
    bootstrap_results: Dict[str, np.ndarray],
    metric: str = "auroc",
    alternative: str = "two-sided",
) -> Dict[str, Dict[str, Dict]]:
    """Pairwise Wilcoxon tests between all model pairs.

    Parameters
    ----------
    bootstrap_results:
        Dict mapping model name → 1D array of bootstrap AUROC estimates
        (or any scalar metric estimates).
    metric:
        Label for the metric being compared (informational only).
    alternative:
        Wilcoxon alternative hypothesis.

    Returns
    -------
    Nested dict: {model_a: {model_b: test_result}}.
    """
    models  = list(bootstrap_results.keys())
    results = {m: {} for m in models}

    for i, m_a in enumerate(models):
        for j, m_b in enumerate(models):
            if i == j:
                results[m_a][m_b] = {"statistic": 0.0, "p_value": 1.0, "significant": False}
                continue
            test = wilcoxon_test(
                bootstrap_results[m_a], bootstrap_results[m_b], alternative
            )
            results[m_a][m_b] = test

    return results


# ---------------------------------------------------------------------------
# Relative improvement analysis
# ---------------------------------------------------------------------------

def compute_relative_improvements(
    table: Dict[str, Dict[str, float]],
    baseline_model: str = "LSTM",
    target_model:   str = "TIDAL",
) -> Dict[str, float]:
    """Compute relative improvement of target over baseline for each metric.

    Positive values mean target is better; sign is adjusted for ↓ metrics.

    Returns
    -------
    Dict mapping metric name → relative improvement (%).
    """
    improvements = {}
    for metric, row in table.items():
        baseline = row.get(baseline_model, float("nan"))
        target   = row.get(target_model,   float("nan"))

        if np.isnan(baseline) or np.isnan(target) or baseline == 0:
            improvements[metric] = float("nan")
            continue

        display = METRIC_DISPLAY.get(metric, "")
        higher_is_better = not display.endswith("↓")

        rel = (target - baseline) / (abs(baseline) + 1e-8) * 100.0
        if not higher_is_better:
            rel = -rel  # Flip sign so positive always means "better"

        improvements[metric] = float(rel)

    return improvements


# ---------------------------------------------------------------------------
# Summary report builder
# ---------------------------------------------------------------------------

def build_benchmark_report(
    results: Dict[str, Dict],
    horizons: Optional[List[str]] = None,
    save_dir: Optional[str | Path] = None,
) -> Dict:
    """Build a complete benchmark comparison report.

    Parameters
    ----------
    results:
        Dict mapping model name → per-horizon evaluation report.
    horizons:
        Horizon keys to include (default: all found in first model's report).
    save_dir:
        If provided, save the report as JSON and formatted text.

    Returns
    -------
    Full benchmark report dict.
    """
    if horizons is None:
        first = next(iter(results.values()))
        horizons = [k for k in first.keys() if k.startswith("h")]

    report: Dict = {"horizons": {}}

    for horizon in horizons:
        table = build_comparison_table(results, horizon)
        formatted = format_comparison_table(table)

        # Relative improvements vs LSTM baseline
        improvements = {}
        if "LSTM" in results and "TIDAL" in results:
            improvements = compute_relative_improvements(table, "LSTM", "TIDAL")

        report["horizons"][horizon] = {
            "table":        table,
            "formatted":    formatted,
            "improvements": improvements,
        }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        def _jsonify(obj):
            if isinstance(obj, dict):
                return {k: _jsonify(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_jsonify(v) for v in obj]
            elif isinstance(obj, (np.floating, float)):
                return round(float(obj), 6) if not np.isnan(obj) else None
            return obj

        with open(save_dir / "benchmark_report.json", "w") as f:
            json.dump(_jsonify(report), f, indent=2)

        # Text tables
        with open(save_dir / "benchmark_tables.txt", "w") as f:
            for horizon in horizons:
                f.write(f"\n{'='*60}\n")
                f.write(f"  Benchmark Comparison — Horizon {horizon}\n")
                f.write(f"{'='*60}\n")
                f.write(report["horizons"][horizon]["formatted"])
                f.write("\n\n")

                if report["horizons"][horizon]["improvements"]:
                    f.write("  TIDAL vs LSTM — Relative Improvement (%):\n")
                    for metric, pct in report["horizons"][horizon]["improvements"].items():
                        if not np.isnan(pct):
                            sign = "+" if pct > 0 else ""
                            f.write(f"    {metric:<28}: {sign}{pct:.1f}%\n")

    return report
