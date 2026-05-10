"""
evaluation/benchmark_compare.py
─────────────────────────────────
Cross-model benchmark comparison for TIDAL.

Loads saved model results, computes statistical comparisons,
and generates publication-ready summary tables.

Includes:
    - Pairwise significance testing
    - Bootstrap confidence intervals
    - LaTeX table generation
    - HTML report generation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

from evaluation.metrics import format_metrics_table


class BenchmarkComparison:
    """
    Aggregate and compare results across all TIDAL baseline models.

    Usage:
        compare = BenchmarkComparison()
        compare.add_result("TIDAL", tidal_metrics)
        compare.add_result("LSTM", lstm_metrics)
        table = compare.generate_latex_table()
        compare.save_all("results/tables/")
    """

    MODEL_ORDER = [
        "Logistic Regression", "XGBoost",
        "LSTM", "Transformer", "SSM", "TIDAL"
    ]

    def __init__(self, horizons: List[int] = [10, 30, 60]):
        self.horizons = horizons
        self.results: Dict[str, Dict] = {}
        self.raw_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Register a model's evaluation results.

        Args:
            model_name: Display name of the model.
            metrics: Metric dictionary from compute_binary_metrics.
            predictions: Optional dict of raw prediction arrays for stat tests.
        """
        self.results[model_name] = metrics
        if predictions:
            self.raw_predictions[model_name] = predictions
        logger.info(f"Registered results: {model_name} | AUROC={metrics.get('auroc', 0):.4f}")

    def generate_main_table(
        self,
        horizon: int = 10,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate the main results comparison DataFrame.

        Args:
            horizon: Horizon to report (default: shortest = h10).
            metrics: Metric names to include.

        Returns:
            DataFrame with models as rows, metrics as columns.
        """
        if metrics is None:
            metrics = ["auroc", "auprc", "f1", "precision", "recall",
                       "false_alarm_rate", "lead_time_mean"]

        rows = []
        for model_name in self.MODEL_ORDER:
            if model_name not in self.results:
                continue

            m = self.results[model_name]
            row = {"Model": model_name}
            for metric in metrics:
                # Try horizon-specific first, fall back to global
                val = m.get(f"{metric}_h{horizon}", m.get(metric, None))
                if val is None:
                    val = 0.0
                row[metric.upper()] = val
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("Model")
        return df

    def generate_latex_table(
        self,
        horizon: int = 10,
        bold_best: bool = True,
    ) -> str:
        """
        Generate LaTeX-formatted main results table.

        Args:
            horizon: Horizon to display.
            bold_best: Bold the best value per column.

        Returns:
            LaTeX table string.
        """
        df = self.generate_main_table(horizon=horizon)
        if df.empty:
            return "% No results to display"

        # Bold best value per column
        if bold_best:
            df_display = df.copy()
            for col in df.columns:
                is_lower_better = col in ["FALSE_ALARM_RATE"]
                if is_lower_better:
                    best_idx = df[col].idxmin()
                else:
                    best_idx = df[col].idxmax()
                df_display.loc[best_idx, col] = f"\\textbf{{{df.loc[best_idx, col]:.3f}}}"
                for idx in df.index:
                    if idx != best_idx:
                        try:
                            df_display.loc[idx, col] = f"{float(df.loc[idx, col]):.3f}"
                        except Exception:
                            pass

            latex = df_display.to_latex(
                escape=False,
                caption=f"Instability Detection Performance at Horizon h={horizon} (TIDAL vs. Baselines)",
                label=f"tab:results_h{horizon}",
            )
        else:
            latex = df.round(3).to_latex(
                caption=f"Results at h={horizon}",
                label=f"tab:results_h{horizon}",
            )

        return latex

    def generate_multi_horizon_table(self) -> pd.DataFrame:
        """
        Generate AUROC comparison across all horizons and models.

        Returns:
            DataFrame with models as rows, horizons as columns.
        """
        rows = []
        for model_name in self.MODEL_ORDER:
            if model_name not in self.results:
                continue
            m = self.results[model_name]
            row = {"Model": model_name}
            for h in self.horizons:
                val = m.get(f"auroc_h{h}", m.get("auroc", 0.0))
                row[f"AUROC h={h}"] = round(val, 3)
            rows.append(row)

        return pd.DataFrame(rows).set_index("Model")

    def pairwise_significance_test(
        self,
        model_a: str,
        model_b: str,
        metric: str = "auroc",
        horizon: int = 10,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Bootstrap significance test comparing two models.

        Uses bootstrap resampling to estimate p-value for the hypothesis
        that model_a outperforms model_b on a given metric.

        Args:
            model_a: First model name (assumed better).
            model_b: Second model name.
            metric: Metric to compare.
            horizon: Prediction horizon.
            n_bootstrap: Number of bootstrap samples.
            seed: Random seed.

        Returns:
            Dict with 'p_value', 'delta_mean', 'delta_ci_lower', 'delta_ci_upper'.
        """
        key = f"{metric}_h{horizon}"

        # Need raw predictions for proper bootstrap
        if model_a in self.raw_predictions and model_b in self.raw_predictions:
            return self._bootstrap_metric_test(
                model_a, model_b, metric, horizon, n_bootstrap, seed
            )

        # Fallback: report point estimate difference only
        val_a = self.results.get(model_a, {}).get(key, self.results.get(model_a, {}).get(metric, 0))
        val_b = self.results.get(model_b, {}).get(key, self.results.get(model_b, {}).get(metric, 0))
        delta = val_a - val_b

        return {
            "delta_mean": float(delta),
            "p_value": None,  # Cannot compute without raw predictions
            "note": "Point estimate only — provide raw predictions for bootstrap test",
        }

    def _bootstrap_metric_test(
        self,
        model_a: str,
        model_b: str,
        metric: str,
        horizon: int,
        n_bootstrap: int,
        seed: int,
    ) -> Dict[str, float]:
        """Bootstrap-based significance test using raw predictions."""
        from sklearn.metrics import roc_auc_score, f1_score
        rng = np.random.default_rng(seed)

        preds_a = self.raw_predictions[model_a]
        preds_b = self.raw_predictions[model_b]

        y_true = preds_a.get(f"y_true_h{horizon}", preds_a.get("y_true"))
        probs_a = preds_a.get(f"probs_h{horizon}", preds_a.get("probs"))
        probs_b = preds_b.get(f"probs_h{horizon}", preds_b.get("probs"))

        if y_true is None or probs_a is None or probs_b is None:
            return {"p_value": None, "note": "Missing raw prediction arrays"}

        N = len(y_true)
        deltas = []

        for _ in range(n_bootstrap):
            idx = rng.choice(N, size=N, replace=True)
            y_b = y_true[idx]
            p_a = probs_a[idx]
            p_b = probs_b[idx]

            if len(np.unique(y_b)) < 2:
                continue

            if metric == "auroc":
                score_a = roc_auc_score(y_b, p_a)
                score_b = roc_auc_score(y_b, p_b)
            elif metric == "f1":
                score_a = f1_score(y_b, (p_a > 0.5).astype(int), zero_division=0)
                score_b = f1_score(y_b, (p_b > 0.5).astype(int), zero_division=0)
            else:
                score_a = roc_auc_score(y_b, p_a)
                score_b = roc_auc_score(y_b, p_b)

            deltas.append(score_a - score_b)

        deltas = np.array(deltas)
        p_value = float((deltas <= 0).mean())  # One-sided: P(delta <= 0)

        return {
            "delta_mean": float(deltas.mean()),
            "delta_std": float(deltas.std()),
            "delta_ci_lower": float(np.percentile(deltas, 2.5)),
            "delta_ci_upper": float(np.percentile(deltas, 97.5)),
            "p_value": p_value,
            "significant_p05": bool(p_value < 0.05),
        }

    def save_all(self, output_dir: str) -> None:
        """
        Save all comparison tables and reports.

        Args:
            output_dir: Directory to save outputs.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Main results tables per horizon
        for h in self.horizons:
            df = self.generate_main_table(horizon=h)
            if not df.empty:
                df.round(3).to_csv(out / f"results_h{h}.csv")
                latex = self.generate_latex_table(horizon=h)
                (out / f"results_h{h}.tex").write_text(latex)

        # Multi-horizon AUROC table
        mh_df = self.generate_multi_horizon_table()
        if not mh_df.empty:
            mh_df.to_csv(out / "multi_horizon_auroc.csv")
            mh_df.round(3).to_latex(
                out / "multi_horizon_auroc.tex",
                caption="AUROC across prediction horizons",
                label="tab:multi_horizon",
            )

        # JSON dump of all results
        with open(out / "all_results.json", "w") as f:
            json.dump(
                {k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, (int, float, np.floating))}
                 for k, v in self.results.items()},
                f, indent=2
            )

        logger.info(f"All comparison tables saved to {out}")

    def print_summary(self) -> None:
        """Print a clean summary of all model results to stdout."""
        if not self.results:
            print("No results registered yet.")
            return

        print("\n" + "="*70)
        print("TIDAL BENCHMARK COMPARISON SUMMARY")
        print("="*70)

        df = self.generate_multi_horizon_table()
        if not df.empty:
            print(df.to_string())

        print("\n" + "="*70)


def main():
    """CLI entrypoint for benchmark comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="TIDAL Benchmark Comparison")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_dir", default="results/tables", help="Output directory")
    args = parser.parse_args()

    # Load saved results
    results_path = Path(args.results_dir)
    compare = BenchmarkComparison()

    for result_file in results_path.glob("**/metrics.json"):
        model_name = result_file.parent.name
        with open(result_file) as f:
            metrics = json.load(f)
        compare.add_result(model_name, metrics)

    compare.print_summary()
    compare.save_all(args.output_dir)


if __name__ == "__main__":
    main()
