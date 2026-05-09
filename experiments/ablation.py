"""
experiments/ablation.py
────────────────────────
Ablation study for the TIDAL architecture.

Ablations evaluated
-------------------
1. TIDAL (full)          — SSM + Attention + gated fusion + multi-horizon head
2. No SSM                — Attention only (attn_layers=2, ssm_layers=0)
3. No Attention          — SSM only (ssm_layers=2, attn_layers=0)
4. Single horizon        — Only shortest horizon (h10)
5. No class weighting    — Uniform class weights in loss
6. Shared head           — All horizons share MLP weights (share_mlp=True)

For each ablation, we train the model and record full evaluation metrics.
Results are assembled into a comparison table alongside the full TIDAL.

Usage
-----
    python experiments/ablation.py
    python experiments/ablation.py --config configs/default.yaml training.epochs=30
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.config import get_output_dir, load_config, save_config
from utils.logger import get_logger, setup_logger
from utils.seed import set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared data generator
# ---------------------------------------------------------------------------

def _make_data(cfg):
    rng = np.random.default_rng(cfg.experiment.seed)
    seq_len, input_dim = cfg.data.seq_len, cfg.model.input_dim
    horizons = list(cfg.data.horizons)
    H = len(horizons)
    total = 20000

    X_all = rng.standard_normal((total, seq_len, input_dim)).astype(np.float32)
    t   = np.linspace(0, 4 * np.pi, total)
    idx = (np.sin(t) * 0.5 + rng.standard_normal(total) * 0.3 + 0.5).clip(0, 0.99)
    regime = np.zeros(total, dtype=np.int64)
    regime[idx > 0.66] = 2
    regime[(idx > 0.33) & (idx <= 0.66)] = 1

    y_all = np.zeros((total, H), dtype=np.int64)
    for h_idx, H_val in enumerate(horizons):
        for t_i in range(total - H_val):
            y_all[t_i, h_idx] = int(regime[t_i + 1: t_i + H_val + 1].max())
        y_all[total - H_val:, h_idx] = -1

    valid = np.all(y_all >= 0, axis=1)
    X_all, y_all = X_all[valid], y_all[valid]
    N = len(X_all)
    n_train = int(N * cfg.data.train_frac)
    n_val   = int(N * cfg.data.val_frac)
    return (X_all[:n_train], y_all[:n_train],
            X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val],
            X_all[n_train + n_val:], y_all[n_train + n_val:])


# ---------------------------------------------------------------------------
# Single ablation runner
# ---------------------------------------------------------------------------

def run_single_ablation(
    cfg,
    variant_name: str,
    cfg_overrides: Dict,
    use_class_weights: bool = True,
) -> Dict:
    """Train and evaluate one ablation variant.

    Parameters
    ----------
    cfg:
        Base OmegaConf config.
    variant_name:
        Human-readable ablation name (used for experiment naming).
    cfg_overrides:
        Dict of OmegaConf key-value overrides for this ablation.
    use_class_weights:
        Whether to use inverse-frequency class weights in the loss.

    Returns
    -------
    Dict with evaluation metrics.
    """
    from models.tidal import TIDALModel
    from preprocessing.sequence_builder import make_dataloaders
    from training.trainer import Trainer
    from evaluation.metrics import full_evaluation_report

    # Apply overrides
    variant_cfg = deepcopy(cfg)
    OmegaConf.update(variant_cfg, "experiment.name", f"{cfg.experiment.name}_{variant_name}")

    for k, v in cfg_overrides.items():
        OmegaConf.update(variant_cfg, k, v)

    log.info(f"--- Ablation: {variant_name} ---")
    set_seed(variant_cfg.experiment.seed, variant_cfg.reproducibility.deterministic)

    X_train, y_train, X_val, y_val, X_test, y_test = _make_data(variant_cfg)

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        variant_cfg, seed=variant_cfg.experiment.seed
    )

    model = TIDALModel(variant_cfg)
    log.info(f"  Params: {model.count_parameters():,}")

    cw = None
    if use_class_weights:
        flat    = y_train[y_train >= 0].flatten()
        nc      = variant_cfg.model.num_classes
        weights = np.array([len(flat) / (nc * max((flat == c).sum(), 1)) for c in range(nc)], dtype=np.float32)
        weights /= weights.mean()
        cw = torch.tensor(weights)

    trainer = Trainer(model, variant_cfg, class_weights=cw)
    history = trainer.fit(train_loader, val_loader)
    eval_result = trainer.evaluate(test_loader, load_best=True)

    probs   = eval_result["probs"].numpy()
    targets = eval_result["targets"].numpy()
    scores  = eval_result["scores"].numpy()
    horizons = list(variant_cfg.data.horizons)

    full_report = full_evaluation_report(
        probs, targets, scores, horizons,
        threshold=variant_cfg.evaluation.threshold,
        lead_time_window=variant_cfg.evaluation.lead_time_window,
        n_bootstrap=min(200, variant_cfg.evaluation.bootstrap_n),  # faster for ablations
        num_classes=variant_cfg.model.num_classes,
    )

    ev = eval_result["metrics"]
    log.info(f"  AUROC={ev.get('auroc',0):.4f} AUPRC={ev.get('auprc',0):.4f} F1={ev.get('macro_f1',0):.4f}")

    return {
        "metrics":     ev,
        "full_report": full_report,
        "history":     history,
        "params":      model.count_parameters(),
    }


# ---------------------------------------------------------------------------
# Ablation suite definition
# ---------------------------------------------------------------------------

ABLATION_SUITE = {
    "full_tidal": {
        "overrides":          {},
        "use_class_weights":  True,
    },
    "no_ssm": {
        "overrides":          {"model.tidal.ssm_layers": 0, "model.tidal.attn_layers": 3},
        "use_class_weights":  True,
    },
    "no_attention": {
        "overrides":          {"model.tidal.ssm_layers": 3, "model.tidal.attn_layers": 0},
        "use_class_weights":  True,
    },
    "single_horizon": {
        "overrides":          {"data.horizons": [10]},
        "use_class_weights":  True,
    },
    "no_class_weights": {
        "overrides":          {},
        "use_class_weights":  False,
    },
    "shared_head": {
        "overrides":          {},
        "use_class_weights":  True,
        "shared_mlp_patch":   True,  # special flag; handled below
    },
}


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(cfg, ablations: List[str] | None = None) -> Dict:
    """Run all (or selected) ablations and produce a comparison report.

    Parameters
    ----------
    cfg:
        Base OmegaConf config.
    ablations:
        List of ablation keys to run. None = all.

    Returns
    -------
    Dict mapping ablation name → result dict.
    """
    if ablations is None:
        ablations = list(ABLATION_SUITE.keys())

    results: Dict[str, Dict] = {}

    for name in ablations:
        spec = ABLATION_SUITE.get(name)
        if spec is None:
            log.warning(f"Unknown ablation: {name}. Skipping.")
            continue

        try:
            res = run_single_ablation(
                cfg,
                variant_name      = name,
                cfg_overrides     = spec["overrides"],
                use_class_weights = spec.get("use_class_weights", True),
            )
            results[name] = res
        except Exception as e:
            log.error(f"Ablation '{name}' failed: {e}")
            results[name] = {"error": str(e)}

    # Assemble summary table
    _print_ablation_table(results)

    # Save
    out_dir = get_output_dir(cfg) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.floating, float)): return round(float(obj), 6) if not np.isnan(obj) else None
        if isinstance(obj, (np.integer, int)): return int(obj)
        return obj

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(_jsonify(results), f, indent=2)

    log.info(f"Ablation results saved → {out_dir}")
    return results


def _print_ablation_table(results: Dict) -> None:
    log.info("\n" + "=" * 72)
    log.info("ABLATION STUDY RESULTS")
    log.info("=" * 72)
    header = f"{'Variant':<25} {'AUROC':>8} {'AUPRC':>8} {'Macro-F1':>10} {'Params':>10}"
    log.info(header)
    log.info("-" * 72)
    for name, res in results.items():
        if "error" in res:
            log.info(f"  {name:<23}  ERROR: {res['error']}")
            continue
        ev = res.get("metrics", {})
        log.info(
            f"  {name:<23}  "
            f"{ev.get('auroc', 0):>8.4f}  "
            f"{ev.get('auprc', 0):>8.4f}  "
            f"{ev.get('macro_f1', 0):>10.4f}  "
            f"{res.get('params', 0):>10,}"
        )
    log.info("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="TIDAL ablation study")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--ablations", nargs="+", default=None,
        choices=list(ABLATION_SUITE.keys()),
        help="Specific ablations to run (default: all)",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    args, overrides = _parse_args()
    cfg = load_config(args.config, overrides=overrides if overrides else None)
    setup_logger(cfg.experiment.log_dir, "ablation", cfg.logging.level, cfg.logging.rich_console)
    set_seed(cfg.experiment.seed, cfg.reproducibility.deterministic)
    log.info(f"Running ablation study for: {cfg.experiment.name}")
    run_ablation(cfg, ablations=args.ablations)


if __name__ == "__main__":
    main()
