"""
experiments/run_tidal.py
─────────────────────────
Main experiment runner for the TIDAL instability surveillance model.

Usage
-----
    python experiments/run_tidal.py
    python experiments/run_tidal.py --config configs/default.yaml
    python experiments/run_tidal.py --config configs/default.yaml model.hidden_dim=256
    python experiments/run_tidal.py --config configs/default.yaml training.epochs=100

All hyper-parameters are driven by OmegaConf config files.
CLI overrides use dot-notation: ``key=value``.

Pipeline
--------
1. Load config + set seed
2. Load synthetic/real data (FI-2010 or Crypto)
3. Run preprocessing: feature engineering + label generation + sequence building
4. Build DataLoaders
5. Instantiate TIDALModel
6. Train with Trainer (checkpointing, early stopping, TensorBoard)
7. Evaluate on test set (AUROC, AUPRC, F1, ECE, lead-time)
8. Run transition analysis
9. Run early-warning analysis
10. Save all results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- make sure project root is on sys.path ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.config import get_device, get_output_dir, load_config, save_config
from utils.logger import get_logger, log_config, setup_logger
from utils.seed import set_seed

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data helpers (synthetic fallback for reproducible runs without raw files)
# ---------------------------------------------------------------------------

def _make_synthetic_data(cfg) -> tuple:
    """Generate synthetic LOB data for reproducible experiments.

    Returns train/val/test numpy arrays (X, y) matching the pipeline API.
    Shape: X (N, seq_len, input_dim), y (N, H).
    """
    rng = np.random.default_rng(cfg.experiment.seed)
    seq_len   = cfg.data.seq_len
    input_dim = cfg.model.input_dim
    horizons  = list(cfg.data.horizons)
    H         = len(horizons)

    total = 20000
    X_all = rng.standard_normal((total, seq_len, input_dim)).astype(np.float32)

    # Synthetic three-regime labels with temporal structure
    t   = np.linspace(0, 4 * np.pi, total)
    idx = (np.sin(t) * 0.5 + rng.standard_normal(total) * 0.3 + 0.5).clip(0, 0.99)
    regime = np.zeros(total, dtype=np.int64)
    regime[idx > 0.66] = 2
    regime[(idx > 0.33) & (idx <= 0.66)] = 1

    # Multi-horizon labels: forward max regime
    y_all = np.zeros((total, H), dtype=np.int64)
    for h_idx, H_val in enumerate(horizons):
        for t_i in range(total - H_val):
            y_all[t_i, h_idx] = int(regime[t_i + 1: t_i + H_val + 1].max())
        y_all[total - H_val:, h_idx] = -1  # mask boundary

    # Keep only valid (non-masked) rows
    valid = np.all(y_all >= 0, axis=1)
    X_all = X_all[valid]
    y_all = y_all[valid]
    N = len(X_all)

    # Split
    n_train = int(N * cfg.data.train_frac)
    n_val   = int(N * cfg.data.val_frac)

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val,   y_val   = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
    X_test,  y_test  = X_all[n_train + n_val:], y_all[n_train + n_val:]

    log.info(
        f"Synthetic data | train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _compute_class_weights(y_train: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights from training labels."""
    flat = y_train[y_train >= 0].flatten()
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n_c = (flat == c).sum()
        if n_c > 0:
            weights[c] = len(flat) / (num_classes * n_c)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_tidal(cfg) -> dict:
    """Full TIDAL training + evaluation pipeline."""
    from models.tidal import TIDALModel
    from preprocessing.sequence_builder import make_dataloaders
    from training.trainer import Trainer
    from evaluation.metrics import full_evaluation_report
    from evaluation.early_warning import multi_horizon_early_warning
    from evaluation.transition_analysis import full_transition_report
    import torch.nn.functional as F

    device = get_device(cfg)
    log.info(f"Device: {device}")

    # 1. Data
    log.info("Generating synthetic data ...")
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data(cfg)

    # 2. DataLoaders
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg, seed=cfg.experiment.seed
    )

    # 3. Model
    model = TIDALModel(cfg)
    log.info(f"Model: {model}")

    # 4. Class weights
    class_weights = _compute_class_weights(y_train, cfg.model.num_classes)

    # 5. Train
    trainer = Trainer(model, cfg, class_weights=class_weights)
    history = trainer.fit(train_loader, val_loader)
    log.info(f"Training complete. Best epoch: {history['best_epoch']}")

    # 6. Evaluate
    eval_result = trainer.evaluate(test_loader, load_best=True)
    probs   = eval_result["probs"].numpy()    # (N, H, C)
    targets = eval_result["targets"].numpy()  # (N, H)
    scores  = eval_result["scores"].numpy()   # (N, H)

    horizons = list(cfg.data.horizons)
    full_report = full_evaluation_report(
        probs, targets, scores, horizons,
        threshold         = cfg.evaluation.threshold,
        lead_time_window  = cfg.evaluation.lead_time_window,
        n_bootstrap       = cfg.evaluation.bootstrap_n,
        num_classes       = cfg.model.num_classes,
    )

    # 7. Early warning analysis
    ew_report = multi_horizon_early_warning(
        scores, targets, horizons,
        threshold        = cfg.evaluation.threshold,
        lead_time_window = cfg.evaluation.lead_time_window,
    )

    # 8. Transition analysis
    pred_labels = probs.argmax(axis=-1)  # (N, H)
    tr_report   = full_transition_report(
        probs, pred_labels, targets, horizons,
        margin      = cfg.evaluation.transition_margin,
        num_classes = cfg.model.num_classes,
    )

    # 9. Save
    out_dir = get_output_dir(cfg)
    save_config(cfg, out_dir / "config.yaml")

    def _jsonify(obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 6) if not np.isnan(obj) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(out_dir / "full_evaluation.json", "w") as f:
        json.dump(_jsonify(full_report), f, indent=2)
    with open(out_dir / "early_warning.json", "w") as f:
        json.dump(_jsonify(ew_report), f, indent=2)
    with open(out_dir / "transition_analysis.json", "w") as f:
        json.dump(_jsonify(tr_report), f, indent=2)

    log.info(f"All results saved to: {out_dir}")

    return {
        "history":      history,
        "evaluation":   eval_result["metrics"],
        "full_report":  full_report,
        "ew_report":    ew_report,
        "tr_report":    tr_report,
        "out_dir":      str(out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="TIDAL experiment runner")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    args, overrides = _parse_args()
    cfg = load_config(args.config, overrides=overrides if overrides else None)

    setup_logger(
        log_dir         = cfg.experiment.log_dir,
        experiment_name = cfg.experiment.name,
        level           = cfg.logging.level,
        rich_console    = cfg.logging.rich_console,
    )

    log.info(f"TIDAL Experiment: {cfg.experiment.name}")
    log_config(cfg, log)

    set_seed(cfg.experiment.seed, deterministic=cfg.reproducibility.deterministic)

    results = run_tidal(cfg)

    # Print summary
    ev = results["evaluation"]
    log.info("=" * 60)
    log.info("TIDAL EXPERIMENT SUMMARY")
    log.info(f"  AUROC      : {ev.get('auroc', 0):.4f}")
    log.info(f"  AUPRC      : {ev.get('auprc', 0):.4f}")
    log.info(f"  Macro-F1   : {ev.get('macro_f1', 0):.4f}")
    log.info(f"  Output dir : {results['out_dir']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
