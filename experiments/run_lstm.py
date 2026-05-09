"""
experiments/run_lstm.py
────────────────────────
LSTM baseline experiment runner.

Trains the bidirectional LSTM baseline on the same data pipeline
as TIDAL for a fair comparison. Uses the lstm.yaml config merged
over default.yaml.

Usage
-----
    python experiments/run_lstm.py
    python experiments/run_lstm.py --config configs/lstm.yaml training.epochs=80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from utils.config import get_device, get_output_dir, load_config, save_config
from utils.logger import get_logger, log_config, setup_logger
from utils.seed import set_seed

log = get_logger(__name__)


def _make_synthetic_data(cfg):
    """Shared synthetic data generator (mirrors run_tidal.py)."""
    rng = np.random.default_rng(cfg.experiment.seed)
    seq_len   = cfg.data.seq_len
    input_dim = cfg.model.input_dim
    horizons  = list(cfg.data.horizons)
    H         = len(horizons)
    total     = 20000

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


def run_lstm(cfg) -> dict:
    """Full LSTM baseline training + evaluation."""
    from models.lstm import LSTMModel
    from preprocessing.sequence_builder import make_dataloaders
    from training.trainer import Trainer
    from evaluation.metrics import full_evaluation_report
    from evaluation.early_warning import multi_horizon_early_warning

    log.info("=== LSTM Baseline Experiment ===")
    X_train, y_train, X_val, y_val, X_test, y_test = _make_synthetic_data(cfg)

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg, seed=cfg.experiment.seed
    )

    model = LSTMModel(cfg)
    log.info(f"LSTM params: {model.count_parameters():,}")

    flat    = y_train[y_train >= 0].flatten()
    nc      = cfg.model.num_classes
    weights = np.array([len(flat) / (nc * max((flat == c).sum(), 1)) for c in range(nc)], dtype=np.float32)
    weights /= weights.mean()
    cw = torch.tensor(weights)

    trainer = Trainer(model, cfg, class_weights=cw)
    history = trainer.fit(train_loader, val_loader)

    eval_result = trainer.evaluate(test_loader, load_best=True)
    probs   = eval_result["probs"].numpy()
    targets = eval_result["targets"].numpy()
    scores  = eval_result["scores"].numpy()

    horizons    = list(cfg.data.horizons)
    full_report = full_evaluation_report(probs, targets, scores, horizons,
                                         threshold=cfg.evaluation.threshold,
                                         lead_time_window=cfg.evaluation.lead_time_window,
                                         n_bootstrap=cfg.evaluation.bootstrap_n,
                                         num_classes=cfg.model.num_classes)
    ew_report   = multi_horizon_early_warning(scores, targets, horizons,
                                              threshold=cfg.evaluation.threshold,
                                              lead_time_window=cfg.evaluation.lead_time_window)

    out_dir = get_output_dir(cfg)
    save_config(cfg, out_dir / "config.yaml")

    def _jsonify(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.floating, float)): return round(float(obj), 6) if not np.isnan(obj) else None
        if isinstance(obj, (np.integer, int)): return int(obj)
        return obj

    with open(out_dir / "full_evaluation.json", "w") as f:
        json.dump(_jsonify(full_report), f, indent=2)
    with open(out_dir / "early_warning.json", "w") as f:
        json.dump(_jsonify(ew_report), f, indent=2)

    ev = eval_result["metrics"]
    log.info(f"LSTM | AUROC={ev.get('auroc',0):.4f} | AUPRC={ev.get('auprc',0):.4f} | F1={ev.get('macro_f1',0):.4f}")
    return {"history": history, "evaluation": ev, "full_report": full_report, "out_dir": str(out_dir)}


def _parse_args():
    parser = argparse.ArgumentParser(description="LSTM baseline runner")
    parser.add_argument("--config", default="configs/default.yaml")
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    args, overrides = _parse_args()
    cfg = load_config(args.config, overrides=["model.name=lstm"] + (overrides or []))
    setup_logger(cfg.experiment.log_dir, cfg.experiment.name + "_lstm",
                 cfg.logging.level, cfg.logging.rich_console)
    set_seed(cfg.experiment.seed, cfg.reproducibility.deterministic)
    run_lstm(cfg)


if __name__ == "__main__":
    main()
