"""
preprocessing/feature_engineering.py
--------------------------------------
Limit Order Book (LOB) feature extraction for TIDAL.

Produces a feature DataFrame consumed by label_generation.py and
sequence_builder.py. All features are computed causally (no look-ahead)
and documented with economic interpretations relevant to instability
surveillance.

Feature groups
--------------
1. Price features      — mid-price, log-returns, price impact
2. Spread features     — bid-ask spread (absolute and relative)
3. Depth features      — visible order book depth per level and total
4. Imbalance features  — bid vs ask volume pressure per level
5. Microprice          — depth-weighted fair value
6. VWAP deviation      — volume-weighted vs mid spread
7. Velocity features   — first differences of the above (rate of change)

Input
-----
Raw LOB array (FI-2010 format): shape (T, 40)
    Columns: [ask_p1, ask_v1, bid_p1, bid_v1, ..., ask_p10, ask_v10, bid_p10, bid_v10]

Output
------
pd.DataFrame with named columns including all label_generation.py requisites:
    mid_price, best_ask, best_bid, bid_depth, ask_depth, bid_vol, ask_vol
    + enriched microstructure features.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column naming convention for FI-2010 LOB
# ---------------------------------------------------------------------------

def _fi2010_column_names(n_levels: int = 10) -> List[str]:
    """Return ordered column names matching FI-2010 data layout."""
    cols = []
    for i in range(1, n_levels + 1):
        cols += [f"ask_p{i}", f"ask_v{i}", f"bid_p{i}", f"bid_v{i}"]
    return cols


# ---------------------------------------------------------------------------
# Main feature engineering class
# ---------------------------------------------------------------------------

class LOBFeatureEngineer:
    """Extract microstructure surveillance features from raw LOB snapshots.

    Parameters
    ----------
    n_levels:
        Number of price levels to use (FI-2010 provides 10).
    velocity_lags:
        List of lags (in steps) for velocity (first-difference) features.
    ewm_spans:
        EWM spans for smoothed versions of key features.
    """

    def __init__(
        self,
        n_levels: int = 10,
        velocity_lags: Optional[List[int]] = None,
        ewm_spans: Optional[List[int]] = None,
    ):
        self.n_levels = n_levels
        self.velocity_lags = velocity_lags or [1, 5]
        self.ewm_spans = ewm_spans or [5, 20]

    def transform(self, raw: np.ndarray) -> pd.DataFrame:
        """Compute all LOB features from raw snapshot array.

        Parameters
        ----------
        raw:
            Array of shape (T, 4·n_levels) in FI-2010 column order.

        Returns
        -------
        pd.DataFrame
            Feature matrix with named columns. Shape (T, F).
        """
        T = raw.shape[0]
        cols = _fi2010_column_names(self.n_levels)
        df = pd.DataFrame(raw, columns=cols)

        features: dict[str, np.ndarray] = {}

        # ── Level prices and volumes ──────────────────────────────────────
        ask_prices = np.stack([df[f"ask_p{i}"].values for i in range(1, self.n_levels + 1)], axis=1)
        ask_vols   = np.stack([df[f"ask_v{i}"].values for i in range(1, self.n_levels + 1)], axis=1)
        bid_prices = np.stack([df[f"bid_p{i}"].values for i in range(1, self.n_levels + 1)], axis=1)
        bid_vols   = np.stack([df[f"bid_v{i}"].values for i in range(1, self.n_levels + 1)], axis=1)

        # ── 1. Price features ─────────────────────────────────────────────
        best_ask = ask_prices[:, 0]
        best_bid = bid_prices[:, 0]
        mid_price = (best_ask + best_bid) / 2.0
        log_return = np.diff(np.log(np.clip(mid_price, 1e-8, None)), prepend=np.nan)

        features["best_ask"]  = best_ask
        features["best_bid"]  = best_bid
        features["mid_price"] = mid_price
        features["log_return"] = log_return

        # Price impact proxy: (ask5 - bid5) / mid  (outer spread)
        features["outer_spread"] = (ask_prices[:, 4] - bid_prices[:, 4]) / np.clip(mid_price, 1e-8, None)

        # ── 2. Spread features ────────────────────────────────────────────
        spread_abs = best_ask - best_bid
        spread_rel = spread_abs / np.clip(mid_price, 1e-8, None)
        features["spread_abs"] = spread_abs
        features["spread_rel"] = spread_rel

        for i in range(1, self.n_levels + 1):
            lev_spread = ask_prices[:, i - 1] - bid_prices[:, i - 1]
            features[f"spread_l{i}"] = lev_spread / np.clip(mid_price, 1e-8, None)

        # ── 3. Depth features ─────────────────────────────────────────────
        total_ask_depth = ask_vols.sum(axis=1)
        total_bid_depth = bid_vols.sum(axis=1)
        features["ask_depth"] = total_ask_depth
        features["bid_depth"] = total_bid_depth
        features["ask_vol"]   = total_ask_depth   # alias for label_generation
        features["bid_vol"]   = total_bid_depth   # alias for label_generation

        for i in range(1, self.n_levels + 1):
            features[f"ask_depth_l{i}"] = ask_vols[:, i - 1]
            features[f"bid_depth_l{i}"] = bid_vols[:, i - 1]

        # Depth decay: depth concentration at top vs total
        features["ask_depth_top3"] = ask_vols[:, :3].sum(axis=1) / np.clip(total_ask_depth, 1e-8, None)
        features["bid_depth_top3"] = bid_vols[:, :3].sum(axis=1) / np.clip(total_bid_depth, 1e-8, None)

        # ── 4. Imbalance features ─────────────────────────────────────────
        total_vol = np.clip(total_ask_depth + total_bid_depth, 1e-8, None)
        ofi = (total_bid_depth - total_ask_depth) / total_vol  # in [-1, 1]
        features["order_flow_imbalance"] = ofi

        for i in range(1, self.n_levels + 1):
            lev_vol = np.clip(ask_vols[:, i - 1] + bid_vols[:, i - 1], 1e-8, None)
            features[f"ofi_l{i}"] = (bid_vols[:, i - 1] - ask_vols[:, i - 1]) / lev_vol

        # Weighted imbalance: higher weight to top levels
        weights = np.array([1.0 / i for i in range(1, self.n_levels + 1)])
        weights /= weights.sum()
        bid_weighted = (bid_vols * weights).sum(axis=1)
        ask_weighted = (ask_vols * weights).sum(axis=1)
        features["weighted_ofi"] = (bid_weighted - ask_weighted) / np.clip(bid_weighted + ask_weighted, 1e-8, None)

        # ── 5. Microprice ─────────────────────────────────────────────────
        # Depth-weighted fair value: closer to ask if ask side is thinner
        denom = np.clip(total_bid_depth + total_ask_depth, 1e-8, None)
        microprice = (best_bid * total_ask_depth + best_ask * total_bid_depth) / denom
        features["microprice"] = microprice
        features["microprice_deviation"] = (microprice - mid_price) / np.clip(spread_abs, 1e-8, None)

        # ── 6. VWAP deviation ─────────────────────────────────────────────
        ask_vwap = (ask_prices * ask_vols).sum(axis=1) / np.clip(total_ask_depth, 1e-8, None)
        bid_vwap = (bid_prices * bid_vols).sum(axis=1) / np.clip(total_bid_depth, 1e-8, None)
        features["ask_vwap_dev"] = (ask_vwap - best_ask) / np.clip(mid_price, 1e-8, None)
        features["bid_vwap_dev"] = (best_bid - bid_vwap) / np.clip(mid_price, 1e-8, None)

        # ── Build DataFrame ───────────────────────────────────────────────
        feat_df = pd.DataFrame(features)

        # ── 7. Velocity features (causal first-differences) ───────────────
        base_cols = [
            "mid_price", "spread_rel", "order_flow_imbalance",
            "microprice_deviation", "ask_depth", "bid_depth",
        ]
        for lag in self.velocity_lags:
            for col in base_cols:
                if col in feat_df.columns:
                    feat_df[f"{col}_vel{lag}"] = feat_df[col].diff(lag)

        # ── 8. EWM-smoothed versions of key instability signals ───────────
        for span in self.ewm_spans:
            feat_df[f"ofi_ewm{span}"] = feat_df["order_flow_imbalance"].ewm(span=span, adjust=False).mean()
            feat_df[f"spread_ewm{span}"] = feat_df["spread_rel"].ewm(span=span, adjust=False).mean()

        return feat_df

    @property
    def feature_dim(self) -> int:
        """Approximate feature count (call after transform() for exact count)."""
        base = 5 + 1 + self.n_levels + 3 * self.n_levels + 7
        velocity = len(self.velocity_lags) * 6
        ewm = len(self.ewm_spans) * 2
        return base + velocity + ewm


# ---------------------------------------------------------------------------
# Normalisation (train-fold-only statistics)
# ---------------------------------------------------------------------------

class TemporalNormalizer:
    """Z-score normaliser that fits on training fold only.

    Prevents data leakage: statistics are computed on the training period
    and applied (without re-fitting) to validation and test folds.
    """

    def __init__(self) -> None:
        self._mean: Optional[pd.Series] = None
        self._std:  Optional[pd.Series] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "TemporalNormalizer":
        """Fit normalisation statistics from training features.

        Parameters
        ----------
        X:
            Training feature DataFrame. Must NOT include val/test data.
        """
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0, ddof=1).clip(lower=1e-8)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training-fold statistics to any fold."""
        if not self._fitted:
            raise RuntimeError("Call fit() on the training fold before transform().")
        return (X - self._mean) / self._std

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the training fold in one call."""
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        """Persist fit statistics for reproducibility."""
        import json
        stats = {
            "mean": self._mean.to_dict(),
            "std":  self._std.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str) -> "TemporalNormalizer":
        """Load previously fitted statistics."""
        import json
        with open(path) as f:
            stats = json.load(f)
        self._mean = pd.Series(stats["mean"])
        self._std  = pd.Series(stats["std"]).clip(lower=1e-8)
        self._fitted = True
        return self
