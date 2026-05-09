"""
preprocessing/label_generation.py
----------------------------------
Composite Instability Index and three-regime label generation for TIDAL.

Scientific framing
------------------
Financial instability does not emerge instantaneously — it accumulates through
detectable microstructure deterioration before observable price disruption.
We model this via a composite Instability Index I_t that integrates four
complementary stress signals from the limit order book:

    I_t = α·V̂_t  +  β·Ŝ_t  +  γ·L̂_t  +  δ·Ô_t

where:
    V̂_t  = rolling realized volatility (normalized, window=vol_window)
    Ŝ_t  = spread stress: (spread_t - μ_spread) / σ_spread (rolling)
    L̂_t  = liquidity deterioration: 1 - depth_ratio_t  (normalized)
    Ô_t  = order imbalance persistence: EWM of |bid_vol - ask_vol| / total_vol

All components are individually z-scored using a **causal rolling window**
(no look-ahead) so that normalization statistics at time t depend only on
history [0, t). This is then min-max mapped to [0, 1] using the same causal
window.

Three regime labels are assigned per step:
    0 → Stable        (I_t < θ_low)
    1 → Transitional  (θ_low ≤ I_t < θ_high)   ← scientific novelty
    2 → Unstable      (I_t ≥ θ_high)

Thresholds θ_low and θ_high are adaptive: computed from a causal rolling
mean ± k·std of I_t, so they track the regime of the current period without
look-ahead.

Multi-horizon labeling
----------------------
For each target horizon H ∈ {10, 30, 60} steps, the label at time t captures
the maximum instability regime over the forward window [t+1, t+H]. This is
the "will instability emerge within H steps?" question.

IMPORTANT: The forward labeling window is only applied AFTER the train/val/test
split is known. Boundary steps where the look-ahead window crosses a fold
boundary are masked and excluded from training / evaluation.

Usage
-----
    from preprocessing.label_generation import InstabilityLabeler

    labeler = InstabilityLabeler(cfg)
    labels, index = labeler.fit_transform(features_df, fold="train")
    # labels shape: (T, len(horizons))
    # index: pd.Series of I_t values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Component computers
# ---------------------------------------------------------------------------

def _rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Causal (non-look-ahead) rolling z-score.

    At each step t, uses only values in [t-window, t-1] (exclusive of t)
    via ``shift(1)`` so that I_t is strictly causal.
    """
    mp = min_periods or max(2, window // 4)
    roll = series.shift(1).rolling(window=window, min_periods=mp)
    mu = roll.mean()
    sigma = roll.std(ddof=1).clip(lower=1e-8)
    return (series - mu) / sigma


def _causal_minmax(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Causal min-max normalisation to [0, 1] using a rolling window."""
    mp = min_periods or max(2, window // 4)
    shifted = series.shift(1)
    roll = shifted.rolling(window=window, min_periods=mp)
    lo = roll.min()
    hi = roll.max()
    rng = (hi - lo).clip(lower=1e-8)
    return ((series - lo) / rng).clip(0.0, 1.0)


def compute_realized_volatility(
    mid_price: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Causal rolling realized volatility from log mid-price returns.

    V_t = std of log-returns over the past `window` steps (causal).
    """
    log_ret = np.log(mid_price.clip(lower=1e-8)).diff()
    # shift(1) → strictly causal: today's vol uses yesterday's returns
    vol = log_ret.shift(1).rolling(window=window, min_periods=max(2, window // 4)).std(ddof=1)
    return vol.fillna(0.0)


def compute_spread_stress(
    best_ask: pd.Series,
    best_bid: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Causal rolling spread stress: normalised (ask - bid) / mid.

    Spread widens before instability events as market makers withdraw.
    """
    mid = (best_ask + best_bid) / 2.0
    spread = (best_ask - best_bid) / mid.clip(lower=1e-8)
    return _rolling_zscore(spread, window=window).clip(-3.0, 6.0)


def compute_liquidity_deterioration(
    bid_depth: pd.Series,
    ask_depth: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Causal liquidity deterioration index.

    Measures the decline in total visible LOB depth relative to its recent
    history. depth_ratio = total_depth_t / rolling_max_depth → deterioration
    = 1 - depth_ratio. High values indicate thinning order books.
    """
    total_depth = (bid_depth + ask_depth).clip(lower=1e-8)
    shifted = total_depth.shift(1)
    roll_max = shifted.rolling(window=window, min_periods=max(2, window // 4)).max().clip(lower=1e-8)
    depth_ratio = total_depth / roll_max
    deterioration = (1.0 - depth_ratio).clip(0.0, 1.0)
    return deterioration


def compute_order_imbalance_persistence(
    bid_vol: pd.Series,
    ask_vol: pd.Series,
    span: int = 20,
) -> pd.Series:
    """Exponentially weighted order imbalance persistence.

    OI_t = |bid_vol - ask_vol| / (bid_vol + ask_vol)
    Persistence = EWM(OI, span=span) to capture sustained one-sided pressure.
    Persistent imbalance is a leading indicator of directional liquidity stress.
    """
    total = (bid_vol + ask_vol).clip(lower=1e-8)
    imbalance = ((bid_vol - ask_vol) / total).abs()
    return imbalance.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Instability Index
# ---------------------------------------------------------------------------

@dataclass
class InstabilityLabeler:
    """Compute the composite Instability Index and derive regime labels.

    Parameters
    ----------
    cfg:
        OmegaConf config with ``instability`` section (see default.yaml).
    """

    cfg: DictConfig
    # Causal normalisation statistics computed during fit() (train fold only)
    _fitted: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        features: pd.DataFrame,
        fold: str = "train",
        horizon_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, pd.Series]:
        """Compute index and labels from feature DataFrame.

        Parameters
        ----------
        features:
            DataFrame produced by feature_engineering.py. Must contain columns:
            ``mid_price``, ``best_ask``, ``best_bid``,
            ``bid_depth``, ``ask_depth``, ``bid_vol``, ``ask_vol``.
        fold:
            "train" | "val" | "test". Determines whether thresholds are
            fitted here or applied from previously fitted values.
        horizon_mask:
            Boolean array of shape (T,). True = this step is near a fold
            boundary and its forward look-ahead would leak. These steps get
            label = -1 (excluded from training/eval).

        Returns
        -------
        labels:
            int8 array of shape (T, H) where H = len(horizons).
            Values: 0=Stable, 1=Transitional, 2=Unstable, -1=masked.
        index:
            pd.Series of I_t (composite instability index, [0, 1]).
        """
        self._check_columns(features)
        ic = self.cfg.instability

        # --- Compute raw components ---
        V = compute_realized_volatility(
            features["mid_price"], window=ic.vol_window
        )
        S = compute_spread_stress(
            features["best_ask"], features["best_bid"], window=ic.spread_window
        )
        L = compute_liquidity_deterioration(
            features["bid_depth"], features["ask_depth"], window=ic.liquidity_window
        )
        O = compute_order_imbalance_persistence(
            features["bid_vol"], features["ask_vol"], span=ic.imbalance_window
        )

        # --- Causal normalisation to [0, 1] ---
        norm_window = ic.get("threshold_window", 500)
        V_n = _causal_minmax(V.fillna(0.0), window=norm_window)
        S_n = _causal_minmax(S.fillna(0.0), window=norm_window)
        L_n = _causal_minmax(L.fillna(0.0), window=norm_window)
        O_n = _causal_minmax(O.fillna(0.0), window=norm_window)

        # --- Composite index ---
        α, β, γ, δ = (
            ic.weights.alpha,
            ic.weights.beta,
            ic.weights.gamma,
            ic.weights.delta,
        )
        I = (α * V_n + β * S_n + γ * L_n + δ * O_n).clip(0.0, 1.0)
        I.name = "instability_index"

        # --- Adaptive regime thresholds (causal) ---
        θ_low, θ_high = self._compute_thresholds(I)

        # --- Assign point-in-time regime labels ---
        regime_t = self._assign_regime(I, θ_low, θ_high)

        # --- Multi-horizon forward labels ---
        horizons: List[int] = list(self.cfg.data.horizons)
        T = len(features)
        labels = np.full((T, len(horizons)), fill_value=-1, dtype=np.int8)

        for h_idx, H in enumerate(horizons):
            for t in range(T - H):
                # Maximum regime in the forward window [t+1, t+H]
                labels[t, h_idx] = int(regime_t.iloc[t + 1 : t + H + 1].max())
            # Last H steps: forward window crosses the end — mask them
            labels[T - H :, h_idx] = -1

        # --- Apply boundary mask (fold crossings) ---
        if horizon_mask is not None:
            labels[horizon_mask] = -1

        self._fitted = True
        return labels, I

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_thresholds(
        self, I: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Causal adaptive thresholds from rolling mean ± k·std of I_t.

        For the first `threshold_window` steps where the rolling window is
        not fully populated, we fall back to the configurable fixed thresholds
        from ``regime_thresholds``.
        """
        ic = self.cfg.instability
        k = ic.threshold_k
        window = ic.threshold_window
        mp = max(10, window // 10)

        shifted = I.shift(1)
        roll = shifted.rolling(window=window, min_periods=mp)
        mu = roll.mean()
        sigma = roll.std(ddof=1).clip(lower=1e-8)

        θ_low_adaptive = (mu - k * sigma).clip(0.0, 1.0)
        θ_high_adaptive = (mu + k * sigma).clip(0.0, 1.0)

        # Fall back to fixed thresholds where rolling window is not yet full
        rt = ic.regime_thresholds
        θ_low = θ_low_adaptive.fillna(rt.stable_upper)
        θ_high = θ_high_adaptive.fillna(rt.unstable_lower)

        # Ensure θ_low < θ_high
        swap = θ_low >= θ_high
        θ_low[swap] = rt.stable_upper
        θ_high[swap] = rt.unstable_lower

        return θ_low, θ_high

    def _assign_regime(
        self,
        I: pd.Series,
        θ_low: pd.Series,
        θ_high: pd.Series,
    ) -> pd.Series:
        """Map index values to {0, 1, 2} using adaptive thresholds."""
        regime = pd.Series(0, index=I.index, dtype=np.int8)
        regime[I >= θ_high] = 2
        regime[(I >= θ_low) & (I < θ_high)] = 1
        return regime

    @staticmethod
    def _check_columns(df: pd.DataFrame) -> None:
        required = {
            "mid_price", "best_ask", "best_bid",
            "bid_depth", "ask_depth", "bid_vol", "ask_vol",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"features DataFrame is missing required columns: {sorted(missing)}\n"
                f"Available columns: {list(df.columns)}"
            )


# ---------------------------------------------------------------------------
# Utility: compute label class frequencies for loss weighting
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 3,
    method: str = "inverse_freq",
) -> np.ndarray:
    """Compute class weights for imbalanced instability labels.

    Parameters
    ----------
    labels:
        Flat or 2D array of integer labels. Masked values (-1) are excluded.
    num_classes:
        Number of target classes (3 for three-regime).
    method:
        "inverse_freq" — weight_c = N / (num_classes · N_c)
        "sqrt_inv"     — weight_c = sqrt(N / N_c)

    Returns
    -------
    weights:
        Float32 array of shape (num_classes,).
    """
    valid = labels[labels >= 0].flatten()
    N = len(valid)
    weights = np.ones(num_classes, dtype=np.float32)

    for c in range(num_classes):
        N_c = int((valid == c).sum())
        if N_c == 0:
            weights[c] = 1.0
            continue
        if method == "inverse_freq":
            weights[c] = N / (num_classes * N_c)
        elif method == "sqrt_inv":
            weights[c] = np.sqrt(N / N_c)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Normalise so mean weight = 1.0
    weights = weights / weights.mean()
    return weights
