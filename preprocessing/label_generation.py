"""
preprocessing/label_generation.py
───────────────────────────────────
Instability label generation pipeline for TIDAL.

Defines financial market instability through a composite of:
    1. Volatility spikes
    2. Spread widening
    3. Liquidity stress (LOB depth reduction)
    4. Order imbalance surges

Labels are generated for multiple prediction horizons (10, 30, 60 steps).
Supports both binary and multi-class instability labels.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class InstabilityThresholds:
    """
    Configurable thresholds for instability detection.

    All thresholds are applied to rolling-normalized metrics.
    """
    volatility_spike: float = 2.0      # Std deviations above rolling mean
    spread_widening: float = 1.5       # Ratio to rolling median spread
    liquidity_stress: float = 0.3      # Fractional LOB depth drop
    order_imbalance: float = 0.7       # Absolute imbalance threshold
    min_duration: int = 3              # Min consecutive steps for instability
    smoothing_window: int = 5          # Label smoothing window


class InstabilityLabelGenerator:
    """
    Generates binary and horizon-specific instability labels.

    The instability signal is computed from a union of market stress indicators.
    Future-horizon labels are created by looking forward in the label sequence.

    Usage:
        gen = InstabilityLabelGenerator(thresholds=InstabilityThresholds())
        labels = gen.generate(features_df, horizons=[10, 30, 60])
    """

    def __init__(
        self,
        thresholds: Optional[InstabilityThresholds] = None,
        rolling_window: int = 50,
    ):
        """
        Initialize label generator.

        Args:
            thresholds: Instability threshold configuration.
            rolling_window: Window for rolling baseline statistics.
        """
        self.thresholds = thresholds or InstabilityThresholds()
        self.rolling_window = rolling_window

    def generate(
        self,
        features: np.ndarray,
        feature_names: List[str],
        horizons: List[int] = [10, 30, 60],
    ) -> Dict[str, np.ndarray]:
        """
        Generate instability labels at multiple prediction horizons.

        Args:
            features: Feature array (T, n_features).
            feature_names: List of feature names (for column lookup).
            horizons: List of forward prediction horizons (steps).

        Returns:
            Dictionary mapping label name → binary array (T,).
            Keys: 'instability_now', 'instability_h10', 'instability_h30', etc.
        """
        T = len(features)
        df = pd.DataFrame(features, columns=feature_names)

        logger.info(f"Generating instability labels: T={T}, horizons={horizons}")

        # ── Compute individual instability signals ──────────────────────────
        vol_signal    = self._volatility_signal(df)
        spread_signal = self._spread_signal(df)
        depth_signal  = self._depth_signal(df)
        imb_signal    = self._imbalance_signal(df)

        # ── Composite instability: any signal firing ────────────────────────
        raw_instability = (
            vol_signal.astype(int)
            + spread_signal.astype(int)
            + depth_signal.astype(int)
            + imb_signal.astype(int)
        ) >= 2  # At least 2 signals must fire

        # ── Apply minimum duration filter ───────────────────────────────────
        instability_now = self._apply_duration_filter(
            raw_instability.values, self.thresholds.min_duration
        )

        # ── Smooth labels to reduce noise ───────────────────────────────────
        instability_now = self._smooth_labels(
            instability_now, self.thresholds.smoothing_window
        )

        # ── Record individual signal components ─────────────────────────────
        labels = {
            "instability_now": instability_now,
            "vol_signal": vol_signal.values.astype(int),
            "spread_signal": spread_signal.values.astype(int),
            "depth_signal": depth_signal.values.astype(int),
            "imbalance_signal": imb_signal.values.astype(int),
        }

        # ── Generate forward-horizon labels ─────────────────────────────────
        for h in horizons:
            horizon_label = self._future_horizon_label(instability_now, horizon=h)
            labels[f"instability_h{h}"] = horizon_label

        # ── Log class statistics ─────────────────────────────────────────────
        n_unstable = instability_now.sum()
        pct = 100 * n_unstable / T
        logger.info(f"Instability: {n_unstable}/{T} steps ({pct:.1f}%) labeled as unstable")

        for h in horizons:
            k = f"instability_h{h}"
            h_pct = 100 * labels[k].sum() / T
            logger.info(f"  Horizon h={h}: {labels[k].sum()} ({h_pct:.1f}%) unstable")

        return labels

    def generate_multiclass(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Generate 3-class regime labels: 0=Stable, 1=Transitional, 2=Unstable.

        Args:
            features: Feature array (T, n_features).
            feature_names: Feature name list.

        Returns:
            Array of shape (T,) with values {0, 1, 2}.
        """
        df = pd.DataFrame(features, columns=feature_names)

        vol_signal    = self._volatility_signal(df)
        spread_signal = self._spread_signal(df)
        depth_signal  = self._depth_signal(df)
        imb_signal    = self._imbalance_signal(df)

        score = (
            vol_signal.astype(int)
            + spread_signal.astype(int)
            + depth_signal.astype(int)
            + imb_signal.astype(int)
        )

        regimes = np.zeros(len(features), dtype=int)
        regimes[score == 1] = 1  # Transitional
        regimes[score >= 2] = 2  # Unstable

        logger.info(
            f"Regime distribution: "
            f"Stable={( regimes==0).sum()} | "
            f"Transitional={(regimes==1).sum()} | "
            f"Unstable={(regimes==2).sum()}"
        )
        return regimes

    def get_transition_points(self, labels: np.ndarray) -> np.ndarray:
        """
        Identify transition points where the regime label changes.

        Args:
            labels: Label array (T,).

        Returns:
            Boolean array of shape (T,), True at transition steps.
        """
        transitions = np.zeros(len(labels), dtype=bool)
        transitions[1:] = labels[1:] != labels[:-1]
        return transitions

    # ── Private signal methods ───────────────────────────────────────────────

    def _volatility_signal(self, df: pd.DataFrame) -> pd.Series:
        """Detect volatility spikes relative to rolling baseline."""
        col = self._find_col(df, ["vol_rolling_10", "vol_rolling_5", "log_return_1"])
        if col is None:
            logger.warning("No volatility feature found. Generating from mid_price.")
            mid = self._find_col_series(df, ["mid_price", "log_mid_price"])
            col_series = mid.pct_change().abs().fillna(0)
        else:
            col_series = df[col]

        rolling_mean = col_series.rolling(self.rolling_window, min_periods=1).mean()
        rolling_std  = col_series.rolling(self.rolling_window, min_periods=1).std().fillna(1e-8)
        z_score = (col_series - rolling_mean) / (rolling_std + 1e-8)
        return z_score > self.thresholds.volatility_spike

    def _spread_signal(self, df: pd.DataFrame) -> pd.Series:
        """Detect spread widening beyond rolling median."""
        col = self._find_col(df, ["spread_L1", "rel_spread_L1", "spread"])
        if col is None:
            return pd.Series(np.zeros(len(df), dtype=bool))
        spread = df[col]
        rolling_median = spread.rolling(self.rolling_window, min_periods=1).median()
        ratio = spread / (rolling_median + 1e-8)
        return ratio > self.thresholds.spread_widening

    def _depth_signal(self, df: pd.DataFrame) -> pd.Series:
        """Detect LOB depth collapse (liquidity stress)."""
        col = self._find_col(df, ["total_depth", "total_bid_depth", "total_ask_depth"])
        if col is None:
            return pd.Series(np.zeros(len(df), dtype=bool))
        depth = df[col]
        rolling_max = depth.rolling(self.rolling_window, min_periods=1).max()
        drop_fraction = 1.0 - (depth / (rolling_max + 1e-8))
        return drop_fraction > self.thresholds.liquidity_stress

    def _imbalance_signal(self, df: pd.DataFrame) -> pd.Series:
        """Detect extreme order imbalance."""
        col = self._find_col(df, ["weighted_imbalance", "mean_imbalance", "depth_imbalance"])
        if col is None:
            return pd.Series(np.zeros(len(df), dtype=bool))
        imbalance = df[col].abs()
        return imbalance > self.thresholds.order_imbalance

    def _apply_duration_filter(self, labels: np.ndarray, min_duration: int) -> np.ndarray:
        """
        Remove short instability episodes below minimum duration.

        Args:
            labels: Binary label array.
            min_duration: Minimum consecutive instability steps.

        Returns:
            Filtered binary label array.
        """
        filtered = labels.copy()
        i = 0
        while i < len(labels):
            if labels[i]:
                j = i
                while j < len(labels) and labels[j]:
                    j += 1
                if j - i < min_duration:
                    filtered[i:j] = False
                i = j
            else:
                i += 1
        return filtered

    def _smooth_labels(self, labels: np.ndarray, window: int) -> np.ndarray:
        """
        Apply majority-vote smoothing over a sliding window.

        Args:
            labels: Binary label array.
            window: Smoothing window size.

        Returns:
            Smoothed binary label array.
        """
        if window <= 1:
            return labels
        smoothed = pd.Series(labels.astype(float)).rolling(
            window, center=True, min_periods=1
        ).mean().values
        return (smoothed >= 0.5).astype(int)

    def _future_horizon_label(self, labels: np.ndarray, horizon: int) -> np.ndarray:
        """
        Create forward-looking labels: was there instability within next `horizon` steps?

        Args:
            labels: Current-time instability labels.
            horizon: Number of steps to look ahead.

        Returns:
            Binary array where 1 = instability occurs within next `horizon` steps.
        """
        T = len(labels)
        future_labels = np.zeros(T, dtype=int)
        for t in range(T - horizon):
            if labels[t + 1: t + horizon + 1].any():
                future_labels[t] = 1
        # Last `horizon` steps cannot be labeled — mark as 0
        future_labels[T - horizon:] = 0
        return future_labels

    def _find_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _find_col_series(self, df: pd.DataFrame, candidates: List[str]) -> pd.Series:
        """Return first matching column series."""
        col = self._find_col(df, candidates)
        if col:
            return df[col]
        return pd.Series(np.zeros(len(df)))
