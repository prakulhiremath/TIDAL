"""
preprocessing/feature_engineering.py
──────────────────────────────────────
LOB microstructure feature extraction for TIDAL.

Extracts a rich set of features from raw limit order book snapshots,
capturing the temporal microstructure signals relevant to instability detection.

Feature categories:
    1. Price features       — mid price, returns, momentum
    2. Spread features      — bid-ask spread, relative spread
    3. Depth features       — total volume, depth imbalance
    4. Imbalance features   — order flow imbalance, queue imbalance
    5. Volatility features  — realized vol, rolling std
    6. LOB shape features   — slope, convexity, weighted mid price
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from loguru import logger


class LOBFeatureEngineer:
    """
    Extracts market microstructure features from raw LOB snapshots.

    Designed for both FI-2010 and crypto order book formats.
    All features are computed in a vectorized, time-efficient manner.

    Usage:
        engineer = LOBFeatureEngineer(n_levels=10, rolling_windows=[5, 10, 30])
        features, names = engineer.transform(raw_lob_data)
    """

    def __init__(
        self,
        n_levels: int = 10,
        rolling_windows: List[int] = [5, 10, 20, 50],
        include_rolling: bool = True,
        include_lob_shape: bool = True,
    ):
        """
        Initialize feature engineer.

        Args:
            n_levels: Number of LOB price levels.
            rolling_windows: Window sizes for rolling statistics.
            include_rolling: Include rolling statistics features.
            include_lob_shape: Include LOB shape/slope features.
        """
        self.n_levels = n_levels
        self.rolling_windows = rolling_windows
        self.include_rolling = include_rolling
        self.include_lob_shape = include_lob_shape
        self._feature_names: Optional[List[str]] = None

    def transform(self, lob_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from raw LOB data.

        Args:
            lob_data: Raw LOB array of shape (T, n_levels * 4).
                      Column format per level: [ask_p, ask_v, bid_p, bid_v]

        Returns:
            Tuple of (features array (T, n_features), feature name list).
        """
        T = lob_data.shape[0]
        logger.info(f"Extracting features from LOB data: shape={lob_data.shape}")

        # ── Parse LOB structure ─────────────────────────────────────────────
        ask_prices, ask_vols, bid_prices, bid_vols = self._parse_lob(lob_data)

        feature_blocks: List[np.ndarray] = []
        name_blocks: List[List[str]] = []

        # ── 1. Price features ───────────────────────────────────────────────
        price_feats, price_names = self._price_features(ask_prices, bid_prices)
        feature_blocks.append(price_feats)
        name_blocks.append(price_names)

        # ── 2. Spread features ──────────────────────────────────────────────
        spread_feats, spread_names = self._spread_features(ask_prices, bid_prices)
        feature_blocks.append(spread_feats)
        name_blocks.append(spread_names)

        # ── 3. Depth / volume features ──────────────────────────────────────
        depth_feats, depth_names = self._depth_features(ask_vols, bid_vols)
        feature_blocks.append(depth_feats)
        name_blocks.append(depth_names)

        # ── 4. Order imbalance features ─────────────────────────────────────
        imb_feats, imb_names = self._imbalance_features(ask_prices, ask_vols, bid_prices, bid_vols)
        feature_blocks.append(imb_feats)
        name_blocks.append(imb_names)

        # ── 5. LOB shape features ───────────────────────────────────────────
        if self.include_lob_shape:
            shape_feats, shape_names = self._lob_shape_features(ask_prices, ask_vols, bid_prices, bid_vols)
            feature_blocks.append(shape_feats)
            name_blocks.append(shape_names)

        # ── 6. Rolling statistics ───────────────────────────────────────────
        if self.include_rolling:
            mid_price = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
            spread = ask_prices[:, 0] - bid_prices[:, 0]
            roll_feats, roll_names = self._rolling_features(mid_price, spread)
            feature_blocks.append(roll_feats)
            name_blocks.append(roll_names)

        # ── Combine and NaN-fill ────────────────────────────────────────────
        features = np.concatenate(feature_blocks, axis=1)
        names = [n for block in name_blocks for n in block]

        # Forward fill NaN (from rolling windows at start)
        df = pd.DataFrame(features, columns=names)
        df = df.ffill().bfill().fillna(0.0)
        features = df.values

        self._feature_names = names
        logger.info(f"Extracted {features.shape[1]} features over {T} time steps")
        return features, names

    def get_feature_names(self) -> List[str]:
        """Return names of last extracted feature set."""
        if self._feature_names is None:
            raise RuntimeError("Call transform() first.")
        return self._feature_names

    # ── Private methods ─────────────────────────────────────────────────────

    def _parse_lob(
        self, lob_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse LOB columns into price/volume arrays."""
        n = self.n_levels
        ask_prices = lob_data[:, 0::4][:, :n]
        ask_vols   = lob_data[:, 1::4][:, :n]
        bid_prices = lob_data[:, 2::4][:, :n]
        bid_vols   = lob_data[:, 3::4][:, :n]
        return ask_prices, ask_vols, bid_prices, bid_vols

    def _price_features(
        self, ask_prices: np.ndarray, bid_prices: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Mid price, returns, log returns."""
        mid = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
        log_mid = np.log(mid + 1e-10)

        returns_1  = np.diff(log_mid, prepend=log_mid[0])
        returns_5  = np.diff(log_mid, n=1, prepend=log_mid[0])  # Same as 1-step
        momentum_5 = pd.Series(log_mid).diff(5).fillna(0).values
        momentum_20 = pd.Series(log_mid).diff(20).fillna(0).values

        feats = np.stack([mid, log_mid, returns_1, momentum_5, momentum_20], axis=1)
        names = ["mid_price", "log_mid_price", "log_return_1", "momentum_5", "momentum_20"]
        return feats, names

    def _spread_features(
        self, ask_prices: np.ndarray, bid_prices: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Bid-ask spread and relative spread."""
        mid = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
        spread = ask_prices[:, 0] - bid_prices[:, 0]
        rel_spread = spread / (mid + 1e-10)

        # Spreads at deeper levels
        spread_L2 = ask_prices[:, 1] - bid_prices[:, 1] if self.n_levels > 1 else spread
        spread_L5 = ask_prices[:, 4] - bid_prices[:, 4] if self.n_levels > 4 else spread

        feats = np.stack([spread, rel_spread, spread_L2, spread_L5], axis=1)
        names = ["spread_L1", "rel_spread_L1", "spread_L2", "spread_L5"]
        return feats, names

    def _depth_features(
        self, ask_vols: np.ndarray, bid_vols: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Total depth, depth imbalance, top-of-book volumes."""
        total_ask = ask_vols.sum(1)
        total_bid = bid_vols.sum(1)
        total_depth = total_ask + total_bid
        depth_imbalance = (total_bid - total_ask) / (total_depth + 1e-8)

        ask_L1 = ask_vols[:, 0]
        bid_L1 = bid_vols[:, 0]
        lob_depth_ratio = total_bid / (total_ask + 1e-8)

        feats = np.stack([
            total_ask, total_bid, total_depth,
            depth_imbalance, ask_L1, bid_L1, lob_depth_ratio
        ], axis=1)
        names = [
            "total_ask_depth", "total_bid_depth", "total_depth",
            "depth_imbalance", "ask_L1_vol", "bid_L1_vol", "depth_ratio"
        ]
        return feats, names

    def _imbalance_features(
        self,
        ask_prices: np.ndarray,
        ask_vols: np.ndarray,
        bid_prices: np.ndarray,
        bid_vols: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Order flow and queue imbalance features."""
        # Level-by-level imbalance
        imb_per_level = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)  # (T, n_levels)
        mean_imbalance = imb_per_level.mean(1)
        top3_imbalance = imb_per_level[:, :3].mean(1)

        # Weighted imbalance (weight by inverse level)
        weights = 1.0 / (np.arange(1, self.n_levels + 1))
        weights /= weights.sum()
        weighted_imbalance = (imb_per_level * weights).sum(1)

        # Price pressure
        ask_price_pressure = (ask_prices * ask_vols).sum(1) / (ask_vols.sum(1) + 1e-8)
        bid_price_pressure = (bid_prices * bid_vols).sum(1) / (bid_vols.sum(1) + 1e-8)

        feats = np.stack([
            mean_imbalance, top3_imbalance, weighted_imbalance,
            ask_price_pressure, bid_price_pressure
        ], axis=1)
        names = [
            "mean_imbalance", "top3_imbalance", "weighted_imbalance",
            "ask_price_pressure", "bid_price_pressure"
        ]
        return feats, names

    def _lob_shape_features(
        self,
        ask_prices: np.ndarray,
        ask_vols: np.ndarray,
        bid_prices: np.ndarray,
        bid_vols: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """LOB slope and curvature features."""
        # Weighted mid price (microprice)
        ask_v1 = ask_vols[:, 0]
        bid_v1 = bid_vols[:, 0]
        total_v1 = ask_v1 + bid_v1 + 1e-8
        microprice = (ask_prices[:, 0] * bid_v1 + bid_prices[:, 0] * ask_v1) / total_v1

        # Ask/bid side volume concentration (Herfindahl-style)
        ask_shares = ask_vols / (ask_vols.sum(1, keepdims=True) + 1e-8)
        bid_shares = bid_vols / (bid_vols.sum(1, keepdims=True) + 1e-8)
        ask_concentration = (ask_shares ** 2).sum(1)
        bid_concentration = (bid_shares ** 2).sum(1)

        feats = np.stack([microprice, ask_concentration, bid_concentration], axis=1)
        names = ["microprice", "ask_concentration", "bid_concentration"]
        return feats, names

    def _rolling_features(
        self,
        mid_price: np.ndarray,
        spread: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Rolling volatility and spread statistics."""
        all_feats = []
        all_names = []

        log_ret = np.diff(np.log(mid_price + 1e-10), prepend=0.0)
        series = pd.DataFrame({"ret": log_ret, "spread": spread})

        for w in self.rolling_windows:
            roll = series.rolling(w, min_periods=1)
            vol = roll["ret"].std().fillna(0).values
            spread_mean = roll["spread"].mean().fillna(spread[0]).values
            spread_std = roll["spread"].std().fillna(0).values

            all_feats.extend([vol, spread_mean, spread_std])
            all_names.extend([
                f"vol_rolling_{w}",
                f"spread_mean_{w}",
                f"spread_std_{w}",
            ])

        feats = np.stack(all_feats, axis=1)
        return feats, all_names
