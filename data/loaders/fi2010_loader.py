"""
data/loaders/fi2010_loader.py
──────────────────────────────
Loader for the FI-2010 Limit Order Book benchmark dataset.

FI-2010 is the standard academic benchmark for LOB prediction tasks.
It contains order book snapshots from 5 Finnish stocks over 10 days.

Reference:
    Ntakaris et al. (2018). Benchmark dataset for mid-price forecasting
    of limit order book data with machine learning methods.

Dataset format:
    - 40 features per snapshot (10 levels × 4: ask price, ask size, bid price, bid size)
    - Available from: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from loguru import logger


class FI2010Loader:
    """
    Loader and preprocessor for the FI-2010 LOB dataset.

    The dataset contains 10 days of order book data for 5 stocks.
    Each row is a snapshot with 40 LOB features.

    Usage:
        loader = FI2010Loader(data_dir="data/raw/fi2010")
        data = loader.load(day=1, stock=0)
        X_train, X_test = loader.train_test_split(data)
    """

    # FI-2010 column structure: 10 levels of [ask_price, ask_vol, bid_price, bid_vol]
    N_LEVELS = 10
    N_FEATURES = 40  # 10 levels × 4 fields

    # Standard stock tickers in dataset
    STOCKS = ["KESBV", "OUT1V", "SAMPO", "RTRKS", "TIETO"]

    def __init__(self, data_dir: str = "data/raw/fi2010", normalization: str = "zscore"):
        """
        Initialize FI-2010 loader.

        Args:
            data_dir: Directory containing FI-2010 .txt files.
            normalization: Normalization method: 'zscore', 'minmax', or 'none'.
        """
        self.data_dir = Path(data_dir)
        self.normalization = normalization
        self._stats: Optional[Dict] = None

    def load_raw(self, day: int, stock_idx: int = 0) -> np.ndarray:
        """
        Load raw FI-2010 data for a given day and stock.

        Args:
            day: Day number (1-10).
            stock_idx: Stock index (0-4).

        Returns:
            Array of shape (T, 40) containing LOB snapshots.
        """
        # Expected file naming convention
        filename = f"Train_Dst_NoAuction_DecPre_CF_{day}.txt"
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"FI-2010 file not found: {filepath}. Generating synthetic data.")
            return self._generate_synthetic(n_steps=5000)

        data = np.loadtxt(filepath)
        # Rows = features, Cols = time steps → transpose to (T, F)
        if data.shape[0] == self.N_FEATURES:
            data = data.T

        logger.info(f"Loaded FI-2010 day={day}, stock={stock_idx}: shape={data.shape}")
        return data[:, :self.N_FEATURES]

    def load_all_days(self, stock_idx: int = 0) -> np.ndarray:
        """
        Load and concatenate all 10 days for a given stock.

        Args:
            stock_idx: Stock index (0-4).

        Returns:
            Concatenated array of shape (T_total, 40).
        """
        all_data = []
        for day in range(1, 11):
            day_data = self.load_raw(day, stock_idx)
            all_data.append(day_data)
        data = np.concatenate(all_data, axis=0)
        logger.info(f"Loaded all days, stock={stock_idx}: total shape={data.shape}")
        return data

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize the LOB data.

        Args:
            data: Raw data array (T, F).
            fit: If True, compute normalization stats from data.
                 If False, use previously fitted stats.

        Returns:
            Normalized data array (T, F).
        """
        if self.normalization == "none":
            return data

        if fit or self._stats is None:
            if self.normalization == "zscore":
                self._stats = {"mean": data.mean(0), "std": data.std(0) + 1e-8}
            elif self.normalization == "minmax":
                self._stats = {"min": data.min(0), "max": data.max(0) + 1e-8}

        if self.normalization == "zscore":
            return (data - self._stats["mean"]) / self._stats["std"]
        elif self.normalization == "minmax":
            denom = self._stats["max"] - self._stats["min"] + 1e-8
            return (data - self._stats["min"]) / denom

        return data

    def get_feature_names(self) -> List[str]:
        """
        Return human-readable feature names for the 40 LOB columns.

        Returns:
            List of 40 feature name strings.
        """
        names = []
        for level in range(1, self.N_LEVELS + 1):
            for side in ["ask", "bid"]:
                for field in ["price", "volume"]:
                    names.append(f"{side}_L{level}_{field}")
        return names

    def compute_derived_features(self, data: np.ndarray) -> np.ndarray:
        """
        Compute additional microstructure features from raw LOB data.

        Derived features:
            - Mid price
            - Spread
            - Order imbalance per level
            - Weighted mid price
            - LOB depth (total volume)

        Args:
            data: Raw LOB data (T, 40).

        Returns:
            Augmented data (T, 40 + n_derived).
        """
        T = data.shape[0]
        derived = []

        # Extract price and volume arrays
        # Column ordering: ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ...
        ask_prices = data[:, 0::4][:, :self.N_LEVELS]   # (T, 10)
        ask_vols   = data[:, 1::4][:, :self.N_LEVELS]   # (T, 10)
        bid_prices = data[:, 2::4][:, :self.N_LEVELS]   # (T, 10)
        bid_vols   = data[:, 3::4][:, :self.N_LEVELS]   # (T, 10)

        # Mid price
        mid_price = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
        derived.append(mid_price.reshape(-1, 1))

        # Bid-ask spread
        spread = ask_prices[:, 0] - bid_prices[:, 0]
        derived.append(spread.reshape(-1, 1))

        # Order imbalance (per level)
        imbalance = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-8)
        derived.append(imbalance)  # (T, 10)

        # Weighted mid price
        w_ask = ask_vols[:, 0] / (ask_vols[:, 0] + bid_vols[:, 0] + 1e-8)
        w_bid = bid_vols[:, 0] / (ask_vols[:, 0] + bid_vols[:, 0] + 1e-8)
        wmp = w_bid * ask_prices[:, 0] + w_ask * bid_prices[:, 0]
        derived.append(wmp.reshape(-1, 1))

        # Total LOB depth
        total_ask_depth = ask_vols.sum(1)
        total_bid_depth = bid_vols.sum(1)
        derived.append(total_ask_depth.reshape(-1, 1))
        derived.append(total_bid_depth.reshape(-1, 1))

        derived_arr = np.concatenate(derived, axis=1)
        augmented = np.concatenate([data, derived_arr], axis=1)
        return augmented

    def train_test_split(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chronological train/val/test split (no shuffling).

        Args:
            data: Full dataset array (T, F).
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.

        Returns:
            Tuple of (train, val, test) arrays.
        """
        T = len(data)
        n_train = int(T * train_ratio)
        n_val = int(T * val_ratio)

        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def _generate_synthetic(self, n_steps: int = 5000) -> np.ndarray:
        """
        Generate synthetic LOB data for testing when real data is unavailable.

        Args:
            n_steps: Number of time steps to generate.

        Returns:
            Synthetic LOB array (n_steps, 40).
        """
        logger.warning("Generating SYNTHETIC FI-2010-like data. For research, use real data.")
        rng = np.random.default_rng(42)

        # Simulate a realistic mid-price process
        returns = rng.normal(0, 0.0001, n_steps)
        mid_price = 10.0 * np.exp(np.cumsum(returns))

        data = np.zeros((n_steps, self.N_FEATURES))
        spread_base = 0.01

        for t in range(n_steps):
            p = mid_price[t]
            # Simulate instability episodes with wider spreads
            spread = spread_base * (1 + 3 * (np.sin(t / 500) ** 2 > 0.8))
            for level in range(self.N_LEVELS):
                tick = (level + 1) * spread
                col = level * 4
                data[t, col + 0] = p + tick          # ask price
                data[t, col + 1] = max(1, rng.poisson(10 - level))  # ask vol
                data[t, col + 2] = p - tick          # bid price
                data[t, col + 3] = max(1, rng.poisson(10 - level))  # bid vol

        return data
