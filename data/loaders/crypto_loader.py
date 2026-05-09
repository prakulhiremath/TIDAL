"""
data/loaders/crypto_loader.py
──────────────────────────────
Loader for cryptocurrency order book data.

Supports:
    1. Synthetic generation (no API key required — for testing/development)
    2. Historical Binance data via REST API
    3. Live streaming (placeholder for production deployment)

The crypto LOB format mirrors the FI-2010 structure for compatibility
with the shared preprocessing pipeline.
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Live data fetching unavailable.")


class CryptoOrderBookLoader:
    """
    Cryptocurrency order book loader supporting multiple data sources.

    Usage:
        # Synthetic (no API required)
        loader = CryptoOrderBookLoader(mode="synthetic")
        data = loader.load(n_steps=50000)

        # Historical Binance
        loader = CryptoOrderBookLoader(mode="binance", symbol="BTCUSDT")
        data = loader.load_historical(start_date="2023-01-01", end_date="2023-06-01")
    """

    N_LEVELS = 10
    N_FEATURES = 40  # 10 levels × 4: [ask_p, ask_v, bid_p, bid_v]

    BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"
    BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

    def __init__(
        self,
        mode: str = "synthetic",
        symbol: str = "BTCUSDT",
        data_dir: str = "data/raw/crypto",
        normalization: str = "zscore",
    ):
        """
        Initialize crypto order book loader.

        Args:
            mode: Data source mode: 'synthetic', 'binance', 'local'.
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            data_dir: Directory for caching/loading local data.
            normalization: Normalization method.
        """
        self.mode = mode
        self.symbol = symbol
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.normalization = normalization
        self._stats: Optional[Dict] = None

    def load(self, n_steps: int = 100000, **kwargs) -> np.ndarray:
        """
        Load data from the configured source.

        Args:
            n_steps: Number of time steps (for synthetic mode).
            **kwargs: Additional arguments passed to specific loaders.

        Returns:
            LOB data array (T, 40).
        """
        if self.mode == "synthetic":
            return self._generate_synthetic(n_steps, **kwargs)
        elif self.mode == "binance":
            return self._load_binance_historical(**kwargs)
        elif self.mode == "local":
            return self._load_local(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_synthetic(
        self,
        n_steps: int = 100000,
        seed: int = 42,
        regime_changes: int = 20,
        instability_fraction: float = 0.15,
    ) -> np.ndarray:
        """
        Generate realistic synthetic crypto order book data.

        Simulates:
            - Background Brownian motion price dynamics
            - Regime changes (stable → transitional → unstable)
            - Liquidity shocks and spread widening episodes
            - Order imbalance surges

        Args:
            n_steps: Total time steps.
            seed: Random seed for reproducibility.
            regime_changes: Number of regime shift events to inject.
            instability_fraction: Fraction of time in instability.

        Returns:
            Synthetic LOB array (n_steps, 40).
        """
        logger.info(f"Generating synthetic crypto LOB data: n_steps={n_steps}")
        rng = np.random.default_rng(seed)

        # ── Base price process ──────────────────────────────────────────────
        base_volatility = 0.00015  # ~1.5bps per tick
        price = 30000.0  # Starting BTC price
        prices = [price]

        # Regime schedule: 0=stable, 1=transitional, 2=unstable
        regime_sequence = np.zeros(n_steps, dtype=int)
        change_points = sorted(rng.choice(n_steps - 100, size=regime_changes, replace=False))

        for i, cp in enumerate(change_points):
            end = change_points[i + 1] if i + 1 < len(change_points) else n_steps
            target_regime = rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])
            regime_sequence[cp:end] = target_regime

        # ── Simulate price path with regime-dependent volatility ────────────
        volatility_map = {0: base_volatility, 1: base_volatility * 2.5, 2: base_volatility * 6.0}
        for t in range(1, n_steps):
            vol = volatility_map[regime_sequence[t]]
            ret = rng.normal(0, vol)
            price = max(100, prices[-1] * (1 + ret))
            prices.append(price)
        prices = np.array(prices)

        # ── Simulate order book ─────────────────────────────────────────────
        data = np.zeros((n_steps, self.N_FEATURES))
        spread_map = {0: 0.5, 1: 2.0, 2: 8.0}  # In price units (USD)
        depth_map   = {0: 1.0, 1: 0.5, 2: 0.15} # Relative depth multiplier

        for t in range(n_steps):
            p = prices[t]
            regime = regime_sequence[t]
            spread = spread_map[regime] * (1 + 0.5 * rng.exponential())
            base_depth = depth_map[regime]

            for level in range(self.N_LEVELS):
                tick = (level + 1) * spread * 0.5
                col = level * 4

                # Asymmetric depth during instability (order imbalance)
                imbalance = rng.uniform(-0.4, 0.4) if regime == 2 else rng.uniform(-0.1, 0.1)
                ask_mult = base_depth * (1 - imbalance)
                bid_mult = base_depth * (1 + imbalance)

                data[t, col + 0] = p + tick                                      # ask price
                data[t, col + 1] = max(0.001, rng.lognormal(2.0, 0.5) * ask_mult)  # ask vol
                data[t, col + 2] = p - tick                                      # bid price
                data[t, col + 3] = max(0.001, rng.lognormal(2.0, 0.5) * bid_mult)  # bid vol

        # Attach regime labels as metadata
        self._regime_labels = regime_sequence
        logger.info(f"Synthetic data: {(regime_sequence == 0).sum()} stable, "
                    f"{(regime_sequence == 1).sum()} transitional, "
                    f"{(regime_sequence == 2).sum()} unstable steps")

        return data

    def get_regime_labels(self) -> Optional[np.ndarray]:
        """
        Return ground-truth regime labels from last synthetic generation.

        Returns:
            Array of shape (T,) with values {0, 1, 2}, or None if not available.
        """
        return getattr(self, "_regime_labels", None)

    def _load_binance_historical(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1m",
    ) -> np.ndarray:
        """
        Load historical OHLCV data from Binance REST API.

        Note: Binance public API provides OHLCV, not full LOB depth history.
        Full LOB requires premium data providers. This method synthesizes
        LOB structure from OHLCV data.

        Args:
            start_date: Start date string "YYYY-MM-DD".
            end_date: End date string "YYYY-MM-DD".
            interval: Kline interval (e.g., '1m', '5m', '1h').

        Returns:
            LOB-structured array (T, 40).
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests package required for Binance loading.")
            return self._generate_synthetic()

        logger.info(f"Fetching Binance data: {self.symbol} {interval}")
        params = {"symbol": self.symbol, "interval": interval, "limit": 1000}

        if start_date:
            params["startTime"] = int(pd.Timestamp(start_date).timestamp() * 1000)
        if end_date:
            params["endTime"] = int(pd.Timestamp(end_date).timestamp() * 1000)

        try:
            resp = requests.get(self.BINANCE_KLINES_URL, params=params, timeout=10)
            resp.raise_for_status()
            klines = resp.json()
        except Exception as e:
            logger.error(f"Binance API error: {e}. Falling back to synthetic.")
            return self._generate_synthetic()

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "n_trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # Reconstruct synthetic LOB from OHLCV
        return self._ohlcv_to_lob(df)

    def _ohlcv_to_lob(self, df: pd.DataFrame) -> np.ndarray:
        """
        Reconstruct pseudo-LOB structure from OHLCV bars.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            LOB-structured array (T, 40).
        """
        T = len(df)
        data = np.zeros((T, self.N_FEATURES))
        rng = np.random.default_rng(123)

        for i, row in df.iterrows():
            mid = (float(row["high"]) + float(row["low"])) / 2
            spread = float(row["high"]) - float(row["low"])
            vol = float(row["volume"])

            for level in range(self.N_LEVELS):
                tick = (level + 1) * spread * 0.05
                col = level * 4
                data[i, col + 0] = mid + tick
                data[i, col + 1] = max(0.001, vol / (level + 1) * rng.uniform(0.5, 1.5))
                data[i, col + 2] = mid - tick
                data[i, col + 3] = max(0.001, vol / (level + 1) * rng.uniform(0.5, 1.5))

        return data

    def _load_local(self, filename: str = "order_book.npy") -> np.ndarray:
        """
        Load locally cached order book data.

        Args:
            filename: Filename within data_dir.

        Returns:
            LOB array.
        """
        path = self.data_dir / filename
        if not path.exists():
            logger.warning(f"Local file not found: {path}. Generating synthetic.")
            return self._generate_synthetic()
        data = np.load(path)
        logger.info(f"Loaded local data from {path}: shape={data.shape}")
        return data

    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize LOB data.

        Args:
            data: Raw LOB array (T, 40).
            fit: If True, fit normalization statistics from data.

        Returns:
            Normalized array.
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

    def save_to_cache(self, data: np.ndarray, filename: str = "order_book.npy") -> None:
        """
        Save loaded data to local cache.

        Args:
            data: Data array to save.
            filename: Output filename.
        """
        path = self.data_dir / filename
        np.save(path, data)
        logger.info(f"Saved {data.shape} array to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Order Book Loader")
    parser.add_argument("--mode", default="synthetic",
                        choices=["synthetic", "binance", "local"])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--output", default="data/raw/crypto/order_book.npy")
    args = parser.parse_args()

    loader = CryptoOrderBookLoader(mode=args.mode, symbol=args.symbol)
    data = loader.load(n_steps=args.n_steps)
    loader.save_to_cache(data, Path(args.output).name)
    print(f"Generated data shape: {data.shape}")
