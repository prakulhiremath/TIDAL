"""
preprocessing/clean_data.py
────────────────────────────
Data cleaning and validation utilities for LOB data.

Handles:
    - NaN / Inf removal
    - Price crossing detection and correction
    - Outlier clipping
    - Stale price detection
    - Data integrity checks
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from loguru import logger


class LOBDataCleaner:
    """
    Cleans and validates raw limit order book data.

    Detects and corrects common data quality issues including:
    crossed markets, stale quotes, extreme outliers, and missing values.

    Usage:
        cleaner = LOBDataCleaner()
        clean_data, report = cleaner.clean(raw_data)
    """

    def __init__(
        self,
        n_levels: int = 10,
        outlier_clip_sigma: float = 6.0,
        stale_threshold: int = 10,
        fill_method: str = "ffill",
    ):
        """
        Initialize data cleaner.

        Args:
            n_levels: Number of LOB price levels.
            outlier_clip_sigma: Clip values beyond this many standard deviations.
            stale_threshold: Flag quotes unchanged for this many consecutive steps.
            fill_method: Method for filling NaN values: 'ffill', 'linear', 'zero'.
        """
        self.n_levels = n_levels
        self.outlier_clip_sigma = outlier_clip_sigma
        self.stale_threshold = stale_threshold
        self.fill_method = fill_method

    def clean(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Full cleaning pipeline.

        Args:
            data: Raw LOB array (T, n_levels * 4).

        Returns:
            Tuple of (cleaned array, quality report dict).
        """
        T, F = data.shape
        report = {"original_shape": data.shape, "issues": []}
        logger.info(f"Cleaning LOB data: shape={data.shape}")

        # Step 1: Remove infinity values
        n_inf = np.isinf(data).sum()
        if n_inf > 0:
            data = np.where(np.isinf(data), np.nan, data)
            report["issues"].append(f"Replaced {n_inf} Inf values with NaN")
            logger.warning(f"Found {n_inf} Inf values — replaced with NaN")

        # Step 2: Fill NaN values
        n_nan = np.isnan(data).sum()
        if n_nan > 0:
            df = pd.DataFrame(data)
            if self.fill_method == "ffill":
                df = df.ffill().bfill()
            elif self.fill_method == "linear":
                df = df.interpolate(method="linear").ffill().bfill()
            elif self.fill_method == "zero":
                df = df.fillna(0.0)
            data = df.values
            report["issues"].append(f"Filled {n_nan} NaN values via {self.fill_method}")
            logger.info(f"Filled {n_nan} NaN values")

        # Step 3: Detect and fix crossed markets
        n_crossed = self._fix_crossed_markets(data)
        if n_crossed > 0:
            report["issues"].append(f"Fixed {n_crossed} crossed market snapshots")

        # Step 4: Clip outliers
        n_clipped = self._clip_outliers(data)
        if n_clipped > 0:
            report["issues"].append(f"Clipped {n_clipped} outlier values")

        # Step 5: Detect stale quotes
        stale_rows = self._detect_stale_quotes(data)
        report["stale_rows"] = int(stale_rows.sum())
        if stale_rows.sum() > 0:
            logger.warning(f"Detected {stale_rows.sum()} rows with stale quotes")
            report["issues"].append(f"Detected {stale_rows.sum()} stale quote rows")

        # Step 6: Validate price monotonicity
        self._enforce_price_monotonicity(data)

        report["final_shape"] = data.shape
        report["n_issues"] = len(report["issues"])
        logger.info(f"Cleaning complete. Issues found: {report['n_issues']}")
        return data, report

    def _fix_crossed_markets(self, data: np.ndarray) -> int:
        """
        Detect and correct crossed L1 markets (ask_p < bid_p).

        Args:
            data: LOB data array (in-place modification).

        Returns:
            Number of crossed snapshots corrected.
        """
        ask_p1 = data[:, 0]
        bid_p1 = data[:, 2]
        crossed = ask_p1 < bid_p1

        if crossed.sum() > 0:
            # Swap crossed prices
            data[crossed, 0], data[crossed, 2] = data[crossed, 2].copy(), data[crossed, 0].copy()

        return int(crossed.sum())

    def _clip_outliers(self, data: np.ndarray) -> int:
        """
        Clip extreme values using rolling z-score.

        Args:
            data: LOB array (modified in-place).

        Returns:
            Number of clipped values.
        """
        df = pd.DataFrame(data)
        mean = df.rolling(100, min_periods=10).mean()
        std  = df.rolling(100, min_periods=10).std().fillna(1.0)

        lower = mean - self.outlier_clip_sigma * std
        upper = mean + self.outlier_clip_sigma * std

        n_clipped = ((df < lower) | (df > upper)).sum().sum()
        data[:] = np.clip(data, lower.values, upper.values)
        return int(n_clipped)

    def _detect_stale_quotes(self, data: np.ndarray) -> np.ndarray:
        """
        Identify rows where the top-of-book hasn't changed.

        Args:
            data: LOB array.

        Returns:
            Boolean array (T,) marking stale rows.
        """
        # Check if top-of-book prices and volumes are unchanged
        top_snapshot = data[:, :4]  # [ask_p1, ask_v1, bid_p1, bid_v1]
        unchanged = np.zeros(len(data), dtype=bool)

        for t in range(1, len(data)):
            if np.allclose(top_snapshot[t], top_snapshot[t - 1]):
                unchanged[t] = True

        # Only flag truly stale (consecutive unchanged)
        stale = np.zeros(len(data), dtype=bool)
        run_length = 0
        for t in range(len(data)):
            if unchanged[t]:
                run_length += 1
                if run_length >= self.stale_threshold:
                    stale[t] = True
            else:
                run_length = 0

        return stale

    def _enforce_price_monotonicity(self, data: np.ndarray) -> None:
        """
        Ensure ask prices are monotonically increasing and
        bid prices monotonically decreasing across levels.

        Corrects minor violations by overwriting with adjacent level prices.

        Args:
            data: LOB array (modified in-place).
        """
        for level in range(1, self.n_levels):
            col = level * 4
            prev_col = (level - 1) * 4
            # Ask prices should increase with level
            violation = data[:, col] < data[:, prev_col]
            if violation.sum() > 0:
                data[violation, col] = data[violation, prev_col] * 1.0001

            # Bid prices should decrease with level
            violation = data[:, col + 2] > data[:, prev_col + 2]
            if violation.sum() > 0:
                data[violation, col + 2] = data[violation, prev_col + 2] * 0.9999

    def validate(self, data: np.ndarray) -> bool:
        """
        Quick validation check on cleaned data.

        Args:
            data: Cleaned LOB array.

        Returns:
            True if data passes all checks.
        """
        checks = [
            ("No NaN", not np.isnan(data).any()),
            ("No Inf", not np.isinf(data).any()),
            ("Positive prices", (data[:, 0::4] > 0).all()),
            ("Positive volumes", (data[:, 1::4] >= 0).all()),
            ("Ask > Bid L1", (data[:, 0] >= data[:, 2]).all()),
        ]
        passed = True
        for name, result in checks:
            status = "PASS" if result else "FAIL"
            if not result:
                logger.warning(f"Validation {status}: {name}")
                passed = False
            else:
                logger.debug(f"Validation {status}: {name}")
        return passed
