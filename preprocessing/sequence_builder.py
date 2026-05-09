"""
preprocessing/sequence_builder.py
───────────────────────────────────
Sliding window sequence construction for temporal models.

Converts time-series features and labels into fixed-length sequences
suitable for LSTM, Transformer, and TIDAL model training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from loguru import logger


class LOBSequenceDataset(Dataset):
    """
    PyTorch Dataset for LOB sequence data with multi-horizon labels.

    Each sample is a tuple:
        (sequence, labels_dict)
    where sequence has shape (window_size, n_features) and
    labels_dict maps horizon → binary scalar.

    Usage:
        dataset = LOBSequenceDataset(features, labels, window_size=50)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
        window_size: int = 50,
        stride: int = 1,
        horizons: List[int] = [10, 30, 60],
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize sequence dataset.

        Args:
            features: Feature array (T, n_features).
            labels: Dictionary of label arrays, each shape (T,).
            window_size: Input sequence length.
            stride: Step size between sequences.
            horizons: Prediction horizons to include as labels.
            dtype: PyTorch data type.
        """
        self.features = features.astype(np.float32)
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons
        self.dtype = dtype

        # Precompute valid sequence start indices
        T = len(features)
        max_horizon = max(horizons) if horizons else 0
        self.indices = list(range(0, T - window_size - max_horizon + 1, stride))

        logger.debug(
            f"Dataset: T={T}, window={window_size}, stride={stride}, "
            f"n_sequences={len(self.indices)}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Retrieve a sequence and its labels.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (sequence_tensor, label_dict).
            sequence_tensor shape: (window_size, n_features)
            label_dict: horizon_name → scalar tensor
        """
        start = self.indices[idx]
        end = start + self.window_size

        seq = torch.tensor(self.features[start:end], dtype=self.dtype)

        label_dict = {}
        # "instability_now" is labeled at the END of the input window
        if "instability_now" in self.labels:
            label_dict["instability_now"] = torch.tensor(
                self.labels["instability_now"][end - 1], dtype=torch.float32
            )

        for h in self.horizons:
            key = f"instability_h{h}"
            if key in self.labels:
                label_dict[key] = torch.tensor(
                    self.labels[key][end - 1], dtype=torch.float32
                )

        return seq, label_dict

    def get_class_weights(self, horizon: int = 10) -> torch.Tensor:
        """
        Compute class weights for imbalanced instability labels.

        Args:
            horizon: Which horizon to compute weights for.

        Returns:
            Tensor of shape (2,) with [weight_stable, weight_unstable].
        """
        key = f"instability_h{horizon}"
        if key not in self.labels:
            key = "instability_now"

        valid_labels = np.array([
            self.labels[key][self.indices[i] + self.window_size - 1]
            for i in range(len(self.indices))
        ])
        n_pos = valid_labels.sum()
        n_neg = len(valid_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor([1.0, 1.0])

        w_neg = len(valid_labels) / (2 * n_neg)
        w_pos = len(valid_labels) / (2 * n_pos)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


class SequenceBuilder:
    """
    High-level API for building train/val/test sequence loaders.

    Usage:
        builder = SequenceBuilder(window_size=50, horizons=[10, 30, 60])
        train_loader, val_loader, test_loader = builder.build_loaders(
            train_features, val_features, test_features,
            train_labels, val_labels, test_labels,
        )
    """

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 1,
        horizons: List[int] = [10, 30, 60],
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        """
        Initialize sequence builder.

        Args:
            window_size: Temporal input window length.
            stride: Sliding window stride.
            horizons: Prediction horizons.
            batch_size: DataLoader batch size.
            num_workers: DataLoader worker count.
        """
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons
        self.batch_size = batch_size
        self.num_workers = num_workers

    def build_loaders(
        self,
        train_feats: np.ndarray,
        val_feats: np.ndarray,
        test_feats: np.ndarray,
        train_labels: Dict[str, np.ndarray],
        val_labels: Dict[str, np.ndarray],
        test_labels: Dict[str, np.ndarray],
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build DataLoaders for train, validation, and test splits.

        Args:
            train_feats: Training features (T_train, F).
            val_feats: Validation features (T_val, F).
            test_feats: Test features (T_test, F).
            train_labels: Training label dictionary.
            val_labels: Validation label dictionary.
            test_labels: Test label dictionary.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_ds = LOBSequenceDataset(
            train_feats, train_labels, self.window_size, self.stride, self.horizons
        )
        val_ds = LOBSequenceDataset(
            val_feats, val_labels, self.window_size, stride=1, horizons=self.horizons
        )
        test_ds = LOBSequenceDataset(
            test_feats, test_labels, self.window_size, stride=1, horizons=self.horizons
        )

        logger.info(
            f"Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    def build_numpy_sequences(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
        horizon: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build flat NumPy sequences for sklearn-compatible models.

        Args:
            features: Feature array (T, F).
            labels: Label dictionary.
            horizon: Which horizon label to use.

        Returns:
            Tuple of (X, y) where X has shape (n_seq, window_size * F).
        """
        T, F = features.shape
        label_key = f"instability_h{horizon}" if f"instability_h{horizon}" in labels else "instability_now"
        label_arr = labels[label_key]

        max_h = horizon
        valid_start = list(range(0, T - self.window_size - max_h + 1, self.stride))
        n_seq = len(valid_start)

        X = np.zeros((n_seq, self.window_size * F), dtype=np.float32)
        y = np.zeros(n_seq, dtype=int)

        for i, start in enumerate(valid_start):
            end = start + self.window_size
            X[i] = features[start:end].flatten()
            y[i] = label_arr[end - 1]

        logger.debug(f"Built {n_seq} numpy sequences: X={X.shape}, y={y.shape}")
        return X, y
