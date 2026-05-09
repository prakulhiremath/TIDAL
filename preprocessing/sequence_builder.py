"""
preprocessing/sequence_builder.py
-----------------------------------
Boundary-safe sliding window sequence construction for TIDAL.

Data leakage prevention
-----------------------
The train/val/test split is performed on **raw timestamps** BEFORE sequences
are built. Sequences are then constructed independently within each fold.
This ensures that no sequence contains data from both sides of a fold boundary.

Sequence structure
------------------
For each valid step t:
    X[t]  = features[t - seq_len + 1 : t + 1]   shape: (seq_len, F)
    y[t]  = labels[t]                             shape: (H,)   H = num horizons

Steps where labels[t] == -1 (masked) are excluded from all folds.

Usage
-----
    builder = SequenceBuilder(cfg)
    X_train, y_train = builder.build(features_train, labels_train)
    # X_train: (N_train, seq_len, F)
    # y_train: (N_train, H)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset


class SequenceBuilder:
    """Build sliding-window sequences from feature and label arrays.

    Parameters
    ----------
    cfg:
        OmegaConf config containing ``data.seq_len`` and ``data.step_size``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.seq_len  = cfg.data.seq_len
        self.step_size = cfg.data.get("step_size", 1)

    def build(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        return_indices: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct aligned (X, y) sequence pairs.

        Parameters
        ----------
        features:
            Float array of shape (T, F). Features for a single fold.
        labels:
            Int8 array of shape (T, H) or (T,). -1 = masked/excluded.
        return_indices:
            If True, also return the original time indices of the sequences.

        Returns
        -------
        X : (N, seq_len, F)
        y : (N, H) or (N,)
        indices : (N,)  — only if return_indices=True
        """
        T, F = features.shape
        if labels.ndim == 1:
            labels = labels[:, np.newaxis]
        H = labels.shape[1]

        # Minimum index where a full sequence can be formed
        start = self.seq_len - 1

        X_list, y_list, idx_list = [], [], []

        for t in range(start, T, self.step_size):
            lab = labels[t]
            # Exclude masked steps (any horizon masked → exclude entire step)
            if np.any(lab == -1):
                continue
            seq = features[t - self.seq_len + 1 : t + 1]  # (seq_len, F)
            X_list.append(seq)
            y_list.append(lab)
            idx_list.append(t)

        if not X_list:
            raise ValueError(
                f"No valid sequences found. Check seq_len={self.seq_len} "
                f"vs T={T} and label masks."
            )

        X = np.stack(X_list, axis=0).astype(np.float32)   # (N, seq_len, F)
        y = np.stack(y_list, axis=0).astype(np.int64)      # (N, H)
        indices = np.array(idx_list, dtype=np.int64)

        if return_indices:
            return X, y, indices
        return X, y


class InstabilityDataset(Dataset):
    """PyTorch Dataset wrapping sequence arrays for DataLoader compatibility.

    Parameters
    ----------
    X:
        Feature sequences, shape (N, seq_len, F).
    y:
        Labels, shape (N, H) where H = number of horizons.
    horizon_idx:
        If not None, extract only one horizon: y becomes (N,).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon_idx: Optional[int] = None,
    ) -> None:
        import torch
        self.X = torch.from_numpy(X)                  # float32
        if horizon_idx is not None:
            self.y = torch.from_numpy(y[:, horizon_idx]).long()
        else:
            self.y = torch.from_numpy(y).long()       # (N, H) or (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: DictConfig,
    seed: int = 42,
):
    """Build train / val / test DataLoaders.

    Returns
    -------
    Tuple of (train_loader, val_loader, test_loader).
    """
    import torch
    from torch.utils.data import DataLoader
    from utils.seed import seed_worker, make_generator

    g = make_generator(seed)

    train_ds = InstabilityDataset(X_train, y_train)
    val_ds   = InstabilityDataset(X_val,   y_val)
    test_ds  = InstabilityDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.evaluation.get("batch_size", 256),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.evaluation.get("batch_size", 256),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
