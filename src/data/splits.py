"""Train/validation/test splitting strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class FoldSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def skab_group_kfold(
    labels: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold_id, train_idx, test_idx) using StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        yield fold_id, train_idx, test_idx


def batadal_temporal_split(
    n_samples: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> FoldSplit:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)
    return FoldSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def split_train_val(train_idx: np.ndarray, labels: np.ndarray, val_fraction: float = 0.15, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split training indices into train/val for SKAB folds (within train files only)."""
    rng = np.random.default_rng(seed)
    train_idx = np.array(train_idx)
    if len(train_idx) < 10:
        mid = len(train_idx) // 2
        return train_idx[:mid], train_idx[mid:]

    unique_labels = np.unique(labels[train_idx])
    val_indices: list[int] = []
    train_indices: list[int] = []

    for label in unique_labels:
        label_idx = train_idx[labels[train_idx] == label]
        rng.shuffle(label_idx)
        n_val = max(1, int(len(label_idx) * val_fraction))
        val_indices.extend(label_idx[:n_val].tolist())
        train_indices.extend(label_idx[n_val:].tolist())

    return np.array(train_indices), np.array(val_indices)
