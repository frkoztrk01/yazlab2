"""Scenario generators: original, noisy, unseen."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(X: np.ndarray, std: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=X.shape)
    return X + noise


def mark_unseen_patterns(test_patterns: list[str], train_dictionary: set[str]) -> list[bool]:
    return [p not in train_dictionary for p in test_patterns]
