"""Piecewise Aggregate Approximation."""

from __future__ import annotations

import numpy as np


def paa_transform(series: np.ndarray, segment_size: int) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n == 0:
        return np.array([])

    n_segments = max(1, n // segment_size)
    usable = n_segments * segment_size
    trimmed = series[:usable]
    reshaped = trimmed.reshape(n_segments, segment_size)
    return reshaped.mean(axis=1)
