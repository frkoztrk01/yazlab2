"""Symbolic Aggregate approXimation."""

from __future__ import annotations

import numpy as np


def _build_breakpoints(alphabet_size: int) -> np.ndarray:
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >= 2")
    quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    return quantiles


def sax_transform(values: np.ndarray, alphabet_size: int, breakpoints: np.ndarray | None = None) -> str:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return ""

    if breakpoints is None:
        breakpoints = np.quantile(values, _build_breakpoints(alphabet_size))

    symbols = []
    for v in values:
        idx = np.searchsorted(breakpoints, v, side="right")
        symbols.append(chr(ord("a") + min(idx, alphabet_size - 1)))
    return "".join(symbols)


def fit_sax_breakpoints(train_values: np.ndarray, alphabet_size: int) -> np.ndarray:
    train_values = np.asarray(train_values, dtype=float)
    return np.quantile(train_values, _build_breakpoints(alphabet_size))
