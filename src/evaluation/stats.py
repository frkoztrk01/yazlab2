"""Statistical tests for model comparison."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar test for paired model predictions."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))

    table = np.array([[0, b_count], [c_count, 0]])
    if b_count + c_count == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "significant_0_05": False}

    result = mcnemar(table, exact=True)
    pvalue = float(result.pvalue)
    return {
        "statistic": float(result.statistic) if result.statistic is not None else 0.0,
        "pvalue": pvalue,
        "significant_0_05": bool(pvalue < 0.05),
        "model_a_better": b_count,
        "model_b_better": c_count,
    }


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return {"statistic": None, "pvalue": None, "significant_0_05": False}

    stat, pvalue = wilcoxon(scores_a, scores_b)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "significant_0_05": bool(pvalue < 0.05),
    }
