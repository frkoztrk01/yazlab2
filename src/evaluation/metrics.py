"""Metrics computation and logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score


@dataclass
class MetricResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    auc: float | None = None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> MetricResult:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = None

    return MetricResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=cm,
        auc=auc,
    )


def aggregate_metrics(results: list[MetricResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = ["accuracy", "precision", "recall", "f1"]
    agg = {}
    for key in keys:
        values = [getattr(r, key) for r in results]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))
    return agg


def save_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records
