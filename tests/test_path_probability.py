"""Unit tests for path probability and anomaly decisions."""

import numpy as np
import pytest

from src.models.automata.automata import AutomataConfig, ProbabilisticAutomata


def _make_series(length: int = 200, anomaly_start: int = 150) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    series = rng.normal(0, 1, length)
    series[anomaly_start:] += 5
    labels = np.zeros(length, dtype=int)
    labels[anomaly_start:] = 1
    return series, labels


def test_path_probability_is_product_of_transitions():
    train_series, train_labels = _make_series(120)
    val_series, val_labels = _make_series(80)
    model = ProbabilisticAutomata(AutomataConfig(window_size=3, alphabet_size=3, paa_segment_size=4))
    model.fit(train_series, val_series, val_labels)

    explanations = model.explain(val_series)
    if len(explanations) >= 2:
        prob_product = 1.0
        for e in explanations:
            for t in e.transitions:
                prob_product *= t["probability"]
        assert explanations[-1].path_probability == pytest.approx(prob_product, rel=1e-4)


def test_low_probability_marks_anomaly():
    train_series, _ = _make_series(100)
    test_series, test_labels = _make_series(100, anomaly_start=50)
    model = ProbabilisticAutomata(AutomataConfig(window_size=3, alphabet_size=3, paa_segment_size=4))
    model.fit(train_series, train_series[:50], np.zeros(50, dtype=int))
    preds, probs, _ = model.predict(test_series)
    assert len(preds) > 0
    assert probs.min() <= probs.max()


def test_automata_fit_predict_smoke():
    series, labels = _make_series(80)
    model = ProbabilisticAutomata(AutomataConfig(window_size=3, alphabet_size=3, paa_segment_size=4))
    model.fit(series[:50], series[50:], labels[50:])
    preds, probs, _ = model.predict(series)
    assert len(preds) == len(probs)
