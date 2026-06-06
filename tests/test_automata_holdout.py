"""Tests for unseen scenario dictionary holdout."""

import numpy as np

from src.models.automata.automata import AutomataConfig, ProbabilisticAutomata


def test_unseen_holdout_creates_unseen_patterns():
    rng = np.random.default_rng(0)
    train_series = rng.normal(size=800)
    test_series = rng.normal(size=200)

    model = ProbabilisticAutomata(AutomataConfig(window_size=4, alphabet_size=3, paa_segment_size=8))
    model.fit(train_series, dictionary_holdout_ratio=0.15, holdout_seed=42)

    assert len(model.held_out_patterns) > 0
    assert len(model.pattern_dictionary) < 81

    explanations = model.explain(test_series)
    unseen = [e for e in explanations if e.status == "unseen"]
    assert len(unseen) > 0
    assert all(e.mapped_to is not None for e in unseen)
