"""Unit tests for transition probabilities."""

import pytest

from src.models.automata.transitions import TransitionModel


def test_transition_probabilities_sum_to_one_with_smoothing():
    model = TransitionModel(smoothing_alpha=1.0)
    patterns = ["abc", "abc", "aab", "aab", "bcc"]
    model.fit(patterns)
    probs = model.outgoing_transitions("abc")
    assert sum(probs.values()) == pytest.approx(1.0, rel=1e-6)


def test_transition_density_non_negative():
    model = TransitionModel()
    model.fit(["abc", "aab", "bcc", "abc"])
    assert model.transition_density >= 0


def test_unknown_transition_uses_smoothing():
    model = TransitionModel(smoothing_alpha=1.0)
    model.fit(["abc", "aab"])
    prob = model.transition_probability("abc", "zzz")
    assert prob > 0
