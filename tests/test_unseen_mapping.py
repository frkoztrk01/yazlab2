"""Unit tests for unseen pattern mapping."""

import pytest

from src.models.automata.unseen import find_nearest_pattern, map_pattern


def test_unseen_pattern_not_in_dictionary():
    train_patterns = {"abc", "aab", "bcc"}
    pattern = "xyz"
    assert pattern not in train_patterns
    mapped, status, distance = map_pattern(pattern, train_patterns)
    assert status == "unseen"
    assert mapped in train_patterns
    assert distance is not None


def test_empty_pattern_nearest():
    dictionary = {"a", "b"}
    nearest, dist = find_nearest_pattern("c", dictionary)
    assert nearest in dictionary
    assert dist == 1


def test_single_char_patterns():
    dictionary = {"aaa", "aab", "aba"}
    mapped, status, distance = map_pattern("aca", dictionary)
    assert status == "unseen"
    assert mapped in dictionary
