"""Unit tests for Levenshtein distance."""

import pytest

from src.models.automata.unseen import find_nearest_pattern, levenshtein_distance, map_pattern


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("abc", "abc", 0),
        ("abc", "adc", 1),
        ("aab", "aaa", 1),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
    ],
)
def test_levenshtein_distance(a, b, expected):
    assert levenshtein_distance(a, b) == expected


def test_levenshtein_symmetry():
    assert levenshtein_distance("abc", "adc") == levenshtein_distance("adc", "abc")


def test_find_nearest_pattern():
    dictionary = {"abc", "aab", "bcc"}
    nearest, dist = find_nearest_pattern("adc", dictionary)
    assert nearest == "abc"
    assert dist == 1


def test_find_nearest_pattern_tie_break():
    dictionary = {"aac", "abc"}
    nearest, dist = find_nearest_pattern("acc", dictionary)
    assert dist == 1
    assert nearest == "aac"


def test_map_pattern_seen():
    mapped, status, distance = map_pattern("abc", {"abc", "aab"})
    assert mapped == "abc"
    assert status == "seen"
    assert distance is None


def test_map_pattern_unseen():
    mapped, status, distance = map_pattern("adc", {"abc", "aab"})
    assert mapped == "abc"
    assert status == "unseen"
    assert distance == 1
