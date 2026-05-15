"""Levenshtein distance and unseen pattern mapping."""

from __future__ import annotations


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def find_nearest_pattern(pattern: str, dictionary: set[str] | list[str]) -> tuple[str, int]:
    if not dictionary:
        raise ValueError("Dictionary must not be empty")

    best_pattern = None
    best_distance = None

    for candidate in dictionary:
        dist = levenshtein_distance(pattern, candidate)
        if best_distance is None or dist < best_distance or (dist == best_distance and candidate < best_pattern):
            best_pattern = candidate
            best_distance = dist

    assert best_pattern is not None and best_distance is not None
    return best_pattern, best_distance


def map_pattern(pattern: str, dictionary: set[str]) -> tuple[str, str, int | None]:
    """Return (mapped_pattern, status, distance). status is 'seen' or 'unseen'."""
    if pattern in dictionary:
        return pattern, "seen", None
    mapped, dist = find_nearest_pattern(pattern, dictionary)
    return mapped, "unseen", dist
