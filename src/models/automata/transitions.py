"""Transition probability computation for probabilistic automata."""

from __future__ import annotations

from collections import defaultdict


class TransitionModel:
    def __init__(self, smoothing_alpha: float = 1.0) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.states: set[str] = set()

    def fit(self, patterns: list[str]) -> "TransitionModel":
        self.counts.clear()
        self.states = set(patterns)
        for i in range(len(patterns) - 1):
            src, dst = patterns[i], patterns[i + 1]
            self.counts[src][dst] += 1
            self.states.add(src)
            self.states.add(dst)
        return self

    def transition_probability(self, src: str, dst: str) -> float:
        outgoing = self.counts.get(src, {})
        total = sum(outgoing.values())
        vocab_size = max(len(self.states), 1)
        numerator = outgoing.get(dst, 0) + self.smoothing_alpha
        denominator = total + self.smoothing_alpha * vocab_size
        return numerator / denominator

    def outgoing_transitions(self, src: str) -> dict[str, float]:
        return {dst: self.transition_probability(src, dst) for dst in self.states}

    def observed_outgoing_transitions(self, src: str) -> dict[str, float]:
        outgoing = self.counts.get(src, {})
        return {dst: self.transition_probability(src, dst) for dst in outgoing}

    @property
    def transition_density(self) -> float:
        if not self.states:
            return 0.0
        total_transitions = sum(sum(d.values()) for d in self.counts.values())
        possible = max(len(self.states) * (len(self.states) - 1), 1)
        return total_transitions / possible
