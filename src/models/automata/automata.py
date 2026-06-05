"""Probabilistic automata for time series anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.automata.transitions import TransitionModel
from src.models.automata.unseen import map_pattern
from src.preprocessing.paa import paa_transform
from src.preprocessing.sax import fit_sax_breakpoints, sax_transform


def align_segment_labels(y: np.ndarray, indices: np.ndarray, segment_size: int) -> np.ndarray:
    aligned = []
    for idx in indices:
        start = max(0, int(idx) - segment_size + 1)
        window = y[start : int(idx) + 1]
        aligned.append(1 if len(window) and window.max() == 1 else 0)
    return np.array(aligned, dtype=int)


@dataclass
class AutomataConfig:
    window_size: int = 4
    alphabet_size: int = 3
    paa_segment_size: int = 8
    smoothing_alpha: float = 1.0


@dataclass
class StepExplanation:
    time_step: int
    state: str
    pattern: str
    status: str
    mapped_to: str | None
    distance: int | None
    transitions: list[dict[str, float | str]]
    path_probability: float
    decision: str
    confidence_score: float


@dataclass
class ProbabilisticAutomata:
    config: AutomataConfig
    breakpoints: np.ndarray | None = None
    pattern_dictionary: set[str] = field(default_factory=set)
    transition_model: TransitionModel = field(default_factory=TransitionModel)
    threshold: float = 0.01

    def _series_to_patterns(self, series: np.ndarray, breakpoints: np.ndarray) -> list[str]:
        paa_values = paa_transform(series, self.config.paa_segment_size)
        sax_string = sax_transform(paa_values, self.config.alphabet_size, breakpoints)
        patterns = []
        w = self.config.window_size
        for i in range(len(sax_string) - w + 1):
            patterns.append(sax_string[i : i + w])
        return patterns

    def _prediction_indices(self, series_length: int) -> np.ndarray:
        segment_size = self.config.paa_segment_size
        window = self.config.window_size
        paa_n = max(1, series_length // segment_size)
        sax_len = paa_n
        n_patterns = max(0, sax_len - window + 1)
        n_preds = max(0, n_patterns - 1)
        indices = []
        for i in range(n_preds):
            sax_pos = i + 1
            orig_idx = min((sax_pos + window - 1) * segment_size - 1, series_length - 1)
            indices.append(orig_idx)
        return np.array(indices, dtype=int)

    def fit(self, train_series: np.ndarray, val_series: np.ndarray | None = None, val_labels: np.ndarray | None = None) -> "ProbabilisticAutomata":
        paa_values = paa_transform(train_series, self.config.paa_segment_size)
        self.breakpoints = fit_sax_breakpoints(paa_values, self.config.alphabet_size)

        train_patterns = self._series_to_patterns(train_series, self.breakpoints)
        self.pattern_dictionary = set(train_patterns)
        self.transition_model = TransitionModel(self.config.smoothing_alpha).fit(train_patterns)

        if val_series is not None and val_labels is not None and len(val_series) > 0:
            val_probs = self._path_probabilities(val_series)
            val_indices = self._prediction_indices(len(val_series))
            if len(val_probs) > 0 and len(val_indices) >= len(val_probs):
                aligned_labels = align_segment_labels(
                    val_labels, val_indices[: len(val_probs)], self.config.paa_segment_size
                )
                self.threshold = self._tune_threshold(val_probs, aligned_labels)
        return self

    def _tune_threshold(self, probs: np.ndarray, labels: np.ndarray) -> float:
        labels = labels.astype(int)
        best_threshold = float(np.median(probs))
        best_f1 = -1.0

        candidates = set(np.unique(probs).tolist())
        attack_rate = float(labels.mean()) if len(labels) else 0.0
        if attack_rate > 0:
            candidates.add(float(np.quantile(probs, max(attack_rate, 0.01))))
            candidates.add(float(np.quantile(probs, min(attack_rate * 2, 0.99))))

        for threshold in candidates:
            preds = (probs < threshold).astype(int)
            f1 = _f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        return best_threshold

    def _path_probabilities(self, series: np.ndarray) -> np.ndarray:
        if self.breakpoints is None:
            raise ValueError("Model not fitted")

        patterns = self._series_to_patterns(series, self.breakpoints)
        if len(patterns) < 2:
            return np.array([])

        probs = []
        prev_mapped, _, _ = map_pattern(patterns[0], self.pattern_dictionary)
        for i in range(1, len(patterns)):
            curr_raw = patterns[i]
            curr_mapped, _, _ = map_pattern(curr_raw, self.pattern_dictionary)
            prob = self.transition_model.transition_probability(prev_mapped, curr_mapped)
            probs.append(prob)
            prev_mapped = curr_mapped
        return np.array(probs)

    def predict(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self._path_probabilities(series)
        indices = self._prediction_indices(len(series))
        if len(probs) == 0:
            return np.array([]), np.array([]), np.array([])

        preds = (probs < self.threshold).astype(int)
        return preds, probs, indices[: len(preds)]

    def explain(self, series: np.ndarray) -> list[StepExplanation]:
        if self.breakpoints is None:
            raise ValueError("Model not fitted")

        patterns = self._series_to_patterns(series, self.breakpoints)
        explanations: list[StepExplanation] = []

        if len(patterns) < 2:
            return explanations

        prev_mapped, prev_status, _ = map_pattern(patterns[0], self.pattern_dictionary)
        path_prob = 1.0

        for i in range(1, len(patterns)):
            raw = patterns[i]
            mapped, status, distance = map_pattern(raw, self.pattern_dictionary)
            trans_prob = self.transition_model.transition_probability(prev_mapped, mapped)
            path_prob *= trans_prob

            decision = "anomaly" if path_prob < self.threshold else "normal"
            explanations.append(
                StepExplanation(
                    time_step=i,
                    state=prev_mapped,
                    pattern=raw,
                    status=status,
                    mapped_to=mapped if status == "unseen" else None,
                    distance=distance,
                    transitions=[
                        {
                            "from": prev_mapped,
                            "to": mapped,
                            "probability": trans_prob,
                        }
                    ],
                    path_probability=path_prob,
                    decision=decision,
                    confidence_score=path_prob,
                )
            )
            prev_mapped = mapped
            prev_status = status

        return explanations

    def to_json_explanations(self, series: np.ndarray) -> list[dict]:
        return [
            {
                "time_step": e.time_step,
                "state": e.state,
                "pattern": e.pattern,
                "status": e.status,
                "mapped_to": e.mapped_to,
                "distance": e.distance,
                "probability": round(e.path_probability, 6),
                "decision": e.decision,
                "confidence_score": round(e.confidence_score, 6),
                "transitions": e.transitions,
            }
            for e in self.explain(series)
        ]

    @property
    def state_count(self) -> int:
        return len(self.pattern_dictionary)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Segment-level label alignment for BATADAL evaluation.
