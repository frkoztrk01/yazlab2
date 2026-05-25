"""Explainability module for probabilistic automata decisions."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.automata.automata import ProbabilisticAutomata, StepExplanation


def format_explanation_text(explanation: StepExplanation) -> str:
    lines = [
        "[SYSTEM DECISION]",
        f"Time Step: t = {explanation.time_step}",
        f'Previous State: "{explanation.state}"',
        f'Incoming Pattern: "{explanation.pattern}"',
        f"Status: {explanation.status.capitalize()}",
    ]
    if explanation.status == "unseen" and explanation.mapped_to is not None:
        lines.append(f'Nearest Pattern: "{explanation.mapped_to}" (distance = {explanation.distance})')
    lines.append("Transitions:")
    for t in explanation.transitions:
        lines.append(f'{t["from"]} -> {t["to"]} : {t["probability"]}')
    lines.append(f"Path Probability: {explanation.path_probability:.6f}")
    lines.append(f"Decision: {explanation.decision.upper()}")
    lines.append(f"Confidence Score: {explanation.confidence_score:.6f}")
    return "\n".join(lines)


def export_explanations(
    automata: ProbabilisticAutomata,
    series,
    output_path: Path | None = None,
) -> list[dict]:
    records = automata.to_json_explanations(series)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    return records
