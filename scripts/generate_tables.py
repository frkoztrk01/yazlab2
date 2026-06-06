"""Generate all 5 report tables matching yazlab2-EK.pdf format."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config, resolve_path


def load_all_results(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.rglob("seed_*.json")):
        with path.open() as f:
            rec = json.load(f)
            rec["_path"] = str(path)
            records.append(rec)
    return records


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def _fmt_mean(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{np.mean(values):.3f}"


def _is_default_automata(rec: dict) -> bool:
    name = Path(rec["_path"]).name
    if rec["model"] != "automata":
        return True
    if "_w" in name and "_a" in name:
        return rec.get("window_size") == 4 and rec.get("alphabet_size") == 3
    return True


def _collect_f1(records: list[dict], model: str, dataset: str, scenario: str) -> list[float]:
    return [
        r.get("f1_mean", 0)
        for r in records
        if r["model"] == model and r["dataset"] == dataset and r.get("scenario") == scenario and _is_default_automata(r)
    ]


def table1_performance(records: list[dict]) -> str:
    lines = [
        "### Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)",
        "",
        "| Model | SKAB | BATADAL |",
        "|-------|------|---------|",
    ]
    labels = {"lstm": "LSTM", "gru": "GRU", "cnn1d": "1D-CNN", "automata": "Automata"}
    for model in ["lstm", "gru", "cnn1d", "automata"]:
        skab = _collect_f1(records, model, "skab", "original")
        batadal = _collect_f1(records, model, "batadal", "original")
        lines.append(f"| {labels[model]} | {_fmt_mean_std(skab)} | {_fmt_mean_std(batadal)} |")
    return "\n".join(lines)


def table2_robustness(records: list[dict]) -> str:
    lines = [
        "### Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi",
        "",
        "| Model | Orijinal (F1) | Gürültülü (F1) | Det. Rate | Map. Acc. |",
        "|-------|---------------|----------------|-----------|-----------|",
    ]
    labels = {"lstm": "LSTM", "gru": "GRU", "cnn1d": "1D-CNN", "automata": "Automata"}

    for model in ["lstm", "gru", "cnn1d", "automata"]:
        orig = _collect_f1(records, model, "skab", "original")
        noisy = _collect_f1(records, model, "skab", "noisy")
        det_rates: list[float] = []
        map_accs: list[float] = []

        if model == "automata":
            unseen_recs = [r for r in records if r["model"] == "automata" and r["dataset"] == "skab" and r.get("scenario") == "unseen" and _is_default_automata(r)]
            for r in unseen_recs:
                for fr in r.get("fold_results", []):
                    if fr.get("unseen_count", 0) > 0:
                        if fr.get("detection_rate_unseen") is not None:
                            det_rates.append(fr["detection_rate_unseen"])
                        if fr.get("mapping_accuracy") is not None:
                            map_accs.append(fr["mapping_accuracy"])

        det_str = _fmt_mean(det_rates) if det_rates else ("—" if model != "automata" else "—")
        map_str = _fmt_mean(map_accs) if map_accs else ("—" if model != "automata" else "—")

        lines.append(
            f"| {labels[model]} | {_fmt_mean_std(orig)} | {_fmt_mean_std(noisy)} | {det_str} | {map_str} |"
        )
    return "\n".join(lines)


def table3_cross_dataset(results_dir: Path) -> str:
    cross_path = results_dir / "cross_dataset.json"
    lines = [
        "### Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-score)",
        "",
        "| Train / Test | SKAB | BATADAL |",
        "|--------------|------|---------|",
    ]

    matrix: dict[tuple[str, str], list[float]] = defaultdict(list)
    if cross_path.exists():
        data = json.loads(cross_path.read_text())
        for rec in data.get("results", []):
            if rec.get("f1") is not None:
                matrix[(rec["train_dataset"], rec["test_dataset"])].append(rec["f1"])

    if not matrix:
        for train in ["skab", "batadal"]:
            row = [f"Train: {train.upper()}"]
            for test in ["skab", "batadal"]:
                row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("> Cross-dataset sonuçları için: `PYTHONPATH=. python scripts/run_cross_dataset.py`")
        return "\n".join(lines)

    # aggregate by model - use best model (LSTM) or average all models
    model_matrix: dict[tuple[str, str], list[float]] = defaultdict(list)
    data = json.loads(cross_path.read_text())
    for rec in data.get("results", []):
        if rec.get("model") == "lstm" and rec.get("f1") is not None:
            model_matrix[(rec["train_dataset"], rec["test_dataset"])].append(rec["f1"])

    for train in ["skab", "batadal"]:
        row = [f"Train: {train.upper()}"]
        for test in ["skab", "batadal"]:
            vals = model_matrix.get((train, test), [])
            row.append(_fmt_mean(vals) if vals else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("> Tablo 3 LSTM modeli ile cross-dataset (PC1 transfer) sonuçlarını gösterir.")
    return "\n".join(lines)


def table4_param_sensitivity(records: list[dict]) -> str:
    lines = [
        "### Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score, SKAB)",
        "",
        "| Parametre | Değer = 3 | Değer = 4 | Değer = 5 | Değer = 6 |",
        "|-----------|-----------|-----------|-----------|-----------|",
    ]

    fixed_alphabet = 3
    fixed_window = 4

    window_row = ["| Window Size"]
    for w in [3, 4, 5, 6]:
        vals = [
            r.get("f1_mean", 0)
            for r in records
            if r["model"] == "automata"
            and r["dataset"] == "skab"
            and r.get("scenario") == "original"
            and r.get("window_size") == w
            and r.get("alphabet_size") == fixed_alphabet
        ]
        window_row.append(_fmt_mean(vals))
    lines.append(" | ".join(window_row) + " |")

    alpha_row = ["| Alphabet Size"]
    for a in [3, 4, 5, 6]:
        vals = [
            r.get("f1_mean", 0)
            for r in records
            if r["model"] == "automata"
            and r["dataset"] == "skab"
            and r.get("scenario") == "original"
            and r.get("window_size") == fixed_window
            and r.get("alphabet_size") == a
        ]
        alpha_row.append(_fmt_mean(vals))
    lines.append(" | ".join(alpha_row) + " |")

    if all("—" in row for row in [window_row, alpha_row]):
        lines.append("")
        lines.append("> Parametre sweep için: `PYTHONPATH=. python experiments/run_experiment.py --param-sweep`")

    return "\n".join(lines)


def table5_runtime(records: list[dict]) -> str:
    lines = [
        "### Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması",
        "",
        "| Model | Training Time (sn) | Inference Time (sn) |",
        "|-------|--------------------|---------------------|",
    ]
    labels = {"lstm": "LSTM", "gru": "GRU", "cnn1d": "1D-CNN", "automata": "Automata"}
    for model in ["lstm", "gru", "cnn1d", "automata"]:
        vals = [
            r for r in records
            if r["model"] == model and r.get("scenario") == "original" and _is_default_automata(r)
        ]
        if not vals:
            continue
        train = np.mean([r.get("training_time_sec", 0) for r in vals])
        infer = np.mean([r.get("inference_time_sec", 0) for r in vals])
        lines.append(f"| {labels[model]} | {train:.3f} | {infer:.3f} |")
    return "\n".join(lines)


def generate_all_tables(results_dir: Path | None = None) -> str:
    config = load_config()
    results_dir = results_dir or resolve_path(config, "results")
    records = load_all_results(results_dir)

    sections = [
        table1_performance(records),
        "",
        table2_robustness(records),
        "",
        table3_cross_dataset(results_dir),
        "",
        table4_param_sensitivity(records),
        "",
        table5_runtime(records),
    ]
    return "\n".join(sections)


def main() -> None:
    print(generate_all_tables())


if __name__ == "__main__":
    main()
