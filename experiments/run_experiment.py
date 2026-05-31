"""Run experiments from CLI."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config, resolve_path
from src.evaluation.metrics import save_result
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_parameter_sensitivity,
    plot_pr_curve,
    plot_roc_curve,
    plot_state_diagram,
    plot_transition_heatmap,
)
from src.evaluation.stats import mcnemar_test, wilcoxon_test
from src.pipeline.experiment import run_experiment


def _result_path(config: dict, dataset: str, model: str, scenario: str, seed: int, window_size: int, alphabet_size: int) -> Path:
    base = resolve_path(config, "results")
    suffix = f"w{window_size}_a{alphabet_size}" if model == "automata" else ""
    name = f"seed_{seed}{('_' + suffix) if suffix else ''}.json"
    return base / dataset / model / scenario / name


def run_single(args: argparse.Namespace) -> dict:
    config = load_config()
    overrides = {}
    if args.window_size:
        overrides.setdefault("automata", {})["window_size"] = args.window_size
    if args.alphabet_size:
        overrides.setdefault("automata", {})["alphabet_size"] = args.alphabet_size
    if overrides:
        config = load_config(overrides=overrides)

    window_size = args.window_size or config["automata"]["window_size"]
    alphabet_size = args.alphabet_size or config["automata"]["alphabet_size"]

    out_path = _result_path(config, args.dataset, args.model, args.scenario, args.seed, window_size, alphabet_size)
    if out_path.exists() and not getattr(args, "force", False):
        print(f"Skip (exists): {out_path}")
        with out_path.open() as f:
            return json.load(f)

    result = run_experiment(
        dataset=args.dataset,
        model_name=args.model,
        scenario=args.scenario,
        seed=args.seed,
        config=config,
        window_size=window_size,
        alphabet_size=alphabet_size,
    )

    save_payload = {k: v for k, v in result.items() if not k.startswith("_")}
    out_path = _result_path(config, args.dataset, args.model, args.scenario, args.seed, window_size, alphabet_size)
    save_result(out_path, save_payload)
    print(f"Saved: {out_path} | F1={result.get('f1_mean', 0):.4f}")

    if args.model == "automata" and "_automata_model" in result:
        figures_dir = resolve_path(config, "figures")
        automata = result["_automata_model"]
        plot_transition_heatmap(automata.transition_model, figures_dir / f"heatmap_{args.dataset}_{args.scenario}.png")
        plot_state_diagram(automata.transition_model, figures_dir / f"state_diagram_{args.dataset}_{args.scenario}.png")

    eval_data = result.get("_eval", {})
    if eval_data.get("y_true") and eval_data.get("y_pred"):
        figures_dir = resolve_path(config, "figures")
        title = f"{args.model} - {args.dataset} - {args.scenario}"
        plot_confusion_matrix(
            eval_data["y_true"],
            eval_data["y_pred"],
            title,
            figures_dir / f"cm_{args.dataset}_{args.model}_{args.scenario}.png",
        )
        if eval_data.get("y_prob"):
            plot_roc_curve(
                eval_data["y_true"],
                eval_data["y_prob"],
                title,
                figures_dir / f"roc_{args.dataset}_{args.model}_{args.scenario}.png",
            )
            plot_pr_curve(
                eval_data["y_true"],
                eval_data["y_prob"],
                title,
                figures_dir / f"pr_{args.dataset}_{args.model}_{args.scenario}.png",
            )

    return result


def run_all(config: dict | None = None) -> None:
    config = config or load_config()
    datasets = ["skab", "batadal"]
    models = ["lstm", "gru", "cnn1d", "automata"]
    scenarios = ["original", "noisy", "unseen"]
    seeds = config["random_seeds"]

    for dataset in datasets:
        for model in models:
            for scenario in scenarios:
                for seed in seeds:
                    args = argparse.Namespace(
                        dataset=dataset,
                        model=model,
                        scenario=scenario,
                        seed=seed,
                        window_size=None,
                        alphabet_size=None,
                    )
                    try:
                        run_single(args)
                    except Exception as exc:
                        print(f"FAILED {dataset}/{model}/{scenario}/seed={seed}: {exc}")


def run_param_sweep(config: dict | None = None) -> None:
    config = config or load_config()
    grid = config["automata"]["param_grid"]
    records = []

    for dataset in ["skab", "batadal"]:
        for window_size in grid["window_size"]:
            for alphabet_size in grid["alphabet_size"]:
                for seed in config["random_seeds"][:2]:
                    args = argparse.Namespace(
                        dataset=dataset,
                        model="automata",
                        scenario="original",
                        seed=seed,
                        window_size=window_size,
                        alphabet_size=alphabet_size,
                    )
                    try:
                        result = run_single(args)
                        records.append(
                            {
                                "window_size": window_size,
                                "alphabet_size": alphabet_size,
                                "f1_mean": result.get("f1_mean", 0),
                                "state_count": result.get("state_count", 0),
                                "transition_density": result.get("transition_density", 0),
                            }
                        )
                    except Exception as exc:
                        print(f"Param sweep failed w={window_size} a={alphabet_size}: {exc}")

    figures_dir = resolve_path(config, "figures")
    if records:
        by_window = {}
        for r in records:
            w = r["window_size"]
            by_window.setdefault(w, []).append(r)
        for w, recs in by_window.items():
            plot_parameter_sensitivity(recs, "alphabet_size", figures_dir / f"param_sensitivity_w{w}.png")


def run_stats_comparison(config: dict | None = None) -> dict:
    config = config or load_config()
    results_dir = resolve_path(config, "results")
    summary: dict = {"wilcoxon": [], "mcnemar": []}

    model_pairs = [("automata", "lstm"), ("gru", "lstm"), ("automata", "gru")]
    for dataset in ["skab", "batadal"]:
        for scenario in ["original"]:
            for model_a, model_b in model_pairs:
                scores_a = _collect_f1_scores(results_dir, dataset, model_a, scenario, config["random_seeds"])
                scores_b = _collect_f1_scores(results_dir, dataset, model_b, scenario, config["random_seeds"])
                if len(scores_a) >= 2 and len(scores_a) == len(scores_b):
                    wt = wilcoxon_test(scores_a, scores_b)
                    summary["wilcoxon"].append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "model_a": model_a,
                            "model_b": model_b,
                            **wt,
                        }
                    )

            mcnemar_result = _run_paired_mcnemar(config, dataset, scenario, "lstm", "automata")
            if mcnemar_result:
                summary["mcnemar"].append(mcnemar_result)

    stats_path = results_dir / "statistical_tests.json"
    save_result(stats_path, summary)
    return summary


def _collect_f1_scores(results_dir: Path, dataset: str, model: str, scenario: str, seeds: list[int]) -> list[float]:
    scores = []
    for seed in seeds:
        paths = list((results_dir / dataset / model / scenario).glob(f"seed_{seed}*.json"))
        if paths:
            with paths[0].open() as f:
                scores.append(json.load(f).get("f1_mean", 0))
    return scores


def _run_paired_mcnemar(config: dict, dataset: str, scenario: str, model_a: str, model_b: str) -> dict | None:
    try:
        from src.pipeline.experiment import run_experiment

        result_a = run_experiment(dataset, model_a, scenario, config["random_seeds"][0], config=config)
        result_b = run_experiment(dataset, model_b, scenario, config["random_seeds"][0], config=config)
        eval_a = result_a.get("_eval", {})
        eval_b = result_b.get("_eval", {})
        if not eval_a.get("y_true") or not eval_b.get("y_pred"):
            return None
        y_true = np.array(eval_a["y_true"])
        pred_a = np.array(eval_a["y_pred"])
        pred_b = np.array(eval_b["y_pred"])
        min_len = min(len(y_true), len(pred_a), len(pred_b))
        test = mcnemar_test(y_true[:min_len], pred_a[:min_len], pred_b[:min_len])
        return {"dataset": dataset, "scenario": scenario, "model_a": model_a, "model_b": model_b, **test}
    except Exception as exc:
        return {"dataset": dataset, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="YazLab2 experiment runner")
    parser.add_argument("--dataset", choices=["skab", "batadal"])
    parser.add_argument("--model", choices=["lstm", "gru", "cnn1d", "automata"])
    parser.add_argument("--scenario", choices=["original", "noisy", "unseen"], default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--alphabet-size", type=int, default=None)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--param-sweep", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all()
    elif args.param_sweep:
        run_param_sweep()
    elif args.stats:
        summary = run_stats_comparison()
        print(json.dumps(summary, indent=2))
    elif args.dataset and args.model:
        run_single(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
