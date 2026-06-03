"""Cross-dataset evaluation (Tablo 3): train on one dataset, test on another."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config, resolve_path
from src.data.splits import batadal_temporal_split, skab_group_kfold, split_train_val
from src.evaluation.metrics import compute_metrics, save_result
from src.models.automata.automata import AutomataConfig, ProbabilisticAutomata, align_segment_labels
from src.models.dl.cnn1d import CNN1DModel
from src.models.dl.gru import GRUModel
from src.models.dl.lstm import LSTMModel
from src.pipeline.experiment import (
    DL_MODELS,
    _labels_from_df,
    get_features_and_labels,
    load_dataset,
)
from src.preprocessing.pipeline import Preprocessor, create_sequences

DEFAULT_SEED = 42


def _get_splits(config: dict, dataset: str, df, labels: np.ndarray, seed: int):
    if dataset == "skab":
        groups = df[config["splits"]["skab"]["group_col"]].values
        fold = next(skab_group_kfold(labels, groups, config["splits"]["skab"]["n_folds"], seed))
        _, train_idx, test_idx = fold
        train_sub, val_sub = split_train_val(train_idx, labels, seed=seed)
        return df.iloc[train_sub], df.iloc[val_sub], df.iloc[test_idx]
    split = batadal_temporal_split(
        len(df),
        config["splits"]["batadal"]["train"],
        config["splits"]["batadal"]["val"],
        config["splits"]["batadal"]["test"],
    )
    return df.iloc[split.train_idx], df.iloc[split.val_idx], df.iloc[split.test_idx]


def _pc1_series(preprocessor: Preprocessor, df, feature_cols: list[str]) -> np.ndarray:
    return preprocessor.transform(df, return_pc1=True)


def run_cross_automata(
    config: dict,
    train_dataset: str,
    test_dataset: str,
    seed: int = DEFAULT_SEED,
) -> dict:
    train_df = load_dataset(config, train_dataset)
    test_df = load_dataset(config, test_dataset)
    train_cols, train_labels = get_features_and_labels(train_df, train_dataset)
    test_cols, _ = get_features_and_labels(test_df, test_dataset)

    tr_train, tr_val, _ = _get_splits(config, train_dataset, train_df, train_labels.values, seed)
    _, _, te_test = _get_splits(config, test_dataset, test_df, get_features_and_labels(test_df, test_dataset)[1].values, seed)

    prep = Preprocessor().fit(tr_train, train_cols, use_pca=True)
    auto_cfg = config["automata"]
    model = ProbabilisticAutomata(
        AutomataConfig(
            window_size=auto_cfg["window_size"],
            alphabet_size=auto_cfg["alphabet_size"],
            paa_segment_size=auto_cfg["paa_segment_size"],
            smoothing_alpha=auto_cfg["smoothing_alpha"],
        )
    )

    train_pc1 = _pc1_series(prep, tr_train, train_cols)
    val_pc1 = _pc1_series(prep, tr_val, train_cols)
    test_pc1 = _pc1_series(
        Preprocessor().fit(te_test.iloc[: max(1, len(te_test) // 2)], test_cols, use_pca=True),
        te_test,
        test_cols,
    )

    y_val = _labels_from_df(tr_val, train_cols)
    y_test = _labels_from_df(te_test, test_cols)

    start = time.perf_counter()
    model.fit(train_pc1, val_pc1, y_val)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    preds, probs, pred_indices = model.predict(test_pc1)
    infer_time = time.perf_counter() - start

    if len(preds):
        y_aligned = align_segment_labels(y_test, pred_indices, auto_cfg["paa_segment_size"])
        metrics = compute_metrics(y_aligned, preds, probs)
    else:
        from src.evaluation.metrics import MetricResult
        metrics = MetricResult(0, 0, 0, 0, [[0, 0], [0, 0]])

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "model": "automata",
        "f1": metrics.f1,
        "training_time_sec": train_time,
        "inference_time_sec": infer_time,
    }


def run_cross_dl(
    config: dict,
    train_dataset: str,
    test_dataset: str,
    model_name: str,
    seed: int = DEFAULT_SEED,
) -> dict:
    train_df = load_dataset(config, train_dataset)
    test_df = load_dataset(config, test_dataset)
    train_cols, train_labels = get_features_and_labels(train_df, train_dataset)
    test_cols, test_labels = get_features_and_labels(test_df, test_dataset)

    tr_train, tr_val, _ = _get_splits(config, train_dataset, train_df, train_labels.values, seed)
    _, _, te_test = _get_splits(config, test_dataset, test_df, test_labels.values, seed)

    prep_train = Preprocessor().fit(tr_train, train_cols, use_pca=True)
    prep_test = Preprocessor().fit(te_test.iloc[: max(1, len(te_test) // 2)], test_cols, use_pca=True)

    def to_seq(df, cols, prep):
        pc1 = prep.transform(df, return_pc1=True).reshape(-1, 1)
        y = _labels_from_df(df, cols)
        return create_sequences(pc1, y, config["dl"]["sequence_length"])

    X_train, y_train = to_seq(tr_train, train_cols, prep_train)
    X_val, y_val = to_seq(tr_val, train_cols, prep_train)
    X_test, y_test = to_seq(te_test, test_cols, prep_test)

    if len(X_train) == 0 or len(X_test) == 0:
        return {"train_dataset": train_dataset, "test_dataset": test_dataset, "model": model_name, "f1": 0.0}

    model = DL_MODELS[model_name](config, seed=seed)
    start = time.perf_counter()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    preds = model.predict(X_test)
    infer_time = time.perf_counter() - start

    metrics = compute_metrics(y_test, preds, model.predict_proba(X_test))
    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "model": model_name,
        "f1": metrics.f1,
        "training_time_sec": train_time,
        "inference_time_sec": infer_time,
    }


def run_all_cross(config: dict | None = None, seed: int = DEFAULT_SEED) -> list[dict]:
    config = config or load_config()
    datasets = ["skab", "batadal"]
    models = ["lstm", "gru", "cnn1d", "automata"]
    results = []

    for train_ds in datasets:
        for test_ds in datasets:
            for model in models:
                try:
                    if model == "automata":
                        rec = run_cross_automata(config, train_ds, test_ds, seed)
                    else:
                        rec = run_cross_dl(config, train_ds, test_ds, model, seed)
                    results.append(rec)
                    print(f"Cross {model}: train={train_ds} test={test_ds} F1={rec['f1']:.4f}")
                except Exception as exc:
                    print(f"Cross failed {model} {train_ds}->{test_ds}: {exc}")
                    results.append(
                        {
                            "train_dataset": train_ds,
                            "test_dataset": test_ds,
                            "model": model,
                            "f1": None,
                            "error": str(exc),
                        }
                    )
    return results


def main() -> None:
    config = load_config()
    results = run_all_cross(config)
    out = resolve_path(config, "results") / "cross_dataset.json"
    save_result(out, {"seed": DEFAULT_SEED, "results": results})
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
