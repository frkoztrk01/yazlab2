"""End-to-end experiment pipeline."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.data.batadal_loader import get_batadal_feature_columns, get_batadal_labels, load_batadal
from src.data.scenarios import add_gaussian_noise
from src.data.skab_loader import get_skab_feature_columns, get_skab_labels, load_skab
from src.data.splits import batadal_temporal_split, skab_group_kfold, split_train_val
from src.evaluation.metrics import MetricResult, compute_metrics
from src.models.automata.automata import AutomataConfig, ProbabilisticAutomata, align_segment_labels
from src.models.dl.cnn1d import CNN1DModel
from src.models.dl.gru import GRUModel
from src.models.dl.lstm import LSTMModel
from src.preprocessing.pipeline import Preprocessor, create_sequences


DL_MODELS = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn1d": CNN1DModel,
}


def load_dataset(config: dict, dataset: str) -> pd.DataFrame:
    if dataset == "skab":
        return load_skab(resolve_path(config, "skab_raw"))
    if dataset == "batadal":
        return load_batadal(resolve_path(config, "batadal_raw"))
    raise ValueError(f"Unknown dataset: {dataset}")


def get_features_and_labels(df: pd.DataFrame, dataset: str) -> tuple[list[str], pd.Series]:
    if dataset == "skab":
        return get_skab_feature_columns(df), get_skab_labels(df)
    return get_batadal_feature_columns(df), get_batadal_labels(df)


def run_automata_fold(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    scenario: str,
    seed: int,
    window_size: int | None = None,
    alphabet_size: int | None = None,
) -> dict[str, Any]:
    auto_cfg = config["automata"]
    preprocessor = Preprocessor()
    preprocessor.fit(train_df, feature_cols, use_pca=True)

    train_pc1 = preprocessor.transform(train_df, return_pc1=True)
    val_pc1 = preprocessor.transform(val_df, return_pc1=True)
    test_pc1 = preprocessor.transform(test_df, return_pc1=True)

    if scenario == "noisy":
        train_pc1 = add_gaussian_noise(train_pc1, config["noise"]["gaussian_std"], seed)
        val_pc1 = add_gaussian_noise(val_pc1, config["noise"]["gaussian_std"], seed + 1)
        test_pc1 = add_gaussian_noise(test_pc1, config["noise"]["gaussian_std"], seed + 2)

    automata_cfg = AutomataConfig(
        window_size=window_size or auto_cfg["window_size"],
        alphabet_size=alphabet_size or auto_cfg["alphabet_size"],
        paa_segment_size=auto_cfg["paa_segment_size"],
        smoothing_alpha=auto_cfg["smoothing_alpha"],
    )
    model = ProbabilisticAutomata(automata_cfg)

    y_val = _labels_from_df(val_df, feature_cols)
    y_test = _labels_from_df(test_df, feature_cols)

    start = time.perf_counter()
    model.fit(train_pc1, val_pc1, y_val)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    preds, probs, pred_indices = model.predict(test_pc1)
    inference_time = time.perf_counter() - start

    if len(preds) > 0:
        y_test_aligned = align_segment_labels(y_test, pred_indices, auto_cfg["paa_segment_size"])
    else:
        y_test_aligned = np.array([])

    metrics = compute_metrics(y_test_aligned, preds, probs)

    unseen_count = 0
    mapping_correct = 0
    detection_rate = 0.0
    if len(preds) > 0:
        explanations = model.explain(test_pc1)
        unseen_explanations = [e for e in explanations if e.status == "unseen"]
        unseen_count = len(unseen_explanations)
        mapping_correct = sum(1 for e in unseen_explanations if e.mapped_to is not None)
        if unseen_count > 0:
            detection_rate = sum(1 for e in unseen_explanations if e.decision == "anomaly") / unseen_count

    return {
        "metrics": asdict(metrics),
        "state_count": model.state_count,
        "transition_density": model.transition_model.transition_density,
        "training_time_sec": train_time,
        "inference_time_sec": inference_time,
        "unseen_count": unseen_count,
        "mapping_accuracy": mapping_correct / max(unseen_count, 1),
        "detection_rate_unseen": detection_rate,
        "threshold": model.threshold,
        "y_true": y_test_aligned.tolist(),
        "y_pred": preds.tolist(),
        "y_prob": probs.tolist(),
        "model_obj": model,
    }


def _labels_from_df(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if "anomaly" in df.columns:
        return df["anomaly"].astype(int).values
    if "ATT_FLAG" in df.columns:
        return df["ATT_FLAG"].astype(int).values
    raise KeyError("Label column not found")


def run_dl_fold(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    preprocessor = Preprocessor()
    preprocessor.fit(train_df, feature_cols, use_pca=False)

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_train = _labels_from_df(train_df, feature_cols)
    y_val = _labels_from_df(val_df, feature_cols)
    y_test = _labels_from_df(test_df, feature_cols)

    if scenario == "noisy":
        X_train = add_gaussian_noise(X_train, config["noise"]["gaussian_std"], seed)
        X_val = add_gaussian_noise(X_val, config["noise"]["gaussian_std"], seed + 1)
        X_test = add_gaussian_noise(X_test, config["noise"]["gaussian_std"], seed + 2)

    seq_len = config["dl"]["sequence_length"]
    X_train, y_train = create_sequences(X_train, y_train, seq_len)
    X_val, y_val = create_sequences(X_val, y_val, seq_len)
    X_test, y_test = create_sequences(X_test, y_test, seq_len)

    if len(X_train) == 0 or len(X_test) == 0:
        empty = MetricResult(0, 0, 0, 0, [[0, 0], [0, 0]])
        return {"metrics": asdict(empty), "training_time_sec": 0, "inference_time_sec": 0}

    model_cls = DL_MODELS[model_name]
    model = model_cls(config, seed=seed)

    start = time.perf_counter()
    history = model.fit(X_train, y_train, X_val, y_val)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    inference_time = time.perf_counter() - start

    metrics = compute_metrics(y_test, preds, probs)
    return {
        "metrics": asdict(metrics),
        "training_time_sec": train_time,
        "inference_time_sec": inference_time,
        "epochs_run": history.get("epochs_run", 0),
        "y_true": y_test.tolist(),
        "y_pred": preds.tolist(),
        "y_prob": probs.tolist(),
    }


def run_experiment(
    dataset: str,
    model_name: str,
    scenario: str,
    seed: int,
    config: dict | None = None,
    window_size: int | None = None,
    alphabet_size: int | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    df = load_dataset(config, dataset)
    feature_cols, labels = get_features_and_labels(df, dataset)
    labels_arr = labels.values

    fold_results = []
    is_automata = model_name == "automata"

    if dataset == "skab":
        groups = df[config["splits"]["skab"]["group_col"]].values
        n_folds = config["splits"]["skab"]["n_folds"]
        for fold_id, train_idx, test_idx in skab_group_kfold(labels_arr, groups, n_folds, seed):
            train_sub, val_sub = split_train_val(train_idx, labels_arr, seed=seed)
            train_df = df.iloc[train_sub]
            val_df = df.iloc[val_sub]
            test_df = df.iloc[test_idx]

            if is_automata:
                result = run_automata_fold(
                    config, train_df, val_df, test_df, feature_cols, scenario, seed, window_size, alphabet_size
                )
            else:
                result = run_dl_fold(config, train_df, val_df, test_df, feature_cols, model_name, scenario, seed)

            result["fold"] = fold_id
            fold_results.append(result)
    else:
        split = batadal_temporal_split(
            len(df),
            config["splits"]["batadal"]["train"],
            config["splits"]["batadal"]["val"],
            config["splits"]["batadal"]["test"],
        )
        train_df = df.iloc[split.train_idx]
        val_df = df.iloc[split.val_idx]
        test_df = df.iloc[split.test_idx]

        if is_automata:
            result = run_automata_fold(
                config, train_df, val_df, test_df, feature_cols, scenario, seed, window_size, alphabet_size
            )
        else:
            result = run_dl_fold(config, train_df, val_df, test_df, feature_cols, model_name, scenario, seed)
        result["fold"] = 0
        fold_results.append(result)

    metrics_list = [MetricResult(**{k: v for k, v in r["metrics"].items() if k != "confusion_matrix"}, confusion_matrix=r["metrics"]["confusion_matrix"]) for r in fold_results]

    from src.evaluation.metrics import aggregate_metrics

    agg = aggregate_metrics(metrics_list)

    payload = {
        "dataset": dataset,
        "model": model_name,
        "scenario": scenario,
        "seed": seed,
        "window_size": window_size or config["automata"]["window_size"],
        "alphabet_size": alphabet_size or config["automata"]["alphabet_size"],
        "fold_results": [
            {k: v for k, v in r.items() if k not in ("model_obj", "y_true", "y_pred", "y_prob")}
            for r in fold_results
        ],
        "aggregate": agg,
        "f1_mean": agg.get("f1_mean", 0),
        "f1_std": agg.get("f1_std", 0),
        "state_count": fold_results[-1].get("state_count"),
        "transition_density": fold_results[-1].get("transition_density"),
        "training_time_sec": float(np.mean([r.get("training_time_sec", 0) for r in fold_results])),
        "inference_time_sec": float(np.mean([r.get("inference_time_sec", 0) for r in fold_results])),
    }

    if fold_results:
        last = fold_results[-1]
        payload["_eval"] = {
            "y_true": last.get("y_true", []),
            "y_pred": last.get("y_pred", []),
            "y_prob": last.get("y_prob", []),
        }
        if "model_obj" in last:
            payload["_automata_model"] = last["model_obj"]

    return payload
