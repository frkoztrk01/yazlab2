"""SKAB dataset loader (valve1 + valve2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METADATA_COLS = {"datetime", "changepoint", "source_group", "source_file", "anomaly"}
EXCLUDE_FROM_FEATURES = METADATA_COLS


def load_skab(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in ("valve1", "valve2"):
        group_dir = raw_dir / group
        if not group_dir.exists():
            continue
        for csv_path in sorted(group_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, sep=";")
            df["source_group"] = group
            df["source_file"] = csv_path.name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No SKAB CSV files found under {raw_dir}")

    data = pd.concat(frames, ignore_index=True)
    if "anomaly" in data.columns:
        data["anomaly"] = data["anomaly"].astype(int)
    return data


def get_skab_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]


def get_skab_labels(df: pd.DataFrame) -> pd.Series:
    return df["anomaly"].astype(int)
