"""BATADAL Training Dataset 2 loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_COLUMN = "ATT_FLAG"
TIME_COLUMN = "DATETIME"
METADATA_COLS = {TIME_COLUMN, LABEL_COLUMN}


def load_batadal(raw_dir: Path) -> pd.DataFrame:
    candidates = [
        raw_dir / "BATADAL_dataset04.csv",
        raw_dir / "Training Dataset 2.csv",
        raw_dir / "training_dataset_2.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df[TIME_COLUMN] = pd.to_datetime(
                df[TIME_COLUMN], format="%m/%d/%y %H", errors="coerce"
            )
            df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
            if LABEL_COLUMN in df.columns:
                df[LABEL_COLUMN] = df[LABEL_COLUMN].replace(-999, 0).astype(int)
            return df

    raise FileNotFoundError(
        f"BATADAL Training Dataset 2 not found in {raw_dir}. "
        "Expected BATADAL_dataset04.csv (Training Dataset 2 with ATT_FLAG labels)"
    )


def get_batadal_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLS]


def get_batadal_labels(df: pd.DataFrame) -> pd.Series:
    return df[LABEL_COLUMN].astype(int)
