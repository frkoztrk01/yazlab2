"""Preprocessing pipeline with train-only fit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    pca: PCA | None = None
    feature_columns: list[str] = field(default_factory=list)

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        use_pca: bool = False,
        pca_random_state: int | None = None,
    ) -> "Preprocessor":
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        if use_pca:
            self.pca = PCA(n_components=1, random_state=pca_random_state)
            self.pca.fit(X)
        return self

    def transform(self, df: pd.DataFrame, return_pc1: bool = False) -> np.ndarray:
        X = df[self.feature_columns].values
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        if return_pc1:
            if self.pca is None:
                raise ValueError("PCA not fitted")
            return self.pca.transform(X).ravel()
        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        use_pca: bool = False,
        pca_random_state: int | None = None,
    ) -> np.ndarray:
        self.fit(df, feature_columns, use_pca=use_pca, pca_random_state=pca_random_state)
        return self.transform(df, return_pc1=use_pca)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))

    sequences = []
    labels = []
    for i in range(seq_len, len(X)):
        sequences.append(X[i - seq_len : i])
        labels.append(y[i])
    return np.array(sequences), np.array(labels)
