"""Base deep learning model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BaseDLModel(ABC):
    def __init__(self, config: dict, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.threshold: float = 0.5
        self._set_seed(seed)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def build_model(self, input_size: int) -> nn.Module:
        raise NotImplementedError

    def _pos_weight(self, y_train: np.ndarray) -> float:
        positives = float((y_train == 1).sum())
        negatives = float((y_train == 0).sum())
        if positives == 0:
            return 1.0
        return max(negatives / positives, 1.0)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        dl_cfg = self.config["dl"]
        input_size = X_train.shape[-1]
        self.model = self.build_model(input_size).to(self.device)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=dl_cfg["batch_size"],
            shuffle=True,
        )
        val_X = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self._pos_weight(y_train)], device=self.device)
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=dl_cfg["learning_rate"])

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(dl_cfg["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= max(len(train_loader), 1)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_X).squeeze(-1)
                val_loss = criterion(val_logits, val_y).item()

            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= dl_cfg["early_stopping_patience"]:
                    break

        if self.model is not None and "best_state" in locals():
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.threshold = self._tune_threshold(X_val, y_val)
        history["epochs_run"] = len(history["train_loss"])
        return history

    def _tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        if len(X_val) == 0:
            return 0.5
        probs = self.predict_proba(X_val)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 19):
            preds = (probs >= threshold).astype(int)
            tp = ((y_val == 1) & (preds == 1)).sum()
            fp = ((y_val == 0) & (preds == 1)).sum()
            fn = ((y_val == 1) & (preds == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        return best_threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        threshold = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= threshold).astype(int)
