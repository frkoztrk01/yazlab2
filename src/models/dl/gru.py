"""GRU model for anomaly detection."""

from __future__ import annotations

import torch.nn as nn

from src.models.dl.base import BaseDLModel


class GRUModel(BaseDLModel):
    def build_model(self, input_size: int) -> nn.Module:
        cfg = self.config["dl"]
        return _GRUNet(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )


class _GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
