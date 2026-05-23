"""1D-CNN model for anomaly detection."""

from __future__ import annotations

import torch.nn as nn

from src.models.dl.base import BaseDLModel


class CNN1DModel(BaseDLModel):
    def build_model(self, input_size: int) -> nn.Module:
        cfg = self.config["dl"]
        return _CNN1DNet(input_size=input_size, hidden_size=cfg["hidden_size"], dropout=cfg["dropout"])


class _CNN1DNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
