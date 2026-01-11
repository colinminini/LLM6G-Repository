"""
PyTorch models for univariate time-series quantile forecasting.

The model expects input shaped as (batch, seq_len) or (batch, seq_len, 1).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3:
        return x
    raise ValueError("Expected input with shape (seq,), (batch, seq), or (batch, seq, channels).")


class LSTMForecast(nn.Module):
    """LSTM forecaster that outputs multiple quantiles."""

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.quantiles = tuple(float(q) for q in quantiles)
        if not self.quantiles:
            raise ValueError("quantiles must contain at least one entry.")
        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError("quantiles must be between 0 and 1 (exclusive).")
        self.num_quantiles = len(self.quantiles)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, forecast_length * self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        if x.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if x.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        preds = self.head(last_hidden)
        return preds.reshape(x.size(0), self.forecast_length, self.num_quantiles)
