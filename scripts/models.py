"""
PyTorch models for univariate time-series forecasting.

Both models expect input shaped as (batch, seq_len) or (batch, seq_len, 1).
"""

from __future__ import annotations

from typing import Tuple

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
    """LSTM forecaster for a single univariate series."""

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, forecast_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        if x.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if x.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class PatchTSTForecast(nn.Module):
    """Patch-based Transformer forecaster for a single univariate series."""

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        patch_len: int = 16,
        stride: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if context_length < patch_len:
            raise ValueError("context_length must be >= patch_len.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.context_length = context_length
        self.forecast_length = forecast_length
        self.patch_len = patch_len
        self.stride = stride
        self.input_channels = input_channels

        num_patches = 1 + (context_length - patch_len) // stride
        patch_dim = patch_len * input_channels

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, forecast_length))

    def _make_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # x: (batch, seq_len, channels)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        num_patches = patches.size(1)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.view(patches.size(0), num_patches, -1)
        return patches, num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        if x.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if x.size(2) != self.input_channels:
            raise ValueError("Input channel count does not match input_channels.")

        patches, num_patches = self._make_patches(x)
        if num_patches > self.pos_emb.size(1):
            raise ValueError("Number of patches exceeds model positional embedding size.")

        tokens = self.patch_proj(patches) + self.pos_emb[:, :num_patches, :]
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)
