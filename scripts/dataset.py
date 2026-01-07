"""
Torch dataset for sliding-window forecasting on the instant_dataset CSV.

Each item returns a 1D context window and a 1D target window for a single sector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


class TrafficWindowDataset(Dataset):
    """Create sector-by-sector windows from a wide traffic dataframe."""

    def __init__(
        self,
        csv_path: str | Path,
        context_length: int = 128,
        forecast_length: int = 1,
        timestamp_col: str = "timestamp",
        sector_cols: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if forecast_length <= 0:
            raise ValueError("forecast_length must be positive.")

        df = pd.read_csv(csv_path)

        if sector_cols is None:
            sector_cols = [col for col in df.columns if col != timestamp_col]
        sector_cols = list(sector_cols)
        if not sector_cols:
            raise ValueError("No sector columns found in dataframe.")

        values = df[sector_cols].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            raise ValueError(
                "NaN values found in sector data. Clean or fill missing values before training."
            )

        self.sectors = sector_cols
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.window_size = context_length + forecast_length
        self.values = torch.tensor(values.to_numpy(), dtype=dtype)

        self.num_steps = self.values.shape[0]
        self.num_sectors = self.values.shape[1]
        self.num_windows = self.num_steps - self.window_size + 1
        if self.num_windows <= 0:
            raise ValueError(
                "Not enough timesteps for the requested context and forecast lengths."
            )

    def __len__(self) -> int:
        return self.num_sectors * self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        sector_idx, window_idx = divmod(idx, self.num_windows)
        start = window_idx
        end_context = start + self.context_length
        end_target = end_context + self.forecast_length

        series = self.values[start:end_target, sector_idx]
        context = series[: self.context_length]
        target = series[self.context_length :]
        return context, target
