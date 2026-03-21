"""Torch dataset for split-aware sliding-window forecasting on the full wide CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from src.config import DEFAULT_TIMESTAMP_COL
from src.experiment import ExperimentManifest, load_manifest, load_wide_dataframe, valid_window_start_indices


class TrafficWindowDataset(Dataset):
    """Create chronological windows from the canonical wide dataset."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        context_length: int = 128,
        forecast_length: int = 1,
        timestamp_col: str = DEFAULT_TIMESTAMP_COL,
        manifest: ExperimentManifest | None = None,
        manifest_path: str | Path | None = None,
        sector_cols: Sequence[str] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if forecast_length <= 0:
            raise ValueError("forecast_length must be positive.")

        if manifest is None and manifest_path is not None:
            manifest = load_manifest(manifest_path)

        self.manifest = manifest
        self.split = split.strip().lower()

        df = load_wide_dataframe(csv_path, timestamp_col=timestamp_col)

        if sector_cols is None:
            sector_cols = [col for col in df.columns if col != timestamp_col]
        sector_cols = list(sector_cols)
        if not sector_cols:
            raise ValueError("No sector columns found in dataframe.")

        self.sectors = sector_cols
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.window_size = context_length + forecast_length
        self.values = torch.tensor(df[sector_cols].to_numpy(), dtype=dtype)

        self.num_steps = self.values.shape[0]
        self.num_sectors = self.values.shape[1]
        if self.manifest is not None:
            self.start_indices = valid_window_start_indices(
                self.manifest,
                self.split,
                context_length=self.context_length,
                horizon=self.forecast_length,
            ).astype(int).tolist()
        else:
            raw_starts = range(self.context_length, self.num_steps - self.forecast_length + 1)
            self.start_indices = [int(start) for start in raw_starts]

        self.num_windows = len(self.start_indices)
        if self.num_windows <= 0:
            raise ValueError(
                f"No valid windows found for split={self.split} with "
                f"context_length={self.context_length} and forecast_length={self.forecast_length}."
            )

    def __len__(self) -> int:
        return self.num_sectors * self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        sector_idx, window_pos = divmod(idx, self.num_windows)
        target_start = int(self.start_indices[window_pos])
        context_start = target_start - self.context_length
        end_context = target_start
        end_target = end_context + self.forecast_length

        series = self.values[context_start:end_target, sector_idx]
        context = series[: self.context_length]
        target = series[self.context_length :]
        return context, target
