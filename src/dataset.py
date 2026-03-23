"""Torch dataset utilities for split-aware forecasting on the canonical wide CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DEFAULT_TIMESTAMP_COL
from src.deepar_support import (
    DeepARFeatureSpec,
    build_feature_spec,
    build_series_item_index,
    build_time_feature_matrix,
)
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
        window_step: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if forecast_length <= 0:
            raise ValueError("forecast_length must be positive.")
        if int(window_step) <= 0:
            raise ValueError("window_step must be positive.")

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
        self.window_step = int(window_step)
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
            )[:: self.window_step].astype(int).tolist()
        else:
            raw_starts = range(
                self.context_length,
                self.num_steps - self.forecast_length + 1,
                self.window_step,
            )
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


class DeepARWindowDataset(Dataset):
    """DeepAR training/eval windows with time covariates, item ids, and optional padding."""

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
        window_step: int = 1,
        allow_left_padding: bool | None = None,
        feature_spec: DeepARFeatureSpec | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if forecast_length <= 0:
            raise ValueError("forecast_length must be positive.")
        if int(window_step) <= 0:
            raise ValueError("window_step must be positive.")

        if manifest is None and manifest_path is not None:
            manifest = load_manifest(manifest_path)
        if manifest is None:
            raise ValueError("DeepARWindowDataset requires an experiment manifest.")

        self.manifest = manifest
        self.split = split.strip().lower()
        self.context_length = int(context_length)
        self.forecast_length = int(forecast_length)
        self.window_step = int(window_step)
        self.total_length = self.context_length + self.forecast_length
        self.allow_left_padding = (
            bool(allow_left_padding)
            if allow_left_padding is not None
            else self.split == "train"
        )

        df = load_wide_dataframe(csv_path, timestamp_col=timestamp_col)
        if sector_cols is None:
            sector_cols = [col for col in df.columns if col != timestamp_col]
        sector_cols = list(sector_cols)
        if not sector_cols:
            raise ValueError("No sector columns found in dataframe.")

        self.sectors = sector_cols
        self.series_to_item_id = build_series_item_index(self.sectors)
        self.values = torch.tensor(df[sector_cols].to_numpy(), dtype=dtype)
        self.num_steps = int(self.values.shape[0])
        self.num_sectors = int(self.values.shape[1])

        if feature_spec is None:
            feature_spec = build_feature_spec(
                df[timestamp_col],
                cadence=manifest.cadence,
                train_end_idx=manifest.train_split.end_idx,
            )
        self.feature_spec = feature_spec
        self.time_features = torch.tensor(
            build_time_feature_matrix(
                df[timestamp_col],
                cadence=manifest.cadence,
                feature_spec=self.feature_spec,
            ),
            dtype=dtype,
        )

        bounds = self.manifest.split_bounds(self.split)
        start_min = bounds.start_idx if self.allow_left_padding else max(self.context_length, bounds.start_idx)
        start_max = bounds.end_idx - self.forecast_length
        if start_max < start_min:
            raise ValueError(
                f"No valid DeepAR windows found for split={self.split} with "
                f"context_length={self.context_length} and forecast_length={self.forecast_length}."
            )
        self.start_indices = list(range(int(start_min), int(start_max) + 1, self.window_step))
        self.num_windows = len(self.start_indices)
        self.sampling_weights = self._build_sampling_weights()

    def _build_sampling_weights(self) -> torch.Tensor:
        starts = np.asarray(self.start_indices, dtype=int)
        ctx_starts = np.maximum(0, starts - self.context_length)
        obs_counts = np.maximum(1, starts - ctx_starts).astype(float)
        abs_values = self.values.abs().cpu().numpy()
        prefix = np.concatenate(
            [np.zeros((1, self.num_sectors), dtype=float), abs_values.cumsum(axis=0)],
            axis=0,
        )
        sum_abs = prefix[starts, :] - prefix[ctx_starts, :]
        scale = (sum_abs / obs_counts[:, None]) + 1.0
        flattened = np.asarray(scale.T.reshape(-1), dtype=np.float64)
        return torch.tensor(flattened, dtype=torch.double)

    def __len__(self) -> int:
        return self.num_sectors * self.num_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        sector_idx, window_pos = divmod(idx, self.num_windows)
        target_start = int(self.start_indices[window_pos])
        target_end = target_start + self.forecast_length
        raw_start = max(0, target_start - self.context_length)
        left_pad = max(0, self.context_length - target_start)

        series = self.values[raw_start:target_end, sector_idx]
        if left_pad > 0:
            padded = torch.zeros(self.total_length, dtype=self.values.dtype)
            observed_mask = torch.zeros(self.total_length, dtype=self.values.dtype)
            padded[left_pad:] = series
            observed_mask[left_pad:] = 1.0
            full_target = padded
        else:
            full_target = series
            observed_mask = torch.ones(self.total_length, dtype=self.values.dtype)

        feature_slice = self.time_features[raw_start:target_end]
        if left_pad > 0:
            padded_features = torch.zeros(
                self.total_length,
                self.time_features.shape[1],
                dtype=self.time_features.dtype,
            )
            padded_features[left_pad:] = feature_slice
            time_features = padded_features
        else:
            time_features = feature_slice

        context = full_target[: self.context_length]
        target = full_target[self.context_length :]
        return {
            "context": context,
            "target": target,
            "full_target": full_target,
            "time_features": time_features,
            "observed_mask": observed_mask,
            "item_id": torch.tensor(self.series_to_item_id[self.sectors[sector_idx]], dtype=torch.long),
            "series_index": torch.tensor(sector_idx, dtype=torch.long),
            "start_index": torch.tensor(target_start, dtype=torch.long),
        }


class TFTWindowDataset(Dataset):
    """TFT windows with static entity ids and timestamp-derived known future covariates."""

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
        window_step: int = 1,
        feature_spec: DeepARFeatureSpec | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if forecast_length <= 0:
            raise ValueError("forecast_length must be positive.")
        if int(window_step) <= 0:
            raise ValueError("window_step must be positive.")

        if manifest is None and manifest_path is not None:
            manifest = load_manifest(manifest_path)
        if manifest is None:
            raise ValueError("TFTWindowDataset requires an experiment manifest.")

        self.manifest = manifest
        self.split = split.strip().lower()
        self.context_length = int(context_length)
        self.forecast_length = int(forecast_length)
        self.window_step = int(window_step)

        df = load_wide_dataframe(csv_path, timestamp_col=timestamp_col)
        if sector_cols is None:
            sector_cols = [col for col in df.columns if col != timestamp_col]
        sector_cols = list(sector_cols)
        if not sector_cols:
            raise ValueError("No sector columns found in dataframe.")

        self.sectors = sector_cols
        self.series_to_item_id = build_series_item_index(self.sectors)
        self.values = torch.tensor(df[sector_cols].to_numpy(), dtype=dtype)

        if feature_spec is None:
            feature_spec = build_feature_spec(
                df[timestamp_col],
                cadence=manifest.cadence,
                train_end_idx=manifest.train_split.end_idx,
            )
        self.feature_spec = feature_spec
        self.time_features = torch.tensor(
            build_time_feature_matrix(
                df[timestamp_col],
                cadence=manifest.cadence,
                feature_spec=self.feature_spec,
            ),
            dtype=dtype,
        )

        self.start_indices = valid_window_start_indices(
            self.manifest,
            self.split,
            context_length=self.context_length,
            horizon=self.forecast_length,
        )[:: self.window_step].astype(int).tolist()
        self.num_windows = len(self.start_indices)
        self.num_sectors = int(self.values.shape[1])
        if self.num_windows <= 0:
            raise ValueError(
                f"No valid TFT windows found for split={self.split} with "
                f"context_length={self.context_length} and forecast_length={self.forecast_length}."
            )

    def __len__(self) -> int:
        return self.num_sectors * self.num_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        sector_idx, window_pos = divmod(idx, self.num_windows)
        target_start = int(self.start_indices[window_pos])
        context_start = target_start - self.context_length
        target_end = target_start + self.forecast_length

        series = self.values[context_start:target_end, sector_idx]
        known = self.time_features[context_start:target_end]

        context = series[: self.context_length]
        target = series[self.context_length :]
        past_known = known[: self.context_length]
        future_known = known[self.context_length :]
        past_inputs = torch.cat([context.unsqueeze(-1), past_known], dim=-1)

        return {
            "context": context,
            "target": target,
            "past_inputs": past_inputs,
            "future_inputs": future_known,
            "static_categorical": torch.tensor(
                [self.series_to_item_id[self.sectors[sector_idx]]],
                dtype=torch.long,
            ),
            "series_index": torch.tensor(sector_idx, dtype=torch.long),
            "start_index": torch.tensor(target_start, dtype=torch.long),
        }
