"""DataLoader helpers for split-aware training on the canonical full dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.config import DEFAULT_DATASET_PATH, DEFAULT_TIMESTAMP_COL
from src.dataset import TrafficWindowDataset
from src.experiment import ExperimentManifest, load_manifest


@dataclass
class DataLoaderConfig:
    data_path: Path = DEFAULT_DATASET_PATH
    manifest_path: Path | None = None
    timestamp_col: str = DEFAULT_TIMESTAMP_COL
    context_length: int = 128
    forecast_length: int = 1
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    dtype: torch.dtype = torch.float32


def build_datasets(config: DataLoaderConfig) -> Tuple[TrafficWindowDataset, ...]:
    manifest: ExperimentManifest | None = (
        load_manifest(config.manifest_path) if config.manifest_path is not None else None
    )

    datasets = (
        TrafficWindowDataset(
            config.data_path,
            split="train",
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            timestamp_col=config.timestamp_col,
            manifest=manifest,
            dtype=config.dtype,
        ),
        TrafficWindowDataset(
            config.data_path,
            split="val",
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            timestamp_col=config.timestamp_col,
            manifest=manifest,
            dtype=config.dtype,
        ),
        TrafficWindowDataset(
            config.data_path,
            split="test",
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            timestamp_col=config.timestamp_col,
            manifest=manifest,
            dtype=config.dtype,
        ),
    )
    return datasets


def build_dataloaders(
    config: DataLoaderConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
