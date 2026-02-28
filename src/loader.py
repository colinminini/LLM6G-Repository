"""
DataLoader helpers for training/validation/testing splits.

Splits are expected under data/datasets as:
  <base>_train.csv, <base>_val.csv, <base>_test.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.dataset import TrafficWindowDataset


@dataclass
class DataLoaderConfig:
    dataset_base: str = "data_instant"
    data_dir: Path = Path("data/datasets")
    context_length: int = 128
    forecast_length: int = 1
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    dtype: torch.dtype = torch.float32


def _split_paths(data_dir: Path, base: str) -> Tuple[Path, Path, Path]:
    return (
        data_dir / f"{base}_train.csv",
        data_dir / f"{base}_val.csv",
        data_dir / f"{base}_test.csv",
    )


def build_datasets(config: DataLoaderConfig) -> Tuple[TrafficWindowDataset, ...]:
    train_path, val_path, test_path = _split_paths(
        Path(config.data_dir), config.dataset_base
    )

    datasets = (
        TrafficWindowDataset(
            train_path,
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            dtype=config.dtype,
        ),
        TrafficWindowDataset(
            val_path,
            context_length=config.context_length,
            forecast_length=config.forecast_length,
            dtype=config.dtype,
        ),
        TrafficWindowDataset(
            test_path,
            context_length=config.context_length,
            forecast_length=config.forecast_length,
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
