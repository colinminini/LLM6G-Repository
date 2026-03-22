"""Experiment manifests, chronological splits, and split-aware window sampling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CADENCE,
    DEFAULT_DATASET_PATH,
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TEST_RATIO,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    normalize_quantiles,
    quantiles_tag,
)


@dataclass(frozen=True)
class SplitBounds:
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class ExperimentManifest:
    dataset_path: str
    timestamp_col: str
    cadence: str
    random_seed: int
    context_length: int
    horizon: int
    quantiles: tuple[float, ...]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    num_rows: int
    num_series: int
    series_columns: tuple[str, ...]
    train_split: SplitBounds
    val_split: SplitBounds
    test_split: SplitBounds
    created_at_utc: str

    @property
    def dataset_name(self) -> str:
        return Path(self.dataset_path).stem

    def split_bounds(self, split: str) -> SplitBounds:
        name = split.strip().lower()
        if name == "train":
            return self.train_split
        if name == "val":
            return self.val_split
        if name == "test":
            return self.test_split
        raise ValueError(f"Unsupported split: {split}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quantiles"] = list(self.quantiles)
        return payload


def parse_timestamp_series(
    values: pd.Series | Sequence[object],
    *,
    data_path: str | Path | None = None,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> pd.Series:
    timestamps = pd.to_datetime(pd.Series(values), errors="coerce")
    if timestamps.isna().any():
        invalid_positions = timestamps.index[timestamps.isna()].tolist()[:5]
        source = f"{data_path}" if data_path is not None else "dataset"
        raise ValueError(
            f"{source} contains malformed timestamps in '{timestamp_col}' at rows "
            f"{invalid_positions}. Expected a regular timestamp column."
        )
    return timestamps


def cadence_from_timestamps(timestamps: pd.Series | Sequence[object]) -> tuple[pd.Timedelta, str]:
    parsed = parse_timestamp_series(timestamps)
    diffs = parsed.diff().dropna()
    if diffs.empty:
        raise ValueError("At least two timestamp rows are required to infer dataset cadence.")
    delta = diffs.mode().iloc[0]
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        raise ValueError("Dataset cadence must be a positive regular interval.")
    return delta, _format_cadence(delta)


def daily_seasonal_period(cadence: str) -> int:
    delta = pd.Timedelta(cadence)
    if delta <= pd.Timedelta(0):
        raise ValueError("cadence must be a positive interval.")
    ratio = pd.Timedelta(days=1) / delta
    rounded = int(round(float(ratio)))
    if not np.isclose(float(ratio), float(rounded)):
        raise ValueError(
            f"cadence '{cadence}' does not divide 24 hours exactly, so a daily seasonal period "
            "cannot be represented as an integer number of steps."
        )
    return rounded


def validate_regular_timestamps(
    values: pd.Series | Sequence[object],
    *,
    data_path: str | Path | None = None,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> tuple[pd.Series, pd.Timedelta, str]:
    parsed = parse_timestamp_series(values, data_path=data_path, timestamp_col=timestamp_col)
    if parsed.duplicated().any():
        duplicate_rows = parsed.index[parsed.duplicated()].tolist()[:5]
        source = f"{data_path}" if data_path is not None else "dataset"
        raise ValueError(
            f"{source} contains duplicate timestamps in '{timestamp_col}' at rows {duplicate_rows}. "
            "The pipeline expects a strictly increasing regular time grid."
        )
    diffs = parsed.diff().dropna()
    if diffs.empty:
        raise ValueError("At least two timestamp rows are required to infer dataset cadence.")
    delta = diffs.mode().iloc[0]
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        raise ValueError("Dataset cadence must be a positive regular interval.")
    mismatch_mask = diffs != delta
    if bool(mismatch_mask.any()):
        mismatch_rows = diffs.index[mismatch_mask].tolist()[:5]
        mismatch_values = [str(value) for value in diffs[mismatch_mask].tolist()[:5]]
        source = f"{data_path}" if data_path is not None else "dataset"
        raise ValueError(
            f"{source} has irregular timestamps in '{timestamp_col}'. Expected a constant step of "
            f"{_format_cadence(delta)}, but found mismatches at rows {mismatch_rows} "
            f"with deltas {mismatch_values}."
        )
    if not bool(parsed.is_monotonic_increasing):
        source = f"{data_path}" if data_path is not None else "dataset"
        raise ValueError(
            f"{source} must be strictly increasing in '{timestamp_col}'. "
            "The pipeline does not reorder rows automatically."
        )
    return parsed, delta, _format_cadence(delta)


def load_wide_dataframe(
    data_path: str | Path,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"{path} must contain a '{timestamp_col}' column.")

    value_df = df.drop(columns=[timestamp_col]).apply(pd.to_numeric, errors="coerce")
    if value_df.empty:
        raise ValueError(f"{path} does not contain any series columns.")
    if value_df.isna().any().any():
        raise ValueError(
            f"{path} contains NaN values in the series columns. Clean the dataset before use."
        )
    validate_regular_timestamps(df[timestamp_col], data_path=path, timestamp_col=timestamp_col)

    return pd.concat([df[[timestamp_col]], value_df], axis=1)


def _format_cadence(delta: pd.Timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        raise ValueError("Dataset cadence must be a positive interval.")
    if total_seconds % 60 == 0:
        minutes = total_seconds // 60
        return f"{minutes}min"
    return f"{total_seconds}s"


def _infer_cadence(df: pd.DataFrame, timestamp_col: str) -> str:
    _, _, cadence = validate_regular_timestamps(df[timestamp_col], timestamp_col=timestamp_col)
    return cadence


def build_experiment_manifest(
    *,
    data_path: str | Path = DEFAULT_DATASET_PATH,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    context_length: int,
    horizon: int,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> ExperimentManifest:
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("split ratios must all be positive.")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    df = load_wide_dataframe(data_path, timestamp_col=timestamp_col)
    num_rows = len(df)
    if num_rows <= context_length + horizon:
        raise ValueError("Dataset is too short for the requested context_length and horizon.")

    train_end = int(np.floor(num_rows * train_ratio))
    val_end = int(np.floor(num_rows * (train_ratio + val_ratio)))
    if train_end <= 0 or val_end <= train_end or val_end >= num_rows:
        raise ValueError("Invalid split boundaries computed from the provided ratios.")

    series_columns = tuple(col for col in df.columns if col != timestamp_col)
    return ExperimentManifest(
        dataset_path=str(Path(data_path)),
        timestamp_col=timestamp_col,
        cadence=_infer_cadence(df, timestamp_col=timestamp_col),
        random_seed=int(random_seed),
        context_length=int(context_length),
        horizon=int(horizon),
        quantiles=normalize_quantiles(quantiles),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        num_rows=int(num_rows),
        num_series=len(series_columns),
        series_columns=series_columns,
        train_split=SplitBounds(start_idx=0, end_idx=int(train_end)),
        val_split=SplitBounds(start_idx=int(train_end), end_idx=int(val_end)),
        test_split=SplitBounds(start_idx=int(val_end), end_idx=int(num_rows)),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_manifest(manifest: ExperimentManifest, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


def load_manifest(path: str | Path) -> ExperimentManifest:
    payload = json.loads(Path(path).read_text())
    return ExperimentManifest(
        dataset_path=str(payload["dataset_path"]),
        timestamp_col=str(payload["timestamp_col"]),
        cadence=str(payload.get("cadence", DEFAULT_CADENCE)),
        random_seed=int(payload.get("random_seed", DEFAULT_RANDOM_SEED)),
        context_length=int(payload["context_length"]),
        horizon=int(payload["horizon"]),
        quantiles=normalize_quantiles(payload["quantiles"]),
        train_ratio=float(payload["train_ratio"]),
        val_ratio=float(payload["val_ratio"]),
        test_ratio=float(payload["test_ratio"]),
        num_rows=int(payload["num_rows"]),
        num_series=int(payload["num_series"]),
        series_columns=tuple(str(col) for col in payload["series_columns"]),
        train_split=SplitBounds(**payload["train_split"]),
        val_split=SplitBounds(**payload["val_split"]),
        test_split=SplitBounds(**payload["test_split"]),
        created_at_utc=str(payload.get("created_at_utc", "")),
    )


def default_experiment_dir(
    manifest: ExperimentManifest,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    return (
        Path(results_root)
        / f"{manifest.dataset_name}_ctx{manifest.context_length}_h{manifest.horizon}"
        / f"{quantiles_tag(manifest.quantiles)}_seed{manifest.random_seed}"
    )


def valid_window_start_indices(
    manifest: ExperimentManifest,
    split: str,
    *,
    context_length: int | None = None,
    horizon: int | None = None,
) -> np.ndarray:
    bounds = manifest.split_bounds(split)
    ctx = int(context_length or manifest.context_length)
    hz = int(horizon or manifest.horizon)
    if ctx <= 0 or hz <= 0:
        raise ValueError("context_length and horizon must be positive.")

    start_min = max(ctx, bounds.start_idx)
    start_max = bounds.end_idx - hz
    if start_max < start_min:
        return np.asarray([], dtype=int)
    return np.arange(start_min, start_max + 1, dtype=int)


def build_sampled_starts_map(
    *,
    manifest: ExperimentManifest,
    split: str,
    series_names: Sequence[str] | None = None,
    sampling_mode: str = "random",
    random_windows_per_series: int = 50,
    random_seed: int | None = None,
    random_with_replacement: bool = True,
    window_step: int = 1,
    max_windows_per_series: int | None = None,
) -> dict[str, list[int]]:
    starts = valid_window_start_indices(manifest, split)
    if starts.size == 0:
        raise ValueError(
            f"No valid windows for split={split} with context_length={manifest.context_length} "
            f"and horizon={manifest.horizon}."
        )

    names = list(series_names) if series_names is not None else list(manifest.series_columns)
    if sampling_mode == "rolling":
        selected = starts[:: max(1, int(window_step))]
        if max_windows_per_series is not None and max_windows_per_series > 0:
            selected = selected[: int(max_windows_per_series)]
        values = selected.astype(int).tolist()
        return {name: list(values) for name in names}

    if sampling_mode != "random":
        raise ValueError("sampling_mode must be 'random' or 'rolling'.")
    if random_windows_per_series <= 0:
        raise ValueError("random_windows_per_series must be positive for random sampling.")

    replace = bool(random_with_replacement)
    sample_size = int(random_windows_per_series)
    if (not replace) and sample_size > starts.size:
        sample_size = int(starts.size)

    rng = np.random.default_rng(
        int(manifest.random_seed if random_seed is None else random_seed)
    )
    return {
        name: rng.choice(starts, size=sample_size, replace=replace).astype(int).tolist()
        for name in names
    }


def split_label_series(manifest: ExperimentManifest) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        bounds = manifest.split_bounds(split)
        rows.append(
            {
                "split": split,
                "start_idx": bounds.start_idx,
                "end_idx": bounds.end_idx,
                "num_rows": bounds.end_idx - bounds.start_idx,
            }
        )
    return pd.DataFrame(rows)
