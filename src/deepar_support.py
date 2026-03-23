"""Shared helpers for the paper-faithful DeepAR implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM,
    DEFAULT_DEEPAR_LIKELIHOOD,
    DEFAULT_DEEPAR_NUM_SAMPLES,
    DEFAULT_RANDOM_SEED,
)
from src.experiment import ExperimentManifest, parse_timestamp_series


@dataclass(frozen=True)
class DeepARFeatureSpec:
    feature_names: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "stats": {
                key: {"mean": float(value["mean"]), "std": float(value["std"])}
                for key, value in self.stats.items()
            },
        }


def resolve_deepar_model_config(
    *,
    manifest: ExperimentManifest,
    likelihood: str = DEFAULT_DEEPAR_LIKELIHOOD,
    num_samples: int = DEFAULT_DEEPAR_NUM_SAMPLES,
    item_embedding_dim: int = DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_spec: DeepARFeatureSpec | None = None,
) -> dict[str, Any]:
    feature_payload = feature_spec.to_dict() if feature_spec is not None else None
    return {
        "model_type": "deepar",
        "dataset_path": manifest.dataset_path,
        "timestamp_col": manifest.timestamp_col,
        "cadence": manifest.cadence,
        "context_length": int(manifest.context_length),
        "forecast_length": int(manifest.horizon),
        "series_columns": list(manifest.series_columns),
        "num_items": int(manifest.num_series),
        "deepar_likelihood": str(likelihood),
        "deepar_num_samples": int(num_samples),
        "deepar_item_embedding_dim": int(item_embedding_dim),
        "deepar_random_seed": int(random_seed),
        "deepar_feature_spec": feature_payload,
    }


def _cadence_timedelta(cadence: str) -> pd.Timedelta:
    delta = pd.Timedelta(str(cadence))
    if delta <= pd.Timedelta(0):
        raise ValueError("cadence must be a positive interval.")
    return delta


def raw_time_feature_frame(
    timestamps: Sequence[object] | pd.Series,
    *,
    cadence: str,
) -> pd.DataFrame:
    parsed = parse_timestamp_series(timestamps)
    delta = _cadence_timedelta(cadence)
    step_minutes = max(int(round(delta.total_seconds() / 60.0)), 1)
    frame: dict[str, np.ndarray] = {
        "age": np.log1p(np.arange(len(parsed), dtype=float)),
    }

    if delta < pd.Timedelta(days=1):
        minute_of_day = (
            parsed.dt.hour.to_numpy(dtype=int) * 60
            + parsed.dt.minute.to_numpy(dtype=int)
        )
        steps_per_day = max(1, int(round(24 * 60 / step_minutes)))
        frame["time_of_day_step"] = (minute_of_day // step_minutes).astype(float)
        frame["day_of_week"] = parsed.dt.dayofweek.to_numpy(dtype=float)
    elif delta < pd.Timedelta(days=7):
        frame["day_of_week"] = parsed.dt.dayofweek.to_numpy(dtype=float)
        frame["day_of_month"] = parsed.dt.day.to_numpy(dtype=float)
    elif delta < pd.Timedelta(days=31):
        iso = parsed.dt.isocalendar()
        frame["week_of_year"] = iso.week.to_numpy(dtype=float)
        frame["month_of_year"] = parsed.dt.month.to_numpy(dtype=float)
    else:
        frame["month_of_year"] = parsed.dt.month.to_numpy(dtype=float)
        frame["year"] = parsed.dt.year.to_numpy(dtype=float)

    return pd.DataFrame(frame)


def build_feature_spec(
    timestamps: Sequence[object] | pd.Series,
    *,
    cadence: str,
    train_end_idx: int,
) -> DeepARFeatureSpec:
    raw = raw_time_feature_frame(timestamps, cadence=cadence)
    stats: dict[str, dict[str, float]] = {}
    train_end = max(1, min(int(train_end_idx), len(raw)))
    train_slice = raw.iloc[:train_end]
    for name in raw.columns:
        mean = float(train_slice[name].mean())
        std = float(train_slice[name].std(ddof=0))
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        stats[name] = {"mean": mean, "std": std}
    return DeepARFeatureSpec(feature_names=tuple(raw.columns.tolist()), stats=stats)


def build_time_feature_matrix(
    timestamps: Sequence[object] | pd.Series,
    *,
    cadence: str,
    feature_spec: DeepARFeatureSpec,
) -> np.ndarray:
    raw = raw_time_feature_frame(timestamps, cadence=cadence)
    ordered = raw.loc[:, list(feature_spec.feature_names)]
    values = ordered.to_numpy(dtype=np.float32)
    normalized = np.empty_like(values, dtype=np.float32)
    for idx, name in enumerate(feature_spec.feature_names):
        stats = feature_spec.stats[name]
        normalized[:, idx] = (values[:, idx] - float(stats["mean"])) / float(stats["std"])
    return normalized


def feature_spec_from_payload(payload: dict[str, Any] | None) -> DeepARFeatureSpec | None:
    if not payload:
        return None
    names = tuple(str(name) for name in payload.get("feature_names", []))
    stats_raw = payload.get("stats", {})
    stats = {
        str(name): {
            "mean": float(values["mean"]),
            "std": float(values["std"]),
        }
        for name, values in stats_raw.items()
    }
    if not names:
        return None
    return DeepARFeatureSpec(feature_names=names, stats=stats)


def build_series_item_index(series_columns: Sequence[str]) -> dict[str, int]:
    return {str(series): idx for idx, series in enumerate(series_columns)}

