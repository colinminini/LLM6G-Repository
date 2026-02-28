"""
AutoGluon Chronos-2 fine-tuning utilities.

This module centralizes the data prep, training, and evaluation logic that the
notebook previously carried inline. It is designed to keep the notebook clean
while still exposing detailed, well-documented hooks for experimentation.

Key ideas
---------
1) We start from wide CSVs (timestamp + one column per sector).
2) We melt to a long format and build an AutoGluon TimeSeriesDataFrame.
3) We train two Chronos-2 variants in one run:
   - ZeroShot: pretrained weights, no fine-tuning.
   - FineTuned: pretrained weights with fine_tune=True.
4) We evaluate each model on train/val/test using:
   - Coverage: one-sided fraction of targets below q95 (q95 is enforced)
   - MAE: mean absolute error using the median quantile (or mean fallback)

Hyperparameter notes
--------------------
The primary AutoGluon knobs used here are:
- prediction_length: Forecast horizon in time steps.
- freq: Data frequency used to reindex/aggregate (e.g., "S" for seconds).
- use_integer_timestamps: If True, ignore dataset timestamps and build a regular index.
- eval_metric: Metric used for AutoGluon tuning (e.g., "MASE").
- time_limit: Training time budget (seconds).
- known_covariates_names: Optional covariates to pass into the model.
- fine_tune (Chronos2 hyperparam): If True, updates Chronos-2 weights.
- ag_args.name_suffix: Tags model names so we can compare ZeroShot vs FineTuned.

We keep Chronos-2 model-level hyperparameters at AutoGluon's defaults unless
explicitly passed in through `Chronos2FineTuneConfig.hyperparameters`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import warnings

import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


def _normalize_quantiles(
    quantiles: Sequence[float],
    *,
    required: Sequence[float] = (0.95,),
) -> tuple[float, ...]:
    """Return a sorted unique quantile tuple and ensure required levels exist."""
    values = [float(q) for q in quantiles]
    if not values:
        raise ValueError("quantiles must contain at least one value.")
    for q in values:
        if q <= 0.0 or q >= 1.0:
            raise ValueError("quantiles must be in the open interval (0, 1).")

    for req in required:
        req_f = float(req)
        if not any(abs(q - req_f) < 1e-8 for q in values):
            warnings.warn(
                f"Required quantile q={req_f} was missing; adding it automatically.",
                RuntimeWarning,
                stacklevel=2,
            )
            values.append(req_f)

    return tuple(sorted(set(values)))


@dataclass
class Chronos2FineTuneConfig:
    """Configuration for Chronos-2 fine-tuning with AutoGluon."""

    dataset_base: str = "data_1to7"
    dataset_dir: Path = Path("data/datasets")
    timestamp_col: str = "timestamp"

    prediction_length: int = 1
    quantiles: Sequence[float] = (0.5, 0.95)

    freq: str | None = None
    use_integer_timestamps: bool = True
    synthetic_start: str = "2000-01-01 00:00:00"
    eval_metric: str = "MAE"
    time_limit: int | None = 300
    known_covariates_names: Sequence[str] = ()
    enable_ensemble: bool = False

    results_dir: Path = Path("results/benchmarks")
    predictor_path: Path | None = None
    auto_version_predictor_path: bool = True

    hyperparameters: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Coverage logic relies on q95; keep it always available.
        self.quantiles = _normalize_quantiles(self.quantiles, required=(0.95,))


def build_default_hyperparameters() -> Dict[str, Any]:
    """Chronos-2 hyperparameters for a zero-shot + fine-tuned comparison."""

    return {
        "Chronos2": [
            {"ag_args": {"name_suffix": "ZeroShot"}},
            {"fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    }


def infer_freq_from_wide(df: pd.DataFrame, timestamp_col: str = "timestamp") -> str:
    """Infer frequency from a wide dataframe with a timestamp column."""

    if timestamp_col not in df.columns:
        raise ValueError(f"Expected '{timestamp_col}' column in dataset.")
    ts = pd.to_datetime(df[timestamp_col]).sort_values()
    deltas = ts.diff().dropna()
    if deltas.empty:
        raise ValueError("Not enough timestamps to infer frequency.")
    # Use the most common delta as the cadence.
    delta = deltas.mode().iloc[0]
    freq = pd.tseries.frequencies.to_offset(delta).freqstr # type: ignore
    if not freq:
        raise ValueError("Failed to infer frequency from timestamps.")
    return freq


def load_wide_csv(path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load a wide CSV (timestamp + sector columns)."""

    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Expected '{timestamp_col}' column in {path}.")
    return df


def load_ts_dataframe(
    dataset_base: str,
    split: str,
    *,
    freq: str,
    dataset_dir: Path = Path("data/datasets"),
    timestamp_col: str = "timestamp",
    use_integer_timestamps: bool = True,
    synthetic_start: str = "2000-01-01 00:00:00",
) -> TimeSeriesDataFrame:
    """Load a split CSV and convert it into a TimeSeriesDataFrame."""

    path = dataset_dir / f"{dataset_base}_{split}.csv"
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Expected '{timestamp_col}' column in {path}.")
    if use_integer_timestamps:
        # Replace timestamps with a synthetic regular index.
        df[timestamp_col] = pd.date_range(
            start=synthetic_start, periods=len(df), freq=freq
        )
    else:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.melt(id_vars=[timestamp_col], var_name="item_id", value_name="target")
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column=timestamp_col
    )
    return ts_df.convert_frequency(freq, agg_numeric="mean")


def load_splits_for_autogluon(
    config: Chronos2FineTuneConfig,
) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, str]:
    """Load train/val/test TimeSeriesDataFrames plus resolved frequency."""

    wide_train = load_wide_csv(
        config.dataset_dir / f"{config.dataset_base}_train.csv",
        timestamp_col=config.timestamp_col,
    )
    freq = config.freq or infer_freq_from_wide(wide_train, timestamp_col=config.timestamp_col)
    if config.use_integer_timestamps and not freq:
        raise ValueError("freq must be set when use_integer_timestamps=True.")

    train_data = load_ts_dataframe(
        config.dataset_base,
        "train",
        freq=freq,
        dataset_dir=config.dataset_dir,
        timestamp_col=config.timestamp_col,
        use_integer_timestamps=config.use_integer_timestamps,
        synthetic_start=config.synthetic_start,
    )
    val_data = load_ts_dataframe(
        config.dataset_base,
        "val",
        freq=freq,
        dataset_dir=config.dataset_dir,
        timestamp_col=config.timestamp_col,
        use_integer_timestamps=config.use_integer_timestamps,
        synthetic_start=config.synthetic_start,
    )
    test_data = load_ts_dataframe(
        config.dataset_base,
        "test",
        freq=freq,
        dataset_dir=config.dataset_dir,
        timestamp_col=config.timestamp_col,
        use_integer_timestamps=config.use_integer_timestamps,
        synthetic_start=config.synthetic_start,
    )

    return train_data, val_data, test_data, freq


def run_chronos2_finetune(
    config: Chronos2FineTuneConfig,
    train_data: TimeSeriesDataFrame,
    val_data: TimeSeriesDataFrame,
    freq: str,
    test_data: TimeSeriesDataFrame | None = None,
) -> Tuple[TimeSeriesPredictor, List[str], pd.DataFrame | None]:
    """Train Chronos-2 zero-shot and fine-tuned models with AutoGluon."""

    hyperparameters = config.hyperparameters or build_default_hyperparameters()
    base_predictor_path = config.predictor_path or (
        config.results_dir / f"autogluon_{config.dataset_base}"
    )
    predictor_path = base_predictor_path
    if config.auto_version_predictor_path and predictor_path.exists():
        idx = 1
        while True:
            candidate = predictor_path.with_name(f"{predictor_path.name}_v{idx}")
            if not candidate.exists():
                predictor_path = candidate
                break
            idx += 1
        print(
            f"Predictor path already existed, using a new run directory: {predictor_path}"
        )
    predictor_path.mkdir(parents=True, exist_ok=True)

    predictor = TimeSeriesPredictor(
        prediction_length=config.prediction_length,
        target="target",
        known_covariates_names=list(config.known_covariates_names),
        quantile_levels=list(config.quantiles),
        eval_metric=config.eval_metric,
        freq=freq,
        path=str(predictor_path),
    ).fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters=hyperparameters, # type: ignore
        time_limit=config.time_limit,
        enable_ensemble=config.enable_ensemble,
        random_seed=np.random.randint(0, 1_000_000),
    )

    model_names = [name for name in _get_model_names(predictor) if "Chronos2" in name]
    if not model_names:
        raise RuntimeError("No Chronos2 models were trained. Check hyperparameters.")

    leaderboard = None
    if test_data is not None:
        leaderboard = predictor.leaderboard(test_data)

    return predictor, model_names, leaderboard


def _get_model_names(predictor: TimeSeriesPredictor) -> List[str]:
    """Return model names across AutoGluon versions."""

    # Newer API
    if hasattr(predictor, "get_model_names"):
        getter = getattr(predictor, "get_model_names")
        if callable(getter):
            return list(getter()) # type: ignore

    # Some versions expose model_names as a property or method
    if hasattr(predictor, "model_names"):
        attr = getattr(predictor, "model_names")
        if callable(attr):
            return list(attr()) # type: ignore
        if isinstance(attr, (list, tuple, set)):
            return list(attr)

    # Fallback to internal trainer if present
    trainer = getattr(predictor, "_trainer", None)
    if trainer is not None:
        if hasattr(trainer, "get_model_names"):
            getter = getattr(trainer, "get_model_names")
            if callable(getter):
                return list(getter()) # type: ignore
        if hasattr(trainer, "model_names"):
            attr = getattr(trainer, "model_names")
            if callable(attr):
                return list(attr()) # type: ignore
            if isinstance(attr, (list, tuple, set)):
                return list(attr)

    raise AttributeError(
        "Unable to locate model names on TimeSeriesPredictor. "
        "Please check AutoGluon version or update this helper."
    )


def _split_context_target(
    ts_df: TimeSeriesDataFrame, prediction_length: int
) -> Tuple[TimeSeriesDataFrame, pd.Series]:
    """Split a TimeSeriesDataFrame into context and target (last horizon)."""

    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")

    df = ts_df.reset_index()
    if not {"item_id", "timestamp", "target"}.issubset(df.columns):
        raise ValueError("Expected TimeSeriesDataFrame with item_id, timestamp, target.")

    context_parts = []
    target_parts = []
    for item_id, group in df.groupby("item_id", sort=False):
        group = group.sort_values("timestamp")
        if len(group) <= prediction_length:
            raise ValueError(
                f"Series '{item_id}' shorter than prediction_length={prediction_length}."
            )
        context_parts.append(group.iloc[:-prediction_length])
        target_parts.append(group.iloc[-prediction_length:])

    context_df = pd.concat(context_parts, ignore_index=True)
    target_df = pd.concat(target_parts, ignore_index=True)

    context_ts = TimeSeriesDataFrame.from_data_frame(
        context_df, id_column="item_id", timestamp_column="timestamp"
    )
    target_series = target_df.set_index(["item_id", "timestamp"])["target"]

    return context_ts, target_series


def _predict_with_quantiles(
    predictor: TimeSeriesPredictor,
    data: TimeSeriesDataFrame,
    quantiles: Sequence[float],
    model: str,
) -> TimeSeriesDataFrame:
    """Predict with quantile levels when supported."""

    try:
        return predictor.predict(data, model=model, quantile_levels=list(quantiles)) # type: ignore
    except TypeError:
        # Older AutoGluon versions may not support quantile_levels in predict.
        return predictor.predict(data, model=model)


def _resolve_quantile_column(pred_df: pd.DataFrame, q: float) -> str:
    """Find the prediction column matching a quantile value."""

    candidates = [str(q), f"{q:g}", f"{q:.3f}"]
    for name in candidates:
        if name in pred_df.columns:
            return name

    numeric_cols: list[tuple[str, float]] = []
    for col in pred_df.columns:
        try:
            numeric_cols.append((str(col), float(col)))
        except (TypeError, ValueError):
            continue

    if numeric_cols:
        best_name, best_q = min(
            numeric_cols,
            key=lambda item: (abs(item[1] - q), -item[1]),
        )
        warnings.warn(
            (
                f"Quantile column for requested q={q} not found; "
                f"using nearest available q={best_q} (column='{best_name}')."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return best_name

    raise KeyError(
        f"Quantile column for {q} not found. Available columns: {list(pred_df.columns)}"
    )


def _select_coverage_quantile(
    quantiles: Sequence[float], target_quantile: float = 0.95
) -> float:
    if not quantiles:
        raise ValueError("quantiles must contain at least one entry.")
    values = [float(q) for q in quantiles]
    idx = min(
        range(len(values)),
        key=lambda i: (abs(values[i] - target_quantile), -values[i]),
    )
    return values[idx]


def compute_coverage_and_mae(
    predictor: TimeSeriesPredictor,
    data: TimeSeriesDataFrame,
    *,
    prediction_length: int,
    quantiles: Sequence[float],
    model: str,
) -> Dict[str, float]:
    """Compute one-sided q95 coverage and MAE for last-horizon evaluation."""
    quantiles = _normalize_quantiles(quantiles, required=(0.95,))

    context, actual = _split_context_target(data, prediction_length)
    preds = _predict_with_quantiles(predictor, context, quantiles, model=model)

    actual = actual.loc[preds.index]

    coverage_q = _select_coverage_quantile(quantiles, target_quantile=0.95)
    coverage_col = _resolve_quantile_column(preds, coverage_q)

    if 0.5 in quantiles:
        median_col = _resolve_quantile_column(preds, 0.5)
    elif "mean" in preds.columns:
        median_col = "mean"
    else:
        # Fall back to the middle quantile if 0.5 is missing.
        mid_idx = len(quantiles) // 2
        median_col = _resolve_quantile_column(preds, sorted(quantiles)[mid_idx])

    coverage = (actual <= preds[coverage_col]).mean()
    mae = float((actual - preds[median_col]).abs().mean())

    return {
        "coverage": float(coverage),
        "coverage_quantile": float(coverage_q),
        "mae": float(mae),
        "count": float(len(actual)),
    }


def compute_coverage_and_rmse(
    predictor: TimeSeriesPredictor,
    data: TimeSeriesDataFrame,
    *,
    prediction_length: int,
    quantiles: Sequence[float],
    model: str,
) -> Dict[str, float]:
    """Backward-compatible alias; returns MAE under key `rmse`."""
    metrics = compute_coverage_and_mae(
        predictor,
        data,
        prediction_length=prediction_length,
        quantiles=quantiles,
        model=model,
    )
    return {
        "coverage": metrics["coverage"],
        "coverage_quantile": metrics["coverage_quantile"],
        "rmse": metrics["mae"],
        "count": metrics["count"],
    }


def evaluate_chronos2_models(
    predictor: TimeSeriesPredictor,
    *,
    model_names: Sequence[str],
    splits: Mapping[str, TimeSeriesDataFrame],
    dataset_label: str,
    prediction_length: int,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    """Evaluate each model on each split and return a tidy dataframe."""

    rows: List[Dict[str, Any]] = []
    for model_name in model_names:
        for split_name, data in splits.items():
            metrics = compute_coverage_and_mae(
                predictor,
                data,
                prediction_length=prediction_length,
                quantiles=quantiles,
                model=model_name,
            )
            rows.append(
                {
                    "dataset": dataset_label,
                    "split": split_name,
                    "model": f"autogluon_{model_name}",
                    "coverage": metrics["coverage"],
                    "coverage_quantile": metrics["coverage_quantile"],
                    "mae": metrics["mae"],
                    "count": metrics["count"],
                }
            )
    return pd.DataFrame(rows)
