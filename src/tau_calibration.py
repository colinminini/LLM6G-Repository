"""Post-hoc tau calibration over saved system-eval window forecasts."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.canonical_eval import canonical_window_policy, evaluate_forecaster_on_split
from src.change_detection import build_change_point_detector
from src.config import (
    DEFAULT_CP_DETECTOR,
    DEFAULT_CP_EXCLUSION_RADIUS,
    DEFAULT_CP_JUMP,
    DEFAULT_CP_MIN_SIZE,
    DEFAULT_CP_MODEL,
    DEFAULT_CP_PENALTY,
    DEFAULT_CP_PERIOD_LENGTH,
    DEFAULT_CP_SCORE_THRESHOLD,
    DEFAULT_DATASET_PATH,
    DEFAULT_DEEPAR_NUM_SAMPLES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TOLERANCE,
)
from src.cp_config import default_cp_config, load_best_cp_configs
from src.device import AUTO_DEVICE, resolve_device_name
from src.experiment import build_canonical_eval_starts_map, daily_seasonal_period, load_manifest, load_wide_dataframe
from src.pipeline import build_forecaster
from src.progress import StageProgressDisplay
from src.system_metrics import clamp_upper_quantile, extract_pre_change_interval, json_array, resolve_tau, safe_ceiling_from_tau


@dataclass
class WindowSample:
    model: str
    split: str
    series: str
    start_index: int
    start_timestamp: pd.Timestamp
    horizon: int
    context_length: int
    history: np.ndarray
    future_true: np.ndarray
    y_pred_median: np.ndarray
    y_pred_95: np.ndarray
    tau_pred: int
    tau_ref: int
    safe_ceiling_raw: float


class GlobalMedianOffsetCalibrator:
    def fit(self, _: pd.DataFrame, delta: np.ndarray, meta: pd.DataFrame) -> "GlobalMedianOffsetCalibrator":
        self.offset_ = float(np.median(delta))
        return self

    def predict_delta(self, features: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.offset_, dtype=float)


class PerSeriesMedianOffsetCalibrator:
    def fit(self, _: pd.DataFrame, delta: np.ndarray, meta: pd.DataFrame) -> "PerSeriesMedianOffsetCalibrator":
        joined = meta[["series"]].copy()
        joined["delta"] = delta
        grouped = joined.groupby("series", sort=False)["delta"].median()
        self.offsets_ = grouped.to_dict()
        self.global_offset_ = float(np.median(delta))
        return self

    def predict_delta(self, features: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [self.offsets_.get(series, self.global_offset_) for series in meta["series"]],
            dtype=float,
        )


class SklearnDeltaCalibrator:
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []

    def _prepare_features(
        self,
        features: pd.DataFrame,
        fit: bool,
    ) -> pd.DataFrame:
        encoded = features.replace([np.inf, -np.inf], np.nan)
        if fit:
            self.feature_columns_ = encoded.columns.tolist()
            return encoded
        return encoded.reindex(columns=self.feature_columns_, fill_value=0.0)

    def fit(self, features: pd.DataFrame, delta: np.ndarray, meta: pd.DataFrame) -> "SklearnDeltaCalibrator":
        del meta
        X = self._prepare_features(features, fit=True)
        self.estimator.fit(X, delta)
        return self

    def predict_delta(self, features: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
        del meta
        X = self._prepare_features(features, fit=False)
        return np.asarray(self.estimator.predict(X), dtype=float)

    def feature_importances(self) -> pd.DataFrame:
        if hasattr(self.estimator, "feature_importances_"):
            values = np.asarray(self.estimator.feature_importances_, dtype=float)
        elif hasattr(self.estimator, "coef_"):
            coef = np.asarray(self.estimator.coef_, dtype=float).reshape(-1)
            values = np.abs(coef)
        elif hasattr(self.estimator, "named_steps") and "ridge" in self.estimator.named_steps:
            coef = np.asarray(self.estimator.named_steps["ridge"].coef_, dtype=float).reshape(-1)
            values = np.abs(coef)
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        if values.size != len(self.feature_columns_):
            return pd.DataFrame(columns=["feature", "importance"])
        df = pd.DataFrame({"feature": self.feature_columns_, "importance": values})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name.")
    return models


def _series_columns(df: pd.DataFrame, timestamp_col: str) -> list[str]:
    cols = [col for col in df.columns if col != timestamp_col]
    if not cols:
        raise ValueError("No series columns found in data.")
    return cols


def _linear_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _lag1_autocorr(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 3:
        return 0.0
    x0 = arr[:-1]
    x1 = arr[1:]
    std0 = float(np.std(x0))
    std1 = float(np.std(x1))
    if std0 <= 1e-12 or std1 <= 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _segment_mean(values: np.ndarray, start: int, end: int) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    start = max(0, min(start, arr.size))
    end = max(start + 1, min(end, arr.size))
    return float(np.mean(arr[start:end]))


def _safe_get(values: np.ndarray, idx: int) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    idx = max(0, min(int(idx), arr.size - 1))
    return float(arr[idx])


def _array_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    diffs = np.diff(arr) if arr.size >= 2 else np.zeros(1, dtype=float)
    quarter = max(1, arr.size // 4)
    return {
        f"{prefix}_first": float(arr[0]),
        f"{prefix}_last": float(arr[-1]),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q10": float(np.quantile(arr, 0.10)),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
        f"{prefix}_argmax_norm": float(np.argmax(arr) / max(1, arr.size - 1)),
        f"{prefix}_argmin_norm": float(np.argmin(arr) / max(1, arr.size - 1)),
        f"{prefix}_slope": _linear_slope(arr),
        f"{prefix}_delta": float(arr[-1] - arr[0]),
        f"{prefix}_diff_mean": float(np.mean(diffs)),
        f"{prefix}_diff_std": float(np.std(diffs)),
        f"{prefix}_absdiff_mean": float(np.mean(np.abs(diffs))),
        f"{prefix}_absdiff_max": float(np.max(np.abs(diffs))),
        f"{prefix}_posdiff_rate": float(np.mean(diffs > 0)),
        f"{prefix}_negdiff_rate": float(np.mean(diffs < 0)),
        f"{prefix}_early_mean": float(np.mean(arr[:quarter])),
        f"{prefix}_late_mean": float(np.mean(arr[-quarter:])),
        f"{prefix}_early_late_gap": float(np.mean(arr[-quarter:]) - np.mean(arr[:quarter])),
        f"{prefix}_zero_frac": float(np.mean(np.isclose(arr, 0.0))),
    }


def _history_features(history: np.ndarray) -> dict[str, float]:
    arr = np.asarray(history, dtype=float).reshape(-1)
    last_10 = arr[-10:] if arr.size >= 10 else arr
    last_20 = arr[-20:] if arr.size >= 20 else arr
    diffs = np.diff(arr) if arr.size >= 2 else np.zeros(1, dtype=float)
    return {
        "hist_mean": float(np.mean(arr)),
        "hist_std": float(np.std(arr)),
        "hist_min": float(np.min(arr)),
        "hist_max": float(np.max(arr)),
        "hist_range": float(np.max(arr) - np.min(arr)),
        "hist_median": float(np.median(arr)),
        "hist_q10": float(np.quantile(arr, 0.10)),
        "hist_q90": float(np.quantile(arr, 0.90)),
        "hist_first": float(arr[0]),
        "hist_last": float(arr[-1]),
        "hist_delta": float(arr[-1] - arr[0]),
        "hist_slope": _linear_slope(arr),
        "hist_slope_last10": _linear_slope(last_10),
        "hist_slope_last20": _linear_slope(last_20),
        "hist_diff_mean": float(np.mean(diffs)),
        "hist_diff_std": float(np.std(diffs)),
        "hist_absdiff_mean": float(np.mean(np.abs(diffs))),
        "hist_zero_frac": float(np.mean(np.isclose(arr, 0.0))),
        "hist_acf_lag1": _lag1_autocorr(arr),
        "hist_recent_mean_10": float(np.mean(last_10)),
        "hist_recent_std_10": float(np.std(last_10)),
        "hist_recent_mean_20": float(np.mean(last_20)),
        "hist_recent_std_20": float(np.std(last_20)),
        "hist_recent_gap_10_20": float(np.mean(last_10) - np.mean(last_20)),
        "hist_last_vs_mean": float(arr[-1] - np.mean(arr)),
        "hist_last_vs_q90": float(arr[-1] - np.quantile(arr, 0.90)),
    }


def _calendar_features(ts: pd.Timestamp) -> dict[str, float]:
    second_of_day = (ts.hour * 3600) + (ts.minute * 60) + ts.second
    day_of_week = int(ts.dayofweek)
    return {
        "calendar_hour": float(ts.hour),
        "calendar_minute": float(ts.minute),
        "calendar_second": float(ts.second),
        "calendar_dayofweek": float(day_of_week),
        "calendar_day_sin": float(np.sin(2 * np.pi * second_of_day / 86400.0)),
        "calendar_day_cos": float(np.cos(2 * np.pi * second_of_day / 86400.0)),
        "calendar_week_sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
        "calendar_week_cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
    }


def _load_samples(
    *,
    model_name: str,
    split: str,
    windows_path: Path,
    detector: Any,
) -> list[WindowSample]:
    samples: list[WindowSample] = []

    windows_df = pd.read_csv(windows_path)
    windows_df["start_timestamp"] = pd.to_datetime(windows_df["start_timestamp"], utc=False)

    for row in windows_df.itertuples(index=False):
        history = json_array(row.history)
        future_true = json_array(row.future_true)
        y50 = json_array(row.y_pred_median)
        y95 = clamp_upper_quantile(y50, json_array(row.y_pred_95))
        horizon = int(row.horizon)
        context_length = int(row.context_length)

        if history.size != context_length or future_true.size != horizon:
            raise ValueError(
                f"Window artifact mismatch for ({row.series}, start={row.start_index})."
            )

        tau_pred = resolve_tau(detector.detect_change_point(y50).tau, horizon)
        tau_ref = resolve_tau(detector.detect_change_point(future_true).tau, horizon)
        safe_ceiling_raw = safe_ceiling_from_tau(y95, tau_pred, horizon)

        samples.append(
            WindowSample(
                model=model_name,
                split=split,
                series=str(row.series),
                start_index=int(row.start_index),
                start_timestamp=pd.Timestamp(row.start_timestamp),
                horizon=horizon,
                context_length=context_length,
                history=history,
                future_true=future_true,
                y_pred_median=y50,
                y_pred_95=y95,
                tau_pred=tau_pred,
                tau_ref=tau_ref,
                safe_ceiling_raw=safe_ceiling_raw,
            )
        )

    if not samples:
        raise ValueError(f"No windows found in {windows_path}.")
    return samples


def _series_static_features(full_df: pd.DataFrame, timestamp_col: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in _series_columns(full_df, timestamp_col):
        arr = pd.to_numeric(full_df[col], errors="coerce").to_numpy(dtype=float)
        stats[col] = {
            "series_global_mean": float(np.mean(arr)),
            "series_global_std": float(np.std(arr)),
            "series_global_q90": float(np.quantile(arr, 0.90)),
            "series_global_zero_frac": float(np.mean(np.isclose(arr, 0.0))),
        }
    return stats


def _build_feature_table(samples: list[WindowSample], series_stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for sample in samples:
        y50 = sample.y_pred_median
        y95 = sample.y_pred_95
        width = np.maximum(y95 - y50, 0.0)
        tau = int(sample.tau_pred)
        before_end = max(1, min(tau, sample.horizon))
        after_start = min(tau, sample.horizon - 1)

        row: dict[str, float | int | str] = {
            "model": sample.model,
            "split": sample.split,
            "series": sample.series,
            "start_index": int(sample.start_index),
            "horizon": int(sample.horizon),
            "tau_pred_raw": int(sample.tau_pred),
            "tau_ref": int(sample.tau_ref),
            "delta_true": int(sample.tau_ref - sample.tau_pred),
            "tau_pred_norm": float(sample.tau_pred / max(1, sample.horizon)),
            "tau_pred_is_horizon": float(sample.tau_pred >= sample.horizon),
            "tau_pred_is_zero": float(sample.tau_pred == 0),
            "safe_ceiling_raw": float(sample.safe_ceiling_raw),
        }
        row.update(_array_features("y50", y50))
        row.update(_array_features("y95", y95))
        row.update(_array_features("width", width))
        row.update(_history_features(sample.history))
        row.update(_calendar_features(sample.start_timestamp))
        row.update(series_stats[sample.series])

        row.update(
            {
                "y50_before_tau_mean": _segment_mean(y50, 0, before_end),
                "y50_after_tau_mean": _segment_mean(y50, after_start, sample.horizon),
                "y50_post_pre_gap": _segment_mean(y50, after_start, sample.horizon)
                - _segment_mean(y50, 0, before_end),
                "width_before_tau_mean": _segment_mean(width, 0, before_end),
                "width_after_tau_mean": _segment_mean(width, after_start, sample.horizon),
                "y50_at_tau": _safe_get(y50, tau),
                "y50_prev_tau": _safe_get(y50, tau - 1),
                "y50_next_tau": _safe_get(y50, tau + 1),
                "y50_jump_tau": _safe_get(y50, tau) - _safe_get(y50, tau - 1),
                "width_at_tau": _safe_get(width, tau),
                "width_prev_tau": _safe_get(width, tau - 1),
                "width_next_tau": _safe_get(width, tau + 1),
                "width_jump_tau": _safe_get(width, tau) - _safe_get(width, tau - 1),
                "hist_last_minus_y50_first": float(sample.history[-1] - y50[0]),
                "hist_last_minus_y95_first": float(sample.history[-1] - y95[0]),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _corrected_tau(tau_pred: np.ndarray, delta_hat: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    corrected = np.rint(tau_pred + delta_hat).astype(int)
    return np.clip(corrected, 0, horizon.astype(int))


def _evaluate_predictions(
    df: pd.DataFrame,
    tau_corr: np.ndarray,
    tolerance: int,
) -> dict[str, float]:
    tau_ref = df["tau_ref"].to_numpy(dtype=int)
    horizon = df["horizon"].to_numpy(dtype=int)
    cp_errors = np.abs(tau_corr - tau_ref)
    tol_hits = cp_errors <= int(tolerance)

    coverage_hits_total = 0
    coverage_count_total = 0
    sharpness_values: list[float] = []
    for row, tau_hat in zip(df.itertuples(index=False), tau_corr):
        y95 = np.asarray(json.loads(row.y_pred_95), dtype=float)
        future_true = np.asarray(json.loads(row.future_true_json), dtype=float)
        safe_ceiling = safe_ceiling_from_tau(y95, int(tau_hat), int(row.horizon))
        true_interval = extract_pre_change_interval(
            future_true,
            tau=int(row.tau_ref),
            horizon=int(row.horizon),
        )
        coverage_hits_total += int(np.sum(true_interval <= safe_ceiling))
        coverage_count_total += int(true_interval.size)
        sharpness_values.append(float(safe_ceiling - np.max(true_interval)))

    coverage_rate = (
        float(coverage_hits_total / coverage_count_total)
        if coverage_count_total
        else float("nan")
    )
    return {
        "MAE_CP": float(np.mean(cp_errors)),
        "tolerance_hit_rate": float(np.mean(tol_hits)),
        "mean_signed_error": float(np.mean(tau_corr - tau_ref)),
        "coverage_rate": coverage_rate,
        "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
    }


def _make_calibrators(random_seed: int) -> dict[str, Any]:
    return {
        "raw_tau": None,
        "global_median_offset": GlobalMedianOffsetCalibrator(),
        "per_series_median_offset": PerSeriesMedianOffsetCalibrator(),
        "random_forest": SklearnDeltaCalibrator(
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=random_seed,
                n_jobs=-1,
            )
        ),
        "extra_trees": SklearnDeltaCalibrator(
            ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=random_seed,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": SklearnDeltaCalibrator(
            GradientBoostingRegressor(
                random_state=random_seed,
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
            )
        ),
    }


def _select_best_val_model(summary_df: pd.DataFrame, model_name: str) -> str:
    subset = summary_df[
        (summary_df["model"] == model_name)
        & (summary_df["split"] == "val")
        & (summary_df["calibrator"] != "raw_tau")
    ].copy()
    if subset.empty:
        raise ValueError(f"No validation rows found for model '{model_name}'.")
    best = subset.sort_values(
        by=["MAE_CP", "tolerance_hit_rate", "coverage_gap"],
        ascending=[True, False, True],
    ).iloc[0]
    return str(best["calibrator"])


def _select_best_rows(summary_df: pd.DataFrame, model_name: str) -> list[pd.Series]:
    subset = summary_df[
        (summary_df["model"] == model_name)
        & (summary_df["split"] == "val")
        & (summary_df["calibrator"] != "raw_tau")
    ].copy()
    if subset.empty:
        raise ValueError(f"No validation rows found for model '{model_name}'.")

    selectors = [
        (
            "best_val_mae_cp",
            ["MAE_CP", "tolerance_hit_rate", "coverage_gap"],
            [True, False, True],
        ),
        (
            "best_val_tolerance_hit_rate",
            ["tolerance_hit_rate", "MAE_CP", "coverage_gap"],
            [False, True, True],
        ),
        (
            "closest_val_coverage",
            ["coverage_gap", "MAE_CP", "tolerance_hit_rate"],
            [True, True, False],
        ),
    ]

    rows: list[pd.Series] = []
    for label, cols, ascending in selectors:
        calibrator_name = str(
            subset.sort_values(by=cols, ascending=ascending).iloc[0]["calibrator"]
        )
        test_row = summary_df[
            (summary_df["model"] == model_name)
            & (summary_df["split"] == "test")
            & (summary_df["calibrator"] == calibrator_name)
        ].iloc[0].copy()
        test_row["selected_on"] = label
        rows.append(test_row)

    raw_row = summary_df[
        (summary_df["model"] == model_name)
        & (summary_df["split"] == "test")
        & (summary_df["calibrator"] == "raw_tau")
    ].iloc[0].copy()
    raw_row["selected_on"] = "raw_test_baseline"
    rows.append(raw_row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate post-hoc tau calibration models over saved forecast windows."
        )
    )
    parser.add_argument("--models", default="lstm,deepar,tft,chronos2,seasonal_naive,sarima")
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    parser.add_argument("--cp-config-manifest", type=Path, default=None)
    parser.add_argument("--cp-detector", default=None)
    parser.add_argument("--cp-period-length", default=DEFAULT_CP_PERIOD_LENGTH)
    parser.add_argument("--cp-exclusion-radius", type=float, default=DEFAULT_CP_EXCLUSION_RADIUS)
    parser.add_argument("--cp-score-threshold", type=float, default=DEFAULT_CP_SCORE_THRESHOLD)
    parser.add_argument("--cp-model", default=DEFAULT_CP_MODEL)
    parser.add_argument("--cp-penalty", type=float, default=DEFAULT_CP_PENALTY)
    parser.add_argument("--cp-min-size", type=int, default=DEFAULT_CP_MIN_SIZE)
    parser.add_argument("--cp-jump", type=int, default=DEFAULT_CP_JUMP)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _copy_existing_windows(
    *,
    source_windows: Path,
    source_metrics: Path,
    target_windows: Path,
    target_metrics: Path,
) -> None:
    target_windows.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_windows, target_windows)
    if source_metrics.exists():
        shutil.copy2(source_metrics, target_metrics)


def _materialize_forecast_cache(
    *,
    forecast_dir: Path,
    output_dir: Path,
    model_names: list[str],
) -> Path:
    cache_root = output_dir / "forecast_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_path = forecast_dir.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for calibration cache: {manifest_path}")
    manifest = load_manifest(manifest_path)
    full_df = load_wide_dataframe(manifest.dataset_path, manifest.timestamp_col)
    forecast_summary = _read_json_if_exists(forecast_dir / "forecast_eval_summary.json") or {}
    device = resolve_device_name(str(forecast_summary.get("device", AUTO_DEVICE)))
    chronos_model_id = str(forecast_summary.get("chronos_model_id", "amazon/chronos-2"))
    chronos_num_samples = int(forecast_summary.get("chronos_num_samples", 100))
    deepar_num_samples = int(forecast_summary.get("deepar_num_samples", DEFAULT_DEEPAR_NUM_SAMPLES))
    policy = forecast_summary.get("window_policy", {}) if isinstance(forecast_summary, dict) else {}
    max_windows_per_series = policy.get("max_windows_per_series")

    for split_name in ("train", "val", "test"):
        split_dir = cache_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        starts_map = build_canonical_eval_starts_map(
            manifest=manifest,
            split=split_name,
            max_windows_per_series=max_windows_per_series,
        )
        (split_dir / "window_policy.json").write_text(
            json.dumps(
                {
                    "split": split_name,
                    **canonical_window_policy(
                        manifest,
                        max_windows_per_series=max_windows_per_series,
                    ),
                    "starts_by_series": starts_map,
                },
                indent=2,
            )
        )

        for model_name in model_names:
            target_windows = split_dir / f"{model_name}_forecast_windows.csv"
            target_metrics = split_dir / f"{model_name}_forecast_metrics.json"
            source_windows = forecast_dir / split_name / f"{model_name}_forecast_windows.csv"
            source_metrics = forecast_dir / split_name / f"{model_name}_forecast_metrics.json"
            if source_windows.exists():
                _copy_existing_windows(
                    source_windows=source_windows,
                    source_metrics=source_metrics,
                    target_windows=target_windows,
                    target_metrics=target_metrics,
                )
                continue

            checkpoint_path = None
            if model_name not in {"chronos2", "seasonal_naive", "sarima"}:
                checkpoint_path = (
                    forecast_dir.parent
                    / "checkpoints"
                    / f"{model_name}_{manifest.dataset_name}_best.pt"
                )
            forecaster = build_forecaster(
                model_name,
                checkpoint_path=checkpoint_path,
                context_length=manifest.context_length,
                forecast_length=manifest.horizon,
                quantiles=manifest.quantiles,
                chronos_model_id=chronos_model_id,
                chronos_num_samples=chronos_num_samples,
                seasonal_period=daily_seasonal_period(manifest.cadence),
                deepar_num_samples=deepar_num_samples,
                device=device,
            )
            rows, metrics = evaluate_forecaster_on_split(
                df=full_df,
                manifest=manifest,
                split=split_name,
                model_name=model_name,
                forecaster=forecaster,
                device=device,
                starts_map=starts_map,
                max_windows_per_series=max_windows_per_series,
                progress_label=f"tau_calibration:forecast_cache:{split_name}:{model_name}",
            )
            pd.DataFrame(rows).to_csv(target_windows, index=False)
            target_metrics.write_text(json.dumps(metrics, indent=2))

    return cache_root


def main() -> None:
    args = parse_args()
    model_names = _parse_models(args.models)
    output_dir = args.output_dir or args.forecast_dir.parent / "tau_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_cache_dir = _materialize_forecast_cache(
        forecast_dir=args.forecast_dir,
        output_dir=output_dir,
        model_names=model_names,
    )

    full_df = pd.read_csv(args.data_path)
    full_df[args.timestamp_col] = pd.to_datetime(full_df[args.timestamp_col], utc=False)
    series_stats = _series_static_features(full_df, args.timestamp_col)
    manifest_configs = (
        load_best_cp_configs(args.cp_config_manifest)
        if args.cp_config_manifest is not None
        else {}
    )
    fallback_config = default_cp_config(
        cp_detector=args.cp_detector,
        cp_model=args.cp_model,
        cp_penalty=args.cp_penalty,
        cp_min_size=args.cp_min_size,
        cp_jump=args.cp_jump,
        cp_period_length=args.cp_period_length,
        cp_exclusion_radius=args.cp_exclusion_radius,
        cp_score_threshold=args.cp_score_threshold,
    )

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    feature_importance_rows: list[dict[str, Any]] = []

    n_calibrators = len(_make_calibrators(random_seed=0))
    progress = StageProgressDisplay(
        "tau_calibration",
        len(model_names) * n_calibrators,
        unit="calibrator",
    )
    progress.write(
        f"[tau_calibration] starting models={len(model_names)} calibrators_per_model={n_calibrators}"
    )

    for model_name in model_names:
        detector_config = dict(manifest_configs.get(model_name, fallback_config))
        detector = build_change_point_detector(
            cp_detector=str(detector_config.get("cp_detector", fallback_config["cp_detector"])),
            cp_period_length=detector_config.get("cp_period_length", fallback_config["cp_period_length"]),
            cp_exclusion_radius=float(
                detector_config.get(
                    "cp_exclusion_radius",
                    fallback_config["cp_exclusion_radius"],
                )
            ),
            cp_score_threshold=float(
                detector_config.get(
                    "cp_score_threshold",
                    fallback_config["cp_score_threshold"],
                )
            ),
            cp_model=str(detector_config.get("cp_model", fallback_config["cp_model"])),
            cp_penalty=float(detector_config.get("cp_penalty", fallback_config["cp_penalty"])),
            cp_min_size=int(detector_config.get("cp_min_size", fallback_config["cp_min_size"])),
            cp_jump=int(detector_config.get("cp_jump", fallback_config["cp_jump"])),
        )
        samples: list[WindowSample] = []
        for split_name in ("train", "val", "test"):
            windows_path = forecast_cache_dir / split_name / f"{model_name}_forecast_windows.csv"
            if not windows_path.exists():
                raise FileNotFoundError(f"Missing windows file: {windows_path}")
            samples.extend(
                _load_samples(
                    model_name=model_name,
                    split=split_name,
                    windows_path=windows_path,
                    detector=detector,
                )
            )
        progress.write(f"[tau_calibration:{model_name}] loaded {len(samples)} windows (train/val/test)")

        features_df = _build_feature_table(samples, series_stats)
        split = features_df["split"].astype(str)

        eval_df = pd.DataFrame(
            {
                "model": features_df["model"],
                "series": features_df["series"],
                "start_index": features_df["start_index"],
                "start_timestamp": [sample.start_timestamp.isoformat() for sample in samples],
                "split": split,
                "horizon": features_df["horizon"],
                "tau_pred_raw": features_df["tau_pred_raw"],
                "tau_ref": features_df["tau_ref"],
                "safe_ceiling_raw": features_df["safe_ceiling_raw"],
                "future_true_json": [json.dumps(sample.future_true.tolist()) for sample in samples],
                "y_pred_95": [json.dumps(sample.y_pred_95.tolist()) for sample in samples],
            }
        )

        target_delta = features_df["delta_true"].to_numpy(dtype=float)
        feature_cols = [
            col
            for col in features_df.columns
            if col
            not in {
                "model",
                "split",
                "series",
                "start_index",
                "tau_ref",
                "delta_true",
            }
        ]

        train_mask = split == "train"
        calibrators = _make_calibrators(random_seed=int(args.random_seed))
        fitted_calibrators: dict[str, Any] = {}

        for calibrator_name, calibrator in calibrators.items():
            if calibrator_name != "raw_tau":
                calibrator.fit(
                    features_df.loc[train_mask, feature_cols],
                    target_delta[train_mask],
                    eval_df.loc[train_mask, ["series"]],
                )
                fitted_calibrators[calibrator_name] = calibrator

            for split_name in ("train", "val", "test"):
                mask = split == split_name
                if not np.any(mask):
                    continue

                if calibrator_name == "raw_tau":
                    delta_hat = np.zeros(np.sum(mask), dtype=float)
                else:
                    delta_hat = calibrator.predict_delta(
                        features_df.loc[mask, feature_cols],
                        eval_df.loc[mask, ["series"]],
                    )

                tau_corr = _corrected_tau(
                    tau_pred=features_df.loc[mask, "tau_pred_raw"].to_numpy(dtype=int),
                    delta_hat=delta_hat,
                    horizon=features_df.loc[mask, "horizon"].to_numpy(dtype=int),
                )
                metrics = _evaluate_predictions(
                    eval_df.loc[mask],
                    tau_corr=tau_corr,
                    tolerance=int(args.tolerance),
                )
                row = {
                    "model": model_name,
                    "calibrator": calibrator_name,
                    "split": split_name,
                    "cp_detector": detector_config.get("cp_detector"),
                    "cp_model": detector_config.get("cp_model"),
                    "cp_penalty": float(detector_config.get("cp_penalty", float("nan"))),
                    "cp_min_size": int(detector_config.get("cp_min_size", DEFAULT_CP_MIN_SIZE)),
                    "cp_jump": int(detector_config.get("cp_jump", DEFAULT_CP_JUMP)),
                    "cp_period_length": detector_config.get("cp_period_length"),
                    "cp_exclusion_radius": float(
                        detector_config.get("cp_exclusion_radius", DEFAULT_CP_EXCLUSION_RADIUS)
                    ),
                    "cp_score_threshold": float(
                        detector_config.get("cp_score_threshold", DEFAULT_CP_SCORE_THRESHOLD)
                    ),
                    "num_windows": int(np.sum(mask)),
                    **metrics,
                }
                summary_rows.append(row)

                for meta_row, tau_hat, delta_value in zip(
                    eval_df.loc[mask].itertuples(index=False),
                    tau_corr,
                    delta_hat,
                ):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "calibrator": calibrator_name,
                            "split": split_name,
                            "series": meta_row.series,
                            "start_index": int(meta_row.start_index),
                            "start_timestamp": meta_row.start_timestamp,
                            "tau_pred_raw": int(meta_row.tau_pred_raw),
                            "tau_ref": int(meta_row.tau_ref),
                            "delta_hat": float(delta_value),
                            "tau_corrected": int(tau_hat),
                        }
                    )

            val_row = next(
                (
                    r
                    for r in reversed(summary_rows)
                    if r["model"] == model_name
                    and r["calibrator"] == calibrator_name
                    and r["split"] == "val"
                ),
                None,
            )
            progress.advance(
                1,
                phase="eval",
                model=model_name,
                calibrator=calibrator_name,
                val_mae_cp=val_row["MAE_CP"] if val_row else None,
            )

        summary_df = pd.DataFrame(summary_rows)
        raw_test = summary_df[
            (summary_df["model"] == model_name)
            & (summary_df["calibrator"] == "raw_tau")
            & (summary_df["split"] == "test")
        ].iloc[0]
        for split_name in ("train", "val", "test"):
            split_mask = (summary_df["model"] == model_name) & (summary_df["split"] == split_name)
            summary_df.loc[split_mask, "delta_MAE_CP"] = (
                summary_df.loc[split_mask, "MAE_CP"] - float(raw_test["MAE_CP"])
            )
            summary_df.loc[split_mask, "delta_tolerance_hit_rate"] = (
                summary_df.loc[split_mask, "tolerance_hit_rate"] - float(raw_test["tolerance_hit_rate"])
            )
            summary_df.loc[split_mask, "delta_coverage_rate"] = (
                summary_df.loc[split_mask, "coverage_rate"] - float(raw_test["coverage_rate"])
            )
            summary_df.loc[split_mask, "delta_sharpness"] = (
                summary_df.loc[split_mask, "sharpness"] - float(raw_test["sharpness"])
            )
            summary_df.loc[split_mask, "coverage_gap"] = (
                summary_df.loc[split_mask, "coverage_rate"] - float(args.coverage_target)
            ).abs()

        best_name = _select_best_val_model(summary_df, model_name)
        best_calibrator = fitted_calibrators.get(best_name)
        if best_calibrator is not None and hasattr(best_calibrator, "feature_importances"):
            importance_df = best_calibrator.feature_importances().head(25).copy()
            if not importance_df.empty:
                importance_df.insert(0, "model", model_name)
                importance_df.insert(1, "calibrator", best_name)
                feature_importance_rows.extend(importance_df.to_dict("records"))

    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    if summary_df.empty:
        raise ValueError("No tau calibration results were produced.")

    for model_name in model_names:
        raw_test = summary_df[
            (summary_df["model"] == model_name)
            & (summary_df["calibrator"] == "raw_tau")
            & (summary_df["split"] == "test")
        ].iloc[0]
        for split_name in ("train", "val", "test"):
            split_mask = (summary_df["model"] == model_name) & (summary_df["split"] == split_name)
            summary_df.loc[split_mask, "delta_MAE_CP"] = (
                summary_df.loc[split_mask, "MAE_CP"] - float(raw_test["MAE_CP"])
            )
            summary_df.loc[split_mask, "delta_tolerance_hit_rate"] = (
                summary_df.loc[split_mask, "tolerance_hit_rate"] - float(raw_test["tolerance_hit_rate"])
            )
            summary_df.loc[split_mask, "delta_coverage_rate"] = (
                summary_df.loc[split_mask, "coverage_rate"] - float(raw_test["coverage_rate"])
            )
            summary_df.loc[split_mask, "delta_sharpness"] = (
                summary_df.loc[split_mask, "sharpness"] - float(raw_test["sharpness"])
            )
            summary_df.loc[split_mask, "coverage_gap"] = (
                summary_df.loc[split_mask, "coverage_rate"] - float(args.coverage_target)
            ).abs()

    best_rows: list[pd.Series] = []
    for model_name in model_names:
        best_rows.extend(_select_best_rows(summary_df, model_name))

    summary_path = output_dir / "tau_calibration_summary.csv"
    summary_df.sort_values(by=["model", "split", "calibrator"]).to_csv(summary_path, index=False)

    best_path = output_dir / "tau_calibration_best_test_rows.csv"
    pd.DataFrame(best_rows).to_csv(best_path, index=False)

    predictions_path = output_dir / "tau_calibration_predictions.csv"
    prediction_df.to_csv(predictions_path, index=False)

    test_predictions_path = output_dir / "tau_calibration_test_predictions.csv"
    prediction_df[prediction_df["split"] == "test"].to_csv(test_predictions_path, index=False)

    feature_importances_path = output_dir / "tau_calibration_feature_importances.csv"
    pd.DataFrame(feature_importance_rows).to_csv(feature_importances_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data_path),
        "forecast_dir": str(args.forecast_dir),
        "forecast_cache_dir": str(forecast_cache_dir),
        "output_dir": str(output_dir),
        "models": model_names,
        "tolerance": int(args.tolerance),
        "coverage_target": float(args.coverage_target),
        "cp_config_manifest": str(args.cp_config_manifest) if args.cp_config_manifest else None,
        "cp_detector": fallback_config,
        "summary_csv": str(summary_path),
        "best_test_rows_csv": str(best_path),
        "test_predictions_csv": str(predictions_path),
        "test_only_predictions_csv": str(test_predictions_path),
        "feature_importances_csv": str(feature_importances_path),
    }
    manifest_path = output_dir / "tau_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    progress.write(f"Saved summary: {summary_path}")
    progress.write(f"Saved best test rows: {best_path}")
    progress.write(f"Saved predictions: {predictions_path}")
    progress.write(f"Saved test predictions: {test_predictions_path}")
    progress.write(f"Saved feature importances: {feature_importances_path}")
    progress.write(f"Saved manifest: {manifest_path}")
    progress.close()


if __name__ == "__main__":
    main()
