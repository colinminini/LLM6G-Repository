"""Hyperparameter sweep for breakpoint detection on cached train forecast artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.canonical_eval import evaluate_forecaster_on_split
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
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TOLERANCE,
)
from src.cp_config import parse_csv_list
from src.device import AUTO_DEVICE, resolve_device_name
from src.experiment import (
    build_experiment_manifest,
    build_window_starts_map,
    daily_seasonal_period,
    default_experiment_dir,
    load_manifest,
    load_wide_dataframe,
    save_manifest,
)
from src.pipeline import build_forecaster
from src.progress import StageProgressDisplay
from src.system_metrics import (
    binary_classification_metrics,
    clamp_upper_quantile,
    extract_pre_change_interval,
    json_array,
    resolve_tau,
    safe_ceiling_from_tau,
)


@dataclass(frozen=True)
class SweepConfig:
    cp_detector: str
    cp_min_size: int
    cp_period_length: int | str | None
    cp_exclusion_radius: float
    cp_score_threshold: float
    cp_model: str
    cp_penalty: float
    cp_jump: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cp_detector": self.cp_detector,
            "cp_min_size": int(self.cp_min_size),
            "cp_period_length": self.cp_period_length,
            "cp_exclusion_radius": float(self.cp_exclusion_radius),
            "cp_score_threshold": float(self.cp_score_threshold),
            "cp_model": self.cp_model,
            "cp_penalty": float(self.cp_penalty),
            "cp_jump": int(self.cp_jump),
        }


@dataclass
class WindowRecord:
    model: str
    split: str
    series: str
    start_index: int
    horizon: int
    y_pred_median: np.ndarray
    y_pred_95: np.ndarray
    future_true: np.ndarray


def _resolve_models(raw: str) -> list[str]:
    return parse_csv_list(raw, str)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _serialize_period_length(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return int(value)


def _load_window_records(
    *,
    model_name: str,
    split: str,
    windows_path: Path,
) -> list[WindowRecord]:
    windows_df = pd.read_csv(windows_path)
    records: list[WindowRecord] = []
    for row in windows_df.itertuples(index=False):
        future_true = json_array(row.future_true)
        y50 = json_array(row.y_pred_median)
        y95 = clamp_upper_quantile(y50, json_array(row.y_pred_95))
        horizon = int(row.horizon)
        records.append(
            WindowRecord(
                model=model_name,
                split=split,
                series=str(row.series),
                start_index=int(row.start_index),
                horizon=horizon,
                y_pred_median=y50,
                y_pred_95=y95,
                future_true=future_true,
            )
        )
    if not records:
        raise ValueError(f"No forecast windows found in {windows_path}.")
    return records


def _conditional_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _rank_best_configs(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    scored = summary_df.copy()
    scored["tau_mae_when_both_break_rank"] = scored["tau_mae_when_both_break"].fillna(np.inf)
    scored["coverage_gap_rank"] = scored["coverage_gap"].fillna(np.inf)
    scored["sharpness_rank"] = scored["sharpness"].fillna(np.inf)
    best_rows: list[pd.Series] = []
    for _, group in scored.groupby("model", sort=False):
        best_row = group.sort_values(
            by=[
                "break_f1",
                "tau_mae_when_both_break_rank",
                "coverage_gap_rank",
                "sharpness_rank",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        ).iloc[0].copy()
        best_row["selection_rule"] = "best_break_f1"
        best_rows.append(best_row)
    return pd.DataFrame(best_rows).drop(
        columns=["tau_mae_when_both_break_rank", "coverage_gap_rank", "sharpness_rank"],
        errors="ignore",
    )


def _materialize_train_forecast_cache(
    *,
    forecast_dir: Path,
    output_dir: Path,
    model_names: list[str],
    train_window_step: int,
    max_windows_per_series: int | None = None,
) -> Path:
    cache_root = output_dir / "train_forecast_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest_path = forecast_dir.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for CP sweep cache: {manifest_path}")
    manifest = load_manifest(manifest_path)
    full_df = load_wide_dataframe(manifest.dataset_path, manifest.timestamp_col)
    forecast_summary = _load_json(forecast_dir / "forecast_eval_summary.json") or {}
    device = resolve_device_name(str(forecast_summary.get("device", AUTO_DEVICE)))
    chronos_model_id = str(forecast_summary.get("chronos_model_id", "amazon/chronos-2"))
    chronos_num_samples = int(forecast_summary.get("chronos_num_samples", 100))
    deepar_num_samples = int(forecast_summary.get("deepar_num_samples", DEFAULT_DEEPAR_NUM_SAMPLES))

    starts_map = build_window_starts_map(
        manifest=manifest,
        split="train",
        step=max(1, int(train_window_step)),
        max_windows_per_series=max_windows_per_series,
    )
    (cache_root / "window_policy.json").write_text(
        json.dumps(
            {
                "split": "train",
                "type": "stepped_train",
                "window_step": int(train_window_step),
                "starts_by_series": starts_map,
            },
            indent=2,
        )
    )

    for model_name in model_names:
        target_windows = cache_root / f"{model_name}_forecast_windows.csv"
        target_metrics = cache_root / f"{model_name}_forecast_metrics.json"
        if target_windows.exists() and target_metrics.exists():
            continue

        checkpoint_path = None
        if model_name not in {"chronos2", "seasonal_naive"}:
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
            split="train",
            model_name=model_name,
            forecaster=forecaster,
            device=device,
            starts_map=starts_map,
            progress_label=f"cp_sweep:forecast_cache:train:{model_name}",
        )
        pd.DataFrame(rows).to_csv(target_windows, index=False)
        target_metrics.write_text(json.dumps(metrics, indent=2))

    return cache_root


def _evaluate_config(
    *,
    records: list[WindowRecord],
    config: SweepConfig,
    tolerance: int,
) -> dict[str, float | int | str | None]:
    detector = build_change_point_detector(**config.as_dict())

    has_break_pred: list[bool] = []
    has_break_ref: list[bool] = []
    resolved_cp_errors: list[float] = []
    conditional_cp_errors: list[float] = []
    conditional_tol_hits: list[float] = []
    sharpness_values: list[float] = []
    coverage_hits_total = 0
    coverage_count_total = 0

    for record in records:
        pred_result = detector.detect_change_point(record.y_pred_median)
        ref_result = detector.detect_change_point(record.future_true)
        has_break_pred.append(bool(pred_result.has_break))
        has_break_ref.append(bool(ref_result.has_break))

        tau_pred_resolved = resolve_tau(pred_result.tau, record.horizon)
        tau_ref_resolved = resolve_tau(ref_result.tau, record.horizon)
        resolved_cp_error = abs(tau_pred_resolved - tau_ref_resolved)
        resolved_cp_errors.append(float(resolved_cp_error))

        if pred_result.has_break and ref_result.has_break and pred_result.tau is not None and ref_result.tau is not None:
            conditional_error = abs(int(pred_result.tau) - int(ref_result.tau))
            conditional_cp_errors.append(float(conditional_error))
            conditional_tol_hits.append(float(conditional_error <= tolerance))

        true_interval = extract_pre_change_interval(
            record.future_true,
            tau=ref_result.tau,
            horizon=record.horizon,
        )
        safe_ceiling = safe_ceiling_from_tau(record.y_pred_95, pred_result.tau, record.horizon)
        coverage_hits_total += int(np.sum(true_interval <= safe_ceiling))
        coverage_count_total += int(true_interval.size)
        sharpness_values.append(float(safe_ceiling - np.max(true_interval)))

    break_metrics = binary_classification_metrics(has_break_ref, has_break_pred)
    coverage_rate = (
        float(coverage_hits_total / coverage_count_total)
        if coverage_count_total
        else float("nan")
    )
    return {
        **config.as_dict(),
        "num_windows_total": int(len(records)),
        "MAE_CP": float(np.mean(resolved_cp_errors)) if resolved_cp_errors else float("nan"),
        "tau_mae_when_both_break": _conditional_mean(conditional_cp_errors),
        "tolerance": int(tolerance),
        "tolerance_hit_rate": float(np.mean(np.asarray(resolved_cp_errors) <= int(tolerance)))
        if resolved_cp_errors
        else float("nan"),
        "tolerance_hit_rate_when_both_break": _conditional_mean(conditional_tol_hits),
        "break_precision": float(break_metrics["precision"]),
        "break_recall": float(break_metrics["recall"]),
        "break_f1": float(break_metrics["f1"]),
        "coverage_rate": coverage_rate,
        "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep breakpoint detector hyperparameters over cached train forecast windows "
            "without rerunning the full forecast_eval stage."
        )
    )
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--models", default="lstm,deepar,tft,chronos2,seasonal_naive")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-window-step", type=int, default=1)
    parser.add_argument("--cp-detector", default=None)
    parser.add_argument("--cp-period-lengths", default="auto,8,16")
    parser.add_argument("--cp-exclusion-radii", default="0.05")
    parser.add_argument("--cp-score-thresholds", default="0.5,0.6,0.7,0.75")
    parser.add_argument("--cp-models", default=DEFAULT_CP_MODEL)
    parser.add_argument("--cp-penalties", default=str(DEFAULT_CP_PENALTY))
    parser.add_argument("--cp-min-sizes", default=str(DEFAULT_CP_MIN_SIZE))
    parser.add_argument("--cp-jumps", default=str(DEFAULT_CP_JUMP))
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    parser.add_argument(
        "--cp-max-windows-per-series",
        type=int,
        default=None,
        help="Limit windows per series in the train forecast cache and sweep. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.split).strip().lower() != "train":
        raise ValueError("cp_sweep only supports split='train' in the canonical pipeline.")
    model_names = _resolve_models(args.models)
    if args.cp_detector not in (None, ""):
        resolved_detector = str(args.cp_detector).strip().lower()
    elif (
        str(args.cp_models) != str(DEFAULT_CP_MODEL)
        or str(args.cp_penalties) != str(DEFAULT_CP_PENALTY)
        or str(args.cp_min_sizes) != str(DEFAULT_CP_MIN_SIZE)
        or str(args.cp_jumps) != str(DEFAULT_CP_JUMP)
    ):
        resolved_detector = "ruptures_pelt"
    else:
        resolved_detector = DEFAULT_CP_DETECTOR

    output_dir = args.output_dir or args.forecast_dir.parent / "cp_sweep" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = _materialize_train_forecast_cache(
        forecast_dir=args.forecast_dir,
        output_dir=output_dir,
        model_names=model_names,
        train_window_step=max(1, int(args.train_window_step)),
        max_windows_per_series=args.cp_max_windows_per_series,
    )

    if resolved_detector == "clasp":
        period_lengths = [
            item.strip()
            for item in str(args.cp_period_lengths).split(",")
            if item.strip()
        ]
        exclusion_radii = parse_csv_list(args.cp_exclusion_radii, float)
        score_thresholds = parse_csv_list(args.cp_score_thresholds, float)
        min_sizes = parse_csv_list(args.cp_min_sizes, int)
        configs = [
            SweepConfig(
                cp_detector="clasp",
                cp_min_size=int(min_size),
                cp_period_length=period_length,
                cp_exclusion_radius=float(exclusion_radius),
                cp_score_threshold=float(score_threshold),
                cp_model=DEFAULT_CP_MODEL,
                cp_penalty=DEFAULT_CP_PENALTY,
                cp_jump=DEFAULT_CP_JUMP,
            )
            for period_length, exclusion_radius, score_threshold, min_size in product(
                period_lengths,
                exclusion_radii,
                score_thresholds,
                min_sizes,
            )
        ]
    else:
        cp_models = parse_csv_list(args.cp_models, str)
        cp_penalties = parse_csv_list(args.cp_penalties, float)
        cp_min_sizes = parse_csv_list(args.cp_min_sizes, int)
        cp_jumps = parse_csv_list(args.cp_jumps, int)
        configs = [
            SweepConfig(
                cp_detector="ruptures_pelt",
                cp_min_size=int(cp_min_size),
                cp_period_length=DEFAULT_CP_PERIOD_LENGTH,
                cp_exclusion_radius=DEFAULT_CP_EXCLUSION_RADIUS,
                cp_score_threshold=DEFAULT_CP_SCORE_THRESHOLD,
                cp_model=str(cp_model),
                cp_penalty=float(cp_penalty),
                cp_jump=int(cp_jump),
            )
            for cp_model, cp_penalty, cp_min_size, cp_jump in product(
                cp_models,
                cp_penalties,
                cp_min_sizes,
                cp_jumps,
            )
        ]

    progress = StageProgressDisplay(
        "cp_sweep",
        max(1, len(model_names) * len(configs)),
        unit="config",
    )
    progress.write(
        f"[cp_sweep] starting models={len(model_names)} configs_per_model={len(configs)} split={args.split}"
    )
    progress.update(0, phase="load")

    rows: list[dict[str, Any]] = []
    completed = 0
    for model_name in model_names:
        windows_path = cache_root / f"{model_name}_forecast_windows.csv"
        records = _load_window_records(
            model_name=model_name,
            split=args.split,
            windows_path=windows_path,
        )
        progress.write(
            f"[cp_sweep:{model_name}] cached_train_windows={len(records)} configs={len(configs)}"
        )
        for config in configs:
            metrics = _evaluate_config(records=records, config=config, tolerance=args.tolerance)
            rows.append(
                {
                    "model": model_name,
                    "split": args.split,
                    "forecast_windows_path": str(windows_path),
                    **metrics,
                    "coverage_gap": abs(float(metrics["coverage_rate"]) - float(args.coverage_target)),
                }
            )
            completed += 1
            progress.update(
                completed,
                phase="sweep",
                model=model_name,
                detector=config.cp_detector,
                break_f1=float(metrics["break_f1"]),
            )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("Sweep produced no rows.")
    summary_df = summary_df.sort_values(
        by=[
            "model",
            "cp_detector",
            "cp_min_size",
            "cp_period_length",
            "cp_score_threshold",
            "cp_penalty",
        ],
        kind="stable",
    ).reset_index(drop=True)

    summary_path = output_dir / "cp_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_df = _rank_best_configs(summary_df)
    best_path = output_dir / "cp_sweep_best_configs.csv"
    best_df.to_csv(best_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_dir": str(args.forecast_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "models": model_names,
        "train_window_step": int(args.train_window_step),
        "cp_detector": resolved_detector,
        "coverage_target": float(args.coverage_target),
        "tolerance": int(args.tolerance),
        "train_forecast_cache_dir": str(cache_root),
        "summary_csv": str(summary_path),
        "best_configs_csv": str(best_path),
        "best_configs": best_df.to_dict("records"),
    }
    manifest_path = output_dir / "cp_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    progress.update(
        completed,
        phase="write",
        best_models=len(best_df),
    )
    progress.write(f"Saved sweep summary: {summary_path}")
    progress.write(f"Saved best configs: {best_path}")
    progress.write(f"Saved sweep manifest: {manifest_path}")
    for row in best_df.itertuples(index=False):
        progress.write(
            f"[cp_sweep:{row.model}] best detector={row.cp_detector} "
            f"break_f1={row.break_f1:.4f} tau_mae_when_both_break={row.tau_mae_when_both_break}"
        )
    progress.close()


if __name__ == "__main__":
    main()
