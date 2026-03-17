"""Hyperparameter sweep for change-point detection over saved system-eval forecasts."""

from __future__ import annotations

import argparse
import csv
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

from src.change_detection import RupturesPeltDetector


@dataclass(frozen=True)
class SweepConfig:
    cp_model: str
    cp_penalty: float
    cp_min_size: int
    cp_jump: int


@dataclass
class WindowRecord:
    model: str
    series: str
    start_index: int
    horizon: int
    y_pred_median: np.ndarray
    y_pred_95: np.ndarray
    future_true: np.ndarray


def _parse_csv_list(raw: str, cast: Any) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return [cast(item) for item in values]


def _series_columns(df: pd.DataFrame, timestamp_col: str) -> list[str]:
    cols = [col for col in df.columns if col != timestamp_col]
    if not cols:
        raise ValueError("No series columns found in data.")
    return cols


def _resolve_tau(tau: int | None, horizon: int) -> int:
    if tau is None:
        return int(horizon)
    return int(np.clip(int(tau), 0, horizon))


def _extract_pre_change_interval(values: np.ndarray, tau: int, horizon: int) -> np.ndarray:
    if tau >= horizon:
        segment = values[:horizon]
    else:
        segment = values[: max(1, tau)]
    if segment.size == 0:
        return values[:1]
    return segment


def _safe_ceiling_from_tau(y95: np.ndarray, tau_pred: int, horizon: int) -> float:
    if tau_pred >= horizon:
        stationary_95 = y95[:horizon]
    else:
        stationary_95 = y95[: max(1, tau_pred)]
    return float(np.max(stationary_95))


def _load_baseline_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text())
    return {
        "baseline_MAE_CP": float(payload.get("MAE_CP", np.nan)),
        "baseline_tolerance_hit_rate": float(payload.get("tolerance_hit_rate", np.nan)),
        "baseline_coverage_rate": float(payload.get("coverage_rate", np.nan)),
        "baseline_sharpness": float(payload.get("sharpness", np.nan)),
    }


def _load_window_records(
    *,
    model_name: str,
    windows_path: Path,
    full_df: pd.DataFrame,
    timestamp_col: str,
) -> list[WindowRecord]:
    series_cols = set(_series_columns(full_df, timestamp_col))
    records: list[WindowRecord] = []

    with windows_path.open() as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            series_name = row["series"]
            if series_name not in series_cols:
                raise ValueError(f"Series '{series_name}' not found in dataset columns.")

            start_index = int(float(row["start_index"]))
            horizon = int(float(row["horizon"]))
            y50 = np.asarray(json.loads(row["y_pred_median"]), dtype=float).reshape(-1)
            y95 = np.asarray(json.loads(row["y_pred_95"]), dtype=float).reshape(-1)
            future_true = pd.to_numeric(
                full_df.loc[start_index : start_index + horizon - 1, series_name],
                errors="coerce",
            ).to_numpy(dtype=float)

            if future_true.size != horizon:
                raise ValueError(
                    f"Window ({series_name}, start={start_index}) does not have {horizon} future points."
                )
            if np.isnan(future_true).any():
                raise ValueError(
                    f"NaN values detected in future_true for ({series_name}, start={start_index})."
                )
            if y50.size != horizon or y95.size != horizon:
                raise ValueError(
                    f"Prediction length mismatch for ({series_name}, start={start_index}): "
                    f"horizon={horizon}, y50={y50.size}, y95={y95.size}."
                )

            records.append(
                WindowRecord(
                    model=model_name,
                    series=series_name,
                    start_index=start_index,
                    horizon=horizon,
                    y_pred_median=y50,
                    y_pred_95=np.maximum(y95, y50),
                    future_true=future_true,
                )
            )

    if not records:
        raise ValueError(f"No rows found in {windows_path}.")
    return records


def _evaluate_config(
    *,
    records: list[WindowRecord],
    config: SweepConfig,
    tolerance: int,
) -> dict[str, float | int | str]:
    detector = RupturesPeltDetector(
        model=config.cp_model,
        penalty=config.cp_penalty,
        min_size=config.cp_min_size,
        jump=config.cp_jump,
    )

    cp_errors: list[float] = []
    tol_hits: list[float] = []
    sharpness_values: list[float] = []
    coverage_hits_total = 0
    coverage_count_total = 0
    tau_pred_horizon_hits = 0
    tau_true_horizon_hits = 0

    for record in records:
        tau_pred = _resolve_tau(
            detector.detect_first_change_point(record.y_pred_median),
            record.horizon,
        )
        tau_true = _resolve_tau(
            detector.detect_first_change_point(record.future_true),
            record.horizon,
        )

        cp_abs_error = abs(tau_pred - tau_true)
        cp_errors.append(float(cp_abs_error))
        tol_hits.append(float(cp_abs_error <= tolerance))

        true_interval = _extract_pre_change_interval(
            record.future_true,
            tau=tau_true,
            horizon=record.horizon,
        )
        safe_ceiling = _safe_ceiling_from_tau(record.y_pred_95, tau_pred, record.horizon)

        coverage_hits_total += int(np.sum(true_interval <= safe_ceiling))
        coverage_count_total += int(true_interval.size)
        sharpness_values.append(float(safe_ceiling - np.max(true_interval)))

        tau_pred_horizon_hits += int(tau_pred >= record.horizon)
        tau_true_horizon_hits += int(tau_true >= record.horizon)

    num_windows = len(records)
    coverage_rate = (
        float(coverage_hits_total / coverage_count_total)
        if coverage_count_total
        else float("nan")
    )
    return {
        "cp_model": config.cp_model,
        "cp_penalty": float(config.cp_penalty),
        "cp_min_size": int(config.cp_min_size),
        "cp_jump": int(config.cp_jump),
        "num_windows_total": int(num_windows),
        "MAE_CP": float(np.mean(cp_errors)) if cp_errors else float("nan"),
        "tolerance": int(tolerance),
        "tolerance_hit_rate": float(np.mean(tol_hits)) if tol_hits else float("nan"),
        "coverage_rate": coverage_rate,
        "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
        "tau_pred_horizon_rate": float(tau_pred_horizon_hits / num_windows) if num_windows else float("nan"),
        "tau_true_horizon_rate": float(tau_true_horizon_hits / num_windows) if num_windows else float("nan"),
    }


def _rank_best_configs(summary_df: pd.DataFrame, coverage_target: float) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    best_rows: list[pd.Series] = []
    for model_name, group in summary_df.groupby("model", sort=False):
        mae_best = group.sort_values(
            by=["MAE_CP", "tolerance_hit_rate", "coverage_gap"],
            ascending=[True, False, True],
        ).iloc[0].copy()
        mae_best["selection_rule"] = "best_mae_cp"
        best_rows.append(mae_best)

        tol_best = group.sort_values(
            by=["tolerance_hit_rate", "MAE_CP", "coverage_gap"],
            ascending=[False, True, True],
        ).iloc[0].copy()
        tol_best["selection_rule"] = "best_tolerance_hit_rate"
        best_rows.append(tol_best)

        cov_best = group.sort_values(
            by=["coverage_gap", "MAE_CP", "tolerance_hit_rate"],
            ascending=[True, True, False],
        ).iloc[0].copy()
        cov_best["selection_rule"] = f"closest_coverage_to_{coverage_target:.2f}"
        best_rows.append(cov_best)

    return pd.DataFrame(best_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ruptures PELT hyperparameters over saved system-eval window forecasts "
            "without rerunning the forecasters."
        )
    )
    parser.add_argument("--models", default="lstm,deepar,chronos2")
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column in the full evaluation dataset.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/data_1to7.csv"),
        help="Full dataset CSV used to reconstruct future_true windows.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory containing <model>_window_metrics.csv from system evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation/cp_sweep"),
        help="Directory where the sweep CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--cp-models",
        default="normal",
        help="Comma-separated ruptures model names.",
    )
    parser.add_argument(
        "--cp-penalties",
        default="10,15,20",
        help="Comma-separated penalty values.",
    )
    parser.add_argument(
        "--cp-min-sizes",
        default="8,10,12",
        help="Comma-separated min_size values.",
    )
    parser.add_argument(
        "--cp-jumps",
        default="1",
        help="Comma-separated jump values.",
    )
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.95,
        help="Coverage target used when reporting the closest-coverage config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_names = _parse_csv_list(args.models, str)
    cp_models = _parse_csv_list(args.cp_models, str)
    cp_penalties = _parse_csv_list(args.cp_penalties, float)
    cp_min_sizes = _parse_csv_list(args.cp_min_sizes, int)
    cp_jumps = _parse_csv_list(args.cp_jumps, int)

    full_df = pd.read_csv(args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        SweepConfig(
            cp_model=cp_model,
            cp_penalty=float(cp_penalty),
            cp_min_size=int(cp_min_size),
            cp_jump=int(cp_jump),
        )
        for cp_model, cp_penalty, cp_min_size, cp_jump in product(
            cp_models,
            cp_penalties,
            cp_min_sizes,
            cp_jumps,
        )
    ]

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        windows_path = args.input_dir / f"{model_name}_window_metrics.csv"
        metrics_path = args.input_dir / f"{model_name}_metrics.json"
        if not windows_path.exists():
            raise FileNotFoundError(
                f"Missing system-eval window metrics for model '{model_name}': {windows_path}"
            )

        records = _load_window_records(
            model_name=model_name,
            windows_path=windows_path,
            full_df=full_df,
            timestamp_col=args.timestamp_col,
        )
        baseline_metrics = _load_baseline_metrics(metrics_path)

        for config in configs:
            metrics = _evaluate_config(
                records=records,
                config=config,
                tolerance=args.tolerance,
            )
            row = {
                "model": model_name,
                "dataset_path": str(args.data_path),
                **metrics,
                **baseline_metrics,
            }
            if baseline_metrics:
                row["delta_MAE_CP"] = float(row["MAE_CP"]) - float(
                    baseline_metrics["baseline_MAE_CP"]
                )
                row["delta_tolerance_hit_rate"] = float(row["tolerance_hit_rate"]) - float(
                    baseline_metrics["baseline_tolerance_hit_rate"]
                )
                row["delta_coverage_rate"] = float(row["coverage_rate"]) - float(
                    baseline_metrics["baseline_coverage_rate"]
                )
                row["delta_sharpness"] = float(row["sharpness"]) - float(
                    baseline_metrics["baseline_sharpness"]
                )
            row["coverage_gap"] = abs(float(row["coverage_rate"]) - float(args.coverage_target))
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("Sweep produced no rows.")

    summary_df = summary_df.sort_values(
        by=["model", "cp_model", "cp_jump", "cp_min_size", "cp_penalty"],
        kind="stable",
    ).reset_index(drop=True)

    summary_path = args.output_dir / "cp_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_df = _rank_best_configs(summary_df, coverage_target=float(args.coverage_target))
    best_path = args.output_dir / "cp_sweep_best_configs.csv"
    best_df.to_csv(best_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data_path),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "models": model_names,
        "cp_models": cp_models,
        "cp_penalties": cp_penalties,
        "cp_min_sizes": cp_min_sizes,
        "cp_jumps": cp_jumps,
        "tolerance": int(args.tolerance),
        "coverage_target": float(args.coverage_target),
        "num_configs_per_model": len(configs),
        "summary_csv": str(summary_path),
        "best_configs_csv": str(best_path),
    }
    manifest_path = args.output_dir / "cp_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Saved sweep summary: {summary_path}")
    print(f"Saved best configs: {best_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
