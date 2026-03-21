"""Hyperparameter sweep for breakpoint detection on saved forecast artifacts."""

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

from src.change_detection import RupturesPeltDetector
from src.config import (
    DEFAULT_CP_JUMP,
    DEFAULT_CP_MIN_SIZE,
    DEFAULT_CP_MODEL,
    DEFAULT_CP_PENALTY,
    DEFAULT_TOLERANCE,
)
from src.system_metrics import (
    clamp_upper_quantile,
    extract_pre_change_interval,
    json_array,
    resolve_tau,
    safe_ceiling_from_tau,
)


@dataclass(frozen=True)
class SweepConfig:
    cp_model: str
    cp_penalty: float
    cp_min_size: int
    cp_jump: int


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


def _parse_csv_list(raw: str, cast: Any) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return [cast(item) for item in values]


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
        if future_true.size != horizon or y50.size != horizon or y95.size != horizon:
            raise ValueError(
                f"Prediction length mismatch for ({row.series}, start={row.start_index})."
            )
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

    for record in records:
        tau_pred = resolve_tau(
            detector.detect_first_change_point(record.y_pred_median),
            record.horizon,
        )
        tau_true = resolve_tau(
            detector.detect_first_change_point(record.future_true),
            record.horizon,
        )

        cp_abs_error = abs(tau_pred - tau_true)
        cp_errors.append(float(cp_abs_error))
        tol_hits.append(float(cp_abs_error <= tolerance))

        true_interval = extract_pre_change_interval(
            record.future_true,
            tau=tau_true,
            horizon=record.horizon,
        )
        safe_ceiling = safe_ceiling_from_tau(record.y_pred_95, tau_pred, record.horizon)

        coverage_hits_total += int(np.sum(true_interval <= safe_ceiling))
        coverage_count_total += int(true_interval.size)
        sharpness_values.append(float(safe_ceiling - np.max(true_interval)))

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
        "num_windows_total": int(len(records)),
        "MAE_CP": float(np.mean(cp_errors)) if cp_errors else float("nan"),
        "tolerance": int(tolerance),
        "tolerance_hit_rate": float(np.mean(tol_hits)) if tol_hits else float("nan"),
        "coverage_rate": coverage_rate,
        "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
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
            "Sweep ruptures PELT hyperparameters over saved forecast windows without rerunning the forecasters."
        )
    )
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--models", default="lstm,deepar,chronos2")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cp-models", default=DEFAULT_CP_MODEL)
    parser.add_argument("--cp-penalties", default=str(DEFAULT_CP_PENALTY))
    parser.add_argument("--cp-min-sizes", default=str(DEFAULT_CP_MIN_SIZE))
    parser.add_argument("--cp-jumps", default=str(DEFAULT_CP_JUMP))
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = _parse_csv_list(args.models, str)
    cp_models = _parse_csv_list(args.cp_models, str)
    cp_penalties = _parse_csv_list(args.cp_penalties, float)
    cp_min_sizes = _parse_csv_list(args.cp_min_sizes, int)
    cp_jumps = _parse_csv_list(args.cp_jumps, int)

    output_dir = args.output_dir or args.forecast_dir.parent / "cp_sweep" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        SweepConfig(
            cp_model=cp_model,
            cp_penalty=float(cp_penalty),
            cp_min_size=int(cp_min_size),
            cp_jump=int(cp_jump),
        )
        for cp_model, cp_penalty, cp_min_size, cp_jump in product(
            cp_models, cp_penalties, cp_min_sizes, cp_jumps
        )
    ]

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        windows_path = args.forecast_dir / args.split / f"{model_name}_forecast_windows.csv"
        if not windows_path.exists():
            raise FileNotFoundError(
                f"Missing forecast windows for model '{model_name}': {windows_path}"
            )
        records = _load_window_records(
            model_name=model_name,
            split=args.split,
            windows_path=windows_path,
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

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise ValueError("Sweep produced no rows.")

    summary_df = summary_df.sort_values(
        by=["model", "cp_model", "cp_jump", "cp_min_size", "cp_penalty"],
        kind="stable",
    ).reset_index(drop=True)

    summary_path = output_dir / "cp_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_df = _rank_best_configs(summary_df, coverage_target=float(args.coverage_target))
    best_path = output_dir / "cp_sweep_best_configs.csv"
    best_df.to_csv(best_path, index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_dir": str(args.forecast_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "models": model_names,
        "cp_models": cp_models,
        "cp_penalties": cp_penalties,
        "cp_min_sizes": cp_min_sizes,
        "cp_jumps": cp_jumps,
        "tolerance": int(args.tolerance),
        "coverage_target": float(args.coverage_target),
        "summary_csv": str(summary_path),
        "best_configs_csv": str(best_path),
    }
    manifest_path = output_dir / "cp_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Saved sweep summary: {summary_path}")
    print(f"Saved best configs: {best_path}")
    print(f"Saved sweep manifest: {manifest_path}")


if __name__ == "__main__":
    main()
