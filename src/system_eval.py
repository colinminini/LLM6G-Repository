"""Consume saved forecast artifacts and run the breakpoint-safe-ceiling pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run q50 -> breakpoint -> safe ceiling evaluation from saved forecast artifacts."
        )
    )
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--models", default="lstm,deepar,chronos2")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--cp-model", default=DEFAULT_CP_MODEL)
    parser.add_argument("--cp-penalty", type=float, default=DEFAULT_CP_PENALTY)
    parser.add_argument("--cp-min-size", type=int, default=DEFAULT_CP_MIN_SIZE)
    parser.add_argument("--cp-jump", type=int, default=DEFAULT_CP_JUMP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecast_root = args.forecast_dir
    output_root = args.output_dir or forecast_root.parent / "system_eval"
    output_root.mkdir(parents=True, exist_ok=True)
    detector = RupturesPeltDetector(
        model=args.cp_model,
        penalty=args.cp_penalty,
        min_size=args.cp_min_size,
        jump=args.cp_jump,
    )

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_dir": str(forecast_root),
        "splits": {},
    }

    for split in _parse_csv(args.splits):
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_summary: dict[str, Any] = {}

        for model_name in _parse_csv(args.models):
            windows_path = forecast_root / split / f"{model_name}_forecast_windows.csv"
            if not windows_path.exists():
                raise FileNotFoundError(f"Missing forecast windows file: {windows_path}")

            windows_df = pd.read_csv(windows_path)
            rows: list[dict[str, Any]] = []
            cp_errors: list[float] = []
            tol_hits: list[float] = []
            sharpness_values: list[float] = []
            coverage_hits_total = 0
            coverage_count_total = 0

            for row in windows_df.itertuples(index=False):
                future_true = json_array(row.future_true)
                y50 = json_array(row.y_pred_median)
                y95 = clamp_upper_quantile(y50, json_array(row.y_pred_95))
                horizon = int(row.horizon)

                tau_pred = resolve_tau(detector.detect_first_change_point(y50), horizon)
                tau_true = resolve_tau(detector.detect_first_change_point(future_true), horizon)
                cp_abs_error = abs(tau_pred - tau_true)
                tol_hits.append(float(cp_abs_error <= args.tolerance))
                cp_errors.append(float(cp_abs_error))

                true_interval = extract_pre_change_interval(future_true, tau=tau_true, horizon=horizon)
                safe_ceiling = safe_ceiling_from_tau(y95, tau_pred, horizon)
                coverage_hits = int(np.sum(true_interval <= safe_ceiling))
                coverage_count = int(true_interval.size)
                coverage_hits_total += coverage_hits
                coverage_count_total += coverage_count
                actual_max = float(np.max(true_interval))
                sharpness = float(safe_ceiling - actual_max)
                sharpness_values.append(sharpness)

                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "series": row.series,
                        "start_index": int(row.start_index),
                        "start_timestamp": row.start_timestamp,
                        "context_length": int(row.context_length),
                        "horizon": horizon,
                        "dataset_path": row.dataset_path,
                        "tau_pred": int(tau_pred),
                        "tau_true": int(tau_true),
                        "cp_abs_error": float(cp_abs_error),
                        "tolerance_hit": int(cp_abs_error <= args.tolerance),
                        "safe_ceiling": safe_ceiling,
                        "actual_interval_max": actual_max,
                        "sharpness": sharpness,
                        "coverage_hits": coverage_hits,
                        "coverage_count": coverage_count,
                        "coverage_window": float(coverage_hits / coverage_count) if coverage_count else float("nan"),
                        "history": row.history,
                        "future_true": row.future_true,
                        "y_pred_median": json.dumps(y50.tolist()),
                        "y_pred_95": json.dumps(y95.tolist()),
                    }
                )

            metrics = {
                "model": model_name,
                "split": split,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "num_windows_total": len(rows),
                "MAE_CP": float(np.mean(cp_errors)) if cp_errors else float("nan"),
                "tolerance": int(args.tolerance),
                "tolerance_hit_rate": float(np.mean(tol_hits)) if tol_hits else float("nan"),
                "coverage_rate": float(coverage_hits_total / coverage_count_total) if coverage_count_total else float("nan"),
                "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
                "cp_detector": {
                    "model": args.cp_model,
                    "penalty": float(args.cp_penalty),
                    "min_size": int(args.cp_min_size),
                    "jump": int(args.cp_jump),
                },
            }
            model_windows_path = split_dir / f"{model_name}_system_windows.csv"
            model_metrics_path = split_dir / f"{model_name}_system_metrics.json"
            pd.DataFrame(rows).to_csv(model_windows_path, index=False)
            model_metrics_path.write_text(json.dumps(metrics, indent=2))
            split_summary[model_name] = {
                "windows_path": str(model_windows_path),
                "metrics_path": str(model_metrics_path),
            }
            print(
                f"[system_eval:{split}:{model_name}] "
                f"MAE_CP={metrics['MAE_CP']:.4f} coverage={metrics['coverage_rate']:.4f}"
            )

        summary["splits"][split] = split_summary

    summary_path = output_root / "system_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved system summary: {summary_path}")


if __name__ == "__main__":
    main()
