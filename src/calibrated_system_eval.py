"""Apply selected tau calibrators and materialize final calibrated system-eval artifacts."""

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

from src.progress import StageProgressDisplay
from src.system_metrics import (
    clamp_upper_quantile,
    covered_sharpness_summary as _covered_sharpness_summary,
    extract_pre_change_interval,
    json_array,
    ratio_of_means as _ratio_of_means,
    safe_ceiling_from_tau,
    undercoverage_summary as _undercoverage_summary,
)


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply validation-selected tau calibrators and recompute system-eval metrics."
    )
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--tau-calibration-dir", type=Path, required=True)
    parser.add_argument("--raw-system-eval-dir", type=Path, default=None)
    parser.add_argument("--models", default="lstm,deepar,tft,chronos2,seasonal_naive,sarima")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--selection-rule", default="best_val_mae_cp")
    return parser.parse_args()


def _select_calibrators(best_rows: pd.DataFrame, selection_rule: str) -> dict[str, str]:
    subset = best_rows[best_rows["selected_on"] == selection_rule].copy()
    if subset.empty:
        raise ValueError(f"No tau calibration rows found for selection rule '{selection_rule}'.")
    return {
        str(row.model): str(row.calibrator)
        for row in subset.itertuples(index=False)
    }


def _raw_metrics(raw_system_eval_dir: Path, split: str, model_name: str) -> dict[str, float]:
    path = raw_system_eval_dir / split / f"{model_name}_system_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw system metrics: {path}")
    payload = json.loads(path.read_text())
    return {
        "MAE_CP": float(payload["MAE_CP"]),
        "tolerance": float(payload["tolerance"]),
        "tolerance_hit_rate": float(payload["tolerance_hit_rate"]),
        "coverage_rate": float(payload["coverage_rate"]),
        "sharpness": float(payload["sharpness"]),
    }


def main() -> None:
    args = parse_args()
    forecast_dir = args.forecast_dir
    tau_dir = args.tau_calibration_dir
    raw_system_eval_dir = args.raw_system_eval_dir or forecast_dir.parent / "system_eval"
    output_dir = args.output_dir or forecast_dir.parent / "calibrated_system_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_rows_path = tau_dir / "tau_calibration_best_test_rows.csv"
    predictions_path = tau_dir / "tau_calibration_predictions.csv"
    if not best_rows_path.exists():
        raise FileNotFoundError(f"Missing tau calibration best rows: {best_rows_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing tau calibration predictions: {predictions_path}")

    best_rows = pd.read_csv(best_rows_path)
    predictions_df = pd.read_csv(predictions_path)
    selected_calibrators = _select_calibrators(best_rows, args.selection_rule)

    splits = _parse_csv(args.splits)
    model_names = _parse_csv(args.models)
    progress = StageProgressDisplay(
        "calibrated_system_eval",
        len(splits) * len(model_names),
        unit="model",
    )
    progress.write(
        f"[calibrated_system_eval] starting splits={splits} models={len(model_names)} rule={args.selection_rule}"
    )

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_dir": str(forecast_dir),
        "tau_calibration_dir": str(tau_dir),
        "raw_system_eval_dir": str(raw_system_eval_dir),
        "selection_rule": args.selection_rule,
        "splits": {},
    }
    comparison_rows: list[dict[str, Any]] = []

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_summary: dict[str, Any] = {}

        for model_name in model_names:
            selected_calibrator = selected_calibrators.get(model_name)
            if selected_calibrator is None:
                continue

            windows_path = forecast_dir / split / f"{model_name}_forecast_windows.csv"
            if not windows_path.exists():
                raise FileNotFoundError(f"Missing forecast windows file: {windows_path}")
            windows_df = pd.read_csv(windows_path)

            pred_subset = predictions_df[
                (predictions_df["model"] == model_name)
                & (predictions_df["split"] == split)
                & (predictions_df["calibrator"] == selected_calibrator)
            ].copy()
            if pred_subset.empty:
                raise ValueError(
                    f"No calibrated tau predictions found for model='{model_name}', split='{split}', "
                    f"calibrator='{selected_calibrator}'."
                )

            merged = windows_df.merge(
                pred_subset,
                on=["model", "series", "start_index"],
                how="inner",
                suffixes=("", "_tau"),
            )
            if merged.empty:
                raise ValueError(
                    f"Could not align forecast windows with calibrated tau predictions for "
                    f"model='{model_name}', split='{split}'."
                )

            rows: list[dict[str, Any]] = []
            cp_errors: list[float] = []
            tol_hits: list[float] = []
            sharpness_values: list[float] = []
            actual_max_values: list[float] = []
            rel_sharpness_values: list[float] = []
            coverage_hits_total = 0
            coverage_count_total = 0
            raw_metrics = _raw_metrics(Path(raw_system_eval_dir), split, model_name)
            tolerance = int(raw_metrics["tolerance"])

            for row in merged.itertuples(index=False):
                future_true = json_array(row.future_true)
                y50 = json_array(row.y_pred_median)
                y95 = clamp_upper_quantile(y50, json_array(row.y_pred_95))
                horizon = int(row.horizon)
                tau_pred_raw = int(row.tau_pred_raw)
                tau_ref = int(row.tau_ref)
                tau_corrected = int(row.tau_corrected)
                cp_abs_error = abs(tau_corrected - tau_ref)
                cp_errors.append(float(cp_abs_error))
                tol_hits.append(float(cp_abs_error <= tolerance))

                true_interval = extract_pre_change_interval(future_true, tau=tau_corrected, horizon=horizon)
                safe_ceiling = safe_ceiling_from_tau(y95, tau_corrected, horizon)
                actual_max = float(np.max(true_interval))
                coverage_hits = int(np.sum(true_interval <= safe_ceiling))
                coverage_count = int(true_interval.size)
                coverage_hits_total += coverage_hits
                coverage_count_total += coverage_count
                sharpness = float(safe_ceiling - actual_max)
                sharpness_values.append(sharpness)
                actual_max_values.append(actual_max)
                rel_sharpness = (
                    float(sharpness / actual_max) if actual_max > 0 else float("nan")
                )
                rel_sharpness_values.append(rel_sharpness)

                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "selected_calibrator": selected_calibrator,
                        "selection_rule": args.selection_rule,
                        "series": row.series,
                        "start_index": int(row.start_index),
                        "start_timestamp": row.start_timestamp,
                        "context_length": int(row.context_length),
                        "horizon": horizon,
                        "dataset_path": row.dataset_path,
                        "tau_pred_raw": tau_pred_raw,
                        "tau_pred_calibrated": tau_corrected,
                        "tau_ref": tau_ref,
                        "cp_abs_error": float(cp_abs_error),
                        "tolerance_hit": int(cp_abs_error <= tolerance),
                        "safe_ceiling": safe_ceiling,
                        "actual_interval_max": actual_max,
                        "sharpness": sharpness,
                        "relative_sharpness": rel_sharpness,
                        "coverage_hits": coverage_hits,
                        "coverage_count": coverage_count,
                        "coverage_window": float(coverage_hits / coverage_count) if coverage_count else float("nan"),
                        "history": row.history,
                        "future_true": row.future_true,
                        "y_pred_median": row.y_pred_median,
                        "y_pred_95": row.y_pred_95,
                    }
                )

            calibrated_metrics = {
                "model": model_name,
                "split": split,
                "selected_calibrator": selected_calibrator,
                "selection_rule": args.selection_rule,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "num_windows_total": len(rows),
                "MAE_CP": float(np.mean(cp_errors)) if cp_errors else float("nan"),
                "tolerance": tolerance,
                "tolerance_hit_rate": float(np.mean(tol_hits)) if tol_hits else float("nan"),
                "coverage_rate": float(coverage_hits_total / coverage_count_total) if coverage_count_total else float("nan"),
                "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
                "relative_sharpness": _ratio_of_means(sharpness_values, actual_max_values),
                "relative_sharpness_mean_of_ratios": (
                    float(np.nanmean(rel_sharpness_values)) if rel_sharpness_values else float("nan")
                ),
                **_covered_sharpness_summary(sharpness_values, actual_max_values),
                **_undercoverage_summary(sharpness_values, actual_max_values),
            }
            comparison_rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "selected_calibrator": selected_calibrator,
                    "selection_rule": args.selection_rule,
                    "raw_MAE_CP": raw_metrics["MAE_CP"],
                    "calibrated_MAE_CP": calibrated_metrics["MAE_CP"],
                    "delta_MAE_CP": calibrated_metrics["MAE_CP"] - raw_metrics["MAE_CP"],
                    "raw_tolerance_hit_rate": raw_metrics["tolerance_hit_rate"],
                    "calibrated_tolerance_hit_rate": calibrated_metrics["tolerance_hit_rate"],
                    "delta_tolerance_hit_rate": calibrated_metrics["tolerance_hit_rate"] - raw_metrics["tolerance_hit_rate"],
                    "raw_coverage_rate": raw_metrics["coverage_rate"],
                    "calibrated_coverage_rate": calibrated_metrics["coverage_rate"],
                    "delta_coverage_rate": calibrated_metrics["coverage_rate"] - raw_metrics["coverage_rate"],
                    "raw_sharpness": raw_metrics["sharpness"],
                    "calibrated_sharpness": calibrated_metrics["sharpness"],
                    "delta_sharpness": calibrated_metrics["sharpness"] - raw_metrics["sharpness"],
                }
            )

            model_windows_path = split_dir / f"{model_name}_calibrated_system_windows.csv"
            model_metrics_path = split_dir / f"{model_name}_calibrated_system_metrics.json"
            pd.DataFrame(rows).to_csv(model_windows_path, index=False)
            model_metrics_path.write_text(json.dumps(calibrated_metrics, indent=2))
            split_summary[model_name] = {
                "windows_path": str(model_windows_path),
                "metrics_path": str(model_metrics_path),
                "selected_calibrator": selected_calibrator,
            }
            progress.advance(
                1,
                phase="eval",
                split=split,
                model=model_name,
                calibrator=selected_calibrator,
                mae_cp=calibrated_metrics["MAE_CP"],
            )

        summary["splits"][split] = split_summary

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "calibrated_system_eval_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    summary["comparison_csv"] = str(comparison_path)

    summary_path = output_dir / "calibrated_system_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    progress.write(f"Saved calibrated system summary: {summary_path}")
    progress.write(f"Saved calibrated system comparison: {comparison_path}")
    progress.close()


if __name__ == "__main__":
    main()
