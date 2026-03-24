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

from src.change_detection import build_change_point_detector
from src.config import (
    DEFAULT_CP_EXCLUSION_RADIUS,
    DEFAULT_CP_JUMP,
    DEFAULT_CP_MIN_SIZE,
    DEFAULT_CP_MODEL,
    DEFAULT_CP_PENALTY,
    DEFAULT_CP_PERIOD_LENGTH,
    DEFAULT_CP_SCORE_THRESHOLD,
    DEFAULT_TOLERANCE,
)
from src.cp_config import default_cp_config, load_best_cp_configs, parse_csv_list
from src.progress import StageProgressDisplay
from src.system_metrics import (
    binary_classification_metrics,
    clamp_upper_quantile,
    extract_pre_change_interval,
    json_array,
    resolve_tau,
    safe_ceiling_from_tau,
)


def _parse_csv(raw: str) -> list[str]:
    return parse_csv_list(raw, str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run q50 -> breakpoint -> safe ceiling evaluation from saved forecast artifacts."
        )
    )
    parser.add_argument("--forecast-dir", type=Path, required=True)
    parser.add_argument("--models", default="lstm,deepar,tft,chronos2,seasonal_naive")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--cp-config-manifest", type=Path, default=None)
    parser.add_argument("--cp-detector", default=None)
    parser.add_argument("--cp-period-length", default=DEFAULT_CP_PERIOD_LENGTH)
    parser.add_argument("--cp-exclusion-radius", type=float, default=DEFAULT_CP_EXCLUSION_RADIUS)
    parser.add_argument("--cp-score-threshold", type=float, default=DEFAULT_CP_SCORE_THRESHOLD)
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

    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_dir": str(forecast_root),
        "cp_config_manifest": str(args.cp_config_manifest) if args.cp_config_manifest else None,
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

            windows_df = pd.read_csv(windows_path)
            # Pre-parse JSON array columns once to avoid repeated json.loads inside the loop.
            future_true_arrays = [json_array(v) for v in windows_df["future_true"]]
            y50_arrays = [json_array(v) for v in windows_df["y_pred_median"]]
            y95_arrays = [
                clamp_upper_quantile(y50, json_array(v))
                for y50, v in zip(y50_arrays, windows_df["y_pred_95"])
            ]
            progress = StageProgressDisplay(
                f"system_eval:{split}:{model_name}",
                len(windows_df),
                unit="window",
            )
            progress.write(
                f"[system_eval:{split}:{model_name}] starting windows={len(windows_df)}"
            )
            progress.update(0, phase="analyze")
            rows: list[dict[str, Any]] = []
            resolved_cp_errors: list[float] = []
            conditional_cp_errors: list[float] = []
            resolved_tol_hits: list[float] = []
            conditional_tol_hits: list[float] = []
            sharpness_values: list[float] = []
            rel_sharpness_values: list[float] = []
            coverage_hits_total = 0
            coverage_count_total = 0
            has_break_pred: list[bool] = []
            has_break_ref: list[bool] = []

            for row, future_true, y50, y95 in zip(
                windows_df.itertuples(index=False),
                future_true_arrays,
                y50_arrays,
                y95_arrays,
            ):
                horizon = int(row.horizon)

                pred_result = detector.detect_change_point(y50)
                ref_result = detector.detect_change_point(future_true)
                has_break_pred.append(bool(pred_result.has_break))
                has_break_ref.append(bool(ref_result.has_break))
                tau_pred_resolved = resolve_tau(pred_result.tau, horizon)
                tau_ref_resolved = resolve_tau(ref_result.tau, horizon)
                cp_abs_error = abs(tau_pred_resolved - tau_ref_resolved)
                resolved_cp_errors.append(float(cp_abs_error))
                resolved_tol_hits.append(float(cp_abs_error <= args.tolerance))

                conditional_cp_abs_error: float | None = None
                if pred_result.has_break and ref_result.has_break and pred_result.tau is not None and ref_result.tau is not None:
                    conditional_cp_abs_error = float(abs(int(pred_result.tau) - int(ref_result.tau)))
                    conditional_cp_errors.append(conditional_cp_abs_error)
                    conditional_tol_hits.append(float(conditional_cp_abs_error <= args.tolerance))

                true_interval = extract_pre_change_interval(
                    future_true,
                    tau=ref_result.tau,
                    horizon=horizon,
                )
                safe_ceiling = safe_ceiling_from_tau(y95, pred_result.tau, horizon)
                coverage_hits = int(np.sum(true_interval <= safe_ceiling))
                coverage_count = int(true_interval.size)
                coverage_hits_total += coverage_hits
                coverage_count_total += coverage_count
                actual_max = float(np.max(true_interval))
                sharpness = float(safe_ceiling - actual_max)
                sharpness_values.append(sharpness)
                rel_sharpness = float(sharpness / actual_max) if actual_max > 0 else float("nan")
                rel_sharpness_values.append(rel_sharpness)

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
                        "tau_pred": pred_result.tau,
                        "tau_pred_resolved": int(tau_pred_resolved),
                        "tau_ref": ref_result.tau,
                        "tau_ref_resolved": int(tau_ref_resolved),
                        "tau_true": ref_result.tau,
                        "tau_true_resolved": int(tau_ref_resolved),
                        "has_break_pred": int(pred_result.has_break),
                        "has_break_ref": int(ref_result.has_break),
                        "cp_score_pred": pred_result.score,
                        "cp_score_ref": ref_result.score,
                        "cp_abs_error": float(cp_abs_error),
                        "cp_abs_error_when_both_break": conditional_cp_abs_error,
                        "tolerance_hit": int(cp_abs_error <= args.tolerance),
                        "tolerance_hit_when_both_break": (
                            int(conditional_cp_abs_error <= args.tolerance)
                            if conditional_cp_abs_error is not None
                            else None
                        ),
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
                        "y_pred_95": json.dumps(y95.tolist()),
                    }
                )
                progress.advance(1, phase="analyze", series=row.series)

            break_metrics = binary_classification_metrics(has_break_ref, has_break_pred)
            metrics = {
                "model": model_name,
                "split": split,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "num_windows_total": len(rows),
                "MAE_CP": float(np.mean(resolved_cp_errors)) if resolved_cp_errors else float("nan"),
                "tau_mae_when_both_break": (
                    float(np.mean(conditional_cp_errors))
                    if conditional_cp_errors
                    else float("nan")
                ),
                "tolerance": int(args.tolerance),
                "tolerance_hit_rate": float(np.mean(resolved_tol_hits)) if resolved_tol_hits else float("nan"),
                "tolerance_hit_rate_when_both_break": (
                    float(np.mean(conditional_tol_hits))
                    if conditional_tol_hits
                    else float("nan")
                ),
                "break_precision": float(break_metrics["precision"]),
                "break_recall": float(break_metrics["recall"]),
                "break_f1": float(break_metrics["f1"]),
                "coverage_rate": float(coverage_hits_total / coverage_count_total) if coverage_count_total else float("nan"),
                "sharpness": float(np.mean(sharpness_values)) if sharpness_values else float("nan"),
                "relative_sharpness": float(np.nanmean(rel_sharpness_values)) if rel_sharpness_values else float("nan"),
                "cp_detector": detector_config,
            }
            model_windows_path = split_dir / f"{model_name}_system_windows.csv"
            model_metrics_path = split_dir / f"{model_name}_system_metrics.json"
            pd.DataFrame(rows).to_csv(model_windows_path, index=False)
            model_metrics_path.write_text(json.dumps(metrics, indent=2))
            split_summary[model_name] = {
                "windows_path": str(model_windows_path),
                "metrics_path": str(model_metrics_path),
            }
            progress.update(
                len(windows_df),
                phase="write",
                mae_cp=float(metrics["MAE_CP"]),
                break_f1=float(metrics["break_f1"]),
                coverage=float(metrics["coverage_rate"]),
            )
            progress.write(
                f"[system_eval:{split}:{model_name}] "
                f"MAE_CP={metrics['MAE_CP']:.4f} break_f1={metrics['break_f1']:.4f} "
                f"coverage={metrics['coverage_rate']:.4f}"
            )
            progress.close()

        summary["splits"][split] = split_summary

    summary_path = output_root / "system_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved system summary: {summary_path}")


if __name__ == "__main__":
    main()
