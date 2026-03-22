"""Run split-aware forecast benchmarking and save reusable forecast artifacts."""

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

from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    SUPPORTED_MAIN_MODELS,
    parse_quantiles,
)
from src.device import AUTO_DEVICE, AUTO_DEVICE_HELP, resolve_device_name
from src.experiment import (
    build_experiment_manifest,
    build_sampled_starts_map,
    daily_seasonal_period,
    default_experiment_dir,
    load_manifest,
    load_wide_dataframe,
    save_manifest,
)
from src.pipeline import build_forecaster
from src.system_metrics import clamp_upper_quantile, pinball_loss


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def _validate_models(models: list[str]) -> list[str]:
    normalized = [model.lower() for model in models]
    invalid = [model for model in normalized if model not in SUPPORTED_MAIN_MODELS]
    if invalid:
        raise ValueError(
            f"Unsupported forecast model(s): {', '.join(invalid)}. "
            f"Supported models: {', '.join(SUPPORTED_MAIN_MODELS)}."
        )
    return normalized


def _resolve_checkpoint(
    explicit_path: Path | None,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
) -> Path | None:
    if model_name in {"chronos2", "seasonal_naive"}:
        return None
    if explicit_path is not None:
        return explicit_path
    return output_dir / "checkpoints" / f"{model_name}_{dataset_name}_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trainable, zero-shot, and seasonal-naive baselines on identical validation/test "
            "windows and save the forecast paths for downstream system evaluation."
        )
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--quantiles", default="0.5,0.95")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--models", default="lstm,deepar,chronos2,seasonal_naive")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--sampling-mode", choices=("random", "rolling"), default="random")
    parser.add_argument("--random-windows-per-series", type=int, default=50)
    parser.add_argument("--random-with-replacement", action="store_true")
    parser.add_argument("--random-without-replacement", dest="random_with_replacement", action="store_false")
    parser.set_defaults(random_with_replacement=True)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--max-windows-per-series", type=int, default=None)
    parser.add_argument("--paired-windows", action="store_true")
    parser.add_argument("--independent-windows", dest="paired_windows", action="store_false")
    parser.set_defaults(paired_windows=True)
    parser.add_argument("--chronos-model-id", default="amazon/chronos-2")
    parser.add_argument("--chronos-num-samples", type=int, default=100)
    parser.add_argument("--lstm-checkpoint", type=Path, default=None)
    parser.add_argument("--deepar-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=AUTO_DEVICE, help=AUTO_DEVICE_HELP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device_name(args.device)
    if args.manifest_path is not None:
        manifest = load_manifest(args.manifest_path)
    else:
        manifest = build_experiment_manifest(
            data_path=args.data_path,
            timestamp_col=args.timestamp_col,
            context_length=args.context_length,
            horizon=args.horizon,
            quantiles=parse_quantiles(args.quantiles) if args.quantiles else DEFAULT_QUANTILES,
            random_seed=args.random_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    output_dir = args.output_dir or default_experiment_dir(manifest)
    forecast_root = output_dir / "forecast_eval"
    forecast_root.mkdir(parents=True, exist_ok=True)
    manifest_path = save_manifest(manifest, output_dir / "manifest.json")

    df = load_wide_dataframe(manifest.dataset_path, manifest.timestamp_col)
    timestamps = df[manifest.timestamp_col].astype(str).to_numpy()
    models = _validate_models(_parse_csv(args.models))
    splits = _parse_csv(args.splits)

    run_summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "device": device,
        "splits": {},
    }

    shared_maps: dict[str, dict[str, list[int]]] = {}
    for split in splits:
        if args.paired_windows:
            shared_maps[split] = build_sampled_starts_map(
                manifest=manifest,
                split=split,
                sampling_mode=args.sampling_mode,
                random_windows_per_series=args.random_windows_per_series,
                random_seed=args.random_seed,
                random_with_replacement=args.random_with_replacement,
                window_step=args.window_step,
                max_windows_per_series=args.max_windows_per_series,
            )

        split_dir = forecast_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if args.paired_windows:
            (split_dir / "sampled_windows_manifest.json").write_text(
                json.dumps(
                    {
                        "split": split,
                        "sampling_mode": args.sampling_mode,
                        "random_seed": args.random_seed,
                        "random_windows_per_series": args.random_windows_per_series,
                        "random_with_replacement": args.random_with_replacement,
                        "window_step": args.window_step,
                        "max_windows_per_series": args.max_windows_per_series,
                        "starts_by_series": shared_maps[split],
                    },
                    indent=2,
                )
            )

        split_summary: dict[str, Any] = {}
        for model_name in models:
            starts_map = (
                shared_maps[split]
                if args.paired_windows
                else build_sampled_starts_map(
                    manifest=manifest,
                    split=split,
                    sampling_mode=args.sampling_mode,
                    random_windows_per_series=args.random_windows_per_series,
                    random_seed=args.random_seed + (abs(hash(model_name)) % 10000),
                    random_with_replacement=args.random_with_replacement,
                    window_step=args.window_step,
                    max_windows_per_series=args.max_windows_per_series,
                )
            )
            checkpoint_path = _resolve_checkpoint(
                explicit_path=args.lstm_checkpoint if model_name == "lstm" else args.deepar_checkpoint,
                output_dir=output_dir,
                model_name=model_name,
                dataset_name=manifest.dataset_name,
            )
            forecaster = build_forecaster(
                model_name,
                checkpoint_path=checkpoint_path,
                context_length=manifest.context_length,
                forecast_length=manifest.horizon,
                quantiles=manifest.quantiles,
                chronos_model_id=args.chronos_model_id,
                chronos_num_samples=args.chronos_num_samples,
                seasonal_period=daily_seasonal_period(manifest.cadence),
                device=device,
            )

            rows: list[dict[str, Any]] = []
            sse_total = 0.0
            count_total = 0
            q50_pinball_total = 0.0
            q95_pinball_total = 0.0
            coverage_hits = 0
            interval_sum = 0.0
            interval_count = 0

            for series_name in manifest.series_columns:
                series = df[series_name].to_numpy(dtype=float)
                for start in starts_map[series_name]:
                    start_idx = int(start)
                    history = series[start_idx - manifest.context_length : start_idx]
                    future_true = series[start_idx : start_idx + manifest.horizon]
                    pred = forecaster.predict_quantiles(history=history, horizon=manifest.horizon)
                    y50 = np.asarray(pred.y_pred_median, dtype=float).reshape(-1)
                    y95 = clamp_upper_quantile(y50, pred.y_pred_95)

                    sse_total += float(np.square(y50 - future_true).sum())
                    count_total += int(future_true.size)
                    q50_pinball_total += pinball_loss(future_true, y50, 0.5) * future_true.size
                    q95_pinball_total += pinball_loss(future_true, y95, 0.95) * future_true.size
                    coverage_hits += int(np.sum(future_true <= y95))
                    interval_sum += float(np.sum(np.maximum(y95 - y50, 0.0)))
                    interval_count += int(y50.size)

                    rows.append(
                        {
                            "model": model_name,
                            "split": split,
                            "series": series_name,
                            "start_index": start_idx,
                            "start_timestamp": timestamps[start_idx],
                            "context_length": manifest.context_length,
                            "horizon": manifest.horizon,
                            "dataset_path": manifest.dataset_path,
                            "history": json.dumps(history.tolist()),
                            "future_true": json.dumps(future_true.tolist()),
                            "y_pred_median": json.dumps(y50.tolist()),
                            "y_pred_95": json.dumps(y95.tolist()),
                        }
                    )

            metrics = {
                "model": model_name,
                "split": split,
                "dataset_path": manifest.dataset_path,
                "context_length": manifest.context_length,
                "horizon": manifest.horizon,
                "quantiles": list(manifest.quantiles),
                "device": device,
                "num_windows_total": len(rows),
                "rmse_q50": float(np.sqrt(sse_total / count_total)) if count_total else float("nan"),
                "pinball_q50": float(q50_pinball_total / count_total) if count_total else float("nan"),
                "pinball_q95": float(q95_pinball_total / count_total) if count_total else float("nan"),
                "coverage_q95": float(coverage_hits / count_total) if count_total else float("nan"),
                "interval_width_mean": float(interval_sum / interval_count) if interval_count else float("nan"),
            }
            windows_path = split_dir / f"{model_name}_forecast_windows.csv"
            metrics_path = split_dir / f"{model_name}_forecast_metrics.json"
            pd.DataFrame(rows).to_csv(windows_path, index=False)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            split_summary[model_name] = {
                "windows_path": str(windows_path),
                "metrics_path": str(metrics_path),
            }
            print(
                f"[forecast_eval:{split}:{model_name}] "
                f"rmse_q50={metrics['rmse_q50']:.4f} coverage_q95={metrics['coverage_q95']:.4f}"
            )

        run_summary["splits"][split] = split_summary

    summary_path = forecast_root / "forecast_eval_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"Saved forecast summary: {summary_path}")


if __name__ == "__main__":
    main()
