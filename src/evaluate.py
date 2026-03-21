"""Rolling-window backtest for probabilistic forecast + change-point evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_DATASET_PATH, DEFAULT_QUANTILES
from src.change_detection import RupturesPeltDetector
from src.pipeline import HybridForecastingChangePointPipeline, build_forecaster


@dataclass
class BacktestConfig:
    model_name: str
    horizon: int
    context_length: int
    window_step: int
    max_windows_per_series: int | None
    sampling_mode: str
    random_windows_per_series: int
    random_seed: int
    random_with_replacement: bool
    tolerance: int
    cp_model: str
    cp_penalty: float
    cp_min_size: int
    cp_jump: int
    forecaster_quantiles: tuple[float, ...]
    chronos_num_samples: int
    lstm_residual_sigma: float | None


def _parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name.")
    return models


def _parse_quantiles(raw: str) -> tuple[float, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("quantiles must be a comma-separated list like 0.5,0.95")
    quantiles = tuple(float(part) for part in parts)
    for q in quantiles:
        if q <= 0.0 or q >= 1.0:
            raise ValueError("quantiles must be between 0 and 1 (exclusive).")
    return quantiles


def _series_columns(df: pd.DataFrame, timestamp_col: str) -> list[str]:
    cols = [col for col in df.columns if col != timestamp_col]
    if not cols:
        raise ValueError("No series columns found in data.")
    return cols


def _resolve_tau(tau: int | None, horizon: int) -> int:
    """Convert None to horizon (i.e., no change in the forecast window)."""
    if tau is None:
        return int(horizon)
    return int(np.clip(int(tau), 0, horizon))


def _extract_pre_change_interval(values: np.ndarray, tau: int, horizon: int) -> np.ndarray:
    """
    Return the interval before the first change-point.

    We interpret tau as the index where the new regime starts, so we keep [0, tau)
    and force at least one point for numerical stability when tau == 0.
    """
    if tau >= horizon:
        segment = values[:horizon]
    else:
        segment = values[: max(1, tau)]
    if segment.size == 0:
        return values[:1]
    return segment


def _compute_start_indices(
    *,
    num_rows: int,
    context_length: int,
    horizon: int,
    evaluation_start_idx: int | None,
) -> tuple[int, np.ndarray]:
    start_first = max(
        int(context_length),
        int(evaluation_start_idx) if evaluation_start_idx is not None else int(context_length),
    )
    start_last = int(num_rows) - int(horizon)
    if start_first > start_last:
        raise ValueError(
            "No valid windows. Check context_length/horizon versus dataset size."
        )
    all_start_indices = np.arange(start_first, start_last + 1, dtype=int)
    if all_start_indices.size == 0:
        raise ValueError(
            "No valid windows. Check context_length/horizon versus dataset size."
        )
    return start_first, all_start_indices


def _build_random_starts_map(
    *,
    series_names: list[str],
    all_start_indices: np.ndarray,
    n_samples: int,
    replace: bool,
    seed: int,
) -> dict[str, list[int]]:
    if n_samples <= 0:
        raise ValueError("random_windows_per_series must be > 0 when sampling_mode=random.")

    sample_size = int(n_samples)
    if (not replace) and sample_size > all_start_indices.size:
        sample_size = int(all_start_indices.size)

    rng = np.random.default_rng(int(seed))
    starts_map: dict[str, list[int]] = {}
    for series_name in series_names:
        sampled = rng.choice(all_start_indices, size=sample_size, replace=replace)
        starts_map[series_name] = np.asarray(sampled, dtype=int).tolist()
    return starts_map


def _load_lstm_residual_sigma(
    explicit_sigma: float | None,
    stats_path: Path | None,
) -> float | None:
    if explicit_sigma is not None:
        return float(explicit_sigma)
    if stats_path is None or not stats_path.exists():
        return None

    payload = json.loads(stats_path.read_text())
    for key in ("sigma_res", "sigma_residual", "residual_sigma", "val_residual_sigma"):
        value = payload.get(key)
        if value is not None:
            return float(value)
    return None


def run_backtest_for_model(
    *,
    full_df: pd.DataFrame,
    timestamp_col: str,
    config: BacktestConfig,
    checkpoint_path: Path | None,
    chronos_model_id: str,
    device: str,
    output_dir: Path,
    evaluation_start_idx: int | None = None,
    dataset_path: str | None = None,
    sampled_starts_map: dict[str, list[int]] | None = None,
    paired_random_windows: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = RupturesPeltDetector(
        model=config.cp_model,
        penalty=config.cp_penalty,
        min_size=config.cp_min_size,
        jump=config.cp_jump,
    )

    residual_sigma = config.lstm_residual_sigma if config.model_name.lower() == "lstm" else None
    forecaster = build_forecaster(
        config.model_name,
        checkpoint_path=checkpoint_path,
        context_length=config.context_length,
        forecast_length=config.horizon,
        quantiles=config.forecaster_quantiles,
        residual_sigma=residual_sigma,
        enable_residual_fallback=True,
        chronos_model_id=chronos_model_id,
        chronos_num_samples=config.chronos_num_samples,
        device=device,
    )

    pipeline = HybridForecastingChangePointPipeline(
        forecaster=forecaster,
        change_detector=detector,
    )

    series_cols = _series_columns(full_df, timestamp_col)
    timestamps = full_df[timestamp_col].astype(str).to_numpy()

    start_first, all_start_indices = _compute_start_indices(
        num_rows=len(full_df),
        context_length=config.context_length,
        horizon=config.horizon,
        evaluation_start_idx=evaluation_start_idx,
    )

    model_seed = int(config.random_seed) + (abs(hash(config.model_name)) % 10_000)
    rng = np.random.default_rng(model_seed)

    rows: list[dict[str, Any]] = []
    cp_errors: list[float] = []
    tol_hits: list[float] = []
    sharpness_values: list[float] = []
    coverage_hits_total = 0
    coverage_count_total = 0

    for series_name in series_cols:
        series = pd.to_numeric(full_df[series_name], errors="coerce").to_numpy(dtype=float)
        if np.isnan(series).any():
            raise ValueError(f"NaN values detected in series {series_name}.")

        if config.sampling_mode == "random" and sampled_starts_map is not None:
            effective_starts = sampled_starts_map.get(series_name, [])
        elif config.sampling_mode == "random":
            n_samples = int(config.random_windows_per_series)
            if n_samples <= 0:
                raise ValueError("random_windows_per_series must be > 0 when sampling_mode=random.")
            replace = bool(config.random_with_replacement)
            if (not replace) and n_samples > all_start_indices.size:
                n_samples = int(all_start_indices.size)
            effective_starts = rng.choice(all_start_indices, size=n_samples, replace=replace)
            effective_starts = np.asarray(effective_starts, dtype=int).tolist()
        else:
            if config.window_step <= 0:
                raise ValueError("window_step must be > 0 when sampling_mode=rolling.")
            effective_starts = all_start_indices[:: config.window_step].tolist()
            if (
                config.max_windows_per_series is not None
                and config.max_windows_per_series > 0
                and len(effective_starts) > config.max_windows_per_series
            ):
                effective_starts = effective_starts[: config.max_windows_per_series]

        for start in effective_starts:
            history = series[start - config.context_length : start]
            future_true = series[start : start + config.horizon]

            pred = pipeline.run(history=history, horizon=config.horizon)
            tau_pred = _resolve_tau(pred.tau_pred, config.horizon)

            tau_true_raw = detector.detect_first_change_point(future_true)
            tau_true = _resolve_tau(tau_true_raw, config.horizon)

            true_interval = _extract_pre_change_interval(
                future_true,
                tau=tau_true,
                horizon=config.horizon,
            )

            cp_abs_error = abs(tau_pred - tau_true)
            cp_errors.append(float(cp_abs_error))
            tol_hits.append(float(cp_abs_error <= config.tolerance))

            safe_ceiling = float(pred.safe_ceiling)
            coverage_hits = int(np.sum(true_interval <= safe_ceiling))
            coverage_count = int(true_interval.size)
            coverage_hits_total += coverage_hits
            coverage_count_total += coverage_count
            coverage_window = coverage_hits / coverage_count if coverage_count else float("nan")

            actual_max = float(np.max(true_interval))
            sharpness = float(safe_ceiling - actual_max)
            sharpness_values.append(sharpness)

            rows.append(
                {
                    "model": config.model_name,
                    "series": series_name,
                    "start_index": int(start),
                    "start_timestamp": timestamps[start],
                    "horizon": int(config.horizon),
                    "tau_pred": int(tau_pred),
                    "tau_true": int(tau_true),
                    "cp_abs_error": float(cp_abs_error),
                    "tolerance_hit": int(cp_abs_error <= config.tolerance),
                    "safe_ceiling": safe_ceiling,
                    "bound_95": safe_ceiling,
                    "actual_interval_max": actual_max,
                    "sharpness": sharpness,
                    "coverage_hits": coverage_hits,
                    "coverage_count": coverage_count,
                    "coverage_window": float(coverage_window),
                    "y_pred_median": json.dumps(pred.y_pred_median.tolist()),
                    "y_pred_95": json.dumps(pred.y_pred_95.tolist()),
                }
            )

    mae_cp = float(np.mean(cp_errors)) if cp_errors else float("nan")
    tolerance_hit_rate = float(np.mean(tol_hits)) if tol_hits else float("nan")
    coverage_rate = (
        float(coverage_hits_total / coverage_count_total)
        if coverage_count_total
        else float("nan")
    )
    sharpness_mean = float(np.mean(sharpness_values)) if sharpness_values else float("nan")

    metrics = {
        "model": config.model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "horizon": config.horizon,
        "context_length": config.context_length,
        "evaluation_start_index": start_first,
        "sampling_mode": config.sampling_mode,
        "random_windows_per_series": config.random_windows_per_series,
        "random_seed": config.random_seed,
        "random_with_replacement": config.random_with_replacement,
        "paired_random_windows": bool(paired_random_windows),
        "window_step": config.window_step,
        "max_windows_per_series": config.max_windows_per_series,
        "num_series": len(series_cols),
        "num_rows_dataset": len(full_df),
        "num_windows_total": len(rows),
        "MAE_CP": mae_cp,
        "tolerance": config.tolerance,
        "tolerance_hit_rate": tolerance_hit_rate,
        "coverage_rate": coverage_rate,
        "sharpness": sharpness_mean,
        "bound_definition": "safe_ceiling=max(y_pred_95[0:tau_pred])",
        "forecaster_quantiles": list(config.forecaster_quantiles),
        "chronos_num_samples": config.chronos_num_samples,
        "lstm_residual_sigma": config.lstm_residual_sigma,
        "cp_detector": {
            "method": "ruptures_pelt",
            "model": config.cp_model,
            "penalty": config.cp_penalty,
            "min_size": config.cp_min_size,
            "jump": config.cp_jump,
        },
    }

    metrics_path = output_dir / f"{config.model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    windows_df = pd.DataFrame(rows)
    windows_path = output_dir / f"{config.model_name}_window_metrics.csv"
    windows_df.to_csv(windows_path, index=False)

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "windows_path": str(windows_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling-window backtest for probabilistic forecast + change-point + safe-ceiling evaluation."
        )
    )
    parser.add_argument("--models", default="lstm,deepar,chronos2")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Full dataset CSV used for evaluation.",
    )
    parser.add_argument(
        "--evaluation-start-index",
        type=int,
        default=None,
        help="Optional index where evaluation starts. If unset, starts at context_length.",
    )

    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument(
        "--forecaster-quantiles",
        default="0.5,0.95",
        help="Comma-separated quantile levels expected from quantile models (e.g., 0.5,0.95).",
    )
    parser.add_argument(
        "--chronos-num-samples",
        type=int,
        default=100,
        help=(
            "Deprecated for Chronos2 direct-quantile output mode. "
            "Kept for backward compatibility."
        ),
    )

    parser.add_argument(
        "--lstm-residual-sigma",
        type=float,
        default=None,
        help="Optional residual sigma for LSTM q95 fallback (Option B).",
    )
    parser.add_argument(
        "--lstm-residual-stats-path",
        type=Path,
        default=None,
        help="Optional JSON stats path containing sigma_res for LSTM fallback.",
    )

    parser.add_argument(
        "--sampling-mode",
        choices=("random", "rolling"),
        default="random",
        help="Use random samples (default) or deterministic rolling windows.",
    )
    parser.add_argument(
        "--random-windows-per-series",
        type=int,
        default=200,
        help="Number of random windows per series when sampling-mode=random.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for window sampling.",
    )
    parser.add_argument(
        "--random-with-replacement",
        action="store_true",
        help="Sample random windows with replacement.",
    )
    parser.add_argument(
        "--random-without-replacement",
        dest="random_with_replacement",
        action="store_false",
        help="Sample random windows without replacement.",
    )
    parser.set_defaults(random_with_replacement=True)

    parser.add_argument(
        "--paired-random-windows",
        action="store_true",
        help="Use the exact same random windows for all models.",
    )
    parser.add_argument(
        "--independent-random-windows",
        dest="paired_random_windows",
        action="store_false",
        help="Use independently sampled windows for each model.",
    )
    parser.set_defaults(paired_random_windows=True)

    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--max-windows-per-series", type=int, default=None)

    parser.add_argument("--tolerance", type=int, default=3)

    parser.add_argument("--cp-model", default="normal")
    parser.add_argument("--cp-penalty", type=float, default=10.0)
    parser.add_argument("--cp-min-size", type=int, default=8)
    parser.add_argument("--cp-jump", type=int, default=1)

    parser.add_argument(
        "--chronos-model-id",
        default="amazon/chronos-2",
        help="Chronos2 zero-shot model identifier.",
    )
    parser.add_argument(
        "--lstm-checkpoint",
        type=Path,
        default=Path("results/experiments/data_1to672_ctx48_h48/q50_q95_seed42/checkpoints/lstm_data_1to672_best.pt"),
    )
    parser.add_argument(
        "--deepar-checkpoint",
        type=Path,
        default=Path("results/experiments/data_1to672_ctx48_h48/q50_q95_seed42/checkpoints/deepar_data_1to672_best.pt"),
    )
    parser.add_argument(
        "--tft-checkpoint",
        type=Path,
        default=Path("results/experiments/data_1to672_ctx48_h48/q50_q95_seed42/checkpoints/tft_data_1to672_best.pt"),
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_df = pd.read_csv(args.data_path)
    series_cols = _series_columns(full_df, args.timestamp_col)

    checkpoint_map = {
        "lstm": args.lstm_checkpoint,
        "deepar": args.deepar_checkpoint,
        "tft": args.tft_checkpoint,
    }

    quantiles = _parse_quantiles(args.forecaster_quantiles)
    lstm_residual_sigma = _load_lstm_residual_sigma(
        explicit_sigma=args.lstm_residual_sigma,
        stats_path=args.lstm_residual_stats_path,
    )

    shared_sampled_starts_map: dict[str, list[int]] | None = None
    if args.sampling_mode == "random" and args.paired_random_windows:
        _, all_start_indices = _compute_start_indices(
            num_rows=len(full_df),
            context_length=args.context_length,
            horizon=args.horizon,
            evaluation_start_idx=args.evaluation_start_index,
        )
        shared_sampled_starts_map = _build_random_starts_map(
            series_names=series_cols,
            all_start_indices=all_start_indices,
            n_samples=args.random_windows_per_series,
            replace=bool(args.random_with_replacement),
            seed=int(args.random_seed),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "paired_random_windows_manifest.json"
        manifest_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(args.data_path),
            "sampling_mode": "random",
            "paired_random_windows": True,
            "random_windows_per_series": int(args.random_windows_per_series),
            "random_seed": int(args.random_seed),
            "random_with_replacement": bool(args.random_with_replacement),
            "num_series": len(series_cols),
            "starts_by_series": shared_sampled_starts_map,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    run_summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "sampling_mode": args.sampling_mode,
        "paired_random_windows": bool(args.paired_random_windows),
        "forecaster_quantiles": list(quantiles),
        "chronos_num_samples": int(args.chronos_num_samples),
        "models": {},
    }

    for model_name in _parse_models(args.models):
        cfg = BacktestConfig(
            model_name=model_name,
            horizon=args.horizon,
            context_length=args.context_length,
            window_step=args.window_step,
            max_windows_per_series=args.max_windows_per_series,
            sampling_mode=args.sampling_mode,
            random_windows_per_series=args.random_windows_per_series,
            random_seed=args.random_seed,
            random_with_replacement=args.random_with_replacement,
            tolerance=args.tolerance,
            cp_model=args.cp_model,
            cp_penalty=args.cp_penalty,
            cp_min_size=args.cp_min_size,
            cp_jump=args.cp_jump,
            forecaster_quantiles=quantiles,
            chronos_num_samples=args.chronos_num_samples,
            lstm_residual_sigma=lstm_residual_sigma,
        )

        checkpoint = checkpoint_map.get(model_name.lower())
        result = run_backtest_for_model(
            full_df=full_df,
            timestamp_col=args.timestamp_col,
            config=cfg,
            checkpoint_path=checkpoint,
            chronos_model_id=args.chronos_model_id,
            device=args.device,
            output_dir=args.output_dir,
            evaluation_start_idx=args.evaluation_start_index,
            dataset_path=str(args.data_path),
            sampled_starts_map=shared_sampled_starts_map,
            paired_random_windows=bool(args.sampling_mode == "random" and args.paired_random_windows),
        )
        run_summary["models"][model_name] = result
        print(
            f"[{model_name}] MAE_CP={result['metrics']['MAE_CP']:.4f} "
            f"Coverage={result['metrics']['coverage_rate']:.4f} "
            f"Sharpness={result['metrics']['sharpness']:.4f}"
        )

    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
