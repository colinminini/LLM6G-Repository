"""Shared canonical forecast evaluation helpers."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.experiment import ExperimentManifest, build_canonical_eval_starts_map
from src.pipeline import Forecaster
from src.progress import StageProgressDisplay
from src.system_metrics import clamp_upper_quantile, pinball_loss


def canonical_window_policy(
    manifest: ExperimentManifest,
    *,
    max_windows_per_series: int | None = None,
) -> dict[str, Any]:
    return {
        "type": "non_overlapping",
        "anchor": "split_start",
        "window_step": int(manifest.horizon),
        "max_windows_per_series": (
            int(max_windows_per_series)
            if max_windows_per_series is not None
            else None
        ),
    }


def evaluate_forecaster_on_split(
    *,
    df: pd.DataFrame,
    manifest: ExperimentManifest,
    split: str,
    model_name: str,
    forecaster: Forecaster,
    device: str,
    starts_map: dict[str, list[int]] | None = None,
    max_windows_per_series: int | None = None,
    progress_label: str | None = None,
    progress_position: int = 0,
    progress_leave: bool = True,
    progress_callback: Callable[[int, int, str, dict[str, Any]], None] | None = None,
    collect_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_name = split.strip().lower()
    starts_map = starts_map or build_canonical_eval_starts_map(
        manifest=manifest,
        split=split_name,
        max_windows_per_series=max_windows_per_series,
    )
    timestamps = df[manifest.timestamp_col].astype(str).to_numpy()
    total_windows = sum(len(starts) for starts in starts_map.values())
    batch_size = max(1, int(getattr(forecaster, "evaluation_batch_size", 64)))
    upper_quantile = float(manifest.quantiles[-1]) if manifest.quantiles else 0.99

    progress: StageProgressDisplay | None = None
    if progress_label is not None:
        progress = StageProgressDisplay(
            progress_label,
            total_windows,
            unit="window",
            position=progress_position,
            leave=progress_leave,
        )
        progress.write(f"[{progress_label}] starting windows={total_windows}")
        progress.update(0, phase="predict")
    if progress_callback is not None:
        progress_callback(0, total_windows, "predict", {})

    rows: list[dict[str, Any]] = []
    sse_total = 0.0
    count_total = 0
    q50_pinball_total = 0.0
    q95_pinball_total = 0.0
    coverage_hits = 0
    interval_sum = 0.0
    interval_count = 0
    callback_started_at = time.perf_counter()
    processed_windows = 0

    for series_name in manifest.series_columns:
        series = df[series_name].to_numpy(dtype=float)
        starts = starts_map[series_name]
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = [int(start) for start in starts[batch_start : batch_start + batch_size]]
            histories = np.stack(
                [
                    series[start_idx - manifest.context_length : start_idx]
                    for start_idx in batch_starts
                ],
                axis=0,
            )
            futures = np.stack(
                [
                    series[start_idx : start_idx + manifest.horizon]
                    for start_idx in batch_starts
                ],
                axis=0,
            )
            pred_batch = forecaster.predict_quantiles_batch(
                histories=histories,
                horizon=manifest.horizon,
                series_names=[series_name] * len(batch_starts),
                start_indices=batch_starts,
            )
            y50_batch = np.asarray(pred_batch.y_pred_median, dtype=float)
            y95_batch = clamp_upper_quantile(y50_batch, pred_batch.y_pred_95)

            sse_total += float(np.square(y50_batch - futures).sum())
            count_total += int(futures.size)
            q50_pinball_total += pinball_loss(futures, y50_batch, 0.5) * futures.size
            q95_pinball_total += pinball_loss(futures, y95_batch, upper_quantile) * futures.size
            coverage_hits += int(np.sum(futures <= y95_batch))
            interval_sum += float(np.sum(np.maximum(y95_batch - y50_batch, 0.0)))
            interval_count += int(y50_batch.size)
            processed_windows += len(batch_starts)

            if collect_rows:
                for row_idx, start_idx in enumerate(batch_starts):
                    rows.append(
                        {
                            "model": model_name,
                            "split": split_name,
                            "series": series_name,
                            "start_index": start_idx,
                            "start_timestamp": timestamps[start_idx],
                            "context_length": manifest.context_length,
                            "horizon": manifest.horizon,
                            "dataset_path": manifest.dataset_path,
                            "history": json.dumps(histories[row_idx].tolist()),
                            "future_true": json.dumps(futures[row_idx].tolist()),
                            "y_pred_median": json.dumps(y50_batch[row_idx].tolist()),
                            "y_pred_95": json.dumps(y95_batch[row_idx].tolist()),
                        }
                    )
            if progress is not None:
                progress.advance(len(batch_starts), phase="predict", series=series_name)
            if progress_callback is not None:
                completed = min(processed_windows, total_windows)
                elapsed = time.perf_counter() - callback_started_at
                estimated_total = (
                    elapsed * total_windows / completed
                    if completed > 0
                    else None
                )
                remaining = (
                    max(0.0, estimated_total - elapsed)
                    if estimated_total is not None
                    else None
                )
                progress_callback(
                    completed,
                    total_windows,
                    "predict",
                    {
                        "series": series_name,
                        "eval_elapsed": elapsed,
                        "eval_eta": remaining,
                        "eval_est_total": estimated_total,
                    },
                )

    upper_tag = f"q{int(round(100.0 * upper_quantile))}"
    pinball_upper = float(q95_pinball_total / count_total) if count_total else float("nan")
    coverage_upper = float(coverage_hits / count_total) if count_total else float("nan")
    metrics = {
        "model": model_name,
        "split": split_name,
        "dataset_path": manifest.dataset_path,
        "context_length": manifest.context_length,
        "horizon": manifest.horizon,
        "quantiles": list(manifest.quantiles),
        "upper_quantile": float(upper_quantile),
        "device": device,
        "num_windows_total": total_windows,
        "rmse_q50": float(np.sqrt(sse_total / count_total)) if count_total else float("nan"),
        "pinball_q50": float(q50_pinball_total / count_total) if count_total else float("nan"),
        f"pinball_{upper_tag}": pinball_upper,
        f"coverage_{upper_tag}": coverage_upper,
        "pinball_q_upper": pinball_upper,
        "coverage_q_upper": coverage_upper,
        # Aliases retained for backward compatibility with downstream readers
        # that key on the hardcoded "q95" suffix (train_utils, older plots).
        "pinball_q95": pinball_upper,
        "coverage_q95": coverage_upper,
        "interval_width_mean": float(interval_sum / interval_count) if interval_count else float("nan"),
        "window_policy": canonical_window_policy(
            manifest,
            max_windows_per_series=max_windows_per_series,
        ),
    }

    if progress is not None:
        progress.update(
            total_windows,
            phase="write",
            rmse_q50=float(metrics["rmse_q50"]),
            coverage_q95=float(metrics["coverage_q95"]),
        )
        progress.write(
            f"[{progress_label}] rmse_q50={metrics['rmse_q50']:.4f} "
            f"coverage_q95={metrics['coverage_q95']:.4f}"
        )
        progress.close()
    if progress_callback is not None:
        final_elapsed = time.perf_counter() - callback_started_at
        progress_callback(
            total_windows,
            total_windows,
            "done",
            {
                "rmse_q50": float(metrics["rmse_q50"]),
                "coverage_q95": float(metrics["coverage_q95"]),
                "eval_elapsed": final_elapsed,
                "eval_eta": 0.0,
                "eval_est_total": final_elapsed,
            },
        )

    return rows, metrics
