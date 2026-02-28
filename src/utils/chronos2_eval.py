"""
Evaluate a Chronos-2 model on a test CSV dataset (wide format).

The dataset should include a timestamp column followed by sector columns.
RMSE is computed using the median quantile forecast, alongside pinball
losses, coverage, and interval width for each requested quantile set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Chronos-2 and report median RMSE, quantile losses, "
            "coverage, and interval width."
        )
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="Chronos-2 model identifier (default: amazon/chronos-2).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/datasets/data_instant_test.csv"),
        help="Testing CSV dataset in wide format.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        help="Number of past points to feed the model.",
    )
    parser.add_argument(
        "--forecast-length",
        type=int,
        default=1,
        help="Prediction length (one-step by default).",
    )
    parser.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated list of quantiles to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/chronos_eval_benchmarking/evals/chronos2_eval.json"),
        help="Where to write the RMSE metrics JSON.",
    )
    return parser.parse_args()


def init_pipeline(model_id: str, device: torch.device) -> Any:
    from chronos import Chronos2Pipeline

    if device.type == "cuda":
        device_map = "cuda"
    else:
        device_map = "cpu"
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)


def _format_quantile_key(q: float) -> str:
    return f"q{q:g}"


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _display_metrics_table(title: str, metrics: Dict[str, Any]) -> None:
    rows = [{"metric": key, "value": _format_metric_value(value)} for key, value in metrics.items()]
    try:
        from IPython.display import display

        if title:
            display(title)
        try:
            import pandas as pd

            display(pd.DataFrame(rows))
            return
        except Exception:
            display(rows)
            return
    except Exception:
        pass

    if title:
        print(title)
    metric_width = max(len("metric"), *(len(row["metric"]) for row in rows)) if rows else 6
    value_width = max(len("value"), *(len(row["value"]) for row in rows)) if rows else 5
    separator = f"+-{'-' * metric_width}-+-{'-' * value_width}-+"
    print(separator)
    print(f"| {'metric'.ljust(metric_width)} | {'value'.ljust(value_width)} |")
    print(separator)
    for row in rows:
        print(f"| {row['metric'].ljust(metric_width)} | {row['value'].ljust(value_width)} |")
    print(separator)


def _parse_quantiles(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("quantiles must be a comma-separated list like 0.1,0.5,0.9")
    quantiles = [float(part) for part in parts]
    for q in quantiles:
        if q <= 0.0 or q >= 1.0:
            raise ValueError("quantiles must be between 0 and 1 (exclusive).")
    return quantiles


def _select_median_index(quantiles: Sequence[float]) -> int:
    if not quantiles:
        raise ValueError("quantiles must contain at least one entry.")
    quantiles = list(quantiles)
    try:
        return quantiles.index(0.5)
    except ValueError:
        return min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))


def forecast_to_samples(forecast: Any) -> np.ndarray:
    if isinstance(forecast, list) and forecast and isinstance(forecast[0], torch.Tensor):
        arr = forecast[0].detach().cpu().numpy()
    elif isinstance(forecast, torch.Tensor):
        arr = forecast.detach().cpu().numpy()
    else:
        arr = np.asarray(forecast)

    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    raise ValueError("Unexpected forecast shape for Chronos-2 samples.")


def load_series(data_path: Path, timestamp_col: str = "timestamp") -> Dict[str, List[float]]:
    df = pd.read_csv(data_path)
    sector_cols = [col for col in df.columns if col != timestamp_col]
    if not sector_cols:
        raise ValueError("No sector columns found in dataset.")
    values = df[sector_cols].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("NaN values found in dataset; clean data before evaluation.")
    return {col: values[col].tolist() for col in sector_cols}


def evaluate(
    pipeline: Any,
    series_map: Dict[str, List[float]],
    context_length: int,
    forecast_length: int,
    rng: np.random.Generator,
    quantiles: Sequence[float],
) -> Dict[str, Any]:
    total_sse = 0.0
    total_count = 0
    quantile_loss_sum = [0.0 for _ in quantiles]
    quantile_count = 0
    coverage_hits = 0
    coverage_count = 0
    interval_sum = 0.0
    interval_count = 0
    per_series: List[Dict[str, Any]] = []
    skipped: List[str] = []
    median_idx = _select_median_index(quantiles)
    quantiles_array = np.asarray(quantiles, dtype=float)
    lower_idx = int(np.argmin(quantiles_array))
    upper_idx = int(np.argmax(quantiles_array))

    with torch.no_grad():
        for sector, series in series_map.items():
            max_start = len(series) - context_length - forecast_length + 1
            if max_start <= 0:
                skipped.append(sector)
                continue

            for start_idx in range(max_start):
                start = int(start_idx)
                ctx_values = series[start : start + context_length]
                target_window = series[
                    start + context_length : start + context_length + forecast_length
                ]
                target = np.asarray(target_window, dtype=float)
                context = torch.tensor(ctx_values, dtype=torch.float32)

                forecast = pipeline.predict(
                    inputs=[context], prediction_length=forecast_length
                )
                samples = forecast_to_samples(forecast)
                quantile_preds = np.quantile(samples, quantiles_array, axis=0)
                median_pred = quantile_preds[median_idx]
                lower_pred = quantile_preds[lower_idx]
                upper_pred = quantile_preds[upper_idx]

                squared_error = (median_pred - target) ** 2
                total_sse += float(squared_error.sum())
                total_count += squared_error.size

                errors = target[None, :] - quantile_preds
                loss_q = np.maximum(
                    quantiles_array[:, None] * errors,
                    (quantiles_array[:, None] - 1) * errors,
                )
                quantile_loss_sum = [
                    total + float(value)
                    for total, value in zip(quantile_loss_sum, loss_q.sum(axis=1))
                ]
                quantile_count += loss_q.shape[1]

                within = (target >= lower_pred) & (target <= upper_pred)
                coverage_hits += int(within.sum())
                coverage_count += within.size
                width = np.maximum(upper_pred - lower_pred, 0.0)
                interval_sum += float(width.sum())
                interval_count += width.size

                if forecast_length == 1:
                    target_value: float | list[float] = float(target[0])
                    prediction_value: float | list[float] = float(median_pred[0])
                    squared_error_value: float | list[float] = float(squared_error[0])
                else:
                    target_value = target.tolist()
                    prediction_value = median_pred.tolist()
                    squared_error_value = squared_error.tolist()

                per_series.append(
                    {
                        "sector": sector,
                        "sample_start_index": start,
                        "target": target_value,
                        "prediction": prediction_value,
                        "squared_error": squared_error_value,
                    }
                )

    if total_count == 0:
        raise RuntimeError("No RMSE samples were generated; check dataset length.")

    rmse = math.sqrt(total_sse / total_count)
    quantile_metrics = {
        _format_quantile_key(q): loss / quantile_count if quantile_count else 0.0
        for q, loss in zip(quantiles, quantile_loss_sum)
    }
    coverage = coverage_hits / coverage_count if coverage_count else 0.0
    interval_width = interval_sum / interval_count if interval_count else 0.0
    return {
        "rmse": rmse,
        "median_quantile": float(quantiles[median_idx]),
        "quantiles": list(quantiles),
        "interval_quantiles": {
            "lower": float(quantiles_array[lower_idx]),
            "upper": float(quantiles_array[upper_idx]),
        },
        "quantile_loss": quantile_metrics,
        "coverage": coverage,
        "interval_width": interval_width,
        "per_series": per_series,
        "skipped_sectors": skipped,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    quantiles = _parse_quantiles(args.quantiles)
    pipeline = init_pipeline(args.model_id, device)
    series_map = load_series(args.data_path)
    results = evaluate(
        pipeline,
        series_map,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        rng=rng,
        quantiles=quantiles,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)

    table_metrics: Dict[str, Any] = {
        "rmse": results["rmse"],
        "median_quantile": results.get("median_quantile"),
        "coverage": results.get("coverage"),
        "interval_width": results.get("interval_width"),
    }
    interval_quantiles = results.get("interval_quantiles", {})
    if interval_quantiles:
        table_metrics["interval_lower"] = interval_quantiles.get("lower")
        table_metrics["interval_upper"] = interval_quantiles.get("upper")
    quantile_loss = results.get("quantile_loss", {})
    for key, value in quantile_loss.items():
        table_metrics[f"pinball_{key}"] = value
    _display_metrics_table("Chronos-2 metrics", table_metrics)
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
