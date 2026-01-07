"""
Evaluate a Chronos-2 model on a test CSV dataset (wide format).

The dataset should include a timestamp column followed by sector columns.
RMSE is computed over one-step forecasts sampled within each sector series.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Chronos-2 on a testing dataset and report RMSE."
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
        default=Path("results/evals/chronos2_test_rmse.json"),
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


def forecast_to_point(forecast: Any) -> float:
    if isinstance(forecast, list) and forecast and isinstance(forecast[0], torch.Tensor):
        arr = forecast[0].detach().cpu().numpy()
    elif isinstance(forecast, torch.Tensor):
        arr = forecast.detach().cpu().numpy()
    else:
        arr = np.asarray(forecast)

    if arr.ndim == 0:
        return float(arr.item())
    if arr.ndim == 1:
        return float(arr[-1])
    if arr.ndim == 2:
        return float(arr.mean(axis=0)[-1])
    return float(arr.mean(axis=1)[..., -1].squeeze()[()])


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
) -> Dict[str, Any]:
    squared_errors: List[float] = []
    per_series: List[Dict[str, Any]] = []
    skipped: List[str] = []

    with torch.no_grad():
        for sector, series in series_map.items():
            max_start = len(series) - context_length - forecast_length + 1
            if max_start <= 0:
                skipped.append(sector)
                continue

            for start_idx in range(max_start):
                start = int(start_idx)
                ctx_values = series[start : start + context_length]
                target = float(series[start + context_length])
                context = torch.tensor(ctx_values, dtype=torch.float32)

                forecast = pipeline.predict(
                    inputs=[context], prediction_length=forecast_length
                )
                prediction = forecast_to_point(forecast)
                squared_error = (prediction - target) ** 2
                squared_errors.append(squared_error)
                per_series.append(
                    {
                        "sector": sector,
                        "sample_start_index": start,
                        "target": target,
                        "prediction": prediction,
                        "squared_error": squared_error,
                    }
                )

    if not squared_errors:
        raise RuntimeError("No RMSE samples were generated; check dataset length.")

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    return {"rmse": rmse, "per_series": per_series, "skipped_sectors": skipped}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    pipeline = init_pipeline(args.model_id, device)
    series_map = load_series(args.data_path)
    results = evaluate(
        pipeline,
        series_map,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        rng=rng,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)

    print(f"RMSE: {results['rmse']:.6f}")
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
