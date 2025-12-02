"""
Run single-step RMSE evaluation of a Chronos model on the processed traffic data.

The dataset must be produced by single_dataprocessing.py and contains one entry
per sector with its full history. The model forecasts a single final step using
all previous measurements as context.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Chronos model on single-step antenna traffic forecasting."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Chronos model identifier (e.g., amazon/chronos-bolt-tiny).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed_trafic.jsonl"),
        help="JSONL produced by dataprocessing.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Where to store the evaluation JSON. Defaults to results/<model>.json.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of forecast samples to draw before averaging.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu; CPU-only is recommended on macOS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open() as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise RuntimeError(f"No data found in {path}")
    return entries


def init_pipeline(model_id: str, device: torch.device) -> Any:
    """
    Load Chronos pipeline depending on the model type.

    Newer versions accept the `dtype` kwarg; older ones expect `torch_dtype`.
    We try the modern API first and fall back for compatibility. When the
    installed chronos package is too old for the model, surface a clear error.
    """
    torch_dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    common_kwargs = {"device_map": device}

    def _try_load(loader):
        try:
            return loader(dtype=torch_dtype, **common_kwargs)
        except TypeError:
            return loader(torch_dtype=torch_dtype, **common_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            if "input_patch_size" in str(exc):
                raise RuntimeError(
                    "This model requires a newer chronos-forecasting build. "
                    "Please upgrade: pip install -U chronos-forecasting"
                ) from exc
            raise

    if "chronos-2" in model_id:
        from chronos import Chronos2Pipeline

        return _try_load(lambda **kw: Chronos2Pipeline.from_pretrained(model_id, **kw))

    if "chronos-bolt" in model_id:
        from chronos import chronos_bolt

        return _try_load(
            lambda **kw: chronos_bolt.ChronosBoltPipeline.from_pretrained(
                model_id, **kw
            )
        )

    from chronos import ChronosPipeline

    return _try_load(lambda **kw: ChronosPipeline.from_pretrained(model_id, **kw))


def forecast_to_point(pipeline: Any, forecast: Any) -> float:
    """
    Convert pipeline output to a single point estimate.

    ChronosPipeline returns samples shaped (batch, num_samples, pred_len).
    ChronosBoltPipeline returns quantiles shaped (batch, num_quantiles, pred_len).
    Chronos2Pipeline returns a list of tensors, one per series.
    """
    # Chronos2: list of tensors
    if isinstance(forecast, list) and forecast and isinstance(forecast[0], torch.Tensor):
        arr = forecast[0].detach().cpu().numpy()
        if arr.ndim == 0:
            return float(arr.item())
        if arr.ndim == 1:
            return float(arr[-1])
        if arr.ndim == 2:
            return float(arr.mean(axis=0)[-1])
        # Expected shape (batch, num_samples, prediction_length)
        return float(arr.mean(axis=1)[..., -1].squeeze())

    if isinstance(forecast, torch.Tensor):
        arr = forecast.detach().cpu().numpy()
    else:
        arr = np.asarray(forecast)

    if hasattr(pipeline, "quantiles"):
        quantiles = np.asarray(getattr(pipeline, "quantiles", []), dtype=float)
        if quantiles.size == 0:
            # Fallback: choose middle quantile index
            idx = arr.shape[1] // 2 if arr.ndim >= 2 else 0
        else:
            idx = int(np.abs(quantiles - 0.5).argmin())
        return float(arr[..., idx, -1].squeeze()[()])

    # Samples case
    if arr.ndim == 0:
        return float(arr.item())
    if arr.ndim == 1:
        return float(arr.mean())
    if arr.ndim == 2:
        return float(arr.mean(axis=0)[-1])
    # Expected shape (batch, num_samples, prediction_length)
    return float(arr.mean(axis=1)[..., -1].squeeze()[()])


def run_evaluation(
    pipeline: Any, dataset: List[Dict[str, Any]], num_samples: int
) -> Dict[str, Any]:
    squared_errors: List[float] = []
    per_series: List[Dict[str, Any]] = []

    with torch.no_grad():
        for entry in dataset:
            context = torch.tensor(entry["context"], dtype=torch.float32)
            target = float(entry["target"])

            if pipeline.__class__.__name__ == "Chronos2Pipeline":
                forecast = pipeline.predict(inputs=[context], prediction_length=1)
            elif hasattr(pipeline, "quantiles"):
                forecast = pipeline.predict(inputs=context, prediction_length=1)
            else:
                forecast = pipeline.predict(
                    inputs=context,
                    prediction_length=1,
                    num_samples=num_samples,
                )
            prediction = forecast_to_point(pipeline, forecast)
            error = (prediction - target) ** 2
            squared_errors.append(error)
            per_series.append(
                {
                    "sector": entry["sector"],
                    "site": entry.get("site"),
                    "target": target,
                    "prediction": prediction,
                    "squared_error": error,
                }
            )

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    return {"rmse": rmse, "per_series": per_series}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = load_dataset(args.data_path)
    pipeline = init_pipeline(args.model_id, device)
    results = run_evaluation(pipeline, dataset, args.num_samples)

    slug = args.model_id.replace("/", "__").replace(":", "_")
    output_path = args.output_path or Path("results") / f"eval_{slug}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": args.model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_series": len(dataset),
        "num_samples": args.num_samples,
        "rmse": results["rmse"],
        "per_series": results["per_series"],
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(
        f"Model {args.model_id} | RMSE={results['rmse']:.4f} "
        f"across {len(dataset)} sectors."
    )
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
