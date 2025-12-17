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
        "--context-length",
        type=int,
        default=128,
        help="Number of past points to feed the model (uniform across models).",
    )
    parser.add_argument(
        "--rmse-samples",
        type=int,
        default=10,
        help="How many random 1-step predictions to score per sector before averaging.",
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
    pipeline: Any,
    dataset: List[Dict[str, Any]],
    num_samples: int,
    context_length: int,
    rmse_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    squared_errors: List[float] = []
    squared_ratio_errors: List[float] = []
    per_series: List[Dict[str, Any]] = []
    skipped_sectors: List[str] = []

    with torch.no_grad():
        for entry in dataset:
            # Reconstruct full history (context plus final target).
            series = list(entry["context"]) + [float(entry["target"])]
            max_start = len(series) - context_length
            if max_start <= 0:
                skipped_sectors.append(entry["sector"])
                continue

            replace = max_start < rmse_samples
            start_indices = rng.choice(max_start, size=rmse_samples, replace=replace)

            for start_idx in start_indices:
                start = int(start_idx)
                ctx_values = series[start : start + context_length]
                target = float(series[start + context_length])
                context = torch.tensor(ctx_values, dtype=torch.float32)

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
                # RMSE uses the raw absolute error (no scaling).
                squared_error = (prediction - target) ** 2
                # Scale only for RMSE%/ratio to avoid exploding percentages on near-zero values.
                scale = max(abs(target), abs(prediction), 1.0)
                percentage_error = (prediction - target) / scale
                squared_ratio = percentage_error**2
                squared_errors.append(squared_error)
                squared_ratio_errors.append(squared_ratio)
                per_series.append(
                    {
                        "sector": entry["sector"],
                        "site": entry.get("site"),
                        "sample_start_index": start,
                        "target": target,
                        "prediction": prediction,
                        "squared_error": squared_error,
                        "percentage_error": percentage_error * 100.0,
                        "squared_percentage_error": (percentage_error * 100.0) ** 2,
                    }
                )

    if not squared_ratio_errors:
        raise RuntimeError(
            "No RMSE samples were generated. "
            f"Skipped sectors (not enough history for context {context_length}): {skipped_sectors}"
        )

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    rmse_ratio = math.sqrt(sum(squared_ratio_errors) / len(squared_ratio_errors))
    rmse_percentage = rmse_ratio * 100.0
    return {
        "rmse": rmse,
        "rmse_percentage": rmse_percentage,
        "rmse_ratio": rmse_ratio,
        "per_series": per_series,
        "skipped_sectors": skipped_sectors,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    dataset = load_dataset(args.data_path)
    pipeline = init_pipeline(args.model_id, device)
    results = run_evaluation(
        pipeline,
        dataset,
        num_samples=args.num_samples,
        context_length=args.context_length,
        rmse_samples=args.rmse_samples,
        rng=rng,
    )

    slug = args.model_id.replace("/", "__").replace(":", "_")
    output_path = args.output_path or Path("results") / "evals" / f"eval_{slug}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_id": args.model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_series": len(dataset),
        "num_samples": args.num_samples,
        "context_length": args.context_length,
        "rmse_samples": args.rmse_samples,
        "rmse": results["rmse"],
        "rmse_percentage": results["rmse_percentage"],
        "rmse_ratio": results["rmse_ratio"],
        "skipped_sectors": results["skipped_sectors"],
        "per_series": results["per_series"],
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(
        f"Model {args.model_id} | RMSE%={results['rmse_percentage']:.4f}% "
        f"(RMSE={results['rmse']:.4f}) across {len(dataset)} sectors "
        f"({args.rmse_samples} samples/sector, context={args.context_length})."
    )
    print(f"Saved detailed results to {output_path}")
    if results["skipped_sectors"]:
        print(
            f"Skipped {len(results['skipped_sectors'])} sectors "
            f"with insufficient history: {sorted(results['skipped_sectors'])}"
        )


if __name__ == "__main__":
    main()
