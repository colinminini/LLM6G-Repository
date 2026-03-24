"""Train benchmark baselines on the canonical split-aware experiment manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import optim

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM,
    DEFAULT_DEEPAR_LIKELIHOOD,
    DEFAULT_DEEPAR_NUM_SAMPLES,
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    SUPPORTED_DEEPAR_LIKELIHOODS,
    TRAINABLE_BASELINE_MODELS,
    parse_quantiles,
)
from src.deepar_support import build_feature_spec, resolve_deepar_model_config
from src.device import AUTO_DEVICE, AUTO_DEVICE_HELP, resolve_torch_device
from src.experiment import build_experiment_manifest, default_experiment_dir, load_manifest, load_wide_dataframe, save_manifest, valid_window_start_indices
from src.loader import DataLoaderConfig, build_dataloaders
from src.models import DeepARForecast, LSTMForecast, TFTForecast
from src.tft_support import resolve_tft_model_config
from src.train_utils import (
    CanonicalEvalConfig,
    DeepARGaussianNLLLoss,
    GaussianNLLLoss,
    NegativeBinomialNLLLoss,
    QuantileLoss,
    Trainer,
    TrainerConfig,
)


def _parse_models(raw: str) -> list[str]:
    models = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name.")
    invalid = [name for name in models if name not in TRAINABLE_BASELINE_MODELS]
    if invalid:
        raise ValueError(
            f"Unsupported training model(s): {', '.join(invalid)}. "
            f"Supported models: {', '.join(TRAINABLE_BASELINE_MODELS)}."
        )
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train benchmark baselines on a regular split-aware experiment manifest."
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
    parser.add_argument("--models", default="lstm,deepar,tft")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-window-step", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--patience-iterations", type=int, default=1000)
    parser.add_argument("--validate-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tft-num-heads", type=int, default=4)
    parser.add_argument("--deepar-likelihood", default=DEFAULT_DEEPAR_LIKELIHOOD, choices=SUPPORTED_DEEPAR_LIKELIHOODS)
    parser.add_argument("--deepar-num-samples", type=int, default=DEFAULT_DEEPAR_NUM_SAMPLES)
    parser.add_argument("--deepar-item-embedding-dim", type=int, default=DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--device", default=AUTO_DEVICE, help=AUTO_DEVICE_HELP)
    return parser.parse_args()


def _build_or_load_manifest(args: argparse.Namespace):
    if args.manifest_path is not None:
        return load_manifest(args.manifest_path)
    return build_experiment_manifest(
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


def _serialize_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.items()}


def _make_model(
    model_name: str,
    *,
    context_length: int,
    horizon: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    quantiles: tuple[float, ...],
    num_time_features: int | None = None,
    num_items: int | None = None,
    deepar_item_embedding_dim: int = DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM,
    deepar_likelihood: str = DEFAULT_DEEPAR_LIKELIHOOD,
):
    if model_name == "lstm":
        return LSTMForecast(
            context_length=context_length,
            forecast_length=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            quantiles=quantiles,
        )
    if model_name == "deepar":
        return DeepARForecast(
            context_length=context_length,
            forecast_length=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_time_features=int(num_time_features or 1),
            num_items=max(1, int(num_items or 1)),
            item_embedding_dim=int(deepar_item_embedding_dim),
            likelihood=deepar_likelihood,
        )
    if model_name == "tft":
        return TFTForecast(
            context_length=context_length,
            forecast_length=horizon,
            hidden_size=hidden_size,
            num_lstm_layers=num_layers,
            num_heads=num_heads,
            quantiles=quantiles,
            num_static_categorical=1,
            static_categorical_cardinalities=(max(1, int(num_items or 1)),),
            num_past_features=1 + int(num_time_features or 0),
            num_future_features=int(num_time_features or 0),
        )
    raise ValueError(f"Unsupported model_name={model_name}")


def main() -> None:
    args = parse_args()
    manifest = _build_or_load_manifest(args)
    output_dir = args.output_dir or default_experiment_dir(manifest)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    training_dir = output_dir / "training"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = save_manifest(manifest, output_dir / "manifest.json")
    device = resolve_torch_device(args.device)

    full_df = load_wide_dataframe(manifest.dataset_path, manifest.timestamp_col)
    deepar_feature_spec = build_feature_spec(
        full_df[manifest.timestamp_col],
        cadence=manifest.cadence,
        train_end_idx=manifest.train_split.end_idx,
    )

    run_summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "device": str(device),
        "models": {},
    }
    for model_name in _parse_models(args.models):
        loader_cfg = DataLoaderConfig(
            data_path=Path(manifest.dataset_path),
            manifest_path=manifest_path,
            timestamp_col=manifest.timestamp_col,
            model_name=model_name,
            context_length=manifest.context_length,
            forecast_length=manifest.horizon,
            train_window_step=max(1, int(args.train_window_step)),
            batch_size=args.batch_size,
            shuffle_train=True,
            pin_memory=device.type == "cuda",
            deepar_feature_spec=deepar_feature_spec if model_name == "deepar" else None,
            tft_feature_spec=deepar_feature_spec if model_name == "tft" else None,
        )
        train_loader, val_loader, test_loader = build_dataloaders(loader_cfg)
        steps_per_epoch = max(1, len(train_loader))
        dense_train_windows = int(len(valid_window_start_indices(manifest, "train")) * manifest.num_series)
        stepped_train_windows = int(len(train_loader.dataset))
        schedule_ratio = (
            float(stepped_train_windows / dense_train_windows)
            if dense_train_windows > 0
            else 1.0
        )
        requested_max_iterations = int(args.max_iterations) if args.max_iterations is not None else 5000
        max_iterations = requested_max_iterations
        if args.max_epochs is not None:
            max_iterations = max(1, int(args.max_epochs) * steps_per_epoch)
        elif int(args.train_window_step) > 1:
            max_iterations = max(1, math.ceil(requested_max_iterations * schedule_ratio))
        validate_every = max(1, int(args.validate_epochs)) * steps_per_epoch
        validate_every = max(1, min(validate_every, max_iterations))
        requested_patience_iterations = int(args.patience_iterations) if args.patience_iterations is not None else 1000
        patience_iterations = requested_patience_iterations
        if args.patience is not None:
            patience_iterations = max(1, int(args.patience) * validate_every)
        elif int(args.train_window_step) > 1:
            patience_iterations = max(1, math.ceil(requested_patience_iterations * schedule_ratio))
        log_every = args.log_every if args.log_every is not None else max(1, validate_every // 5)

        actual_validate_epochs = validate_every / steps_per_epoch
        print(
            f"[train:{model_name}] starting "
            f"max_iterations={max_iterations} "
            f"validate_every={validate_every} (every {actual_validate_epochs:.2f} epoch(s)) "
            f"patience_iterations={patience_iterations}"
        )
        model = _make_model(
            model_name,
            context_length=manifest.context_length,
            horizon=manifest.horizon,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.tft_num_heads,
            quantiles=manifest.quantiles,
            num_time_features=(
                deepar_feature_spec.num_features if model_name in {"deepar", "tft"} else None
            ),
            num_items=manifest.num_series if model_name in {"deepar", "tft"} else None,
            deepar_item_embedding_dim=args.deepar_item_embedding_dim,
            deepar_likelihood=args.deepar_likelihood,
        )
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        if model_name == "deepar":
            loss_fn = (
                DeepARGaussianNLLLoss()
                if args.deepar_likelihood == "gaussian"
                else NegativeBinomialNLLLoss()
            )
        else:
            loss_fn = QuantileLoss(manifest.quantiles)

        model_config = {
            "model_type": model_name,
            "context_length": manifest.context_length,
            "forecast_length": manifest.horizon,
            "quantiles": list(manifest.quantiles),
            "dataset_path": manifest.dataset_path,
            "timestamp_col": manifest.timestamp_col,
            "cadence": manifest.cadence,
            "series_columns": list(manifest.series_columns),
        }
        if model_name == "deepar":
            model_config = resolve_deepar_model_config(
                manifest=manifest,
                likelihood=args.deepar_likelihood,
                num_samples=args.deepar_num_samples,
                item_embedding_dim=args.deepar_item_embedding_dim,
                random_seed=args.random_seed,
                feature_spec=deepar_feature_spec,
            )
        elif model_name == "tft":
            model_config = resolve_tft_model_config(
                manifest=manifest,
                hidden_size=args.hidden_size,
                num_heads=args.tft_num_heads,
                num_lstm_layers=args.num_layers,
                random_seed=args.random_seed,
                feature_spec=deepar_feature_spec,
            )
        trainer_cfg = TrainerConfig(
            max_iterations=max_iterations,
            patience_iterations=patience_iterations,
            validate_every=validate_every,
            log_every=log_every,
            grad_clip=args.grad_clip,
            save_dir=checkpoints_dir,
            log_dir=logs_dir,
            run_name=f"{model_name}_{manifest.dataset_name}",
            monitor_metric="pinball",
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            config=trainer_cfg,
            model_config=model_config,
            quantiles=manifest.quantiles,
            canonical_eval_config=CanonicalEvalConfig(
                manifest=manifest,
                full_df=full_df,
            ),
        )
        history = trainer.fit(train_loader, val_loader, test_loader)
        history_path = training_dir / f"{model_name}_history.json"
        history_path.write_text(json.dumps(_serialize_history(history), indent=2))
        checkpoint_path = checkpoints_dir / f"{model_name}_{manifest.dataset_name}_best.pt"
        run_summary["models"][model_name] = {
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
            "max_iterations": int(max_iterations),
            "patience_iterations": int(patience_iterations),
            "validate_epochs": int(args.validate_epochs),
            "validate_every": int(validate_every),
            "train_window_step": int(args.train_window_step),
            "training_schedule": {
                "steps_per_epoch": int(steps_per_epoch),
                "dense_train_windows": int(dense_train_windows),
                "train_windows": int(stepped_train_windows),
                "schedule_ratio": float(schedule_ratio),
                "log_every": int(log_every),
            },
            "deepar_likelihood": (
                args.deepar_likelihood if model_name == "deepar" else None
            ),
            "deepar_num_samples": (
                int(args.deepar_num_samples) if model_name == "deepar" else None
            ),
            "tft_num_heads": int(args.tft_num_heads) if model_name == "tft" else None,
            "monitor_metrics": trainer.final_monitor_metrics,
            "canonical_metrics": trainer.final_canonical_metrics,
        }

    summary_path = training_dir / "training_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()
