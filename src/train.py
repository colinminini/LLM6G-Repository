"""Train LSTM and DeepAR baselines on the canonical split-aware experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import optim

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    TRAINABLE_BASELINE_MODELS,
    parse_quantiles,
)
from src.experiment import build_experiment_manifest, default_experiment_dir, load_manifest, save_manifest
from src.loader import DataLoaderConfig, build_dataloaders
from src.models import DeepARForecast, LSTMForecast
from src.train_utils import GaussianNLLLoss, QuantileLoss, Trainer, TrainerConfig


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
        description="Train LSTM and DeepAR baselines on a regular split-aware experiment manifest."
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
    parser.add_argument("--models", default="lstm,deepar")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--patience-iterations", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--device", default=None)
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


def _make_model(model_name: str, *, context_length: int, horizon: int, hidden_size: int, num_layers: int, quantiles: tuple[float, ...]):
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
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    loader_cfg = DataLoaderConfig(
        data_path=Path(manifest.dataset_path),
        manifest_path=manifest_path,
        timestamp_col=manifest.timestamp_col,
        context_length=manifest.context_length,
        forecast_length=manifest.horizon,
        batch_size=args.batch_size,
        shuffle_train=True,
        pin_memory=device.type == "cuda",
    )
    train_loader, val_loader, test_loader = build_dataloaders(loader_cfg)
    steps_per_epoch = max(1, len(train_loader))
    max_iterations = (
        args.max_iterations
        if args.max_iterations is not None
        else (args.max_epochs * steps_per_epoch if args.max_epochs is not None else 5000)
    )
    validate_every = (
        args.validate_every
        if args.validate_every is not None
        else min(250, max_iterations)
    )
    validate_every = max(1, min(validate_every, max_iterations))
    patience_iterations = (
        args.patience_iterations
        if args.patience_iterations is not None
        else (
            args.patience * validate_every
            if args.patience is not None
            else max(validate_every, 1000)
        )
    )
    log_every = args.log_every if args.log_every is not None else max(1, validate_every // 5)

    run_summary: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "device": str(device),
        "training_schedule": {
            "max_iterations": int(max_iterations),
            "patience_iterations": int(patience_iterations),
            "validate_every": int(validate_every),
            "log_every": int(log_every),
            "steps_per_epoch": int(steps_per_epoch),
        },
        "models": {},
    }
    for model_name in _parse_models(args.models):
        print(
            f"[train:{model_name}] starting "
            f"max_iterations={max_iterations} "
            f"validate_every={validate_every} "
            f"patience_iterations={patience_iterations}"
        )
        model = _make_model(
            model_name,
            context_length=manifest.context_length,
            horizon=manifest.horizon,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            quantiles=manifest.quantiles,
        )
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_fn = GaussianNLLLoss() if model_name == "deepar" else QuantileLoss(manifest.quantiles)
        trainer_cfg = TrainerConfig(
            max_iterations=max_iterations,
            patience_iterations=patience_iterations,
            validate_every=validate_every,
            log_every=log_every,
            grad_clip=args.grad_clip,
            save_dir=checkpoints_dir,
            log_dir=logs_dir,
            run_name=f"{model_name}_{manifest.dataset_name}",
            monitor_metric="loss",
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            config=trainer_cfg,
            model_config={
                "model_type": model_name,
                "context_length": manifest.context_length,
                "forecast_length": manifest.horizon,
                "quantiles": list(manifest.quantiles),
                "dataset_path": manifest.dataset_path,
            },
            quantiles=manifest.quantiles,
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
            "validate_every": int(validate_every),
        }

    summary_path = training_dir / "training_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"Saved training summary: {summary_path}")


if __name__ == "__main__":
    main()
