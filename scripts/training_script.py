"""
Training entrypoint for quantile forecasting baselines.
"""

from __future__ import annotations

import argparse
from typing import Dict, Sequence

import torch
from torch import optim

from scripts.loader import DataLoaderConfig, build_dataloaders
from scripts.models import DeepARForecast, LSTMForecast, TFTForecast
from scripts.trainer import GaussianNLLLoss, QuantileLoss, Trainer, TrainerConfig


def train_model(
    model_type: str = "lstm",
    dataset_base: str = "data_instant",
    context_length: int = 128,
    forecast_length: int = 1,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> Dict[str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = model_type.lower()
    if model_type not in {"lstm", "deepar", "tft"}:
        raise ValueError("model_type must be one of: lstm, deepar, tft.")
    data_cfg = DataLoaderConfig(
        dataset_base=dataset_base,
        context_length=context_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        shuffle_train=True,
        pin_memory=device.type == "cuda",
    )
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg)

    if model_type == "lstm":
        model = LSTMForecast(
            context_length=context_length,
            forecast_length=forecast_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            quantiles=quantiles,
        )
    elif model_type == "deepar":
        model = DeepARForecast(
            context_length=context_length,
            forecast_length=forecast_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
    else:
        model = TFTForecast(
            context_length=context_length,
            forecast_length=forecast_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            quantiles=quantiles,
        )
    model_config = {
        "model_type": model_type,
        "context_length": context_length,
        "forecast_length": forecast_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "quantiles": list(quantiles),
    }

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if model_type == "deepar":
        loss_fn = GaussianNLLLoss()
    else:
        loss_fn = QuantileLoss(quantiles)

    trainer_cfg = TrainerConfig(
        max_epochs=max_epochs,
        patience=patience,
        run_name=f"{model_type}_quantile_{dataset_base}",
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=trainer_cfg,
        model_config=model_config,
        quantiles=quantiles,
    )
    return trainer.fit(train_loader, val_loader, test_loader)


def _parse_quantiles(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("quantiles must be a comma-separated list like 0.1,0.5,0.9")
    quantiles = [float(part) for part in parts]
    for q in quantiles:
        if q <= 0.0 or q >= 1.0:
            raise ValueError("quantiles must be between 0 and 1 (exclusive).")
    return quantiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a quantile forecasting model (LSTM, DeepAR, or TFT)."
    )
    parser.add_argument(
        "--model-type",
        default="lstm",
        help="Model family to train: lstm or deepar.",
    )
    parser.add_argument("--dataset-base", default="data_instant")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads for TFT (must divide hidden-size).",
    )
    parser.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated list of quantiles to predict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = dict(
        model_type=args.model_type,
        dataset_base=args.dataset_base,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        quantiles=_parse_quantiles(args.quantiles),
    )
    train_model(**common)


if __name__ == "__main__":
    main()
