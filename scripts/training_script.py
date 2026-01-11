"""
Training entrypoint for LSTM quantile forecasting.
"""

from __future__ import annotations

import argparse
from typing import Dict, Sequence

import torch
from torch import optim

from scripts.loader import DataLoaderConfig, build_dataloaders
from scripts.models import LSTMForecast
from scripts.trainer import QuantileLoss, Trainer, TrainerConfig


def train_model(
    dataset_base: str = "data_instant",
    context_length: int = 128,
    forecast_length: int = 1,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> Dict[str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = DataLoaderConfig(
        dataset_base=dataset_base,
        context_length=context_length,
        forecast_length=forecast_length,
        batch_size=batch_size,
        shuffle_train=True,
        pin_memory=device.type == "cuda",
    )
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg)

    model = LSTMForecast(
        context_length=context_length,
        forecast_length=forecast_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        quantiles=quantiles,
    )
    model_config = {
        "context_length": context_length,
        "forecast_length": forecast_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "quantiles": list(quantiles),
    }

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = QuantileLoss(quantiles)

    trainer_cfg = TrainerConfig(
        max_epochs=max_epochs,
        patience=patience,
        run_name=f"lstm_quantile_{dataset_base}",
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
    parser = argparse.ArgumentParser(description="Train an LSTM quantile model.")
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
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated list of quantiles to predict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = dict(
        dataset_base=args.dataset_base,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        quantiles=_parse_quantiles(args.quantiles),
    )
    train_model(**common)


if __name__ == "__main__":
    main()
