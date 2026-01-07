"""
Training entrypoint for LSTM and PatchTST models.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable

import torch
from torch import nn, optim

from scripts.loader import DataLoaderConfig, build_dataloaders
from scripts.models import LSTMForecast, PatchTSTForecast
from scripts.trainer import Trainer, TrainerConfig


def train_model(
    model_name: str,
    dataset_base: str = "data_instant",
    context_length: int = 128,
    forecast_length: int = 1,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 3,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    patch_len: int = 16,
    stride: int = 16,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
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

    model_name_lower = model_name.lower()
    if model_name_lower == "lstm":
        model = LSTMForecast(
            context_length=context_length,
            forecast_length=forecast_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        model_config = {
            "context_length": context_length,
            "forecast_length": forecast_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }
    elif model_name_lower == "patchtst":
        model = PatchTSTForecast(
            context_length=context_length,
            forecast_length=forecast_length,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        model_config = {
            "context_length": context_length,
            "forecast_length": forecast_length,
            "patch_len": patch_len,
            "stride": stride,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
        }
    else:
        raise ValueError("model_name must be 'lstm' or 'patchtst'.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    trainer_cfg = TrainerConfig(
        max_epochs=max_epochs,
        patience=patience,
        run_name=f"{model_name_lower}_{dataset_base}",
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=trainer_cfg,
        model_config=model_config,
    )
    return trainer.fit(train_loader, val_loader, test_loader)


def run_all(models: Iterable[str] = ("lstm", "patchtst"), **kwargs: object):
    histories: Dict[str, Dict[str, list[float]]] = {}
    for name in models:
        histories[name] = train_model(name, **kwargs) # pyright: ignore[reportArgumentType]
    return histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM/PatchTST models.")
    parser.add_argument(
        "--model",
        choices=("lstm", "patchtst", "all"),
        default="all",
        help="Which model to train.",
    )
    parser.add_argument("--dataset-base", default="data_instant")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
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
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )
    if args.model == "all":
        run_all(**common)
    else:
        train_model(args.model, **common)


if __name__ == "__main__":
    main()
