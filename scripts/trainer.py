"""
Training utilities for time-series models.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    patience: int = 3
    grad_clip: float | None = None
    save_dir: Path = Path("results/models")
    log_dir: Path = Path("results/logs")
    run_name: str = "run"


class Trainer:
    """Train a model with early stopping based on validation loss increases."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        config: TrainerConfig | None = None,
        model_config: Dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config or TrainerConfig()
        self.model_config = model_config or {}

        self.model.to(self.device)

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        epoch: int,
        phase: str,
        writer: SummaryWriter,
    ) -> tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_sse = 0.0
        total_count = 0
        num_batches = 0
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None

        for batch_idx, batch in enumerate(loader, start=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Expected batch to be a (inputs, targets) tuple.")

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            diff = outputs - targets
            total_sse += float(diff.pow(2).sum().item())
            total_count += diff.numel()

            if train:
                loss.backward()
                if self.config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            step = (
                (epoch - 1) * total_batches + (batch_idx - 1)
                if total_batches
                else batch_idx - 1
            )
            writer.add_scalar(f"loss/{phase}_batch", loss.item(), step)
            if batch_idx % 1000 == 0:
                print(
                    f"{phase.capitalize()} epoch {epoch} batch {batch_idx} "
                    f"loss={loss.item():.6f}"
                )

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches found in DataLoader.")
        avg_loss = total_loss / num_batches
        rmse = math.sqrt(total_sse / total_count) if total_count else 0.0
        return avg_loss, rmse

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_rmse": [],
            "val_loss": [],
            "val_rmse": [],
        }
        best_val = float("inf")
        num_increase = 0
        writer = SummaryWriter(log_dir=self.config.log_dir / self.config.run_name)
        best_path = self.config.save_dir / f"{self.config.run_name}_best.pt"

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                epoch_start = time.perf_counter()
                train_loss, train_rmse = self._run_epoch(
                    train_loader, train=True, epoch=epoch, phase="train", writer=writer
                )
                val_loss, val_rmse = self._run_epoch(
                    val_loader, train=False, epoch=epoch, phase="val", writer=writer
                )
                epoch_time = time.perf_counter() - epoch_start
                history["train_loss"].append(train_loss)
                history["train_rmse"].append(train_rmse)
                history["val_loss"].append(val_loss)
                history["val_rmse"].append(val_rmse)
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("rmse/train", train_rmse, epoch)
                writer.add_scalar("rmse/val", val_rmse, epoch)
                writer.add_scalar("metrics/epoch_time_sec", epoch_time, epoch)
                writer.add_scalar("metrics/best_val", best_val, epoch)
                writer.add_scalar("metrics/patience_count", num_increase, epoch)
                writer.add_scalar("metrics/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

                if val_loss < best_val:
                    best_val = val_loss
                    num_increase = 0
                    self._save_weights(best_path)
                else:
                    num_increase += 1
                    if num_increase >= self.config.patience:
                        print(
                            "Early stopping at epoch "
                            f"{epoch}: val_loss={val_loss:.6f}, best_val={best_val:.6f}."
                        )
                        break

                print(
                    "Epoch "
                    f"{epoch}/{self.config.max_epochs} "
                    f"- train_loss={train_loss:.6f} "
                    f"- train_rmse={train_rmse:.6f} "
                    f"- val_loss={val_loss:.6f} "
                    f"- val_rmse={val_rmse:.6f} "
                    f"- best_val={best_val:.6f} "
                    f"- patience={num_increase}/{self.config.patience}"
                )

            if test_loader is not None:
                test_loss, test_rmse = self.evaluate(
                    test_loader, epoch=len(history["train_loss"]), writer=writer
                )
                history["test_loss"] = [test_loss]
                history["test_rmse"] = [test_rmse]
                writer.add_scalar("loss/test", test_loss, len(history["train_loss"]))
                writer.add_scalar("rmse/test", test_rmse, len(history["train_loss"]))
        finally:
            writer.close()

        self._save_config(history, best_val, best_path)
        return history

    def evaluate(
        self, loader: DataLoader, epoch: int, writer: SummaryWriter
    ) -> tuple[float, float]:
        """Evaluate on a loader without updating weights."""
        return self._run_epoch(loader, train=False, epoch=epoch, phase="test", writer=writer)

    def _save_weights(self, out_path: Path) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path)

    def _save_config(
        self, history: Dict[str, List[float]], best_val: float, best_path: Path
    ) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "trainer": asdict(self.config),
            "model": self.model.__class__.__name__,
            "model_config": self.model_config,
            "best_val_loss": best_val,
            "best_checkpoint": str(best_path),
            "epochs_ran": len(history["train_loss"]),
            "history": history,
        }
        out_path = self.config.save_dir / f"{self.config.run_name}_config.json"
        with out_path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2)
