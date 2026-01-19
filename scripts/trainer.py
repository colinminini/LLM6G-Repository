"""
Training utilities for quantile time-series models.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


def _format_quantile_key(q: float) -> str:
    return f"q{q:g}"


def _select_median_index(quantiles: Sequence[float]) -> int:
    if not quantiles:
        raise ValueError("quantiles must contain at least one entry.")
    quantiles = list(quantiles)
    try:
        return quantiles.index(0.5)
    except ValueError:
        return min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
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


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        if not quantiles:
            raise ValueError("quantiles must contain at least one entry.")
        quantiles = [float(q) for q in quantiles]
        for q in quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError("quantiles must be between 0 and 1 (exclusive).")
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 3:
            raise ValueError(
                "Expected predictions with shape (batch, forecast_len, num_quantiles)."
            )
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        if target.dim() != 3:
            raise ValueError(
                "Expected targets with shape (batch, forecast_len) or (batch, forecast_len, 1)."
            )
        if target.size(0) != preds.size(0) or target.size(1) != preds.size(1):
            raise ValueError("Target shape does not match predictions.")

        q = self.quantiles.view(1, 1, -1) # type: ignore
        if q.size(-1) != preds.size(-1):
            raise ValueError("Quantile count does not match predictions.")

        errors = target - preds
        loss = torch.maximum(q * errors, (q - 1) * errors)
        return loss.mean()


class GaussianNLLLoss(nn.Module):
    """Negative log-likelihood for Gaussian outputs."""

    output_type = "gaussian"

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 3 or preds.size(-1) != 2:
            raise ValueError(
                "Expected predictions with shape (batch, forecast_len, 2) for Gaussian parameters."
            )
        mu = preds[..., 0]
        sigma = torch.clamp(preds[..., 1], min=self.eps)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.dim() == 2:
            target = target
        if target.dim() != 2:
            raise ValueError(
                "Expected targets with shape (batch, forecast_len)."
            )
        if target.size(0) != mu.size(0) or target.size(1) != mu.size(1):
            raise ValueError("Target shape does not match predictions.")
        loss = 0.5 * ((target - mu) / sigma).pow(2) + torch.log(sigma) + 0.5 * math.log(
            2 * math.pi
        )
        return loss.mean()


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
        quantiles: Sequence[float] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config or TrainerConfig()
        self.model_config = model_config or {}
        self.quantiles = tuple(float(q) for q in quantiles) if quantiles else ()
        self.median_index = (
            _select_median_index(self.quantiles) if self.quantiles else None
        )
        if self.quantiles and len(self.quantiles) >= 2:
            self.interval_indices = (
                min(range(len(self.quantiles)), key=self.quantiles.__getitem__),
                max(range(len(self.quantiles)), key=self.quantiles.__getitem__),
            )
        else:
            self.interval_indices = None

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def _format_quantile_metrics(
        self, phase: str, metrics: Sequence[float]
    ) -> str:
        if not metrics or not self.quantiles:
            return ""
        parts = [
            f"{phase}_{_format_quantile_key(q)}={value:.3f}"
            for q, value in zip(self.quantiles, metrics)
        ]
        return " ".join(parts)

    def _format_interval_metrics(
        self,
        phase: str,
        coverage: float | None,
        interval_width: float | None,
    ) -> str:
        parts: list[str] = []
        if coverage is not None:
            parts.append(f"{phase}_coverage={coverage:.3f}")
        if interval_width is not None:
            parts.append(f"{phase}_interval_width={interval_width:.3f}")
        return " ".join(parts)

    def _log_quantile_metrics(
        self, writer: SummaryWriter, phase: str, metrics: Sequence[float], epoch: int
    ) -> None:
        if not metrics or not self.quantiles:
            return
        for q, value in zip(self.quantiles, metrics):
            writer.add_scalar(
                f"quantile_loss/{phase}/{_format_quantile_key(q)}", value, epoch
            )

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        epoch: int,
        phase: str,
        writer: SummaryWriter,
    ) -> tuple[float, float, list[float], float | None, float | None]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_sse = 0.0
        total_count = 0
        quantile_loss_sum: list[float] | None = None
        quantile_count = 0
        coverage_hits = 0.0
        coverage_count = 0
        interval_sum = 0.0
        interval_count = 0
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

            if getattr(self.model, "uses_targets", False):
                outputs = self.model(inputs, targets=targets)
            else:
                outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            output_type = getattr(self.loss_fn, "output_type", None)
            preds_for_rmse = outputs
            quantile_preds: torch.Tensor | None = None
            if output_type == "gaussian":
                mu = outputs[..., 0]
                sigma = torch.clamp(outputs[..., 1], min=1e-6)
                preds_for_rmse = mu
                if self.quantiles:
                    q = torch.tensor(
                        self.quantiles, device=outputs.device, dtype=mu.dtype
                    ).view(1, 1, -1)
                    dist = torch.distributions.Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1))
                    quantile_preds = dist.icdf(q)
            elif outputs.dim() == 3:
                if self.quantiles and outputs.size(-1) != len(self.quantiles):
                    raise ValueError("Quantile count does not match model outputs.")
                # Track RMSE using the median quantile when available.
                if self.median_index is None:
                    median_idx = outputs.size(-1) // 2
                else:
                    median_idx = self.median_index
                preds_for_rmse = outputs[..., median_idx]
                if self.quantiles:
                    quantile_preds = outputs

            if quantile_preds is not None and self.quantiles:
                targets_3d = targets.unsqueeze(-1) if targets.dim() == 2 else targets
                errors = targets_3d - quantile_preds
                q = torch.tensor(self.quantiles, device=outputs.device).view(1, 1, -1)
                loss_q = torch.maximum(q * errors, (q - 1) * errors)
                batch_sum = loss_q.detach().sum(dim=(0, 1)).cpu().tolist()
                if quantile_loss_sum is None:
                    quantile_loss_sum = [0.0 for _ in batch_sum]
                for idx, value in enumerate(batch_sum):
                    quantile_loss_sum[idx] += float(value)
                quantile_count += quantile_preds.size(0) * quantile_preds.size(1)
                if self.interval_indices is not None:
                    targets_2d = targets.squeeze(-1) if targets.dim() == 3 else targets
                    lower_idx, upper_idx = self.interval_indices
                    lower_pred = quantile_preds[..., lower_idx]
                    upper_pred = quantile_preds[..., upper_idx]
                    within = (targets_2d >= lower_pred) & (targets_2d <= upper_pred)
                    coverage_hits += float(within.sum().item())
                    coverage_count += targets_2d.numel()
                    width = torch.clamp(upper_pred - lower_pred, min=0.0)
                    interval_sum += float(width.sum().item())
                    interval_count += width.numel()
            diff = preds_for_rmse - targets
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
            if batch_idx % 5000 == 0:
                print(
                    f"{phase.capitalize()} epoch {epoch} batch {batch_idx} "
                    f"loss={loss.item():.3f}"
                )

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No batches found in DataLoader.")
        avg_loss = total_loss / num_batches
        rmse = math.sqrt(total_sse / total_count) if total_count else 0.0
        if quantile_loss_sum is None or quantile_count == 0:
            quantile_metrics: list[float] = []
        else:
            quantile_metrics = [value / quantile_count for value in quantile_loss_sum]
        coverage = coverage_hits / coverage_count if coverage_count else None
        interval_width = interval_sum / interval_count if interval_count else None
        return avg_loss, rmse, quantile_metrics, coverage, interval_width

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
            "train_coverage": [],
            "val_coverage": [],
            "train_interval_width": [],
            "val_interval_width": [],
        }
        if test_loader is not None:
            history["test_loss"] = []
            history["test_rmse"] = []
            history["test_coverage"] = []
            history["test_interval_width"] = []
        best_val = float("inf")
        num_increase = 0
        writer = SummaryWriter(log_dir=self.config.log_dir / self.config.run_name)
        best_path = self.config.save_dir / f"{self.config.run_name}_best.pt"
        last_summary: Dict[str, Any] | None = None

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                epoch_start = time.perf_counter()
                (
                    train_loss,
                    train_rmse,
                    train_quantiles,
                    train_coverage,
                    train_width,
                ) = self._run_epoch(
                    train_loader, train=True, epoch=epoch, phase="train", writer=writer
                )
                (
                    val_loss,
                    val_rmse,
                    val_quantiles,
                    val_coverage,
                    val_width,
                ) = self._run_epoch(
                    val_loader, train=False, epoch=epoch, phase="val", writer=writer
                )
                history["train_loss"].append(train_loss)
                history["train_rmse"].append(train_rmse)
                history["val_loss"].append(val_loss)
                history["val_rmse"].append(val_rmse)
                history["train_coverage"].append(
                    train_coverage if train_coverage is not None else float("nan")
                )
                history["val_coverage"].append(
                    val_coverage if val_coverage is not None else float("nan")
                )
                history["train_interval_width"].append(
                    train_width if train_width is not None else float("nan")
                )
                history["val_interval_width"].append(
                    val_width if val_width is not None else float("nan")
                )
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("rmse/train", train_rmse, epoch)
                writer.add_scalar("rmse/val", val_rmse, epoch)
                self._log_quantile_metrics(writer, "train", train_quantiles, epoch)
                self._log_quantile_metrics(writer, "val", val_quantiles, epoch)
                if train_coverage is not None:
                    writer.add_scalar("coverage/train", train_coverage, epoch)
                if val_coverage is not None:
                    writer.add_scalar("coverage/val", val_coverage, epoch)
                if train_width is not None:
                    writer.add_scalar("interval_width/train", train_width, epoch)
                if val_width is not None:
                    writer.add_scalar("interval_width/val", val_width, epoch)
                writer.add_scalar("metrics/best_val", best_val, epoch)
                writer.add_scalar("metrics/patience_count", num_increase, epoch)
                writer.add_scalar("metrics/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

                if test_loader is not None:
                    (
                        test_loss,
                        test_rmse,
                        test_quantiles,
                        test_coverage,
                        test_width,
                    ) = self.evaluate(
                        test_loader, epoch=epoch, writer=writer
                    )
                    history["test_loss"].append(test_loss)
                    history["test_rmse"].append(test_rmse)
                    history["test_coverage"].append(
                        test_coverage if test_coverage is not None else float("nan")
                    )
                    history["test_interval_width"].append(
                        test_width if test_width is not None else float("nan")
                    )
                    writer.add_scalar("loss/test", test_loss, epoch)
                    writer.add_scalar("rmse/test", test_rmse, epoch)
                    self._log_quantile_metrics(writer, "test", test_quantiles, epoch)
                    if test_coverage is not None:
                        writer.add_scalar("coverage/test", test_coverage, epoch)
                    if test_width is not None:
                        writer.add_scalar("interval_width/test", test_width, epoch)

                epoch_time = time.perf_counter() - epoch_start
                writer.add_scalar("metrics/epoch_time_sec", epoch_time, epoch)

                if val_loss < best_val:
                    best_val = val_loss
                    num_increase = 0
                    self._save_weights(best_path)
                else:
                    num_increase += 1

                last_summary = {
                    "train_loss": train_loss,
                    "train_rmse": train_rmse,
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                    "train_quantiles": train_quantiles,
                    "val_quantiles": val_quantiles,
                    "train_coverage": train_coverage,
                    "val_coverage": val_coverage,
                    "train_width": train_width,
                    "val_width": val_width,
                    "best_val": best_val,
                    "patience": num_increase,
                }
                if test_loader is not None:
                    last_summary.update(
                        {
                            "test_loss": test_loss, # pyright: ignore[reportPossiblyUnboundVariable]
                            "test_rmse": test_rmse, # type: ignore
                            "test_quantiles": test_quantiles, # pyright: ignore[reportPossiblyUnboundVariable]
                            "test_coverage": test_coverage, # pyright: ignore[reportPossiblyUnboundVariable]
                            "test_width": test_width, # type: ignore
                        }
                    )

                if num_increase >= self.config.patience:
                    print(
                        "Early stopping at epoch "
                        f"{epoch}: train_rmse={train_rmse:.3f}, val_rmse={val_rmse:.3f}, test_rmse={test_rmse:.3f}" # pyright: ignore[reportPossiblyUnboundVariable]
                    )
                    break

                message = (
                    "Epoch "
                    f"{epoch}/{self.config.max_epochs} "
                    f"- train_loss={train_loss:.3f} "
                    f"- train_rmse={train_rmse:.3f} "
                    f"- val_loss={val_loss:.3f} "
                    f"- val_rmse={val_rmse:.3f} "
                )
                if test_loader is not None:
                    message += (
                        f"- test_loss={test_loss:.3f} " # pyright: ignore[reportPossiblyUnboundVariable]
                        f"- test_rmse={test_rmse:.3f} " # pyright: ignore[reportPossiblyUnboundVariable]
                    )
                print(message)
        finally:
            writer.close()

        if last_summary is not None:
            metrics: Dict[str, Any] = {
                "train_loss": last_summary["train_loss"],
                "train_rmse": last_summary["train_rmse"],
                "val_loss": last_summary["val_loss"],
                "val_rmse": last_summary["val_rmse"],
            }
            for q, value in zip(self.quantiles, last_summary["train_quantiles"]):
                metrics[f"train_{_format_quantile_key(q)}"] = value
            for q, value in zip(self.quantiles, last_summary["val_quantiles"]):
                metrics[f"val_{_format_quantile_key(q)}"] = value
            if last_summary.get("train_coverage") is not None:
                metrics["train_coverage"] = last_summary["train_coverage"]
            if last_summary.get("val_coverage") is not None:
                metrics["val_coverage"] = last_summary["val_coverage"]
            if last_summary.get("train_width") is not None:
                metrics["train_interval_width"] = last_summary["train_width"]
            if last_summary.get("val_width") is not None:
                metrics["val_interval_width"] = last_summary["val_width"]
            if test_loader is not None:
                metrics["test_loss"] = last_summary.get("test_loss")
                metrics["test_rmse"] = last_summary.get("test_rmse")
                for q, value in zip(self.quantiles, last_summary.get("test_quantiles", [])):
                    metrics[f"test_{_format_quantile_key(q)}"] = value
                if last_summary.get("test_coverage") is not None:
                    metrics["test_coverage"] = last_summary.get("test_coverage")
                if last_summary.get("test_width") is not None:
                    metrics["test_interval_width"] = last_summary.get("test_width")
            metrics["best_val"] = last_summary["best_val"]
            metrics["patience"] = f"{last_summary['patience']}/{self.config.patience}"
            _display_metrics_table("Final metrics", metrics)

        return history

    def evaluate(
        self, loader: DataLoader, epoch: int, writer: SummaryWriter
    ) -> tuple[float, float, list[float], float | None, float | None]:
        """Evaluate on a loader without updating weights."""
        return self._run_epoch(loader, train=False, epoch=epoch, phase="test", writer=writer)

    def _save_weights(self, out_path: Path) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path)
