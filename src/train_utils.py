"""
Training utilities for quantile time-series models and baseline training helpers.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_DATASET_PATH, DEFAULT_QUANTILES, DEFAULT_TIMESTAMP_COL, TRAINABLE_BASELINE_MODELS, normalize_quantiles, parse_quantiles
from src.experiment import build_experiment_manifest, default_experiment_dir, load_manifest, save_manifest
from src.loader import DataLoaderConfig, build_dataloaders
from src.models import DeepARForecast, LSTMForecast


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    patience_epochs: int = 10
    log_every: int = 50
    grad_clip: float | None = None
    show_batch_logs: bool = False
    save_dir: Path = Path("results/models")
    log_dir: Path = Path("results/logs")
    run_name: str = "run"
    monitor_metric: str = "loss"


_SUPPORTED_MODEL_TYPES = TRAINABLE_BASELINE_MODELS


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


def _select_upper_coverage_index(
    quantiles: Sequence[float], target_quantile: float = 0.95
) -> int:
    """Pick the quantile index used for one-sided upper coverage."""
    if not quantiles:
        raise ValueError("quantiles must contain at least one entry.")
    values = list(float(q) for q in quantiles)
    return min(
        range(len(values)),
        key=lambda idx: (abs(values[idx] - target_quantile), -values[idx]),
    )


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


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _TrainingProgressDisplay:
    def __init__(self, run_name: str, total_epochs: int, total_train_batches: int) -> None:
        self.run_name = run_name
        self.total_epochs = max(1, int(total_epochs))
        self.total_train_batches = max(1, int(total_train_batches))
        self.started_at = time.perf_counter()
        self.completed_progress = 0.0
        self._bar = (
            tqdm(
                total=self.total_epochs,
                desc=f"train:{run_name}",
                unit="epoch",
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.15,
                leave=True,
                bar_format=(
                    "{l_bar}{bar}| {n:.2f}/{total:.0f} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
            )
            if tqdm is not None
            else None
        )
        self.update(0, phase="warmup")

    def _timings(self, completed_progress: float) -> tuple[float, float | None, float | None]:
        elapsed = time.perf_counter() - self.started_at
        if completed_progress <= 0:
            return elapsed, None, None
        estimated_total = elapsed * self.total_epochs / completed_progress
        remaining = max(0.0, estimated_total - elapsed)
        return elapsed, estimated_total, remaining

    def update(
        self,
        progress_epoch: float,
        *,
        phase: str,
        epoch: int | None = None,
        batch_idx: int | None = None,
        batch_total: int | None = None,
        batch_loss: float | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        test_loss: float | None = None,
        patience: str | None = None,
    ) -> None:
        target_progress = max(0.0, min(float(progress_epoch), float(self.total_epochs)))
        delta = target_progress - self.completed_progress
        if self._bar is not None and delta > 0:
            self._bar.update(delta)
        self.completed_progress = target_progress

        elapsed, estimated_total, remaining = self._timings(self.completed_progress)
        postfix: dict[str, str] = {
            "phase": phase,
            "elapsed": _format_duration(elapsed),
            "eta": _format_duration(remaining),
            "est_total": _format_duration(estimated_total),
        }
        if epoch is not None:
            postfix["epoch"] = f"{min(int(epoch), self.total_epochs)}/{self.total_epochs}"
        if batch_idx is not None and batch_total is not None:
            postfix["batch"] = f"{batch_idx}/{batch_total}"
        if batch_loss is not None:
            postfix["batch_loss"] = f"{batch_loss:.3f}"
        if train_loss is not None:
            postfix["train_loss"] = f"{train_loss:.3f}"
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.3f}"
        if test_loss is not None:
            postfix["test_loss"] = f"{test_loss:.3f}"
        if patience is not None:
            postfix["patience"] = patience

        if self._bar is not None:
            self._bar.set_postfix(postfix, refresh=False)
        elif target_progress > 0:
            print(
                f"[{self.run_name}] progress "
                f"{target_progress:.2f}/{self.total_epochs} "
                f"phase={phase} elapsed={postfix['elapsed']} "
                f"eta={postfix['eta']} est_total={postfix['est_total']}"
            )

    def write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message)
        else:
            print(message)

    def close(self) -> None:
        total_elapsed = time.perf_counter() - self.started_at
        completed_epochs = min(self.total_epochs, int(self.completed_progress + 1e-9))
        self.update(
            self.completed_progress,
            phase="done" if completed_epochs >= self.total_epochs else "stopped",
            epoch=max(1, completed_epochs) if completed_epochs > 0 else 1,
            batch_idx=self.total_train_batches,
            batch_total=self.total_train_batches,
        )
        self.write(
            f"[{self.run_name}] wall_time={_format_duration(total_elapsed)} "
            f"completed_epochs={completed_epochs}/{self.total_epochs}"
        )
        if self._bar is not None:
            self._bar.close()


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

        q = self.quantiles.view(1, 1, -1)  # type: ignore
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
    """Train a model with epoch-based early stopping."""

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
        self.coverage_index = (
            _select_upper_coverage_index(self.quantiles) if self.quantiles else None
        )
        self.coverage_quantile = (
            self.quantiles[self.coverage_index]
            if self.quantiles and self.coverage_index is not None
            else None
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

    def _log_quantile_metrics(
        self, writer: SummaryWriter, phase: str, metrics: Sequence[float], step: int
    ) -> None:
        if not metrics or not self.quantiles:
            return
        for q, value in zip(self.quantiles, metrics):
            writer.add_scalar(
                f"quantile_loss/{phase}/{_format_quantile_key(q)}", value, step
            )

    def _make_metrics_accumulator(self) -> dict[str, Any]:
        return {
            "loss_sum": 0.0,
            "num_batches": 0,
            "sse": 0.0,
            "count": 0,
            "quantile_loss_sum": None,
            "quantile_count": 0,
            "coverage_hits": 0.0,
            "coverage_count": 0,
            "interval_sum": 0.0,
            "interval_count": 0,
        }

    def _forward_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
            if self.median_index is None:
                median_idx = outputs.size(-1) // 2
            else:
                median_idx = self.median_index
            preds_for_rmse = outputs[..., median_idx]
            if self.quantiles:
                quantile_preds = outputs
        return loss, preds_for_rmse, quantile_preds

    def _update_metrics_accumulator(
        self,
        acc: dict[str, Any],
        *,
        loss: torch.Tensor,
        preds_for_rmse: torch.Tensor,
        targets: torch.Tensor,
        quantile_preds: torch.Tensor | None,
    ) -> None:
        acc["loss_sum"] += float(loss.item())
        acc["num_batches"] += 1
        diff = preds_for_rmse - targets
        acc["sse"] += float(diff.pow(2).sum().item())
        acc["count"] += diff.numel()

        if quantile_preds is not None and self.quantiles:
            targets_3d = targets.unsqueeze(-1) if targets.dim() == 2 else targets
            errors = targets_3d - quantile_preds
            q = torch.tensor(
                self.quantiles,
                device=quantile_preds.device,
                dtype=quantile_preds.dtype,
            ).view(1, 1, -1)
            loss_q = torch.maximum(q * errors, (q - 1) * errors)
            batch_sum = loss_q.detach().sum(dim=(0, 1)).cpu().tolist()
            if acc["quantile_loss_sum"] is None:
                acc["quantile_loss_sum"] = [0.0 for _ in batch_sum]
            for idx, value in enumerate(batch_sum):
                acc["quantile_loss_sum"][idx] += float(value)
            acc["quantile_count"] += quantile_preds.size(0) * quantile_preds.size(1)
            if self.coverage_index is not None:
                targets_2d = targets.squeeze(-1) if targets.dim() == 3 else targets
                upper_pred = quantile_preds[..., self.coverage_index]
                within = targets_2d <= upper_pred
                acc["coverage_hits"] += float(within.sum().item())
                acc["coverage_count"] += targets_2d.numel()
            if self.interval_indices is not None:
                lower_idx, upper_idx = self.interval_indices
                lower_pred = quantile_preds[..., lower_idx]
                upper_pred = quantile_preds[..., upper_idx]
                width = torch.clamp(upper_pred - lower_pred, min=0.0)
                acc["interval_sum"] += float(width.sum().item())
                acc["interval_count"] += width.numel()

    def _finalize_metrics_accumulator(
        self,
        acc: dict[str, Any],
    ) -> tuple[float, float, list[float], float | None, float | None]:
        if acc["num_batches"] == 0:
            raise RuntimeError("No batches were processed.")
        avg_loss = acc["loss_sum"] / acc["num_batches"]
        rmse = math.sqrt(acc["sse"] / acc["count"]) if acc["count"] else 0.0
        if acc["quantile_loss_sum"] is None or acc["quantile_count"] == 0:
            quantile_metrics: list[float] = []
        else:
            quantile_metrics = [
                float(value) / acc["quantile_count"] for value in acc["quantile_loss_sum"]
            ]
        coverage = (
            acc["coverage_hits"] / acc["coverage_count"]
            if acc["coverage_count"]
            else None
        )
        interval_width = (
            acc["interval_sum"] / acc["interval_count"]
            if acc["interval_count"]
            else None
        )
        return avg_loss, rmse, quantile_metrics, coverage, interval_width

    def _run_loader(
        self,
        loader: DataLoader,
        *,
        phase: str,
        writer: SummaryWriter,
        step: int,
    ) -> tuple[float, float, list[float], float | None, float | None]:
        self.model.eval()
        acc = self._make_metrics_accumulator()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    raise ValueError("Expected batch to be a (inputs, targets) tuple.")

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(-1)

                loss, preds_for_rmse, quantile_preds = self._forward_batch(inputs, targets)
                self._update_metrics_accumulator(
                    acc,
                    loss=loss.detach(),
                    preds_for_rmse=preds_for_rmse.detach(),
                    targets=targets.detach(),
                    quantile_preds=quantile_preds.detach() if quantile_preds is not None else None,
                )
                writer.add_scalar(f"loss/{phase}_batch", loss.item(), step)
                if self.config.show_batch_logs and batch_idx % max(1, self.config.log_every) == 0:
                    print(
                        f"[{self.config.run_name}] {phase} batch {batch_idx} "
                        f"step={step} loss={loss.item():.3f}"
                    )
        return self._finalize_metrics_accumulator(acc)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "epoch": [],
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
        best_metric: float | None = None
        patience_epochs_used = 0
        writer = SummaryWriter(log_dir=self.config.log_dir / self.config.run_name)
        best_path = self.config.save_dir / f"{self.config.run_name}_best.pt"
        last_summary: Dict[str, Any] | None = None
        total_train_batches = max(1, len(train_loader))
        progress = _TrainingProgressDisplay(
            self.config.run_name,
            self.config.max_epochs,
            total_train_batches,
        )

        try:
            log_every = max(1, self.config.log_every)
            global_step = 0

            for epoch in range(1, self.config.max_epochs + 1):
                interval_acc = self._make_metrics_accumulator()
                interval_start = time.perf_counter()
                last_batch_loss: float | None = None

                for batch_idx, batch in enumerate(train_loader, start=1):
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs, targets = batch
                    else:
                        raise ValueError("Expected batch to be a (inputs, targets) tuple.")

                    self.model.train()
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(-1)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss, preds_for_rmse, quantile_preds = self._forward_batch(inputs, targets)
                    loss.backward()
                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    global_step += 1
                    last_batch_loss = float(loss.item())

                    self._update_metrics_accumulator(
                        interval_acc,
                        loss=loss.detach(),
                        preds_for_rmse=preds_for_rmse.detach(),
                        targets=targets.detach(),
                        quantile_preds=quantile_preds.detach() if quantile_preds is not None else None,
                    )
                    writer.add_scalar("loss/train_batch", loss.item(), global_step)
                    writer.add_scalar(
                        "metrics/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        global_step,
                    )
                    progress.update(
                        (epoch - 1) + (batch_idx / total_train_batches),
                        phase="train",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch_total=total_train_batches,
                        batch_loss=last_batch_loss,
                    )
                    if self.config.show_batch_logs and batch_idx % log_every == 0:
                        progress.write(
                            f"[{self.config.run_name}] epoch {epoch}/{self.config.max_epochs} "
                            f"batch {batch_idx}/{total_train_batches} "
                            f"batch_loss={loss.item():.3f}"
                        )

                (
                    train_loss,
                    train_rmse,
                    train_quantiles,
                    train_coverage,
                    train_width,
                ) = self._finalize_metrics_accumulator(interval_acc)
                eval_start = time.perf_counter()
                (
                    val_loss,
                    val_rmse,
                    val_quantiles,
                    val_coverage,
                    val_width,
                ) = self._run_loader(
                    val_loader,
                    phase="val",
                    writer=writer,
                    step=epoch,
                )

                history["epoch"].append(epoch)
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

                if test_loader is not None:
                    (
                        test_loss,
                        test_rmse,
                        test_quantiles,
                        test_coverage,
                        test_width,
                    ) = self.evaluate(
                        test_loader,
                        step=epoch,
                        writer=writer,
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

                interval_time = time.perf_counter() - interval_start
                eval_time = time.perf_counter() - eval_start
                writer.add_scalar("metrics/train_epoch_time_sec", interval_time, epoch)
                writer.add_scalar("metrics/eval_time_sec", eval_time, epoch)

                monitor_name, monitor_value, monitor_mode = self._select_monitor_value(
                    val_loss, val_coverage
                )
                if best_metric is None:
                    best_metric = monitor_value
                    patience_epochs_used = 0
                    self._save_weights(best_path)
                else:
                    improved = (
                        monitor_value < best_metric
                        if monitor_mode == "min"
                        else monitor_value > best_metric
                    )
                    if improved:
                        best_metric = monitor_value
                        patience_epochs_used = 0
                        self._save_weights(best_path)
                    else:
                        patience_epochs_used += 1
                writer.add_scalar("metrics/best_val", best_metric, epoch)
                writer.add_scalar(
                    "metrics/patience_epochs",
                    patience_epochs_used,
                    epoch,
                )
                progress.update(
                    float(epoch),
                    phase="eval",
                    epoch=epoch,
                    batch_idx=total_train_batches,
                    batch_total=total_train_batches,
                    batch_loss=last_batch_loss,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    test_loss=float(test_loss) if test_loader is not None else None,  # pyright: ignore[reportPossiblyUnboundVariable]
                    patience=f"{patience_epochs_used}/{self.config.patience_epochs}",
                )

                last_summary = {
                    "epoch": epoch,
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
                    "best_val": best_metric,
                    "monitor_name": monitor_name,
                    "patience_epochs": patience_epochs_used,
                }
                if test_loader is not None:
                    last_summary.update(
                        {
                            "test_loss": test_loss,  # pyright: ignore[reportPossiblyUnboundVariable]
                            "test_rmse": test_rmse,  # type: ignore
                            "test_quantiles": test_quantiles,  # pyright: ignore[reportPossiblyUnboundVariable]
                            "test_coverage": test_coverage,  # pyright: ignore[reportPossiblyUnboundVariable]
                            "test_width": test_width,  # type: ignore
                        }
                    )

                message = (
                    f"[{self.config.run_name}] epoch {epoch}/{self.config.max_epochs} "
                    f"- train_loss={train_loss:.3f} "
                    f"- train_rmse={train_rmse:.3f} "
                    f"- val_loss={val_loss:.3f} "
                    f"- val_rmse={val_rmse:.3f} "
                )
                if test_loader is not None:
                    message += (
                        f"- test_loss={test_loss:.3f} "  # pyright: ignore[reportPossiblyUnboundVariable]
                        f"- test_rmse={test_rmse:.3f} "  # pyright: ignore[reportPossiblyUnboundVariable]
                )
                message += (
                    f"- patience_epochs={patience_epochs_used}/{self.config.patience_epochs}"
                )
                progress.write(message)

                if patience_epochs_used >= self.config.patience_epochs:
                    progress.write(
                        f"[{self.config.run_name}] Early stopping at epoch "
                        f"{epoch}: train_rmse={train_rmse:.3f}, val_rmse={val_rmse:.3f}"
                        + (
                            f", test_rmse={test_rmse:.3f}"  # pyright: ignore[reportPossiblyUnboundVariable]
                            if test_loader is not None
                            else ""
                        )
                    )
                    break
        finally:
            progress.close()
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
            if self.coverage_quantile is not None:
                metrics["coverage_quantile"] = self.coverage_quantile
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
            metrics["last_epoch"] = last_summary["epoch"]
            metrics["patience_epochs"] = (
                f"{last_summary['patience_epochs']}/{self.config.patience_epochs}"
            )
            if last_summary.get("monitor_name"):
                metrics["monitor"] = last_summary["monitor_name"]
            _display_metrics_table("Final metrics", metrics)

        return history

    def evaluate(
        self, loader: DataLoader, step: int, writer: SummaryWriter
    ) -> tuple[float, float, list[float], float | None, float | None]:
        """Evaluate on a loader without updating weights."""
        return self._run_loader(loader, phase="test", writer=writer, step=step)

    def _save_weights(self, out_path: Path) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_path)

    def _select_monitor_value(
        self, val_loss: float, val_coverage: float | None
    ) -> tuple[str, float, str]:
        metric = self.config.monitor_metric.lower()
        if metric == "coverage" and val_coverage is not None and not math.isnan(val_coverage):
            return "coverage", float(val_coverage), "max"
        return "loss", float(val_loss), "min"


def _parse_model_types(raw: str) -> list[str]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("No model type provided.")
    if len(parts) == 1 and parts[0] == "all":
        return list(_SUPPORTED_MODEL_TYPES)
    invalid = [part for part in parts if part not in _SUPPORTED_MODEL_TYPES]
    if invalid:
        raise ValueError(
            f"Unsupported model type(s): {', '.join(invalid)}. "
            f"Use one or more of: {', '.join(_SUPPORTED_MODEL_TYPES)}, or 'all'."
    )
    return parts


def train_model(
    model_type: str = "lstm",
    manifest_path: str | Path | None = None,
    full_data_path: str | Path = DEFAULT_DATASET_PATH,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    context_length: int = 128,
    forecast_length: int = 1,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience_epochs: int = 10,
    max_iterations: int | None = None,
    patience_iterations: int | None = None,
    log_every: int = 50,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    output_dir: str | Path | None = None,
) -> Dict[str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = model_type.lower()
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError("model_type must be one of: lstm, deepar.")

    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
    else:
        manifest = build_experiment_manifest(
            data_path=full_data_path,
            timestamp_col=timestamp_col,
            context_length=context_length,
            horizon=forecast_length,
            quantiles=normalize_quantiles(quantiles),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    root_dir = Path(output_dir) if output_dir is not None else default_experiment_dir(manifest)
    manifest_path = save_manifest(manifest, root_dir / "manifest.json")

    data_cfg = DataLoaderConfig(
        data_path=Path(manifest.dataset_path),
        manifest_path=manifest_path,
        timestamp_col=manifest.timestamp_col,
        context_length=manifest.context_length,
        forecast_length=manifest.horizon,
        batch_size=batch_size,
        shuffle_train=True,
        pin_memory=device.type == "cuda",
    )
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg)
    steps_per_epoch = max(1, len(train_loader))
    if max_iterations is not None:
        max_epochs = max(1, math.ceil(int(max_iterations) / steps_per_epoch))
    if patience_iterations is not None:
        patience_epochs = max(1, math.ceil(int(patience_iterations) / steps_per_epoch))

    if model_type == "lstm":
        model = LSTMForecast(
            context_length=manifest.context_length,
            forecast_length=manifest.horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            quantiles=manifest.quantiles,
        )
    else:
        model = DeepARForecast(
            context_length=manifest.context_length,
            forecast_length=manifest.horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    model_config = {
        "model_type": model_type,
        "context_length": manifest.context_length,
        "forecast_length": manifest.horizon,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "quantiles": list(manifest.quantiles),
        "dataset_path": manifest.dataset_path,
    }

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if model_type == "deepar":
        # Explicit loss function for Gaussian outputs.
        loss_fn = GaussianNLLLoss()
    else:
        # Explicit pinball loss for quantile forecasting.
        loss_fn = QuantileLoss(manifest.quantiles)

    trainer_cfg = TrainerConfig(
        max_epochs=max_epochs,
        patience_epochs=patience_epochs,
        log_every=log_every,
        save_dir=root_dir / "checkpoints",
        log_dir=root_dir / "logs",
        run_name=f"{model_type}_{Path(manifest.dataset_path).stem}",
        monitor_metric="loss",
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        config=trainer_cfg,
        model_config=model_config,
        quantiles=manifest.quantiles,
    )
    return trainer.fit(train_loader, val_loader, test_loader)


def extract_last_coverage(
    history: Dict[str, list[float]],
    splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, float]:
    return {split: history[f"{split}_coverage"][-1] for split in splits}


def train_all(
    model_types: Sequence[str],
    *,
    manifest_path: str | Path | None = None,
    full_data_path: str | Path = DEFAULT_DATASET_PATH,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    context_length: int,
    forecast_length: int,
    quantiles: Sequence[float],
    max_epochs: int,
    patience_epochs: int,
    log_every: int,
    hidden_size: int,
    num_layers: int,
    output_dir: str | Path | None = None,
) -> tuple[Dict[tuple[str, str], Dict[str, float]], Dict[tuple[str, str], Dict[str, list[float]]]]:
    train_results: Dict[tuple[str, str], Dict[str, float]] = {}
    train_histories: Dict[tuple[str, str], Dict[str, list[float]]] = {}
    for model in model_types:
        history = train_model(
            model_type=model,
            manifest_path=manifest_path,
            full_data_path=full_data_path,
            timestamp_col=timestamp_col,
            context_length=context_length,
            forecast_length=forecast_length,
            quantiles=quantiles,
            max_epochs=max_epochs,
            patience_epochs=patience_epochs,
            log_every=log_every,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dir=output_dir,
        )
        key = (Path(full_data_path).stem, model)
        train_histories[key] = history
        train_results[key] = extract_last_coverage(history)
    return train_results, train_histories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy wrapper around the new split-aware training utilities."
    )
    parser.add_argument(
        "--model-type",
        default="lstm",
        help=(
            "Model family to train: lstm or deepar. "
            "Can also be a comma-separated list or 'all'."
        ),
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional alias for --model-type (supports comma-separated values and 'all').",
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--full-data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--patience-iterations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--validate-every", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--quantiles",
        default="0.5,0.95",
        help="Comma-separated list of quantiles to predict.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_types = _parse_model_types(args.models or args.model_type)
    common = dict(
        manifest_path=args.manifest_path,
        full_data_path=args.full_data_path,
        timestamp_col=args.timestamp_col,
        context_length=args.context_length,
        forecast_length=args.forecast_length,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience_epochs=args.patience,
        max_iterations=args.max_iterations,
        patience_iterations=args.patience_iterations,
        log_every=args.log_every if args.log_every is not None else 50,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        quantiles=parse_quantiles(args.quantiles),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir,
    )
    for model_type in model_types:
        run_common = dict(common)
        run_common["model_type"] = model_type
        print(f"Training model: {model_type}")
        train_model(**run_common)


if __name__ == "__main__":
    main()
