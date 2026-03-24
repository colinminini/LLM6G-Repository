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

import pandas as pd
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

from src.canonical_eval import evaluate_forecaster_on_split
from src.config import DEFAULT_DATASET_PATH, DEFAULT_QUANTILES, DEFAULT_TIMESTAMP_COL, TRAINABLE_BASELINE_MODELS, normalize_quantiles, parse_quantiles
from src.deepar_support import build_feature_spec, resolve_deepar_model_config
from src.experiment import ExperimentManifest, build_experiment_manifest, default_experiment_dir, load_manifest, load_wide_dataframe, save_manifest
from src.loader import DataLoaderConfig, build_dataloaders
from src.models import DeepARForecast, LSTMForecast, TFTForecast
from src.pipeline import build_forecaster, build_model_forecaster
from src.tft_support import resolve_tft_model_config


@dataclass
class TrainerConfig:
    max_iterations: int = 5000
    patience_iterations: int = 1000
    validate_every: int = 250
    log_every: int = 50
    grad_clip: float | None = None
    show_batch_logs: bool = False
    save_dir: Path = Path("results/models")
    log_dir: Path = Path("results/logs")
    run_name: str = "run"
    monitor_metric: str = "loss"


@dataclass
class CanonicalEvalConfig:
    manifest: ExperimentManifest
    full_df: pd.DataFrame
    splits: tuple[str, ...] = ("val", "test")
    max_windows_per_series: int | None = None


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
    def __init__(self, run_name: str, total_iterations: int, total_train_batches: int) -> None:
        self.run_name = run_name
        self.total_iterations = max(1, int(total_iterations))
        self.total_train_batches = max(1, int(total_train_batches))
        self.started_at = time.perf_counter()
        self.completed_progress = 0
        self._bar = (
            tqdm(
                total=self.total_iterations,
                desc=f"train:{run_name}",
                unit="iter",
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.15,
                position=0,
                leave=True,
                bar_format=(
                    "{l_bar}{bar}| {n:.0f}/{total:.0f} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
            )
            if tqdm is not None
            else None
        )
        self.update(0, phase="warmup")

    def _timings(self, completed_progress: int) -> tuple[float, float | None, float | None]:
        elapsed = time.perf_counter() - self.started_at
        if completed_progress <= 0:
            return elapsed, None, None
        estimated_total = elapsed * self.total_iterations / completed_progress
        remaining = max(0.0, estimated_total - elapsed)
        return elapsed, estimated_total, remaining

    def update(
        self,
        progress_iteration: int,
        *,
        phase: str,
        iteration: int | None = None,
        epoch: int | None = None,
        batch_idx: int | None = None,
        batch_total: int | None = None,
        batch_loss: float | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        test_loss: float | None = None,
        patience: str | None = None,
        **extra_fields: Any,
    ) -> None:
        target_progress = max(0, min(int(progress_iteration), self.total_iterations))
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
        if iteration is not None:
            postfix["iter"] = f"{min(int(iteration), self.total_iterations)}/{self.total_iterations}"
        if epoch is not None:
            postfix["epoch"] = str(int(epoch))
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
        for key, value in extra_fields.items():
            if value is None:
                continue
            if isinstance(value, float):
                postfix[str(key)] = f"{value:.3f}"
            else:
                postfix[str(key)] = str(value)

        if self._bar is not None:
            self._bar.set_postfix(postfix, refresh=False)
        elif target_progress > 0:
            print(
                f"[{self.run_name}] progress "
                f"{target_progress}/{self.total_iterations} "
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
        completed_iterations = min(self.total_iterations, int(self.completed_progress))
        self.update(
            self.completed_progress,
            phase="done" if completed_iterations >= self.total_iterations else "stopped",
            iteration=max(1, completed_iterations) if completed_iterations > 0 else 1,
            batch_idx=self.total_train_batches,
            batch_total=self.total_train_batches,
        )
        self.write(
            f"[{self.run_name}] wall_time={_format_duration(total_elapsed)} "
            f"completed_iterations={completed_iterations}/{self.total_iterations}"
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


class DeepARGaussianNLLLoss(nn.Module):
    """Sequence NLL for the paper-style Gaussian DeepAR outputs."""

    output_type = "gaussian"
    deepar_sequence_loss = True

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = outputs["mean"]
        sigma = torch.clamp(outputs["aux"], min=self.eps)
        if target.dim() != 2:
            raise ValueError("DeepAR sequence targets must have shape (batch, sequence_length).")
        loss = 0.5 * ((target - mean) / sigma).pow(2) + torch.log(sigma) + 0.5 * math.log(
            2 * math.pi
        )
        if observed_mask is None:
            return loss.mean()
        denom = torch.clamp(observed_mask.sum(), min=1.0)
        return (loss * observed_mask).sum() / denom


class NegativeBinomialNLLLoss(nn.Module):
    """Sequence NLL for the paper-style Negative Binomial DeepAR outputs."""

    output_type = "negative_binomial"
    deepar_sequence_loss = True

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean = torch.clamp(outputs["mean"], min=self.eps)
        alpha = torch.clamp(outputs["aux"], min=self.eps)
        total_count = torch.clamp(1.0 / alpha, min=self.eps)
        probs = total_count / (total_count + mean)
        dist = torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)
        non_negative_target = torch.clamp(target, min=0.0).round()
        loss = -dist.log_prob(non_negative_target)
        if observed_mask is None:
            return loss.mean()
        denom = torch.clamp(observed_mask.sum(), min=1.0)
        return (loss * observed_mask).sum() / denom


class Trainer:
    """Train a model with iteration-based early stopping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        config: TrainerConfig | None = None,
        model_config: Dict[str, Any] | None = None,
        quantiles: Sequence[float] | None = None,
        canonical_eval_config: CanonicalEvalConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config or TrainerConfig()
        self.model_config = model_config or {}
        self.canonical_eval_config = canonical_eval_config
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
        self.final_canonical_metrics: dict[str, dict[str, Any]] = {}
        self.final_monitor_metrics: dict[str, Any] = {}

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

    def _normalize_batch(self, batch: Any) -> dict[str, Any]:
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            return {"context": inputs, "target": targets}
        raise ValueError("Expected batch to be a dict or a (inputs, targets) tuple.")

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _forward_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if getattr(self.model, "batch_format", None) == "deepar":
            outputs = self.model.forward_train(
                full_target=batch["full_target"],
                time_features=batch["time_features"],
                item_ids=batch["item_id"],
                observed_mask=batch.get("observed_mask"),
            )
            loss = self.loss_fn(
                outputs,
                batch["full_target"],
                batch.get("observed_mask"),
            )
            future_target = batch["target"]
            preds_for_rmse = outputs["mean"][:, -future_target.size(1) :]
            quantile_preds: torch.Tensor | None = None
            if self.quantiles and not self.model.training:
                # Validation path: use the actual inference path (autoregressive sampling)
                # to produce honest metrics comparable to forecast_eval.
                # Teacher-forced analytical outputs inflate coverage by ~3x and
                # understate RMSE vs true autoregressive inference.
                with torch.no_grad():
                    context = batch["full_target"][:, : self.model.context_length]
                    num_val_samples = int(self.model_config.get("deepar_num_samples", 200))
                    draws = self.model.sample_forecasts(
                        context=context,
                        context_time_features=batch["time_features"][:, : self.model.context_length, :],
                        future_time_features=batch["time_features"][:, self.model.context_length :, :],
                        item_ids=batch["item_id"],
                        num_samples=num_val_samples,
                        random_seed=42,
                    )
                    # Use MC median as RMSE base — matches forecast_eval's rmse_q50.
                    preds_for_rmse = torch.quantile(draws, 0.5, dim=1)
                    q = torch.tensor(self.quantiles, device=draws.device, dtype=draws.dtype)
                    # draws: (batch, num_samples, horizon) → (num_q, batch, horizon) → (batch, horizon, num_q)
                    quantile_preds = torch.quantile(draws, q, dim=1).permute(1, 2, 0)
            return loss, preds_for_rmse, quantile_preds

        if getattr(self.model, "batch_format", None) == "tft":
            outputs = self.model(
                batch["past_inputs"],
                future_inputs=batch["future_inputs"],
                static_categorical=batch["static_categorical"],
            )
            loss = self.loss_fn(outputs, batch["target"])
            if self.quantiles and outputs.size(-1) != len(self.quantiles):
                raise ValueError("Quantile count does not match TFT outputs.")
            median_idx = self.median_index if self.median_index is not None else outputs.size(-1) // 2
            preds_for_rmse = outputs[..., median_idx]
            return loss, preds_for_rmse, outputs

        inputs = batch["context"]
        targets = batch["target"]
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
                batch_dict = self._move_batch_to_device(self._normalize_batch(batch))
                targets = batch_dict["target"]
                if targets.dim() == 1:
                    targets = targets.unsqueeze(-1)
                batch_dict["target"] = targets

                loss, preds_for_rmse, quantile_preds = self._forward_batch(batch_dict)
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

    def _canonical_splits(self) -> tuple[str, ...]:
        if self.canonical_eval_config is None:
            return ()
        return tuple(str(split).strip().lower() for split in self.canonical_eval_config.splits)

    def _run_canonical_eval_from_forecaster(
        self,
        forecaster: Any,
        *,
        training_progress: _TrainingProgressDisplay | None = None,
        training_iteration: int | None = None,
        training_epoch: int | None = None,
        training_batch_idx: int | None = None,
        training_batch_total: int | None = None,
        training_loss: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        if self.canonical_eval_config is None:
            return {}
        results: dict[str, dict[str, Any]] = {}
        model_name = str(self.model_config.get("model_type", self.config.run_name))
        show_progress = str(model_name).strip().lower() == "deepar"
        for split in self._canonical_splits():
            started_at = time.perf_counter()
            if show_progress:
                print(
                    f"[{self.config.run_name}] canonical_{split} starting "
                    f"(sampled forecast replay)"
                )
            progress_callback = None
            progress_label = None
            progress_position = 0
            progress_leave = True
            if show_progress and training_progress is not None and training_iteration is not None:
                def _callback(
                    completed: int,
                    total: int,
                    phase_name: str,
                    fields: dict[str, Any],
                ) -> None:
                    training_progress.update(
                        training_iteration,
                        phase=f"canonical_{split}",
                        iteration=training_iteration,
                        epoch=training_epoch,
                        batch_idx=training_batch_idx,
                        batch_total=training_batch_total,
                        train_loss=training_loss,
                        eval_split=split,
                        eval_window=f"{completed}/{total}",
                        eval_elapsed=_format_duration(fields.get("eval_elapsed")),
                        eval_eta=_format_duration(fields.get("eval_eta")),
                        eval_est_total=_format_duration(fields.get("eval_est_total")),
                        eval_phase=phase_name,
                        eval_series=fields.get("series"),
                        eval_rmse=fields.get("rmse_q50"),
                        eval_coverage=fields.get("coverage_q95"),
                    )

                progress_callback = _callback
            elif show_progress:
                progress_label = f"{self.config.run_name}:canonical_{split}"
                progress_position = 0
                progress_leave = True
            _, metrics = evaluate_forecaster_on_split(
                df=self.canonical_eval_config.full_df,
                manifest=self.canonical_eval_config.manifest,
                split=split,
                model_name=model_name,
                forecaster=forecaster,
                device=str(self.device),
                max_windows_per_series=self.canonical_eval_config.max_windows_per_series,
                collect_rows=False,
                progress_label=progress_label,
                progress_position=progress_position,
                progress_leave=progress_leave,
                progress_callback=progress_callback,
            )
            results[split] = metrics
            if show_progress:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[{self.config.run_name}] canonical_{split} completed in "
                    f"{elapsed:.1f}s rmse_q50={float(metrics['rmse_q50']):.3f}"
                )
        return results

    def _run_current_model_canonical_eval(
        self,
        *,
        training_progress: _TrainingProgressDisplay | None = None,
        training_iteration: int | None = None,
        training_epoch: int | None = None,
        training_batch_idx: int | None = None,
        training_batch_total: int | None = None,
        training_loss: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        model_name = str(self.model_config.get("model_type", "")).strip().lower()
        if not model_name or self.canonical_eval_config is None:
            return {}
        forecaster = build_model_forecaster(
            model_name,
            model=self.model,
            context_length=int(
                self.model_config.get(
                    "context_length",
                    getattr(self.model, "context_length", 1),
                )
            ),
            forecast_length=int(
                self.model_config.get(
                    "forecast_length",
                    getattr(self.model, "forecast_length", 1),
                )
            ),
            quantiles=self.quantiles or DEFAULT_QUANTILES,
            model_config=self.model_config,
            deepar_num_samples=int(self.model_config.get("deepar_num_samples", 200)),
            device=self.device,
        )
        return self._run_canonical_eval_from_forecaster(
            forecaster,
            training_progress=training_progress,
            training_iteration=training_iteration,
            training_epoch=training_epoch,
            training_batch_idx=training_batch_idx,
            training_batch_total=training_batch_total,
            training_loss=training_loss,
        )

    def _run_best_checkpoint_canonical_eval(
        self,
        *,
        checkpoint_path: Path,
    ) -> dict[str, dict[str, Any]]:
        model_name = str(self.model_config.get("model_type", "")).strip().lower()
        if not model_name or self.canonical_eval_config is None:
            return {}
        forecaster = build_forecaster(
            model_name,
            checkpoint_path=checkpoint_path,
            context_length=int(
                self.model_config.get(
                    "context_length",
                    getattr(self.model, "context_length", 1),
                )
            ),
            forecast_length=int(
                self.model_config.get(
                    "forecast_length",
                    getattr(self.model, "forecast_length", 1),
                )
            ),
            quantiles=self.quantiles or DEFAULT_QUANTILES,
            deepar_num_samples=int(self.model_config.get("deepar_num_samples", 200)),
            device=self.device,
        )
        return self._run_canonical_eval_from_forecaster(forecaster)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {
            "iteration": [],
            "train_loss": [],
            "train_rmse": [],
            "val_loss": [],
            "val_rmse": [],
            "val_pinball_q50": [],
            "val_pinball_q95": [],
            "train_coverage": [],
            "val_coverage": [],
            "train_interval_width": [],
            "val_interval_width": [],
        }
        best_metric: float | None = None
        patience_iterations_used = 0
        writer = SummaryWriter(log_dir=self.config.log_dir / self.config.run_name)
        best_path = self.config.save_dir / f"{self.config.run_name}_best.pt"
        last_summary: Dict[str, Any] | None = None
        total_train_batches = max(1, len(train_loader))
        progress = _TrainingProgressDisplay(
            self.config.run_name,
            self.config.max_iterations,
            total_train_batches,
        )

        try:
            train_iter = iter(train_loader)
            interval_acc = self._make_metrics_accumulator()
            interval_start = time.perf_counter()
            validate_every = max(1, min(self.config.validate_every, self.config.max_iterations))
            log_every = max(1, self.config.log_every)
            last_batch_loss: float | None = None

            for iteration in range(1, self.config.max_iterations + 1):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                self.model.train()
                batch_dict = self._move_batch_to_device(self._normalize_batch(batch))
                targets = batch_dict["target"]
                if targets.dim() == 1:
                    targets = targets.unsqueeze(-1)
                batch_dict["target"] = targets

                self.optimizer.zero_grad(set_to_none=True)
                loss, preds_for_rmse, quantile_preds = self._forward_batch(batch_dict)
                loss.backward()
                if self.config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                last_batch_loss = float(loss.item())

                self._update_metrics_accumulator(
                    interval_acc,
                    loss=loss.detach(),
                    preds_for_rmse=preds_for_rmse.detach(),
                    targets=targets.detach(),
                    quantile_preds=quantile_preds.detach() if quantile_preds is not None else None,
                )
                writer.add_scalar("loss/train_batch", loss.item(), iteration)
                writer.add_scalar(
                    "metrics/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    iteration,
                )
                epoch = ((iteration - 1) // total_train_batches) + 1
                batch_idx = ((iteration - 1) % total_train_batches) + 1
                progress.update(
                    iteration,
                    phase="train",
                    iteration=iteration,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    batch_total=total_train_batches,
                    batch_loss=last_batch_loss,
                )
                if self.config.show_batch_logs and (iteration == 1 or iteration % log_every == 0):
                    progress.write(
                        f"[{self.config.run_name}] train iter "
                        f"{iteration}/{self.config.max_iterations} "
                        f"batch_loss={loss.item():.3f}"
                    )

                if iteration % validate_every != 0 and iteration != self.config.max_iterations:
                    continue

                (
                    train_loss,
                    train_rmse,
                    train_quantiles,
                    train_coverage,
                    train_width,
                ) = self._finalize_metrics_accumulator(interval_acc)
                eval_start = time.perf_counter()
                (
                    monitor_val_loss,
                    monitor_val_rmse,
                    monitor_val_quantiles,
                    monitor_val_coverage,
                    monitor_val_width,
                ) = self._run_loader(
                    val_loader,
                    phase="val_monitor",
                    writer=writer,
                    step=iteration,
                )
                val_metrics = {
                    "rmse_q50": float(monitor_val_rmse),
                    "pinball_q50": (
                        float(monitor_val_quantiles[self.median_index])
                        if monitor_val_quantiles and self.median_index is not None
                        else float("nan")
                    ),
                    "pinball_q95": (
                        float(monitor_val_quantiles[self.coverage_index])
                        if monitor_val_quantiles and self.coverage_index is not None
                        else float("nan")
                    ),
                    "coverage_q95": (
                        float(monitor_val_coverage)
                        if monitor_val_coverage is not None
                        else float("nan")
                    ),
                    "interval_width_mean": (
                        float(monitor_val_width)
                        if monitor_val_width is not None
                        else float("nan")
                    ),
                }

                history["iteration"].append(iteration)
                history["train_loss"].append(train_loss)
                history["train_rmse"].append(train_rmse)
                history["val_loss"].append(monitor_val_loss)
                history["val_rmse"].append(float(val_metrics["rmse_q50"]))
                history["val_pinball_q50"].append(float(val_metrics["pinball_q50"]))
                history["val_pinball_q95"].append(float(val_metrics["pinball_q95"]))
                history["train_coverage"].append(
                    train_coverage if train_coverage is not None else float("nan")
                )
                history["val_coverage"].append(
                    float(val_metrics["coverage_q95"])
                )
                history["train_interval_width"].append(
                    train_width if train_width is not None else float("nan")
                )
                history["val_interval_width"].append(
                    float(val_metrics["interval_width_mean"])
                )

                writer.add_scalar("loss/train", train_loss, iteration)
                writer.add_scalar("loss/val_monitor", monitor_val_loss, iteration)
                writer.add_scalar("rmse/train", train_rmse, iteration)
                writer.add_scalar("rmse/val", float(val_metrics["rmse_q50"]), iteration)
                self._log_quantile_metrics(writer, "train", train_quantiles, iteration)
                writer.add_scalar("pinball/val/q50", float(val_metrics["pinball_q50"]), iteration)
                writer.add_scalar("pinball/val/q95", float(val_metrics["pinball_q95"]), iteration)
                if train_coverage is not None:
                    writer.add_scalar("coverage/train", train_coverage, iteration)
                writer.add_scalar("coverage/val", float(val_metrics["coverage_q95"]), iteration)
                if train_width is not None:
                    writer.add_scalar("interval_width/train", train_width, iteration)
                writer.add_scalar(
                    "interval_width/val",
                    float(val_metrics["interval_width_mean"]),
                    iteration,
                )

                interval_time = time.perf_counter() - interval_start
                eval_time = time.perf_counter() - eval_start
                writer.add_scalar("metrics/train_interval_time_sec", interval_time, iteration)
                writer.add_scalar("metrics/eval_time_sec", eval_time, iteration)

                monitor_name, monitor_value, monitor_mode = self._select_monitor_value(
                    monitor_val_loss,
                    monitor_val_coverage,
                    val_metrics.get("pinball_q50"),
                    val_metrics.get("pinball_q95"),
                )
                interval_iterations = int(interval_acc["num_batches"])
                if best_metric is None:
                    best_metric = monitor_value
                    patience_iterations_used = 0
                    self._save_weights(best_path)
                else:
                    improved = (
                        monitor_value < best_metric
                        if monitor_mode == "min"
                        else monitor_value > best_metric
                    )
                    if improved:
                        best_metric = monitor_value
                        patience_iterations_used = 0
                        self._save_weights(best_path)
                    else:
                        patience_iterations_used += interval_iterations
                writer.add_scalar("metrics/best_val", best_metric, iteration)
                writer.add_scalar(
                    "metrics/patience_iterations",
                    patience_iterations_used,
                    iteration,
                )
                progress.update(
                    iteration,
                    phase="eval",
                    iteration=iteration,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    batch_total=total_train_batches,
                    batch_loss=last_batch_loss,
                    train_loss=float(train_loss),
                    val_loss=float(monitor_val_loss),
                    patience=f"{patience_iterations_used}/{self.config.patience_iterations}",
                )

                last_summary = {
                    "iteration": iteration,
                    "train_loss": train_loss,
                    "train_rmse": train_rmse,
                    "monitor_val_loss": monitor_val_loss,
                    "monitor_val_rmse": float(monitor_val_rmse),
                    "train_quantiles": train_quantiles,
                    "train_coverage": train_coverage,
                    "train_width": train_width,
                    "val_metrics": val_metrics,
                    "best_val": best_metric,
                    "monitor_name": monitor_name,
                    "patience_iterations": patience_iterations_used,
                }

                message = (
                    f"[{self.config.run_name}] iter {iteration}/{self.config.max_iterations} "
                    f"- train_loss={train_loss:.3f} "
                    f"- train_rmse={train_rmse:.3f} "
                    f"- monitor_{monitor_name}={monitor_value:.3f} "
                    f"- val_rmse={float(val_metrics['rmse_q50']):.3f} "
                )
                message += (
                    f"- patience_iters={patience_iterations_used}/{self.config.patience_iterations}"
                )
                progress.write(message)

                if patience_iterations_used >= self.config.patience_iterations:
                    progress.write(
                        f"[{self.config.run_name}] Early stopping at iteration "
                        f"{iteration}: train_rmse={train_rmse:.3f}, "
                        f"val_rmse={float(val_metrics['rmse_q50']):.3f}"
                    )
                    break

                interval_acc = self._make_metrics_accumulator()
                interval_start = time.perf_counter()
        finally:
            progress.close()
            writer.close()

        self.final_canonical_metrics = {}

        if last_summary is not None:
            monitor_label = last_summary.get("monitor_name") or "loss"
            metrics: Dict[str, Any] = {
                "train_loss": last_summary["train_loss"],
                "train_rmse": last_summary["train_rmse"],
                f"monitor_val_{monitor_label}": last_summary["monitor_val_loss"],
                "monitor_val_rmse": last_summary["monitor_val_rmse"],
            }
            for q, value in zip(self.quantiles, last_summary["train_quantiles"]):
                metrics[f"train_{_format_quantile_key(q)}"] = value
            if last_summary.get("train_coverage") is not None:
                metrics["train_coverage"] = last_summary["train_coverage"]
            if self.coverage_quantile is not None:
                metrics["coverage_quantile"] = self.coverage_quantile
            if last_summary.get("train_width") is not None:
                metrics["train_interval_width"] = last_summary["train_width"]

            for split_name, payload in {"val": last_summary["val_metrics"]}.items():
                metrics[f"{split_name}_rmse_q50"] = payload.get("rmse_q50")
                metrics[f"{split_name}_pinball_q50"] = payload.get("pinball_q50")
                metrics[f"{split_name}_pinball_q95"] = payload.get("pinball_q95")
                metrics[f"{split_name}_coverage_q95"] = payload.get("coverage_q95")
                metrics[f"{split_name}_interval_width_mean"] = payload.get("interval_width_mean")
            metrics["best_val"] = last_summary["best_val"]
            metrics["last_iteration"] = last_summary["iteration"]
            metrics["patience_iterations"] = (
                f"{last_summary['patience_iterations']}/{self.config.patience_iterations}"
            )
            if last_summary.get("monitor_name"):
                metrics["monitor"] = last_summary["monitor_name"]

            self.final_monitor_metrics = {
                "monitor_val_loss": last_summary["monitor_val_loss"],
                "monitor_val_rmse": last_summary["monitor_val_rmse"],
                "val": dict(last_summary["val_metrics"]),
                "best_val": last_summary["best_val"],
                "last_iteration": last_summary["iteration"],
                "monitor": last_summary.get("monitor_name"),
            }
            _display_metrics_table("Final metrics", metrics)

        return history

    def evaluate(
        self, loader: DataLoader, step: int, writer: SummaryWriter
    ) -> tuple[float, float, list[float], float | None, float | None]:
        """Evaluate on a loader without updating weights."""
        return self._run_loader(loader, phase="test", writer=writer, step=step)

    def _save_weights(self, out_path: Path) -> None:
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": self.model_config,
            },
            out_path,
        )

    def _select_monitor_value(
        self,
        val_loss: float,
        val_coverage: float | None,
        val_pinball_q50: float | None = None,
        val_pinball_q95: float | None = None,
    ) -> tuple[str, float, str]:
        metric = self.config.monitor_metric.lower()
        if metric == "coverage" and val_coverage is not None and not math.isnan(val_coverage):
            return "coverage", float(val_coverage), "max"
        if metric == "pinball":
            q50 = float(val_pinball_q50) if val_pinball_q50 is not None and not math.isnan(float(val_pinball_q50)) else None
            q95 = float(val_pinball_q95) if val_pinball_q95 is not None and not math.isnan(float(val_pinball_q95)) else None
            if q50 is not None and q95 is not None:
                return "pinball", q50 + q95, "min"
            if q95 is not None:
                return "pinball_q95", q95, "min"
            if q50 is not None:
                return "pinball_q50", q50, "min"
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
    train_window_step: int = 1,
    max_iterations: int = 5000,
    patience_iterations: int = 1000,
    validate_every: int = 250,
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
        raise ValueError("model_type must be one of: lstm, deepar, tft.")

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

    full_df = load_wide_dataframe(manifest.dataset_path, manifest.timestamp_col)
    feature_spec = build_feature_spec(
        full_df[manifest.timestamp_col],
        cadence=manifest.cadence,
        train_end_idx=manifest.train_split.end_idx,
    )

    data_cfg = DataLoaderConfig(
        data_path=Path(manifest.dataset_path),
        manifest_path=manifest_path,
        timestamp_col=manifest.timestamp_col,
        model_name=model_type,
        context_length=manifest.context_length,
        forecast_length=manifest.horizon,
        train_window_step=max(1, int(train_window_step)),
        batch_size=batch_size,
        shuffle_train=True,
        pin_memory=device.type == "cuda",
        deepar_feature_spec=feature_spec if model_type == "deepar" else None,
        tft_feature_spec=feature_spec if model_type == "tft" else None,
    )
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg)
    if model_type == "lstm":
        model = LSTMForecast(
            context_length=manifest.context_length,
            forecast_length=manifest.horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            quantiles=manifest.quantiles,
        )
    elif model_type == "deepar":
        model = DeepARForecast(
            context_length=manifest.context_length,
            forecast_length=manifest.horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_time_features=feature_spec.num_features,
            num_items=manifest.num_series,
        )
    else:
        model = TFTForecast(
            context_length=manifest.context_length,
            forecast_length=manifest.horizon,
            hidden_size=hidden_size,
            num_lstm_layers=num_layers,
            num_heads=4,
            quantiles=manifest.quantiles,
            num_static_categorical=1,
            static_categorical_cardinalities=(manifest.num_series,),
            num_past_features=1 + feature_spec.num_features,
            num_future_features=feature_spec.num_features,
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
    if model_type == "deepar":
        model_config = resolve_deepar_model_config(
            manifest=manifest,
            feature_spec=feature_spec,
        )
    elif model_type == "tft":
        model_config = resolve_tft_model_config(
            manifest=manifest,
            hidden_size=hidden_size,
            num_heads=4,
            num_lstm_layers=num_layers,
            feature_spec=feature_spec,
        )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if model_type == "deepar":
        # Explicit loss function for Gaussian outputs.
        loss_fn = GaussianNLLLoss()
    else:
        # Explicit pinball loss for quantile forecasting.
        loss_fn = QuantileLoss(manifest.quantiles)

    trainer_cfg = TrainerConfig(
        max_iterations=max_iterations,
        patience_iterations=patience_iterations,
        validate_every=validate_every,
        log_every=log_every,
        save_dir=root_dir / "checkpoints",
        log_dir=root_dir / "logs",
        run_name=f"{model_type}_{Path(manifest.dataset_path).stem}",
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
    train_window_step: int,
    max_iterations: int,
    patience_iterations: int,
    validate_every: int,
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
            train_window_step=train_window_step,
            max_iterations=max_iterations,
            patience_iterations=patience_iterations,
            validate_every=validate_every,
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
            "Model family to train: lstm, deepar, or tft. "
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
    parser.add_argument("--train-window-step", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--patience-iterations", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=None, help=argparse.SUPPRESS)
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
        train_window_step=args.train_window_step,
        max_iterations=args.max_iterations if args.max_iterations is not None else 5000,
        patience_iterations=args.patience_iterations if args.patience_iterations is not None else 1000,
        validate_every=args.validate_every if args.validate_every is not None else 250,
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
