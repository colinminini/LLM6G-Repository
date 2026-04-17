"""Hybrid probabilistic forecasting + change-point pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.config import DEFAULT_DATASET_PATH, DEFAULT_QUANTILES, DEFAULT_SEASONAL_NAIVE_PERIOD
from src.deepar_support import (
    build_series_item_index,
    build_time_feature_matrix,
    feature_spec_from_payload,
)
from src.device import AUTO_DEVICE, resolve_torch_device
from src.experiment import load_wide_dataframe
from src.models import DeepARForecast, LSTMForecast, LegacyDeepARForecast, TFTForecast
from src.tft_support import build_tft_time_feature_matrix
from src.change_detection import ChangePointDetector, RupturesPeltDetector

Z95 = 1.6448536269514722


@dataclass
class ProbabilisticForecast:
    """Probabilistic forecast container.

    The field name ``y_pred_95`` is historical; it holds whichever upper
    quantile is configured (DEFAULT_QUANTILES[-1], now 0.99). Treat it as
    ``y_pred_upper``.
    """

    y_pred_median: np.ndarray
    y_pred_95: np.ndarray


@dataclass
class ProbabilisticForecastBatch:
    """Batched ``ProbabilisticForecast`` (see that class for field semantics)."""

    y_pred_median: np.ndarray
    y_pred_95: np.ndarray


@dataclass
class DeepARInferenceSupport:
    dataset_path: str
    timestamp_col: str
    cadence: str
    series_columns: tuple[str, ...]
    series_to_item_id: dict[str, int]
    time_feature_matrix: np.ndarray
    num_samples: int
    random_seed: int


@dataclass
class TFTInferenceSupport:
    dataset_path: str
    timestamp_col: str
    cadence: str
    series_columns: tuple[str, ...]
    series_to_item_id: dict[str, int]
    time_feature_matrix: np.ndarray


@dataclass
class PipelineOutput:
    y_pred_median: np.ndarray
    y_pred_95: np.ndarray
    tau_pred: int | None
    safe_ceiling: float
    stationary_pred_95: np.ndarray

    @property
    def upper_bound_95(self) -> float:
        """Backward-compatible alias."""
        return self.safe_ceiling


class Forecaster(ABC):
    """Interface for probabilistic forecasters."""

    @abstractmethod
    def predict_quantiles(
        self,
        history: np.ndarray | list[float],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecast:
        """Return q50 and q95 trajectories."""

    def predict_quantiles_batch(
        self,
        histories: np.ndarray | Sequence[np.ndarray] | Sequence[list[float]],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecastBatch:
        histories_arr = _normalize_history_batch(histories)
        series_names = kwargs.get("series_names")
        start_indices = kwargs.get("start_indices")
        forecasts = []
        for row_idx, history in enumerate(histories_arr):
            row_kwargs: dict[str, Any] = {}
            if series_names is not None:
                row_kwargs["series_name"] = series_names[row_idx]
            if start_indices is not None:
                row_kwargs["start_index"] = start_indices[row_idx]
            forecasts.append(self.predict_quantiles(history=history, horizon=horizon, **row_kwargs))
        if not forecasts:
            empty = np.empty((0, int(horizon)), dtype=float)
            return ProbabilisticForecastBatch(y_pred_median=empty, y_pred_95=empty.copy())
        return ProbabilisticForecastBatch(
            y_pred_median=np.stack([forecast.y_pred_median for forecast in forecasts], axis=0),
            y_pred_95=np.stack([forecast.y_pred_95 for forecast in forecasts], axis=0),
        )


def _normalize_history_batch(
    histories: np.ndarray | Sequence[np.ndarray] | Sequence[list[float]],
) -> np.ndarray:
    if isinstance(histories, np.ndarray):
        arr = np.asarray(histories, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        raise ValueError("Expected histories with shape (seq,) or (batch, seq).")

    rows = [np.asarray(history, dtype=float).reshape(-1) for history in histories]
    if not rows:
        return np.empty((0, 0), dtype=float)
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError("All batched histories must have the same length.")
    return np.stack(rows, axis=0)


def _slice_with_left_padding(
    matrix: np.ndarray,
    *,
    start: int,
    end: int,
    width: int,
) -> np.ndarray:
    slice_start = max(0, int(start))
    slice_end = max(slice_start, int(end))
    values = np.asarray(matrix[slice_start:slice_end], dtype=float)
    if values.shape[0] >= width:
        return values[-width:]
    padded = np.zeros((int(width), values.shape[1]), dtype=float)
    padded[-values.shape[0] :] = values
    return padded


def _build_deepar_inference_support(
    model_config: dict[str, Any],
    *,
    num_samples_override: int | None = None,
) -> DeepARInferenceSupport | None:
    payload = dict(model_config or {})
    if payload.get("model_type") != "deepar":
        return None
    feature_spec = feature_spec_from_payload(payload.get("deepar_feature_spec"))
    if feature_spec is None:
        return None
    dataset_path = str(payload["dataset_path"])
    timestamp_col = str(payload["timestamp_col"])
    cadence = str(payload["cadence"])
    series_columns = tuple(str(name) for name in payload.get("series_columns", ()))
    df = load_wide_dataframe(dataset_path, timestamp_col=timestamp_col)
    time_feature_matrix = build_time_feature_matrix(
        df[timestamp_col],
        cadence=cadence,
        feature_spec=feature_spec,
    )
    return DeepARInferenceSupport(
        dataset_path=dataset_path,
        timestamp_col=timestamp_col,
        cadence=cadence,
        series_columns=series_columns,
        series_to_item_id=build_series_item_index(series_columns),
        time_feature_matrix=np.asarray(time_feature_matrix, dtype=float),
        num_samples=int(
            num_samples_override
            if num_samples_override is not None
            else payload.get("deepar_num_samples", 200)
        ),
        random_seed=int(payload.get("deepar_random_seed", 42)),
    )


def _build_tft_inference_support(
    model_config: dict[str, Any],
) -> TFTInferenceSupport | None:
    payload = dict(model_config or {})
    if payload.get("model_type") != "tft":
        return None
    feature_spec = feature_spec_from_payload(payload.get("tft_known_feature_spec"))
    if feature_spec is None:
        return None
    dataset_path = str(payload["dataset_path"])
    timestamp_col = str(payload["timestamp_col"])
    cadence = str(payload["cadence"])
    series_columns = tuple(str(name) for name in payload.get("series_columns", ()))
    df = load_wide_dataframe(dataset_path, timestamp_col=timestamp_col)
    time_feature_matrix = build_tft_time_feature_matrix(
        df[timestamp_col],
        cadence=cadence,
        feature_spec=feature_spec,
    )
    return TFTInferenceSupport(
        dataset_path=dataset_path,
        timestamp_col=timestamp_col,
        cadence=cadence,
        series_columns=series_columns,
        series_to_item_id=build_series_item_index(series_columns),
        time_feature_matrix=np.asarray(time_feature_matrix, dtype=float),
    )


class _TorchProbabilisticMixin:
    @staticmethod
    def _extract_quantile_path(
        quantile_preds: np.ndarray,
        quantile_levels: Sequence[float],
        target_q: float,
    ) -> np.ndarray:
        if quantile_preds.ndim != 2:
            raise ValueError("quantile_preds must have shape (horizon, num_quantiles).")
        levels = np.asarray(list(quantile_levels), dtype=float)
        if levels.size != quantile_preds.shape[1]:
            raise ValueError("quantile levels size does not match prediction dimension.")

        order = np.argsort(levels)
        levels = levels[order]
        preds = quantile_preds[:, order]

        if target_q <= levels[0]:
            return preds[:, 0]
        if target_q >= levels[-1]:
            return preds[:, -1]

        return np.asarray(
            [np.interp(target_q, levels, row) for row in preds],
            dtype=float,
        )

    def _extract_quantile_batch(
        self,
        quantile_preds: np.ndarray,
        quantile_levels: Sequence[float],
        target_q: float,
    ) -> np.ndarray:
        preds = np.asarray(quantile_preds, dtype=float)
        if preds.ndim != 3:
            raise ValueError(
                "quantile_preds must have shape (batch, horizon, num_quantiles)."
            )
        return np.stack(
            [
                self._extract_quantile_path(sample_preds, quantile_levels, target_q)
                for sample_preds in preds
            ],
            axis=0,
        )

    def _predict_block_batch(
        self,
        context_batch: np.ndarray,
        *,
        series_names: Sequence[str] | None = None,
        start_indices: Sequence[int] | None = None,
    ) -> ProbabilisticForecastBatch:
        contexts = _normalize_history_batch(context_batch)
        context_tensor = torch.tensor(contexts, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.model_type == "deepar":
                if self.deepar_support is None or start_indices is None or series_names is None:
                    context_time = np.zeros(
                        (contexts.shape[0], self.context_length, self.model.num_time_features),
                        dtype=float,
                    )
                    future_time = np.zeros(
                        (contexts.shape[0], self.forecast_length, self.model.num_time_features),
                        dtype=float,
                    )
                    item_ids = np.zeros(contexts.shape[0], dtype=int)
                else:
                    context_rows = []
                    future_rows = []
                    item_ids = []
                    for series_name, start_idx in zip(series_names, start_indices):
                        start_idx = int(start_idx)
                        context_rows.append(
                            _slice_with_left_padding(
                                self.deepar_support.time_feature_matrix,
                                start=start_idx - self.context_length,
                                end=start_idx,
                                width=self.context_length,
                            )
                        )
                        future_rows.append(
                            _slice_with_left_padding(
                                self.deepar_support.time_feature_matrix,
                                start=start_idx,
                                end=start_idx + self.forecast_length,
                                width=self.forecast_length,
                            )
                        )
                        item_ids.append(self.deepar_support.series_to_item_id[str(series_name)])
                    context_time = np.stack(context_rows, axis=0)
                    future_time = np.stack(future_rows, axis=0)

                upper_quantile = float(self.output_quantiles[-1]) if self.output_quantiles else 0.99
                likelihood = str(getattr(self.model, "likelihood", "gaussian")).lower()
                if likelihood == "gaussian":
                    # Analytic Gaussian quantiles via a deterministic greedy rollout:
                    # feeds per-step mean back into the LSTM and reads (mu_t, sigma_t)
                    # from the head. This isolates "Gaussian NLL miscalibration" from
                    # Monte-Carlo noise and is what paper_v2 compares against pinball.
                    params = self.model.analytic_gaussian_forecast(
                        context=context_tensor,
                        context_time_features=torch.tensor(
                            context_time, dtype=torch.float32, device=self.device
                        ),
                        future_time_features=torch.tensor(
                            future_time, dtype=torch.float32, device=self.device
                        ),
                        item_ids=torch.tensor(item_ids, dtype=torch.long, device=self.device),
                    )
                    mean = params["mean"].detach().cpu().numpy().astype(float)
                    sigma = params["sigma"].detach().cpu().numpy().astype(float)
                    from scipy.stats import norm

                    z_upper = float(norm.ppf(upper_quantile))
                    y50 = mean
                    y95 = mean + z_upper * sigma
                else:
                    draws = self.model.sample_forecasts(
                        context=context_tensor,
                        context_time_features=torch.tensor(
                            context_time, dtype=torch.float32, device=self.device
                        ),
                        future_time_features=torch.tensor(
                            future_time, dtype=torch.float32, device=self.device
                        ),
                        item_ids=torch.tensor(item_ids, dtype=torch.long, device=self.device),
                        num_samples=(
                            self.deepar_support.num_samples
                            if self.deepar_support is not None
                            else 200
                        ),
                        random_seed=(
                            self.deepar_support.random_seed if self.deepar_support is not None else 42
                        ),
                    )
                    y50 = torch.quantile(draws, 0.5, dim=1).detach().cpu().numpy().astype(float)
                    y95 = (
                        torch.quantile(draws, upper_quantile, dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
            elif self.model_type == "tft":
                if self.tft_support is None or start_indices is None or series_names is None:
                    context_known = np.zeros(
                        (
                            contexts.shape[0],
                            self.context_length,
                            max(0, int(self.model_config.get("tft_num_known_features", 0))),
                        ),
                        dtype=float,
                    )
                    future_known = np.zeros(
                        (
                            contexts.shape[0],
                            self.forecast_length,
                            max(1, int(self.model_config.get("tft_num_future_features", 0))),
                        ),
                        dtype=float,
                    )
                    static_categorical = np.zeros((contexts.shape[0], 1), dtype=int)
                else:
                    context_rows = []
                    future_rows = []
                    item_ids = []
                    for series_name, start_idx in zip(series_names, start_indices):
                        start_idx = int(start_idx)
                        context_rows.append(
                            _slice_with_left_padding(
                                self.tft_support.time_feature_matrix,
                                start=start_idx - self.context_length,
                                end=start_idx,
                                width=self.context_length,
                            )
                        )
                        future_rows.append(
                            _slice_with_left_padding(
                                self.tft_support.time_feature_matrix,
                                start=start_idx,
                                end=start_idx + self.forecast_length,
                                width=self.forecast_length,
                            )
                        )
                        item_ids.append(self.tft_support.series_to_item_id[str(series_name)])
                    context_known = np.stack(context_rows, axis=0)
                    future_known = np.stack(future_rows, axis=0)
                    static_categorical = np.asarray(item_ids, dtype=int).reshape(-1, 1)

                past_inputs = np.concatenate(
                    [contexts[..., None], context_known],
                    axis=-1,
                )
                outputs = self.model(
                    torch.tensor(past_inputs, dtype=torch.float32, device=self.device),
                    future_inputs=torch.tensor(
                        future_known,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    static_categorical=torch.tensor(
                        static_categorical,
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
                quantile_preds = outputs.detach().cpu().numpy().astype(float)
                y50 = self._extract_quantile_batch(
                    quantile_preds,
                    self.output_quantiles,
                    0.5,
                )
                max_trained_q = max(self.output_quantiles)
                if 0.95 <= max_trained_q + 1e-8:
                    y95 = self._extract_quantile_batch(
                        quantile_preds,
                        self.output_quantiles,
                        0.95,
                    )
                elif self.enable_residual_fallback and self.residual_sigma is not None:
                    y95 = y50 + Z95 * float(self.residual_sigma)
                else:
                    y95 = self._extract_quantile_batch(
                        quantile_preds,
                        self.output_quantiles,
                        max_trained_q,
                    )
            else:
                outputs = self.model(context_tensor)
                quantile_preds = outputs.detach().cpu().numpy().astype(float)
                y50 = self._extract_quantile_batch(
                    quantile_preds,
                    self.output_quantiles,
                    0.5,
                )
                max_trained_q = max(self.output_quantiles)
                if 0.95 <= max_trained_q + 1e-8:
                    y95 = self._extract_quantile_batch(
                        quantile_preds,
                        self.output_quantiles,
                        0.95,
                    )
                elif self.enable_residual_fallback and self.residual_sigma is not None:
                    y95 = y50 + Z95 * float(self.residual_sigma)
                else:
                    y95 = self._extract_quantile_batch(
                        quantile_preds,
                        self.output_quantiles,
                        max_trained_q,
                    )

        y50 = np.asarray(y50, dtype=float)
        y95 = np.asarray(y95, dtype=float)
        y95 = np.maximum(y95, y50)

        if y50.size == 0 or y95.size == 0:
            raise ValueError("Model returned an empty forecast block.")
        if y50.size != y95.size:
            raise ValueError("q50 and q95 block sizes do not match.")

        return ProbabilisticForecastBatch(y_pred_median=y50, y_pred_95=y95)

    def _predict_block(self, context: np.ndarray) -> ProbabilisticForecast:
        batch = self._predict_block_batch(np.asarray(context, dtype=float).reshape(1, -1))
        return ProbabilisticForecast(
            y_pred_median=batch.y_pred_median[0],
            y_pred_95=batch.y_pred_95[0],
        )

    def predict_quantiles(
        self,
        history: np.ndarray | list[float],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecast:
        batch = self.predict_quantiles_batch(
            np.asarray(history, dtype=float).reshape(1, -1),
            horizon,
            series_names=[kwargs["series_name"]] if "series_name" in kwargs else None,
            start_indices=[kwargs["start_index"]] if "start_index" in kwargs else None,
        )
        return ProbabilisticForecast(
            y_pred_median=batch.y_pred_median[0],
            y_pred_95=batch.y_pred_95[0],
        )

    def predict_quantiles_batch(
        self,
        histories: np.ndarray | Sequence[np.ndarray] | Sequence[list[float]],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecastBatch:
        histories_arr = _normalize_history_batch(histories)
        if histories_arr.shape[1] < self.context_length:
            raise ValueError(
                f"History too short ({histories_arr.shape[1]}) for context_length={self.context_length}."
            )
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        generated_50: list[np.ndarray] = []
        generated_95: list[np.ndarray] = []
        rolling_history = histories_arr.copy()
        remaining = int(horizon)
        series_names = kwargs.get("series_names")
        start_indices = kwargs.get("start_indices")

        while remaining > 0:
            contexts = rolling_history[:, -self.context_length :]
            block = self._predict_block_batch(
                contexts,
                series_names=series_names,
                start_indices=start_indices,
            )

            take = min(remaining, int(block.y_pred_median.shape[1]))
            chunk_50 = block.y_pred_median[:, :take]
            chunk_95 = block.y_pred_95[:, :take]

            generated_50.append(chunk_50)
            generated_95.append(chunk_95)
            remaining -= take

            if remaining > 0:
                # Roll with central tendency trajectory for autoregressive extension.
                rolling_history = np.concatenate([rolling_history, chunk_50], axis=1)

        y50 = np.concatenate(generated_50, axis=1)
        y95 = np.concatenate(generated_95, axis=1)
        y95 = np.maximum(y95, y50)
        return ProbabilisticForecastBatch(y_pred_median=y50, y_pred_95=y95)


class TorchModelForecaster(_TorchProbabilisticMixin, Forecaster):
    """Adapter for in-memory PyTorch baselines."""

    def __init__(
        self,
        model_type: str,
        model: torch.nn.Module,
        *,
        context_length: int = 128,
        forecast_length: int | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        output_quantiles: Sequence[float] | None = None,
        residual_sigma: float | None = None,
        enable_residual_fallback: bool = True,
        model_config: dict[str, Any] | None = None,
        deepar_num_samples: int | None = None,
        device: str | torch.device = AUTO_DEVICE,
    ) -> None:
        self.model_type = model_type.lower()
        if self.model_type not in {"lstm", "deepar", "tft"}:
            raise ValueError("model_type must be one of: lstm, deepar, tft.")

        self.context_length = int(context_length)
        self.forecast_length = int(
            forecast_length
            if forecast_length is not None
            else getattr(model, "forecast_length", 1)
        )
        self.input_quantiles = tuple(float(q) for q in quantiles)
        if output_quantiles is None:
            if self.model_type == "deepar":
                self.output_quantiles = self.input_quantiles
            else:
                self.output_quantiles = tuple(
                    float(q) for q in getattr(model, "quantiles", self.input_quantiles)
                )
        else:
            self.output_quantiles = tuple(float(q) for q in output_quantiles)
        self.residual_sigma = float(residual_sigma) if residual_sigma is not None else None
        self.enable_residual_fallback = bool(enable_residual_fallback)
        self.model_config = dict(model_config or {})
        self.device = resolve_torch_device(device)
        self.deepar_support = _build_deepar_inference_support(
            self.model_config,
            num_samples_override=deepar_num_samples,
        )
        self.tft_support = _build_tft_inference_support(self.model_config)
        self._prediction_calls = 0
        self.evaluation_batch_size = 64 if self.model_type == "deepar" else 256
        self.model = model.to(self.device)
        self.model.eval()


class TorchCheckpointForecaster(TorchModelForecaster):
    """Adapter for saved PyTorch baselines in `results/models/*.pt`."""

    def __init__(
        self,
        model_type: str,
        checkpoint_path: str | Path,
        context_length: int = 128,
        forecast_length: int | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        residual_sigma: float | None = None,
        enable_residual_fallback: bool = True,
        deepar_num_samples: int | None = None,
        device: str | torch.device = AUTO_DEVICE,
    ) -> None:
        self.model_type = model_type.lower()
        if self.model_type not in {"lstm", "deepar", "tft"}:
            raise ValueError("model_type must be one of: lstm, deepar, tft.")

        self.checkpoint_path = Path(checkpoint_path)
        self.context_length = int(context_length)
        self.input_quantiles = tuple(float(q) for q in quantiles)
        self.residual_sigma = float(residual_sigma) if residual_sigma is not None else None
        self.enable_residual_fallback = bool(enable_residual_fallback)
        self.device = resolve_torch_device(device)

        checkpoint_payload = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload:
            state_dict = checkpoint_payload["state_dict"]
            self.model_config = dict(checkpoint_payload.get("model_config", {}))
        else:
            state_dict = checkpoint_payload
            self.model_config = {}
        if forecast_length is None and self.model_config.get("forecast_length") is not None:
            forecast_length = int(self.model_config["forecast_length"])
        if context_length == 128 and self.model_config.get("context_length") is not None:
            context_length = int(self.model_config["context_length"])
        self.context_length = int(context_length)
        self.forecast_length = self._resolve_forecast_length(state_dict, forecast_length)
        self.model, self.output_quantiles = self._build_model(state_dict)
        super().__init__(
            model_type=self.model_type,
            model=self.model,
            context_length=self.context_length,
            forecast_length=self.forecast_length,
            quantiles=self.input_quantiles,
            output_quantiles=self.output_quantiles,
            residual_sigma=self.residual_sigma,
            enable_residual_fallback=self.enable_residual_fallback,
            model_config=self.model_config,
            deepar_num_samples=deepar_num_samples,
            device=self.device,
        )

    def _resolve_forecast_length(
        self,
        state_dict: dict[str, torch.Tensor],
        requested: int | None,
    ) -> int:
        if self.model_type == "lstm":
            out_dim = int(state_dict["head.weight"].shape[0])
            if requested is not None:
                requested = int(requested)
                if requested <= 0:
                    raise ValueError("forecast_length must be positive.")
                if out_dim % requested == 0:
                    return requested
                raise ValueError(
                    "Requested forecast_length is incompatible with the LSTM checkpoint: "
                    f"head output dim={out_dim}, requested forecast_length={requested}."
                )

            num_input_quantiles = len(self.input_quantiles)
            if num_input_quantiles > 0 and out_dim % num_input_quantiles == 0:
                return out_dim // num_input_quantiles

            candidate_num_quantiles = (1, 2, 3, 4, 5, 7, 9)
            candidate_lengths = sorted(
                {
                    out_dim // num_q
                    for num_q in candidate_num_quantiles
                    if out_dim % num_q == 0
                }
            )
            if len(candidate_lengths) == 1:
                return candidate_lengths[0]

            raise ValueError(
                "Cannot infer LSTM forecast_length unambiguously from the checkpoint. "
                f"head output dim={out_dim}, candidate forecast lengths={candidate_lengths}. "
                "Pass the training forecast_length explicitly."
            )

        return int(requested) if requested is not None else 16

    @staticmethod
    def _infer_num_layers_from_keys(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
        layers: list[int] = []
        for key in state_dict:
            if key.startswith(prefix) and "weight_ih_l" in key:
                layer_str = key.split("weight_ih_l", 1)[1].split(".", 1)[0]
                if layer_str.isdigit():
                    layers.append(int(layer_str))
        return (max(layers) + 1) if layers else 2

    @staticmethod
    def _infer_tft_num_heads(state_dict: dict[str, torch.Tensor]) -> int:
        heads: set[int] = set()
        marker = "attention.q_layers."
        for key in state_dict:
            if key.startswith(marker):
                suffix = key[len(marker) :]
                head_idx = suffix.split(".", 1)[0]
                if head_idx.isdigit():
                    heads.add(int(head_idx))
        return len(heads) if heads else 4

    @staticmethod
    def _default_quantile_levels(num_quantiles: int) -> tuple[float, ...]:
        if num_quantiles == 1:
            return (0.5,)
        if num_quantiles == 2:
            return (0.5, 0.95)
        if num_quantiles == 3:
            return (0.1, 0.5, 0.9)
        return tuple(float(q) for q in np.linspace(0.1, 0.9, num_quantiles))

    def _build_model(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.nn.Module, tuple[float, ...]]:
        if self.model_type in {"lstm", "deepar"}:
            hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
            input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
            num_layers = self._infer_num_layers_from_keys(state_dict, "lstm")
        else:
            hidden_size = int(state_dict["encoder.weight_hh_l0"].shape[1])
            input_size = int(state_dict["past_vsn.var_projs.0.weight"].shape[1])
            num_layers = self._infer_num_layers_from_keys(state_dict, "encoder")

        output_quantiles: tuple[float, ...]
        if self.model_type == "lstm":
            out_dim = int(state_dict["head.weight"].shape[0])
            if out_dim % self.forecast_length != 0:
                raise ValueError(
                    "LSTM checkpoint output dim is not divisible by forecast_length: "
                    f"out_dim={out_dim}, forecast_length={self.forecast_length}."
                )
            num_quantiles = max(1, out_dim // self.forecast_length)
            if len(self.input_quantiles) == num_quantiles:
                output_quantiles = self.input_quantiles
            else:
                output_quantiles = self._default_quantile_levels(num_quantiles)
            model = LSTMForecast(
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                quantiles=output_quantiles,
            )
        elif self.model_type == "deepar":
            output_quantiles = self.input_quantiles
            if "item_embedding.weight" in state_dict:
                feature_payload = self.model_config.get("deepar_feature_spec") or {}
                inferred_embedding_dim = int(state_dict["item_embedding.weight"].shape[1])
                inferred_time_features = max(1, input_size - 1 - inferred_embedding_dim)
                num_time_features = len(feature_payload.get("feature_names", ())) or inferred_time_features
                model = DeepARForecast(
                    context_length=self.context_length,
                    forecast_length=self.forecast_length,
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_time_features=max(1, int(num_time_features)),
                    num_items=int(self.model_config.get("num_items", state_dict["item_embedding.weight"].shape[0])),
                    item_embedding_dim=int(self.model_config.get("deepar_item_embedding_dim", inferred_embedding_dim)),
                    likelihood=str(self.model_config.get("deepar_likelihood", "gaussian")),
                )
            else:
                model = LegacyDeepARForecast(
                    context_length=self.context_length,
                    forecast_length=self.forecast_length,
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )
        else:
            num_quantiles = int(state_dict["output_layer.weight"].shape[0])
            if len(self.input_quantiles) == num_quantiles:
                output_quantiles = self.input_quantiles
            else:
                output_quantiles = self._default_quantile_levels(num_quantiles)
            tft_hidden_size = int(self.model_config.get("tft_hidden_size", hidden_size))
            tft_num_heads = int(
                self.model_config.get("tft_num_heads", self._infer_tft_num_heads(state_dict))
            )
            tft_num_lstm_layers = int(self.model_config.get("tft_num_lstm_layers", num_layers))
            num_static_categorical = int(self.model_config.get("tft_num_static_categorical", 0))
            static_cardinalities = tuple(
                int(value)
                for value in self.model_config.get("tft_static_categorical_cardinalities", [])
            )
            num_static_continuous = int(self.model_config.get("tft_num_static_continuous", 0))
            num_past_features = int(self.model_config.get("tft_num_past_features", 1))
            num_future_features = int(
                self.model_config.get(
                    "tft_num_future_features",
                    self.model_config.get("tft_num_known_features", 0),
                )
            )
            model = TFTForecast(
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                hidden_size=tft_hidden_size,
                num_lstm_layers=tft_num_lstm_layers,
                num_heads=tft_num_heads,
                quantiles=output_quantiles,
                num_static_categorical=num_static_categorical,
                static_categorical_cardinalities=static_cardinalities,
                num_static_continuous=num_static_continuous,
                num_past_features=num_past_features,
                num_future_features=num_future_features,
            )

        model.load_state_dict(state_dict)
        return model, output_quantiles

class Chronos2ZeroShotForecaster(Forecaster):
    """Adapter for Chronos2 zero-shot probabilistic forecasting."""

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str | torch.device = AUTO_DEVICE,
        num_samples: int = 100,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "chronos-forecasting is required for Chronos2 zero-shot inference."
            ) from exc

        self.device = resolve_torch_device(device)
        # Kept only for backward compatibility with existing CLI args.
        # Chronos2 quantiles are extracted directly from model outputs.
        self.num_samples = int(num_samples)
        self.evaluation_batch_size = 32
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.upper_quantile = float(self.quantile_levels[-1])

        self.pipeline = Chronos2Pipeline.from_pretrained(model_id)
        self.pipeline.model = self.pipeline.model.to(self.device)
        self.pipeline.model.eval()

    @staticmethod
    def _extract_primary_array(forecast: Any) -> np.ndarray:
        if isinstance(forecast, list) and forecast and isinstance(forecast[0], torch.Tensor):
            arr = forecast[0].detach().cpu().numpy()
        elif isinstance(forecast, torch.Tensor):
            arr = forecast.detach().cpu().numpy()
        else:
            arr = np.asarray(forecast)
        return np.asarray(arr)

    def _extract_batch_quantile_matrices(
        self,
        forecast: Any,
        expected_num_quantiles: int | None,
    ) -> list[np.ndarray]:
        if isinstance(forecast, list):
            return [
                self._extract_quantile_matrix(
                    self._extract_primary_array(item),
                    expected_num_quantiles,
                )
                for item in forecast
            ]

        arr = self._extract_primary_array(forecast)
        if arr.ndim == 2:
            return [self._extract_quantile_matrix(arr, expected_num_quantiles)]
        if arr.ndim != 3:
            raise ValueError("Expected Chronos2 batch output with 2D/3D tensor.")
        return [
            self._extract_quantile_matrix(sample, expected_num_quantiles)
            for sample in arr
        ]

    @staticmethod
    def _resolve_quantile_indices(
        quantile_levels: Sequence[float] | None,
        upper_quantile: float = 0.99,
    ) -> tuple[int, int]:
        """
        Resolve indices for q50 and the configured upper quantile.

        Chronos-2's wrapper exposes 21 native quantiles
        ``[0.01, 0.05, 0.10, ..., 0.95, 0.99]``; we pick the closest match.
        Falls back to (10, last index) when the wrapper does not expose levels.
        """
        if quantile_levels is None:
            return 10, 20
        q = np.asarray(list(quantile_levels), dtype=float)
        if q.ndim != 1 or q.size < 2:
            return 10, q.size - 1
        idx_50 = int(np.argmin(np.abs(q - 0.50)))
        idx_upper = int(np.argmin(np.abs(q - float(upper_quantile))))
        return idx_50, idx_upper

    @staticmethod
    def _extract_quantile_matrix(
        raw_preds: np.ndarray,
        expected_num_quantiles: int | None,
    ) -> np.ndarray:
        """
        Normalize Chronos2 output to shape (num_quantiles, horizon).

        Common wrapper output is (n_variates, num_quantiles, horizon) or
        (n_variates, horizon, num_quantiles). We keep first variate.
        """
        preds = np.asarray(raw_preds, dtype=float)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.ndim != 2:
            raise ValueError("Expected Chronos2 output with 2D/3D quantile tensor.")

        if expected_num_quantiles is not None:
            if preds.shape[0] == expected_num_quantiles:
                return preds
            if preds.shape[1] == expected_num_quantiles:
                return preds.T

        # Fallback to the known Chronos2 21-quantile layout.
        if preds.shape[0] == 21:
            return preds
        if preds.shape[1] == 21:
            return preds.T

        raise ValueError(
            f"Cannot infer quantile axis from Chronos2 output shape {preds.shape}. "
            "Expected one dimension to equal the quantile count."
        )

    def predict_quantiles(
        self,
        history: np.ndarray | list[float],
        horizon: int,
    ) -> ProbabilisticForecast:
        batch = self.predict_quantiles_batch(
            np.asarray(history, dtype=float).reshape(1, -1),
            horizon,
        )
        return ProbabilisticForecast(
            y_pred_median=batch.y_pred_median[0],
            y_pred_95=batch.y_pred_95[0],
        )

    def predict_quantiles_batch(
        self,
        histories: np.ndarray | Sequence[np.ndarray] | Sequence[list[float]],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecastBatch:
        histories_arr = _normalize_history_batch(histories)
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        contexts = [
            torch.tensor(history, dtype=torch.float32)
            for history in histories_arr
        ]
        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=contexts,
                prediction_length=horizon,
            )

        quantile_levels = getattr(self.pipeline, "quantiles", None)
        expected_num_quantiles = (
            len(quantile_levels)
            if quantile_levels is not None
            else None
        )
        idx_50, idx_upper = self._resolve_quantile_indices(
            quantile_levels, upper_quantile=self.upper_quantile
        )
        matrices = self._extract_batch_quantile_matrices(forecast, expected_num_quantiles)
        y50 = np.stack(
            [np.asarray(matrix[idx_50][:horizon], dtype=float) for matrix in matrices],
            axis=0,
        )
        y95 = np.stack(
            [np.asarray(matrix[idx_upper][:horizon], dtype=float) for matrix in matrices],
            axis=0,
        )
        y95 = np.maximum(y95, y50)
        return ProbabilisticForecastBatch(y_pred_median=y50, y_pred_95=y95)


class SeasonalNaiveForecaster(Forecaster):
    """Daily seasonal-naive baseline with a history-derived upper-quantile margin.

    The upper quantile target is configurable via ``quantile_levels[-1]``
    (default 0.99) and is used to compute the margin from seasonal residuals.
    """

    def __init__(
        self,
        season_length: int = DEFAULT_SEASONAL_NAIVE_PERIOD,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    ) -> None:
        season_length = int(season_length)
        if season_length <= 0:
            raise ValueError("season_length must be positive.")
        self.season_length = season_length
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        if not self.quantile_levels:
            raise ValueError("quantile_levels must contain at least one value.")
        self.upper_quantile = float(self.quantile_levels[-1])
        self.evaluation_batch_size = 1024

    def _estimate_upper_margin(self, history: np.ndarray) -> float:
        if history.size > self.season_length:
            residuals = history[self.season_length :] - history[: -self.season_length]
        elif history.size > 1:
            residuals = np.diff(history)
        else:
            return 0.0

        residuals = np.asarray(residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            return 0.0
        return float(max(np.quantile(residuals, self.upper_quantile), 0.0))

    def predict_quantiles(
        self,
        history: np.ndarray | list[float],
        horizon: int,
    ) -> ProbabilisticForecast:
        batch = self.predict_quantiles_batch(
            np.asarray(history, dtype=float).reshape(1, -1),
            horizon,
        )
        return ProbabilisticForecast(
            y_pred_median=batch.y_pred_median[0],
            y_pred_95=batch.y_pred_95[0],
        )

    def predict_quantiles_batch(
        self,
        histories: np.ndarray | Sequence[np.ndarray] | Sequence[list[float]],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecastBatch:
        histories_arr = _normalize_history_batch(histories)
        if histories_arr.size == 0 or histories_arr.shape[1] == 0:
            raise ValueError("History must contain at least one value.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        if histories_arr.shape[1] < self.season_length:
            y50 = np.repeat(histories_arr[:, -1:], int(horizon), axis=1).astype(float)
        else:
            extended = histories_arr.astype(float)
            y50_values: list[np.ndarray] = []
            for _ in range(int(horizon)):
                pred = extended[:, -self.season_length]
                y50_values.append(pred[:, None])
                extended = np.concatenate([extended, pred[:, None]], axis=1)
            y50 = np.concatenate(y50_values, axis=1).astype(float)

        margin = np.asarray(
            [self._estimate_upper_margin(history) for history in histories_arr],
            dtype=float,
        )[:, None]
        y95 = np.maximum(y50 + margin, y50)
        return ProbabilisticForecastBatch(y_pred_median=y50, y_pred_95=y95)


class SarimaForecaster(Forecaster):
    """SARIMAX(1,0,1)x(1,0,1,s) with Gaussian-interval upper-quantile bounds.

    Fits per-window on the provided history (no global training stage). The
    upper quantile is set from ``quantile_levels[-1]`` (default 0.99) and the
    interval bound is derived from ``predicted_mean + z_q * sqrt(var)``.
    """

    def __init__(
        self,
        season_length: int = DEFAULT_SEASONAL_NAIVE_PERIOD,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int] | None = None,
    ) -> None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from scipy.stats import norm
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "statsmodels and scipy are required for SarimaForecaster."
            ) from exc

        self._SARIMAX = SARIMAX
        self._norm = norm
        self.season_length = int(season_length)
        self.order = tuple(int(x) for x in order)
        self.seasonal_order = (
            tuple(int(x) for x in seasonal_order)
            if seasonal_order is not None
            else (1, 0, 1, self.season_length)
        )
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.upper_quantile = float(self.quantile_levels[-1])
        # Per-window L-BFGS fit is expensive (state dim grows with seasonal
        # period). Speed knobs below cut a 144-period fit substantially:
        #   simple_differencing=True  -> drop the seasonal lags from the state
        #   maxiter=20                -> mobile traffic fits saturate by ~15
        # Parallelism over windows is exposed via predict_quantiles_batch.
        # (concentrate_scale is intentionally NOT enabled: it interacts poorly
        # with simple_differencing on short clean histories and yields NaN
        # var_pred_mean, which collapses the upper quantile to the fallback.)
        self._fit_kwargs = {"disp": False, "maxiter": 20, "method": "lbfgs"}
        self._sarimax_kwargs = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "enforce_stationarity": False,
            "enforce_invertibility": False,
            "simple_differencing": True,
        }
        self._n_jobs = -1
        self.evaluation_batch_size = 32

    def _fit_one(
        self,
        history_arr: np.ndarray,
        horizon: int,
        z_upper: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Residual-std Gaussian band with sqrt(h) growth. statsmodels'
        # analytical var_pred_mean accumulates unboundedly when AR roots
        # are not constrained (enforce_stationarity=False), producing
        # upper bounds 1000x above the median; residuals are stable.
        try:
            model = self._SARIMAX(history_arr, **self._sarimax_kwargs).fit(
                **self._fit_kwargs
            )
            forecast = model.get_forecast(steps=int(horizon))
            mean = np.asarray(forecast.predicted_mean, dtype=float).reshape(-1)
            resid = np.asarray(model.resid, dtype=float).reshape(-1)
            sigma = float(np.nanstd(resid)) if resid.size > 0 else 0.0
            steps = np.arange(1, int(horizon) + 1, dtype=float)
            y50 = mean
            y_upper = mean + z_upper * sigma * np.sqrt(steps)
        except Exception:
            last = float(history_arr[-1])
            std_hist = (
                float(np.std(np.diff(history_arr))) if history_arr.size > 1 else 0.0
            )
            steps = np.arange(1, int(horizon) + 1, dtype=float)
            y50 = np.full(int(horizon), last, dtype=float)
            y_upper = y50 + z_upper * std_hist * np.sqrt(steps)
        y_upper = np.maximum(y_upper, y50)
        return y50.astype(float), y_upper.astype(float)

    def predict_quantiles(
        self,
        history: np.ndarray | list[float],
        horizon: int,
    ) -> ProbabilisticForecast:
        history_arr = np.asarray(history, dtype=float).reshape(-1)
        if history_arr.size == 0:
            raise ValueError("History must contain at least one value.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        z_upper = float(self._norm.ppf(self.upper_quantile))
        y50, y_upper = self._fit_one(history_arr, int(horizon), z_upper)
        return ProbabilisticForecast(y_pred_median=y50, y_pred_95=y_upper)

    def predict_quantiles_batch(
        self,
        histories: np.ndarray | Sequence[np.ndarray] | Sequence[list[float]],
        horizon: int,
        **kwargs: Any,
    ) -> ProbabilisticForecastBatch:
        histories_arr = _normalize_history_batch(histories)
        z_upper = float(self._norm.ppf(self.upper_quantile))
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self._n_jobs, prefer="processes")(
                delayed(self._fit_one)(h, int(horizon), z_upper)
                for h in histories_arr
            )
        except ImportError:
            results = [
                self._fit_one(h, int(horizon), z_upper) for h in histories_arr
            ]
        y50_rows = [r[0] for r in results]
        y_upper_rows = [r[1] for r in results]
        return ProbabilisticForecastBatch(
            y_pred_median=np.stack(y50_rows, axis=0),
            y_pred_95=np.stack(y_upper_rows, axis=0),
        )


class HybridForecastingChangePointPipeline:
    """Runs q50/q95 forecast -> first change-point -> safe ceiling."""

    def __init__(
        self,
        forecaster: Forecaster,
        change_detector: ChangePointDetector | None = None,
        detection_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.change_detector = change_detector or RupturesPeltDetector()
        self.detection_kwargs = detection_kwargs or {}

    def run(self, history: np.ndarray | list[float], horizon: int) -> PipelineOutput:
        forecast = self.forecaster.predict_quantiles(history=history, horizon=horizon)
        y50 = np.asarray(forecast.y_pred_median, dtype=float).reshape(-1)
        y95 = np.asarray(forecast.y_pred_95, dtype=float).reshape(-1)

        tau_pred = self.change_detector.detect_first_change_point(
            y50,
            **self.detection_kwargs,
        )

        if tau_pred is None:
            stationary_95 = y95
        else:
            end = max(1, min(int(tau_pred), y95.size))
            stationary_95 = y95[:end]

        safe_ceiling = float(np.max(stationary_95))

        return PipelineOutput(
            y_pred_median=y50,
            y_pred_95=y95,
            tau_pred=tau_pred,
            safe_ceiling=safe_ceiling,
            stationary_pred_95=np.asarray(stationary_95, dtype=float),
        )


def build_forecaster(
    model_name: str,
    *,
    checkpoint_path: str | Path | None = None,
    context_length: int = 128,
    forecast_length: int | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    residual_sigma: float | None = None,
    enable_residual_fallback: bool = True,
    chronos_model_id: str = "amazon/chronos-2",
    chronos_num_samples: int = 100,
    seasonal_period: int = DEFAULT_SEASONAL_NAIVE_PERIOD,
    deepar_num_samples: int | None = None,
    device: str | torch.device = AUTO_DEVICE,
) -> Forecaster:
    """Factory for supported forecasters."""
    name = model_name.lower()
    if name in {"lstm", "deepar", "tft"}:
        if checkpoint_path is None:
            checkpoint_path = Path("results/experiments") / f"{DEFAULT_DATASET_PATH.stem}_ctx48_h48" / "q50_q95_seed42" / "checkpoints" / f"{name}_{DEFAULT_DATASET_PATH.stem}_best.pt"
        return TorchCheckpointForecaster(
            model_type=name,
            checkpoint_path=checkpoint_path,
            context_length=context_length,
            forecast_length=forecast_length,
            quantiles=quantiles,
            residual_sigma=residual_sigma,
            enable_residual_fallback=enable_residual_fallback,
            deepar_num_samples=deepar_num_samples,
            device=device,
        )
    if name in {"chronos2", "chronos-2", "chronos2_zero_shot", "chronos2-zero-shot"}:
        return Chronos2ZeroShotForecaster(
            model_id=chronos_model_id,
            device=device,
            num_samples=chronos_num_samples,
            quantile_levels=quantiles,
        )
    if name in {"seasonal_naive", "seasonal-naive", "seasonalnaive"}:
        return SeasonalNaiveForecaster(
            season_length=seasonal_period,
            quantile_levels=quantiles,
        )
    if name in {"sarima", "sarimax"}:
        return SarimaForecaster(
            season_length=seasonal_period,
            quantile_levels=quantiles,
        )

    raise ValueError(
        "Unsupported model_name. Expected one of: "
        "lstm, deepar, tft, chronos2, seasonal_naive, sarima."
    )


def build_model_forecaster(
    model_name: str,
    *,
    model: torch.nn.Module,
    context_length: int = 128,
    forecast_length: int | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    residual_sigma: float | None = None,
    enable_residual_fallback: bool = True,
    model_config: dict[str, Any] | None = None,
    deepar_num_samples: int | None = None,
    device: str | torch.device = AUTO_DEVICE,
) -> Forecaster:
    name = model_name.lower()
    if name not in {"lstm", "deepar", "tft"}:
        raise ValueError("build_model_forecaster only supports torch baseline models.")
    return TorchModelForecaster(
        model_type=name,
        model=model,
        context_length=context_length,
        forecast_length=forecast_length,
        quantiles=quantiles,
        residual_sigma=residual_sigma,
        enable_residual_fallback=enable_residual_fallback,
        model_config=model_config,
        deepar_num_samples=deepar_num_samples,
        device=device,
    )
