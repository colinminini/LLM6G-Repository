"""Hybrid probabilistic forecasting + change-point pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.config import DEFAULT_DATASET_PATH, DEFAULT_QUANTILES, DEFAULT_SEASONAL_NAIVE_PERIOD
from src.device import AUTO_DEVICE, resolve_torch_device
from src.models import DeepARForecast, LSTMForecast, TFTForecast
from src.change_detection import ChangePointDetector, RupturesPeltDetector

Z95 = 1.6448536269514722


@dataclass
class ProbabilisticForecast:
    y_pred_median: np.ndarray
    y_pred_95: np.ndarray


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
    ) -> ProbabilisticForecast:
        """Return q50 and q95 trajectories."""


class TorchCheckpointForecaster(Forecaster):
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

        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self.forecast_length = self._resolve_forecast_length(state_dict, forecast_length)
        self.model, self.output_quantiles = self._build_model(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

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
            input_size = int(state_dict["past_vsn.var_embeds.0.weight"].shape[1])
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
            output_quantiles = (0.5, 0.95)
            model = DeepARForecast(
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        else:
            num_quantiles = int(state_dict["output_layer.weight"].shape[0])
            if len(self.input_quantiles) == num_quantiles:
                output_quantiles = self.input_quantiles
            else:
                output_quantiles = self._default_quantile_levels(num_quantiles)
            model = TFTForecast(
                context_length=self.context_length,
                forecast_length=self.forecast_length,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=self._infer_tft_num_heads(state_dict),
                quantiles=output_quantiles,
            )

        model.load_state_dict(state_dict)
        return model, output_quantiles

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

    def _predict_block(self, context: np.ndarray) -> ProbabilisticForecast:
        context_tensor = (
            torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        )

        with torch.no_grad():
            if self.model_type == "deepar":
                outputs = self.model(context_tensor, targets=None, sample=False)
                mu = outputs[0, :, 0].detach().cpu().numpy().astype(float)
                sigma = outputs[0, :, 1].detach().cpu().numpy().astype(float)
                y50 = mu
                y95 = mu + Z95 * sigma
            else:
                outputs = self.model(context_tensor)
                quantile_preds = outputs[0].detach().cpu().numpy().astype(float)
                y50 = self._extract_quantile_path(quantile_preds, self.output_quantiles, 0.5)
                max_trained_q = max(self.output_quantiles)
                if 0.95 <= max_trained_q + 1e-8:
                    y95 = self._extract_quantile_path(
                        quantile_preds,
                        self.output_quantiles,
                        0.95,
                    )
                elif self.enable_residual_fallback and self.residual_sigma is not None:
                    # Option B fallback when 0.95 is unavailable in checkpoint outputs.
                    y95 = y50 + Z95 * float(self.residual_sigma)
                else:
                    y95 = self._extract_quantile_path(
                        quantile_preds,
                        self.output_quantiles,
                        max_trained_q,
                    )

        y50 = np.asarray(y50, dtype=float).reshape(-1)
        y95 = np.asarray(y95, dtype=float).reshape(-1)
        y95 = np.maximum(y95, y50)

        if y50.size == 0 or y95.size == 0:
            raise ValueError("Model returned an empty forecast block.")
        if y50.size != y95.size:
            raise ValueError("q50 and q95 block sizes do not match.")

        return ProbabilisticForecast(y_pred_median=y50, y_pred_95=y95)

    def predict_quantiles(
        self,
        history: np.ndarray | list[float],
        horizon: int,
    ) -> ProbabilisticForecast:
        history_arr = np.asarray(history, dtype=float).reshape(-1)
        if history_arr.size < self.context_length:
            raise ValueError(
                f"History too short ({history_arr.size}) for context_length={self.context_length}."
            )
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        generated_50: list[np.ndarray] = []
        generated_95: list[np.ndarray] = []
        rolling_history = history_arr.copy()
        remaining = int(horizon)

        while remaining > 0:
            context = rolling_history[-self.context_length :]
            block = self._predict_block(context)

            take = min(remaining, int(block.y_pred_median.size))
            chunk_50 = block.y_pred_median[:take]
            chunk_95 = block.y_pred_95[:take]

            generated_50.append(chunk_50)
            generated_95.append(chunk_95)
            remaining -= take

            if remaining > 0:
                # Roll with central tendency trajectory for autoregressive extension.
                rolling_history = np.concatenate([rolling_history, chunk_50])

        y50 = np.concatenate(generated_50, axis=0)
        y95 = np.concatenate(generated_95, axis=0)
        y95 = np.maximum(y95, y50)
        return ProbabilisticForecast(y_pred_median=y50, y_pred_95=y95)


class Chronos2ZeroShotForecaster(Forecaster):
    """Adapter for Chronos2 zero-shot probabilistic forecasting."""

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str | torch.device = AUTO_DEVICE,
        num_samples: int = 100,
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

    @staticmethod
    def _resolve_quantile_indices(quantile_levels: Sequence[float] | None) -> tuple[int, int]:
        """
        Resolve q50/q95 indices from wrapper metadata when available.

        Falls back to Chronos2's 21-quantile convention:
        - q50 at index 10 (11th quantile)
        - q95 at index -2 (2nd-to-last quantile)
        """
        if quantile_levels is None:
            return 10, -2
        q = np.asarray(list(quantile_levels), dtype=float)
        if q.ndim != 1 or q.size < 2:
            return 10, -2
        idx_50 = int(np.argmin(np.abs(q - 0.50)))
        idx_95 = int(np.argmin(np.abs(q - 0.95)))
        return idx_50, idx_95

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
        history_arr = np.asarray(history, dtype=float).reshape(-1)
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        context = torch.tensor(history_arr, dtype=torch.float32)
        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=[context],
                prediction_length=horizon,
            )

        arr = self._extract_primary_array(forecast)
        quantile_levels = getattr(self.pipeline, "quantiles", None)
        expected_num_quantiles = (
            len(quantile_levels)
            if quantile_levels is not None
            else None
        )
        quantile_matrix = self._extract_quantile_matrix(arr, expected_num_quantiles)
        idx_50, idx_95 = self._resolve_quantile_indices(quantile_levels)
        y50 = quantile_matrix[idx_50]
        y95 = quantile_matrix[idx_95]

        y50 = np.asarray(y50[:horizon], dtype=float)
        y95 = np.asarray(y95[:horizon], dtype=float)
        y95 = np.maximum(y95, y50)
        return ProbabilisticForecast(y_pred_median=y50, y_pred_95=y95)


class SeasonalNaiveForecaster(Forecaster):
    """Daily seasonal-naive baseline with a history-derived q95 margin."""

    def __init__(self, season_length: int = DEFAULT_SEASONAL_NAIVE_PERIOD) -> None:
        season_length = int(season_length)
        if season_length <= 0:
            raise ValueError("season_length must be positive.")
        self.season_length = season_length

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
        return float(max(np.quantile(residuals, 0.95), 0.0))

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

        if history_arr.size < self.season_length:
            y50 = np.repeat(float(history_arr[-1]), int(horizon)).astype(float)
        else:
            extended = history_arr.astype(float).tolist()
            y50_values: list[float] = []
            for _ in range(int(horizon)):
                pred = float(extended[-self.season_length])
                y50_values.append(pred)
                extended.append(pred)
            y50 = np.asarray(y50_values, dtype=float)

        margin = self._estimate_upper_margin(history_arr)
        y95 = np.maximum(y50 + margin, y50)
        return ProbabilisticForecast(y_pred_median=y50, y_pred_95=y95)


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
            device=device,
        )
    if name in {"chronos2", "chronos-2", "chronos2_zero_shot", "chronos2-zero-shot"}:
        return Chronos2ZeroShotForecaster(
            model_id=chronos_model_id,
            device=device,
            num_samples=chronos_num_samples,
        )
    if name in {"seasonal_naive", "seasonal-naive", "seasonalnaive"}:
        return SeasonalNaiveForecaster(season_length=seasonal_period)

    raise ValueError(
        "Unsupported model_name. Expected one of: lstm, deepar, tft, chronos2, seasonal_naive."
    )
