"""PyTorch models for univariate time-series forecasting."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3:
        return x
    raise ValueError("Expected input with shape (seq,), (batch, seq), or (batch, seq, channels).")


class LSTMForecast(nn.Module):
    """LSTM forecaster that outputs multiple quantiles."""

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        quantiles: Sequence[float] = (0.5, 0.95),
        scale_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.quantiles = tuple(float(q) for q in quantiles)
        self.scale_eps = float(scale_eps)
        if not self.quantiles:
            raise ValueError("quantiles must contain at least one entry.")
        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError("quantiles must be between 0 and 1 (exclusive).")
        self.num_quantiles = len(self.quantiles)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, forecast_length * self.num_quantiles)

    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().mean(dim=1, keepdim=True) + 1.0
        return torch.clamp(scale, min=self.scale_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        if x.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if x.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

        scale = self._compute_scale(x)
        normalized = x / scale
        output, _ = self.lstm(normalized)
        last_hidden = output[:, -1, :]
        preds = self.head(last_hidden)
        preds = preds.reshape(x.size(0), self.forecast_length, self.num_quantiles)
        return preds * scale


class DeepARForecast(nn.Module):
    """Paper-style DeepAR with item embeddings, time covariates, and sample rollout."""

    uses_targets = True
    batch_format = "deepar"

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_time_features: int = 3,
        num_items: int = 1,
        item_embedding_dim: int = 20,
        likelihood: str = "gaussian",
        scale_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_size != 1:
            raise ValueError("DeepARForecast currently supports input_size=1.")
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.num_time_features = int(num_time_features)
        self.num_items = int(num_items)
        self.item_embedding_dim = int(item_embedding_dim)
        self.likelihood = str(likelihood).strip().lower()
        if self.likelihood not in {"gaussian", "negative_binomial"}:
            raise ValueError("likelihood must be 'gaussian' or 'negative_binomial'.")
        self.scale_eps = float(scale_eps)

        self.item_embedding = nn.Embedding(self.num_items, self.item_embedding_dim)
        lstm_input_size = 1 + self.num_time_features + self.item_embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.param_proj = nn.Linear(hidden_size, 2)

    def _compute_scale(
        self,
        context: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observed_mask is None:
            observed_mask = torch.ones_like(context)
        denom = torch.clamp(observed_mask.sum(dim=1, keepdim=True), min=1.0)
        scale = (context.abs() * observed_mask).sum(dim=1, keepdim=True) / denom + 1.0
        return torch.clamp(scale, min=self.scale_eps)

    def _project_params(self, outputs: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
        if scale.dim() == 2 and outputs.dim() == 3:
            scale = scale.unsqueeze(1)
        params = self.param_proj(outputs)
        if self.likelihood == "gaussian":
            mean = params[..., :1] * scale
            sigma = F.softplus(params[..., 1:2]) * scale + self.scale_eps
            aux = sigma
        else:
            mean = F.softplus(params[..., :1]) * scale + self.scale_eps
            aux = F.softplus(params[..., 1:2]) + self.scale_eps
        return {
            "mean": mean.squeeze(-1),
            "aux": aux.squeeze(-1),
        }

    def _distribution(
        self,
        mean: torch.Tensor,
        aux: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.distributions.Distribution:
        if device is not None:
            mean = mean.to(device)
            aux = aux.to(device)
        if self.likelihood == "gaussian":
            return torch.distributions.Normal(mean, torch.clamp(aux, min=self.scale_eps))
        total_count = torch.clamp(1.0 / torch.clamp(aux, min=self.scale_eps), min=self.scale_eps)
        probs = total_count / (total_count + torch.clamp(mean, min=self.scale_eps))
        return torch.distributions.NegativeBinomial(total_count=total_count, probs=probs)

    def _build_step_inputs(
        self,
        prev_values: torch.Tensor,
        step_time_features: torch.Tensor,
        item_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if prev_values.dim() == 1:
            prev_feature = prev_values.unsqueeze(-1)
        elif prev_values.dim() == 2:
            prev_feature = prev_values.unsqueeze(-1)
        else:
            raise ValueError("prev_values must have shape (batch,) or (batch, seq).")
        return torch.cat([prev_feature, step_time_features, item_embedding], dim=-1)

    def forward_train(
        self,
        *,
        full_target: torch.Tensor,
        time_features: torch.Tensor,
        item_ids: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if full_target.dim() != 2:
            raise ValueError("full_target must have shape (batch, sequence_length).")
        if time_features.dim() != 3:
            raise ValueError("time_features must have shape (batch, sequence_length, num_features).")
        if full_target.size(1) != self.context_length + self.forecast_length:
            raise ValueError("full_target length does not match context_length + forecast_length.")
        if time_features.size(1) != full_target.size(1):
            raise ValueError("time_features sequence length does not match targets.")
        if time_features.size(2) != self.num_time_features:
            raise ValueError("time_features feature size does not match num_time_features.")

        context = full_target[:, : self.context_length]
        context_mask = None
        if observed_mask is not None:
            context_mask = observed_mask[:, : self.context_length]
        scale = self._compute_scale(context, observed_mask=context_mask)
        scaled = full_target / scale
        prev_values = torch.cat(
            [
                torch.zeros(
                    scaled.size(0),
                    1,
                    device=scaled.device,
                    dtype=scaled.dtype,
                ),
                scaled[:, :-1],
            ],
            dim=1,
        )
        item_embed = self.item_embedding(item_ids)
        repeated_item = item_embed.unsqueeze(1).expand(-1, full_target.size(1), -1)
        inputs = self._build_step_inputs(prev_values, time_features, repeated_item)
        outputs, _ = self.lstm(inputs)
        projected = self._project_params(outputs, scale)
        return {
            "mean": projected["mean"],
            "aux": projected["aux"],
            "scale": scale.squeeze(-1),
        }

    def sample_forecasts(
        self,
        *,
        context: torch.Tensor,
        context_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        item_ids: torch.Tensor,
        num_samples: int,
        random_seed: int,
    ) -> torch.Tensor:
        if context.dim() != 2:
            raise ValueError("context must have shape (batch, context_length).")
        if context_time_features.dim() != 3 or future_time_features.dim() != 3:
            raise ValueError("time feature tensors must have shape (batch, sequence, num_features).")
        if context.size(1) != self.context_length:
            raise ValueError("context length does not match model context_length.")
        if future_time_features.size(1) != self.forecast_length:
            raise ValueError("future_time_features length does not match forecast_length.")

        if self.likelihood == "gaussian":
            return self._sample_gaussian_forecasts_vectorized(
                context=context,
                context_time_features=context_time_features,
                future_time_features=future_time_features,
                item_ids=item_ids,
                num_samples=num_samples,
                random_seed=random_seed,
            )

        return self._sample_forecasts_rowwise(
            context=context,
            context_time_features=context_time_features,
            future_time_features=future_time_features,
            item_ids=item_ids,
            num_samples=num_samples,
            random_seed=random_seed,
        )

    def _sample_forecasts_rowwise(
        self,
        *,
        context: torch.Tensor,
        context_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        item_ids: torch.Tensor,
        num_samples: int,
        random_seed: int,
    ) -> torch.Tensor:
        """Fallback sampler for non-Gaussian likelihoods."""

        batch_size = context.size(0)
        scale = self._compute_scale(context)
        scaled_context = context / scale
        item_embed = self.item_embedding(item_ids)
        repeated_item = item_embed.unsqueeze(1).expand(-1, self.context_length, -1)
        prev_context = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    device=context.device,
                    dtype=context.dtype,
                ),
                scaled_context[:, :-1],
            ],
            dim=1,
        )
        encode_inputs = self._build_step_inputs(prev_context, context_time_features, repeated_item)
        _, state = self.lstm(encode_inputs)

        hidden, cell = state
        sample_rows: list[torch.Tensor] = []
        # NegativeBinomial.sample() uses _standard_gamma, not implemented on MPS.
        # Sample on CPU and move results back to the original device.
        mps_nb_fallback = (
            self.likelihood == "negative_binomial"
            and str(context.device).startswith("mps")
        )
        sampling_device = torch.device("cpu") if mps_nb_fallback else context.device

        for row_idx in range(batch_size):
            row_hidden = hidden[:, row_idx : row_idx + 1, :].repeat_interleave(num_samples, dim=1)
            row_cell = cell[:, row_idx : row_idx + 1, :].repeat_interleave(num_samples, dim=1)
            row_scale = scale[row_idx : row_idx + 1].repeat_interleave(num_samples, dim=0)
            row_item = item_embed[row_idx : row_idx + 1].repeat_interleave(num_samples, dim=0)
            row_future_features = future_time_features[row_idx : row_idx + 1].repeat_interleave(num_samples, dim=0)
            prev = scaled_context[row_idx, -1].repeat(num_samples)
            row_draws: list[torch.Tensor] = []

            with torch.random.fork_rng():
                torch.manual_seed(int(random_seed) + row_idx)
                for step_idx in range(self.forecast_length):
                    step_features = row_future_features[:, step_idx, :]
                    step_input = self._build_step_inputs(prev, step_features, row_item).unsqueeze(1)
                    output, (row_hidden, row_cell) = self.lstm(step_input, (row_hidden, row_cell))
                    projected = self._project_params(output[:, -1:, :], row_scale)
                    mean = projected["mean"].squeeze(1)
                    aux = projected["aux"].squeeze(1)
                    dist = self._distribution(mean, aux, device=sampling_device)
                    sampled = dist.sample().to(context.device, dtype=context.dtype)
                    row_draws.append(sampled)
                    prev = sampled / row_scale.squeeze(-1)

            sample_rows.append(torch.stack(row_draws, dim=-1))

        return torch.stack(sample_rows, dim=0)

    def _sample_gaussian_forecasts_vectorized(
        self,
        *,
        context: torch.Tensor,
        context_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        item_ids: torch.Tensor,
        num_samples: int,
        random_seed: int,
    ) -> torch.Tensor:
        batch_size = context.size(0)
        scale = self._compute_scale(context)
        scaled_context = context / scale
        item_embed = self.item_embedding(item_ids)
        repeated_item = item_embed.unsqueeze(1).expand(-1, self.context_length, -1)
        prev_context = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    device=context.device,
                    dtype=context.dtype,
                ),
                scaled_context[:, :-1],
            ],
            dim=1,
        )
        encode_inputs = self._build_step_inputs(prev_context, context_time_features, repeated_item)
        _, state = self.lstm(encode_inputs)

        hidden, cell = state
        hidden = hidden.repeat_interleave(num_samples, dim=1)
        cell = cell.repeat_interleave(num_samples, dim=1)
        scale_expanded = scale.repeat_interleave(num_samples, dim=0)
        future_features_expanded = future_time_features.repeat_interleave(num_samples, dim=0)
        item_expanded = item_embed.repeat_interleave(num_samples, dim=0)
        prev = scaled_context[:, -1].repeat_interleave(num_samples)

        cpu_gen = torch.Generator()
        cpu_gen.manual_seed(int(random_seed))
        all_noise = torch.randn(
            self.forecast_length,
            batch_size,
            num_samples,
            generator=cpu_gen,
            dtype=context.dtype,
        ).to(context.device)

        step_draws: list[torch.Tensor] = []
        for step_idx in range(self.forecast_length):
            step_features = future_features_expanded[:, step_idx, :]
            step_input = self._build_step_inputs(prev, step_features, item_expanded).unsqueeze(1)
            output, (hidden, cell) = self.lstm(step_input, (hidden, cell))
            projected = self._project_params(output[:, -1:, :], scale_expanded)
            mean = projected["mean"].squeeze(1).reshape(batch_size, num_samples)
            sigma = projected["aux"].squeeze(1).reshape(batch_size, num_samples)
            sampled_step = mean + sigma * all_noise[step_idx]
            step_draws.append(sampled_step)
            prev = sampled_step.reshape(batch_size * num_samples) / scale_expanded.squeeze(-1)

        return torch.stack(step_draws, dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        targets: torch.Tensor | None = None,
        time_features: torch.Tensor | None = None,
        item_ids: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
        sample: bool = False,
        num_samples: int = 100,
        random_seed: int = 42,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        context = context.squeeze(-1) if context.dim() == 3 else context
        if targets is not None:
            targets = targets.squeeze(-1) if targets.dim() == 3 else targets
            if time_features is None:
                zeros = torch.zeros(
                    context.size(0),
                    self.context_length + self.forecast_length,
                    self.num_time_features,
                    device=context.device,
                    dtype=context.dtype,
                )
                time_features = zeros
            if item_ids is None:
                item_ids = torch.zeros(context.size(0), dtype=torch.long, device=context.device)
            full_target = torch.cat([context, targets], dim=1)
            return self.forward_train(
                full_target=full_target,
                time_features=time_features,
                item_ids=item_ids,
                observed_mask=observed_mask,
            )
        if time_features is None:
            time_features = torch.zeros(
                context.size(0),
                self.context_length + self.forecast_length,
                self.num_time_features,
                device=context.device,
                dtype=context.dtype,
            )
        if item_ids is None:
            item_ids = torch.zeros(context.size(0), dtype=torch.long, device=context.device)
        if sample:
            draws = self.sample_forecasts(
                context=context,
                context_time_features=time_features[:, : self.context_length, :],
                future_time_features=time_features[:, self.context_length :, :],
                item_ids=item_ids,
                num_samples=num_samples,
                random_seed=random_seed,
            )
            q50 = torch.quantile(draws, 0.5, dim=1)
            q95 = torch.quantile(draws, 0.95, dim=1)
            return torch.stack([q50, q95], dim=-1)

        train_outputs = self.forward_train(
            full_target=torch.cat(
                [
                    context,
                    torch.zeros(
                        context.size(0),
                        self.forecast_length,
                        device=context.device,
                        dtype=context.dtype,
                    ),
                ],
                dim=1,
            ),
            time_features=time_features,
            item_ids=item_ids,
            observed_mask=observed_mask,
        )
        mean = train_outputs["mean"][:, -self.forecast_length :]
        aux = train_outputs["aux"][:, -self.forecast_length :]
        return torch.stack([mean, aux], dim=-1)


class LegacyDeepARForecast(nn.Module):
    """Compatibility wrapper for loading checkpoints from the old simplified DeepAR."""

    uses_targets = True

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        scale_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.scale_eps = scale_eps
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.param_proj = nn.Linear(hidden_size, 2)

    def _compute_scale(self, context: torch.Tensor) -> torch.Tensor:
        scale = context.abs().mean(dim=1, keepdim=True) + 1.0
        return torch.clamp(scale, min=self.scale_eps)

    def _params_from_outputs(self, outputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        params = self.param_proj(outputs)
        mu = params[..., :1] * scale
        sigma = F.softplus(params[..., 1:2]) * scale + self.scale_eps
        return torch.cat([mu, sigma], dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        targets: torch.Tensor | None = None,
        sample: bool = False,
    ) -> torch.Tensor:
        context = _ensure_3d(context)
        if targets is not None:
            targets = _ensure_3d(targets)
            scale = self._compute_scale(context)
            full = torch.cat([context, targets], dim=1) / scale
            zeros = torch.zeros(full.size(0), 1, full.size(2), device=full.device, dtype=full.dtype)
            inputs = torch.cat([zeros, full[:, :-1, :]], dim=1)
            outputs, _ = self.lstm(inputs)
            params = self._params_from_outputs(outputs, scale)
            start = self.context_length
            end = start + self.forecast_length
            return params[:, start:end, :]

        scale = self._compute_scale(context)
        scaled = context / scale
        zeros = torch.zeros(
            scaled.size(0), 1, scaled.size(2), device=scaled.device, dtype=scaled.dtype
        )
        inputs = torch.cat([zeros, scaled[:, :-1, :]], dim=1)
        _, (hidden, cell) = self.lstm(inputs)
        prev = scaled[:, -1, :]
        preds = []
        for _ in range(self.forecast_length):
            output, (hidden, cell) = self.lstm(prev.unsqueeze(1), (hidden, cell))
            params = self._params_from_outputs(output[:, -1:, :], scale)
            mu = params[..., 0:1]
            sigma = params[..., 1:2]
            preds.append(torch.cat([mu, sigma], dim=-1).squeeze(1))
            next_val = self._distribution(mu.squeeze(-1), sigma.squeeze(-1)).sample() if sample else mu
            prev = (next_val / scale).squeeze(-1)
        return torch.stack(preds, dim=1)

    def _distribution(self, mean: torch.Tensor, sigma: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Normal(mean, sigma)


class GatedLinearUnit(nn.Module):
    """Gated linear unit used in TFT."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        value, gate = out.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Gated residual network (GRN) block from TFT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_size = output_size or input_size
        self.context_size = context_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.context_proj = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None
            else None
        )
        self.hidden = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(output_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.norm = nn.LayerNorm(output_size)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x
        out = self.input_proj(x)
        if self.context_proj is not None and context is not None:
            ctx = context
            if ctx.dim() == 2 and out.dim() == 3:
                ctx = ctx.unsqueeze(1)
            out = out + self.context_proj(ctx)
        out = F.elu(out)
        out = self.hidden(self.dropout(out))
        out = self.gate(out)
        return self.norm(residual + out)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network from TFT."""

    def __init__(
        self,
        input_sizes: Sequence[int],
        hidden_size: int,
        context_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_vars = len(tuple(input_sizes))
        if num_vars <= 0:
            raise ValueError("num_vars must be positive.")
        self.num_vars = num_vars
        self.input_sizes = tuple(int(size) for size in input_sizes)
        self.var_projs = nn.ModuleList(
            [nn.Linear(size, hidden_size) for size in self.input_sizes]
        )
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout) for _ in range(num_vars)]
        )
        self.weight_grn = GatedResidualNetwork(
            num_vars * hidden_size,
            hidden_size,
            output_size=num_vars,
            context_size=context_size,
            dropout=dropout,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_time = True
            else:
                squeeze_time = False
            if x.size(-1) != self.num_vars:
                raise ValueError("Input feature count does not match num_vars.")
            inputs = [x[..., idx : idx + 1] for idx in range(self.num_vars)]
        else:
            inputs = list(x)
            if len(inputs) != self.num_vars:
                raise ValueError("Input variable count does not match num_vars.")
            squeeze_time = inputs[0].dim() == 2

        projected = []
        var_embeddings = []
        for idx, var in enumerate(inputs):
            emb = self.var_projs[idx](var)
            projected.append(emb)
            emb = self.var_grns[idx](emb)
            var_embeddings.append(emb)
        stacked = torch.stack(var_embeddings, dim=-2)
        flat = torch.cat(projected, dim=-1)
        weights = self.weight_grn(flat, context=context)
        weights = self.softmax(weights)
        combined = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        if squeeze_time:
            combined = combined.squeeze(1)
            weights = weights.squeeze(1)
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention from TFT."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scale = self.head_size**-0.5

        self.q_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_size, bias=False) for _ in range(num_heads)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_size, bias=False) for _ in range(num_heads)]
        )
        self.v_layer = nn.Linear(hidden_size, self.head_size, bias=False)
        self.out_proj = nn.Linear(self.head_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = self.v_layer(x)  # (batch, seq, head_size)
        # Stack all head projections, then compute attention in a single batched call.
        q = torch.stack([layer(x) for layer in self.q_layers], dim=1)  # (batch, heads, seq, head_size)
        k = torch.stack([layer(x) for layer in self.k_layers], dim=1)  # (batch, heads, seq, head_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq, seq)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))  # (1, 1, seq, seq) broadcast
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        # Shared V across heads; average over heads to match paper (Eq. 14-16).
        attn = torch.matmul(weights, values.unsqueeze(1)).mean(dim=1)  # (batch, seq, head_size)
        return self.out_proj(attn)


class TFTForecast(nn.Module):
    """Temporal Fusion Transformer for quantile forecasting."""

    batch_format = "tft"

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        hidden_size: int = 128,
        num_lstm_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        num_static_categorical: int = 0,
        static_categorical_cardinalities: Sequence[int] = (),
        num_static_continuous: int = 0,
        num_past_features: int = 1,
        num_future_features: int = 0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.quantiles = tuple(float(q) for q in quantiles)
        if not self.quantiles:
            raise ValueError("quantiles must contain at least one entry.")
        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError("quantiles must be between 0 and 1 (exclusive).")
        self.num_quantiles = len(self.quantiles)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_lstm_layers = int(num_lstm_layers)
        self.num_static_categorical = int(num_static_categorical)
        self.static_categorical_cardinalities = tuple(
            int(value) for value in static_categorical_cardinalities
        )
        self.num_static_continuous = int(num_static_continuous)
        self.num_past_features = int(num_past_features)
        self.num_future_features = num_future_features
        self.input_size = self.num_past_features

        if self.num_static_categorical != len(self.static_categorical_cardinalities):
            raise ValueError(
                "num_static_categorical must match static_categorical_cardinalities length."
            )

        self.static_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, hidden_size)
                for cardinality in self.static_categorical_cardinalities
            ]
        )
        self.static_vsn = (
            VariableSelectionNetwork(
                [hidden_size] * self.num_static_categorical + [1] * self.num_static_continuous,
                hidden_size,
                dropout=dropout,
            )
            if (self.num_static_categorical + self.num_static_continuous) > 0
            else None
        )
        self.static_context = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if self.static_vsn is not None
            else None
        )
        self.static_enrichment = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if self.static_vsn is not None
            else None
        )
        self.static_state_h = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if self.static_vsn is not None
            else None
        )
        self.static_state_c = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if self.static_vsn is not None
            else None
        )

        self.past_vsn = VariableSelectionNetwork(
            [1] * self.num_past_features,
            hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.future_vsn = VariableSelectionNetwork(
            [1] * max(1, int(num_future_features)),
            hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=dropout if self.num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=dropout if self.num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.local_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.local_norm = nn.LayerNorm(hidden_size)
        self.enrichment_grn = GatedResidualNetwork(
            hidden_size,
            hidden_size,
            output_size=hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout=dropout
        )
        self.attention_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.position_grn = GatedResidualNetwork(
            hidden_size, hidden_size, output_size=hidden_size, dropout=dropout
        )
        self.position_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.position_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.num_quantiles)

    def _build_mask(self, length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
        )
        return mask.unsqueeze(0)

    def _repeat_state(self, state: torch.Tensor, batch_size: int) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() != 2:
            raise ValueError("State must have shape (batch, hidden).")
        if state.size(0) == 1 and batch_size > 1:
            state = state.expand(batch_size, -1)
        elif state.size(0) != batch_size:
            raise ValueError("State batch size does not match inputs.")
        return state.unsqueeze(0).repeat(self.encoder.num_layers, 1, 1)

    def _split_time_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.size(-1) <= 0:
            return [x.new_zeros(x.shape[:-1] + (1,))]
        return [x[..., idx : idx + 1] for idx in range(x.size(-1))]

    def _static_inputs(
        self,
        batch_size: int,
        *,
        static_categorical: torch.Tensor | None = None,
        static_continuous: torch.Tensor | None = None,
        device: torch.device,
    ) -> list[torch.Tensor]:
        inputs: list[torch.Tensor] = []
        if self.num_static_categorical > 0:
            if static_categorical is None:
                raise ValueError("static_categorical must be provided for TFT.")
            if static_categorical.dim() != 2 or static_categorical.size(1) != self.num_static_categorical:
                raise ValueError("static_categorical has incorrect shape.")
            for idx, embedding in enumerate(self.static_embeddings):
                inputs.append(embedding(static_categorical[:, idx]))
        if self.num_static_continuous > 0:
            if static_continuous is None:
                raise ValueError("static_continuous must be provided for TFT.")
            if static_continuous.dim() != 2 or static_continuous.size(1) != self.num_static_continuous:
                raise ValueError("static_continuous has incorrect shape.")
            for idx in range(self.num_static_continuous):
                inputs.append(static_continuous[:, idx : idx + 1])
        if not inputs and self.static_vsn is not None:
            raise ValueError("TFT static_vsn is configured but no static inputs were provided.")
        if not inputs and self.static_vsn is None:
            return [torch.zeros(batch_size, 1, device=device)]
        return inputs

    def forward(
        self,
        past_inputs: torch.Tensor,
        future_inputs: torch.Tensor | None = None,
        static_categorical: torch.Tensor | None = None,
        static_continuous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        past_inputs = _ensure_3d(past_inputs)
        if past_inputs.size(1) != self.context_length:
            raise ValueError("past_inputs length does not match context_length.")
        if past_inputs.size(2) != self.num_past_features:
            raise ValueError("past_inputs feature size does not match num_past_features.")

        # Instance normalization on the raw-value channel (index 0) only.
        # Time-feature channels (1+) are already bounded and must not be scaled.
        # Mirrors LSTMForecast._compute_scale for consistent scale-invariance.
        scale = past_inputs[:, :, :1].abs().mean(dim=1, keepdim=True) + 1.0  # (B, 1, 1)
        past_inputs = past_inputs.clone()
        past_inputs[:, :, :1] = past_inputs[:, :, :1] / scale

        batch_size = past_inputs.size(0)
        if future_inputs is None:
            future_inputs = past_inputs.new_zeros(
                batch_size,
                self.forecast_length,
                max(1, self.num_future_features),
            )
        future_inputs = _ensure_3d(future_inputs)
        if future_inputs.size(1) != self.forecast_length:
            raise ValueError("Future input length does not match forecast_length.")
        if future_inputs.size(2) != max(1, self.num_future_features):
            raise ValueError("Future input feature size does not match.")

        if self.static_vsn is not None:
            static_emb, _ = self.static_vsn(
                self._static_inputs(
                    batch_size,
                    static_categorical=static_categorical,
                    static_continuous=static_continuous,
                    device=past_inputs.device,
                )
            )
            static_context = self.static_context(static_emb)  # type: ignore[operator]
            static_enrichment = self.static_enrichment(static_emb)  # type: ignore[operator]
            init_h = self.static_state_h(static_emb)  # type: ignore[operator]
            init_c = self.static_state_c(static_emb)  # type: ignore[operator]
            h0 = self._repeat_state(init_h, batch_size)
            c0 = self._repeat_state(init_c, batch_size)
            init_state = (h0, c0)
        else:
            static_context = None
            static_enrichment = None
            init_state = None

        past_embed, _ = self.past_vsn(
            self._split_time_features(past_inputs),
            context=static_context,
        )
        future_embed, _ = self.future_vsn(
            self._split_time_features(future_inputs),
            context=static_context,
        )

        past_encoded, state = self.encoder(past_embed, init_state)
        future_encoded, _ = self.decoder(future_embed, state)

        selected_inputs = torch.cat([past_embed, future_embed], dim=1)
        temporal = torch.cat([past_encoded, future_encoded], dim=1)
        temporal = self.local_norm(selected_inputs + self.local_gate(temporal))
        if static_enrichment is not None:
            temporal = self.enrichment_grn(temporal, context=static_enrichment)
        else:
            temporal = self.enrichment_grn(temporal, context=None)

        mask = self._build_mask(temporal.size(1), temporal.device)
        attn_out = self.attention(temporal, mask=mask)
        attn_out = self.attention_norm(temporal + self.attention_gate(attn_out))
        position_out = self.position_grn(attn_out)
        position_out = self.position_norm(selected_inputs + self.position_gate(position_out))

        future_out = position_out[:, -self.forecast_length :, :]
        return self.output_layer(future_out) * scale
