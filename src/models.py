"""
PyTorch models for univariate time-series forecasting.

The model expects input shaped as (batch, seq_len) or (batch, seq_len, 1).
"""

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
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.quantiles = tuple(float(q) for q in quantiles)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_3d(x)
        if x.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if x.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        preds = self.head(last_hidden)
        return preds.reshape(x.size(0), self.forecast_length, self.num_quantiles)


class DeepARForecast(nn.Module):
    """Autoregressive LSTM forecaster that outputs Gaussian parameters."""

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
        if input_size != 1:
            raise ValueError("DeepARForecast currently supports input_size=1.")
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

    def _params_from_outputs(
        self, outputs: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        params = self.param_proj(outputs)
        mu = params[..., :1] * scale
        sigma = F.softplus(params[..., 1:2]) * scale + self.scale_eps
        return torch.cat([mu, sigma], dim=-1)

    def _forward_teacher(self, context: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        context = _ensure_3d(context)
        targets = _ensure_3d(targets)
        if context.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if targets.size(1) != self.forecast_length:
            raise ValueError("Target sequence length does not match forecast_length.")
        if context.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

        scale = self._compute_scale(context)
        full = torch.cat([context, targets], dim=1)
        full = full / scale
        zeros = torch.zeros(full.size(0), 1, full.size(2), device=full.device, dtype=full.dtype)
        inputs = torch.cat([zeros, full[:, :-1, :]], dim=1)
        outputs, _ = self.lstm(inputs)
        params = self._params_from_outputs(outputs, scale)
        start = self.context_length
        end = start + self.forecast_length
        return params[:, start:end, :]

    def _forward_autoregressive(self, context: torch.Tensor, sample: bool) -> torch.Tensor:
        context = _ensure_3d(context)
        if context.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if context.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

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
            if sample:
                dist = torch.distributions.Normal(mu, sigma)
                next_val = dist.sample()
            else:
                next_val = mu
            prev = (next_val / scale).squeeze(-1)
        return torch.stack(preds, dim=1)

    def forward(
        self,
        context: torch.Tensor,
        targets: torch.Tensor | None = None,
        sample: bool = False,
    ) -> torch.Tensor:
        if targets is not None:
            return self._forward_teacher(context, targets)
        return self._forward_autoregressive(context, sample=sample)


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
        num_vars: int,
        hidden_size: int,
        context_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_vars <= 0:
            raise ValueError("num_vars must be positive.")
        self.num_vars = num_vars
        self.var_embeds = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_vars)]
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
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_time = True
        else:
            squeeze_time = False
        if x.size(-1) != self.num_vars:
            raise ValueError("Input feature count does not match num_vars.")

        var_embeddings = []
        for idx in range(self.num_vars):
            var = x[..., idx : idx + 1]
            emb = self.var_embeds[idx](var)
            emb = self.var_grns[idx](emb)
            var_embeddings.append(emb)
        stacked = torch.stack(var_embeddings, dim=-2)
        flat = stacked.reshape(stacked.size(0), stacked.size(1), -1)
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
        values = self.v_layer(x)
        head_outputs = []
        for q_layer, k_layer in zip(self.q_layers, self.k_layers):
            q = q_layer(x)
            k = k_layer(x)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            head_outputs.append(torch.matmul(weights, values))
        stacked = torch.stack(head_outputs, dim=0)
        attn = stacked.mean(dim=0)
        return self.out_proj(attn)


class TFTForecast(nn.Module):
    """Temporal Fusion Transformer for quantile forecasting."""

    def __init__(
        self,
        context_length: int = 128,
        forecast_length: int = 1,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        num_future_features: int = 0,
        num_static_features: int = 0,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.input_size = input_size
        self.quantiles = tuple(float(q) for q in quantiles)
        if not self.quantiles:
            raise ValueError("quantiles must contain at least one entry.")
        for q in self.quantiles:
            if q <= 0.0 or q >= 1.0:
                raise ValueError("quantiles must be between 0 and 1 (exclusive).")
        self.num_quantiles = len(self.quantiles)
        self.num_future_features = num_future_features
        self.num_static_features = num_static_features
        self.use_dummy_future = num_future_features == 0
        future_features = num_future_features if num_future_features > 0 else 1

        self.static_vsn = (
            VariableSelectionNetwork(num_static_features, hidden_size, dropout=dropout)
            if num_static_features > 0
            else None
        )
        self.static_context = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if num_static_features > 0
            else None
        )
        self.static_enrichment = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if num_static_features > 0
            else None
        )
        self.static_state_h = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if num_static_features > 0
            else None
        )
        self.static_state_c = (
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            if num_static_features > 0
            else None
        )

        self.past_vsn = VariableSelectionNetwork(
            input_size, hidden_size, context_size=hidden_size, dropout=dropout
        )
        self.future_vsn = VariableSelectionNetwork(
            future_features, hidden_size, context_size=hidden_size, dropout=dropout
        )

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
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

    def forward(
        self,
        context: torch.Tensor,
        future_known: torch.Tensor | None = None,
        static_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = _ensure_3d(context)
        if context.size(1) != self.context_length:
            raise ValueError("Input sequence length does not match context_length.")
        if context.size(2) != self.input_size:
            raise ValueError("Input feature size does not match input_size.")

        batch_size = context.size(0)
        if future_known is None:
            if self.use_dummy_future:
                future_known = context.new_zeros(batch_size, self.forecast_length, 1)
            else:
                future_known = context.new_zeros(
                    batch_size, self.forecast_length, self.num_future_features
                )
        future_known = _ensure_3d(future_known)
        if future_known.size(1) != self.forecast_length:
            raise ValueError("Future input length does not match forecast_length.")
        if future_known.size(2) != (
            self.num_future_features if not self.use_dummy_future else 1
        ):
            raise ValueError("Future input feature size does not match.")

        if self.num_static_features > 0:
            if static_inputs is None:
                raise ValueError("static_inputs must be provided when num_static_features > 0.")
            if static_inputs.dim() != 2 or static_inputs.size(1) != self.num_static_features:
                raise ValueError("static_inputs has incorrect shape.")
            static_emb, _ = self.static_vsn(static_inputs) # pyright: ignore[reportOptionalCall]
            static_context = self.static_context(static_emb) # type: ignore
            static_enrichment = self.static_enrichment(static_emb) # type: ignore
            init_h = self.static_state_h(static_emb) # type: ignore
            init_c = self.static_state_c(static_emb) # type: ignore
            h0 = self._repeat_state(init_h, batch_size)
            c0 = self._repeat_state(init_c, batch_size)
            init_state = (h0, c0)
        else:
            static_context = None
            static_enrichment = None
            init_state = None

        past_embed, _ = self.past_vsn(context, context=static_context)
        future_embed, _ = self.future_vsn(future_known, context=static_context)

        past_encoded, state = self.encoder(past_embed, init_state)
        future_encoded, _ = self.decoder(future_embed, state)

        temporal = torch.cat([past_encoded, future_encoded], dim=1)
        if static_enrichment is not None:
            temporal = self.enrichment_grn(temporal, context=static_enrichment)
        else:
            temporal = self.enrichment_grn(temporal, context=None)

        mask = self._build_mask(temporal.size(1), temporal.device)
        attn_out = self.attention(temporal, mask=mask)
        attn_out = self.attention_norm(temporal + self.attention_gate(attn_out))
        position_out = self.position_grn(attn_out)
        position_out = self.position_norm(temporal + self.position_gate(position_out))

        future_out = position_out[:, -self.forecast_length :, :]
        return self.output_layer(future_out)
