"""Analytical FLOPs and deployment-energy estimates per forecaster.

The accounting is intentionally simple. We first estimate:

- one full training run for trainable models
- one live production inference call for a single series

We then scale those primitives into annual deployment scenarios across markets.
The goal is order-of-magnitude operational comparison, not calibrated power
meter fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.experiment import daily_seasonal_period

J_PER_FLOP_GPU = 1.3e-12
J_PER_FLOP_CPU = 1.0e-9
_TRAINABLE_ENERGY_MODELS = {"lstm", "deepar", "tft"}


@dataclass
class ModelEnergyComponents:
    model: str
    base_train_flops: float
    infer_flops_per_call: float
    arch: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_train_flops": float(self.base_train_flops),
            "infer_flops_per_call": float(self.infer_flops_per_call),
            "arch": self.arch,
        }


@dataclass
class DeploymentEnergyScenario:
    markets: int
    annual_retrain_flops: float
    annual_infer_flops_total: float
    annual_total_flops: float
    annual_energy_J_gpu: float
    annual_energy_J_cpu: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "markets": int(self.markets),
            "annual_retrain_flops": float(self.annual_retrain_flops),
            "annual_infer_flops_total": float(self.annual_infer_flops_total),
            "annual_total_flops": float(self.annual_total_flops),
            "annual_energy_J_gpu": float(self.annual_energy_J_gpu),
            "annual_energy_J_cpu": float(self.annual_energy_J_cpu),
        }


@dataclass
class EnergyReport:
    model: str
    base_train_flops: float
    infer_flops_per_call: float
    annual_infer_calls_per_market: int
    annual_infer_flops_per_market: float
    headline_markets: int
    annual_retrain_flops_headline: float
    annual_total_flops_headline: float
    annual_energy_J_gpu_headline: float
    annual_energy_J_cpu_headline: float
    scenarios: list[DeploymentEnergyScenario]
    arch: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_train_flops": float(self.base_train_flops),
            "infer_flops_per_call": float(self.infer_flops_per_call),
            "annual_infer_calls_per_market": int(self.annual_infer_calls_per_market),
            "annual_infer_flops_per_market": float(self.annual_infer_flops_per_market),
            "headline_markets": int(self.headline_markets),
            "annual_retrain_flops_headline": float(self.annual_retrain_flops_headline),
            "annual_total_flops_headline": float(self.annual_total_flops_headline),
            "annual_energy_J_gpu_headline": float(self.annual_energy_J_gpu_headline),
            "annual_energy_J_cpu_headline": float(self.annual_energy_J_cpu_headline),
            "energy_constants": {
                "J_per_flop_gpu": J_PER_FLOP_GPU,
                "J_per_flop_cpu": J_PER_FLOP_CPU,
            },
            "scenarios": {
                str(scenario.markets): scenario.as_dict()
                for scenario in self.scenarios
            },
            "arch": self.arch,
        }


def _params_lstm(input_dim: int, hidden: int, n_layers: int, n_quantiles: int) -> int:
    # 4 gates * (W_x + W_h + b) per layer; plus a quantile head on hidden->n_quantiles.
    p_layer1 = 4 * (input_dim * hidden + hidden * hidden + hidden)
    p_layer_other = 4 * (hidden * hidden + hidden * hidden + hidden)
    return p_layer1 + (n_layers - 1) * p_layer_other + (hidden + 1) * n_quantiles


def _params_tft(d_model: int, n_heads: int, n_layers: int, n_quantiles: int) -> int:
    # Approximation: variable selection + GRNs + encoder/decoder + attention.
    block = 12 * d_model * d_model + 4 * d_model * d_model
    head = (d_model + 1) * n_quantiles
    return n_layers * block + head


def _params_deepar(input_dim: int, hidden: int, n_layers: int) -> int:
    return _params_lstm(input_dim, hidden, n_layers, n_quantiles=2)


def _forward_flops_from_params(params: int, tokens: int) -> tuple[float, dict[str, Any]]:
    fwd = 2.0 * float(params) * float(tokens)
    return fwd, {"params": int(params), "tokens": int(tokens)}


def annual_infer_calls_per_series(cadence: str, deployment_days: int = 365) -> int:
    if deployment_days <= 0:
        raise ValueError("deployment_days must be positive.")
    return int(daily_seasonal_period(cadence) * int(deployment_days))


def annual_infer_calls_per_market(
    *,
    cadence: str,
    num_series: int,
    deployment_days: int = 365,
) -> int:
    if num_series <= 0:
        raise ValueError("num_series must be positive.")
    return annual_infer_calls_per_series(cadence, deployment_days) * int(num_series)


def lstm_forward_flops(
    *,
    input_dim: int,
    hidden: int,
    n_layers: int,
    ctx: int,
    horizon: int,
    n_quantiles: int = 2,
    params: int | None = None,
) -> tuple[float, dict[str, Any]]:
    tokens = ctx + horizon
    resolved_params = (
        int(params)
        if params is not None
        else _params_lstm(input_dim, hidden, n_layers, n_quantiles)
    )
    return _forward_flops_from_params(resolved_params, tokens)


def tft_forward_flops(
    *,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ctx: int,
    horizon: int,
    n_quantiles: int = 2,
    params: int | None = None,
) -> tuple[float, dict[str, Any]]:
    tokens = ctx + horizon
    resolved_params = (
        int(params)
        if params is not None
        else _params_tft(d_model, n_heads, n_layers, n_quantiles)
    )
    return _forward_flops_from_params(resolved_params, tokens)


def deepar_forward_flops(
    *,
    input_dim: int,
    hidden: int,
    n_layers: int,
    ctx: int,
    horizon: int,
    n_samples: int = 100,
    params: int | None = None,
) -> tuple[float, dict[str, Any]]:
    tokens = ctx + horizon * max(1, n_samples)
    resolved_params = (
        int(params)
        if params is not None
        else _params_deepar(input_dim, hidden, n_layers)
    )
    fwd, meta = _forward_flops_from_params(resolved_params, tokens)
    return fwd, {**meta, "n_samples": int(n_samples)}


def chronos2_forward_flops(
    *,
    params: float = 120e6,
    ctx: int,
    horizon: int,
    patch: int = 16,
) -> tuple[float, dict[str, Any]]:
    tokens = max(1, (ctx + horizon) // max(1, patch))
    fwd = 2.0 * float(params) * float(tokens)
    return fwd, {"params": int(params), "tokens": int(tokens), "patch": int(patch)}


def sarima_fit_flops(
    *,
    n_obs: int,
    p: int = 1,
    q: int = 1,
    P: int = 1,
    Q: int = 1,
    s: int = 144,
    max_iter: int = 50,
) -> tuple[float, dict[str, Any]]:
    # Kalman-filter SARIMAX: O(state^2 * n_obs) per likelihood eval; max_iter LBFGS evals.
    state = 1 + p + q + P + Q + s
    flops_per_eval = float(state) * float(state) * float(n_obs)
    fwd = float(max_iter) * flops_per_eval
    return fwd, {"state_dim": int(state), "max_iter": int(max_iter), "n_obs": int(n_obs)}


def seasonal_naive_flops(*, horizon: int, ctx: int) -> tuple[float, dict[str, Any]]:
    # Indexing + residual quantile (one sort): O(ctx log ctx + horizon).
    import math

    fwd = float(ctx) * math.log2(max(2, ctx)) + float(horizon)
    return fwd, {"ctx": int(ctx), "horizon": int(horizon)}


def model_energy_components(
    *,
    model_name: str,
    n_train_iters: int,
    batch_size: int,
    ctx: int,
    horizon: int,
    n_quantiles: int = 2,
    arch: dict[str, Any] | None = None,
) -> ModelEnergyComponents:
    """Compute one training run and one live inference call for a single model."""
    arch = dict(arch or {})
    name = model_name.lower()

    if name == "lstm":
        fwd, meta = lstm_forward_flops(
            input_dim=arch.get("input_dim", 1),
            hidden=arch.get("hidden", 128),
            n_layers=arch.get("n_layers", 2),
            ctx=ctx,
            horizon=horizon,
            n_quantiles=n_quantiles,
            params=arch.get("params"),
        )
        base_train_flops = 3.0 * fwd * float(batch_size) * float(n_train_iters)
        infer_flops_per_call = fwd
    elif name == "tft":
        fwd, meta = tft_forward_flops(
            d_model=arch.get("d_model", 128),
            n_heads=arch.get("n_heads", 4),
            n_layers=arch.get("n_layers", 2),
            ctx=ctx,
            horizon=horizon,
            n_quantiles=n_quantiles,
            params=arch.get("params"),
        )
        base_train_flops = 3.0 * fwd * float(batch_size) * float(n_train_iters)
        infer_flops_per_call = fwd
    elif name == "deepar":
        # Deployment uses the analytic-Gaussian deterministic rollout actually
        # implemented in the inference wrapper, so infer_flops is one rollout.
        fwd, meta = deepar_forward_flops(
            input_dim=arch.get("input_dim", 1),
            hidden=arch.get("hidden", 128),
            n_layers=arch.get("n_layers", 2),
            ctx=ctx,
            horizon=horizon,
            n_samples=1,
            params=arch.get("params"),
        )
        base_train_flops = 3.0 * fwd * float(batch_size) * float(n_train_iters)
        infer_flops_per_call = fwd
    elif name in {"chronos2", "chronos-2"}:
        fwd, meta = chronos2_forward_flops(
            params=arch.get("params", 120e6),
            ctx=ctx,
            horizon=horizon,
            patch=arch.get("patch", 16),
        )
        base_train_flops = 0.0
        infer_flops_per_call = fwd
    elif name == "sarima":
        fwd, meta = sarima_fit_flops(
            n_obs=ctx,
            p=arch.get("p", 1),
            q=arch.get("q", 1),
            P=arch.get("P", 1),
            Q=arch.get("Q", 1),
            s=arch.get("s", 144),
            max_iter=arch.get("max_iter", 50),
        )
        base_train_flops = 0.0
        infer_flops_per_call = fwd
    elif name in {"seasonal_naive", "seasonal-naive"}:
        fwd, meta = seasonal_naive_flops(ctx=ctx, horizon=horizon)
        base_train_flops = 0.0
        infer_flops_per_call = fwd
    else:
        raise ValueError(f"Unknown model for energy estimate: {model_name}")

    return ModelEnergyComponents(
        model=model_name,
        base_train_flops=float(base_train_flops),
        infer_flops_per_call=float(infer_flops_per_call),
        arch={**arch, **meta},
    )


def deployment_energy_report(
    *,
    model_name: str,
    n_train_iters: int,
    batch_size: int,
    ctx: int,
    horizon: int,
    num_series: int,
    cadence: str,
    deployment_days: int,
    retrain_cycles_per_year: int,
    market_scenarios: Sequence[int],
    headline_markets: int,
    n_quantiles: int = 2,
    arch: dict[str, Any] | None = None,
) -> EnergyReport:
    """Lift simple training/inference FLOP estimates into deployment scenarios."""
    if retrain_cycles_per_year <= 0:
        raise ValueError("retrain_cycles_per_year must be positive.")

    components = model_energy_components(
        model_name=model_name,
        n_train_iters=n_train_iters,
        batch_size=batch_size,
        ctx=ctx,
        horizon=horizon,
        n_quantiles=n_quantiles,
        arch=arch,
    )

    annual_calls = annual_infer_calls_per_market(
        cadence=cadence,
        num_series=num_series,
        deployment_days=deployment_days,
    )
    annual_infer_per_market = components.infer_flops_per_call * float(annual_calls)

    normalized_scenarios = sorted({int(m) for m in market_scenarios} | {int(headline_markets)})
    scenarios: list[DeploymentEnergyScenario] = []
    is_trainable = model_name.lower() in _TRAINABLE_ENERGY_MODELS
    for markets in normalized_scenarios:
        annual_retrain_flops = (
            components.base_train_flops * float(retrain_cycles_per_year) * float(markets)
            if is_trainable
            else 0.0
        )
        annual_infer_flops_total = annual_infer_per_market * float(markets)
        annual_total_flops = annual_retrain_flops + annual_infer_flops_total
        scenarios.append(
            DeploymentEnergyScenario(
                markets=int(markets),
                annual_retrain_flops=float(annual_retrain_flops),
                annual_infer_flops_total=float(annual_infer_flops_total),
                annual_total_flops=float(annual_total_flops),
                annual_energy_J_gpu=float(annual_total_flops * J_PER_FLOP_GPU),
                annual_energy_J_cpu=float(annual_total_flops * J_PER_FLOP_CPU),
            )
        )

    scenario_map = {scenario.markets: scenario for scenario in scenarios}
    headline = scenario_map[int(headline_markets)]
    return EnergyReport(
        model=model_name,
        base_train_flops=float(components.base_train_flops),
        infer_flops_per_call=float(components.infer_flops_per_call),
        annual_infer_calls_per_market=int(annual_calls),
        annual_infer_flops_per_market=float(annual_infer_per_market),
        headline_markets=int(headline_markets),
        annual_retrain_flops_headline=float(headline.annual_retrain_flops),
        annual_total_flops_headline=float(headline.annual_total_flops),
        annual_energy_J_gpu_headline=float(headline.annual_energy_J_gpu),
        annual_energy_J_cpu_headline=float(headline.annual_energy_J_cpu),
        scenarios=scenarios,
        arch={
            **components.arch,
            "deployment_days": int(deployment_days),
            "retrain_cycles_per_year": int(retrain_cycles_per_year),
            "num_series_per_market": int(num_series),
            "cadence": str(cadence),
            "is_trainable": bool(is_trainable),
        },
    )
