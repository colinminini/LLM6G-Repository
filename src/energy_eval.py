"""Deployment-oriented energy/FLOPs evaluation stage.

Reads an experiment manifest and training summary from a previous run, then
emits per-model annual deployment cost estimates under two J/FLOP constants.
Output:
    <output-dir>/deployment_assumptions.json
    <output-dir>/<split>/<model>_energy.json
    <output-dir>/<split>/energy_summary.csv
    <output-dir>/<split>/energy_scenarios.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DEFAULT_DEEPAR_NUM_SAMPLES,
    SUPPORTED_MAIN_MODELS,
    TRAINABLE_BASELINE_MODELS,
)
from src.energy import (
    EnergyReport,
    annual_infer_calls_per_market,
    annual_infer_calls_per_series,
    deployment_energy_report,
)
from src.experiment import daily_seasonal_period, load_manifest


def _parse_models(raw: str) -> list[str]:
    models = [part.strip().lower() for part in raw.split(",") if part.strip()]
    invalid = [m for m in models if m not in SUPPORTED_MAIN_MODELS]
    if invalid:
        raise ValueError(
            f"Unsupported model(s): {', '.join(invalid)}. "
            f"Supported: {', '.join(SUPPORTED_MAIN_MODELS)}."
        )
    return models


def _parse_positive_int_csv(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    parsed: list[int] = []
    for value in values:
        try:
            parsed_value = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid integer value in comma-separated list: {value}") from exc
        if parsed_value <= 0:
            raise ValueError("Scenario values must be positive integers.")
        parsed.append(parsed_value)
    return sorted(set(parsed))


def _load_training_meta(training_dir: Path, model_name: str) -> dict:
    summary = training_dir / "training_summary.json"
    if not summary.exists():
        return {}
    payload = json.loads(summary.read_text())
    return payload.get("models", {}).get(model_name, {})


def _arch_for_model(
    model_name: str,
    *,
    hidden: int,
    n_layers: int,
    tft_num_heads: int,
    deepar_num_samples: int,
    sarima_s: int,
) -> dict:
    name = model_name.lower()
    if name == "lstm":
        return {"input_dim": 1, "hidden": hidden, "n_layers": n_layers}
    if name == "tft":
        return {"d_model": hidden, "n_heads": tft_num_heads, "n_layers": n_layers}
    if name == "deepar":
        return {
            "input_dim": 1,
            "hidden": hidden,
            "n_layers": n_layers,
            "n_samples": deepar_num_samples,
        }
    if name in {"chronos2", "chronos-2"}:
        return {"params": 120e6, "patch": 16}
    if name == "sarima":
        return {"p": 1, "q": 1, "P": 1, "Q": 1, "s": sarima_s, "max_iter": 50}
    if name in {"seasonal_naive", "seasonal-naive"}:
        return {}
    raise ValueError(f"Unknown model: {model_name}")


def _train_iters(meta: dict) -> int:
    schedule = meta.get("training_schedule") or {}
    steps_per_epoch = int(schedule.get("steps_per_epoch", 0))
    history = meta.get("monitor_metrics") or {}
    for field in ("last_iteration", "monitor_iter"):
        monitor_iter = history.get(field)
        if monitor_iter is None:
            continue
        try:
            return int(monitor_iter)
        except (TypeError, ValueError):
            continue
    max_iter = int(meta.get("max_iterations", 0))
    return max_iter or steps_per_epoch


def _checkpoint_param_count(checkpoint_path: Path) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    params = 0
    for value in state_dict.values():
        if hasattr(value, "numel"):
            params += int(value.numel())
    if params <= 0:
        raise ValueError(f"No parameter tensors found in checkpoint: {checkpoint_path}")
    return params


def _attach_checkpoint_params(model_name: str, arch: dict, train_meta: dict) -> dict:
    if model_name not in TRAINABLE_BASELINE_MODELS:
        return arch
    checkpoint_raw = train_meta.get("checkpoint_path")
    if not checkpoint_raw:
        raise FileNotFoundError(
            f"Missing checkpoint path in training summary for model='{model_name}'."
        )
    checkpoint_path = Path(checkpoint_raw)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint referenced by training summary does not exist for "
            f"model='{model_name}': {checkpoint_path}"
        )
    return {
        **arch,
        "params": _checkpoint_param_count(checkpoint_path),
        "params_source": "checkpoint",
    }


def _summary_row(report: EnergyReport) -> dict[str, float | int | str | None]:
    row: dict[str, float | int | str | None] = {
        "model": report.model,
        "params": report.arch.get("params"),
        "base_train_flops": report.base_train_flops,
        "infer_flops_per_call": report.infer_flops_per_call,
        "annual_infer_calls_per_market": int(report.annual_infer_calls_per_market),
        "annual_infer_flops_per_market": report.annual_infer_flops_per_market,
        "headline_markets": int(report.headline_markets),
        "annual_retrain_flops_headline": report.annual_retrain_flops_headline,
        "annual_total_flops_headline": report.annual_total_flops_headline,
        "annual_energy_J_gpu_headline": report.annual_energy_J_gpu_headline,
        "annual_energy_J_cpu_headline": report.annual_energy_J_cpu_headline,
    }
    for scenario in report.scenarios:
        suffix = f"market_{scenario.markets}"
        row[f"annual_retrain_flops_{suffix}"] = scenario.annual_retrain_flops
        row[f"annual_infer_flops_{suffix}"] = scenario.annual_infer_flops_total
        row[f"annual_total_flops_{suffix}"] = scenario.annual_total_flops
        row[f"annual_energy_J_gpu_{suffix}"] = scenario.annual_energy_J_gpu
        row[f"annual_energy_J_cpu_{suffix}"] = scenario.annual_energy_J_cpu
    return row


def _scenario_rows(report: EnergyReport) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for scenario in report.scenarios:
        rows.append(
            {
                "model": report.model,
                "markets": int(scenario.markets),
                "params": report.arch.get("params"),
                "base_train_flops": report.base_train_flops,
                "infer_flops_per_call": report.infer_flops_per_call,
                "annual_infer_calls_per_market": int(report.annual_infer_calls_per_market),
                "annual_infer_flops_per_market": report.annual_infer_flops_per_market,
                "annual_retrain_flops": scenario.annual_retrain_flops,
                "annual_infer_flops_total": scenario.annual_infer_flops_total,
                "annual_total_flops": scenario.annual_total_flops,
                "annual_energy_J_gpu": scenario.annual_energy_J_gpu,
                "annual_energy_J_cpu": scenario.annual_energy_J_cpu,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute analytical annual deployment FLOPs and energy estimates per forecaster."
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--forecast-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        default="lstm,deepar,tft,chronos2,seasonal_naive,sarima",
    )
    parser.add_argument("--splits", default="test")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tft-num-heads", type=int, default=4)
    parser.add_argument("--deepar-num-samples", type=int, default=DEFAULT_DEEPAR_NUM_SAMPLES)
    parser.add_argument("--sarima-season-length", type=int, default=144)
    parser.add_argument("--n-quantiles", type=int, default=None)
    parser.add_argument("--market-scenarios", default="1,10,100")
    parser.add_argument("--headline-markets", type=int, default=10)
    parser.add_argument("--retrain-cycles-per-year", type=int, default=4)
    parser.add_argument("--deployment-days", type=int, default=365)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_path)
    models = _parse_models(args.models)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    market_scenarios = _parse_positive_int_csv(args.market_scenarios)
    if args.headline_markets <= 0:
        raise ValueError("--headline-markets must be positive.")

    ctx = int(args.context_length or manifest.context_length)
    horizon = int(args.horizon or manifest.horizon)
    n_quantiles = int(args.n_quantiles or len(manifest.quantiles))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    live_calls_per_series_per_day = int(daily_seasonal_period(manifest.cadence))
    annual_calls_series = annual_infer_calls_per_series(
        manifest.cadence,
        args.deployment_days,
    )
    annual_calls_market = annual_infer_calls_per_market(
        cadence=manifest.cadence,
        num_series=manifest.num_series,
        deployment_days=args.deployment_days,
    )
    assumptions_payload = {
        "manifest_path": str(args.manifest_path),
        "dataset_path": manifest.dataset_path,
        "cadence": manifest.cadence,
        "context_length": int(ctx),
        "horizon": int(horizon),
        "quantiles": list(manifest.quantiles),
        "num_series_per_market": int(manifest.num_series),
        "live_calls_per_series_per_day": live_calls_per_series_per_day,
        "annual_infer_calls_per_series": int(annual_calls_series),
        "annual_infer_calls_per_market": int(annual_calls_market),
        "deployment_days": int(args.deployment_days),
        "retrain_cycles_per_year": int(args.retrain_cycles_per_year),
        "market_scenarios": market_scenarios,
        "headline_markets": int(args.headline_markets),
        "trainable_models": list(TRAINABLE_BASELINE_MODELS),
        "models": models,
    }
    (args.output_dir / "deployment_assumptions.json").write_text(
        json.dumps(assumptions_payload, indent=2)
    )

    for split in splits:
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, float | int | str | None]] = []
        scenario_rows: list[dict[str, float | int | str | None]] = []
        for model_name in models:
            train_meta = _load_training_meta(args.training_dir, model_name)
            arch = _arch_for_model(
                model_name,
                hidden=args.hidden_size,
                n_layers=args.num_layers,
                tft_num_heads=args.tft_num_heads,
                deepar_num_samples=args.deepar_num_samples,
                sarima_s=args.sarima_season_length,
            )
            arch = _attach_checkpoint_params(model_name, arch, train_meta)
            n_iter = _train_iters(train_meta)

            report = deployment_energy_report(
                model_name=model_name,
                n_train_iters=n_iter,
                batch_size=int(args.batch_size),
                ctx=int(ctx),
                horizon=int(horizon),
                num_series=int(manifest.num_series),
                cadence=manifest.cadence,
                deployment_days=int(args.deployment_days),
                retrain_cycles_per_year=int(args.retrain_cycles_per_year),
                market_scenarios=market_scenarios,
                headline_markets=int(args.headline_markets),
                n_quantiles=int(n_quantiles),
                arch=arch,
            )

            (split_dir / f"{model_name}_energy.json").write_text(
                json.dumps(report.as_dict(), indent=2)
            )
            rows.append(_summary_row(report))
            scenario_rows.extend(_scenario_rows(report))
            print(
                f"[energy_eval] {split}/{model_name}: "
                f"base_train={report.base_train_flops:.2e} "
                f"infer_per_call={report.infer_flops_per_call:.2e} "
                f"annual_total_{report.headline_markets}m={report.annual_total_flops_headline:.2e} "
                f"E_gpu={report.annual_energy_J_gpu_headline:.2e}J "
                f"E_cpu={report.annual_energy_J_cpu_headline:.2e}J"
            )

        pd.DataFrame(rows).to_csv(split_dir / "energy_summary.csv", index=False)
        pd.DataFrame(scenario_rows).to_csv(split_dir / "energy_scenarios.csv", index=False)
        print(f"[energy_eval] wrote {split_dir / 'energy_summary.csv'}")
        print(f"[energy_eval] wrote {split_dir / 'energy_scenarios.csv'}")


if __name__ == "__main__":
    main()
