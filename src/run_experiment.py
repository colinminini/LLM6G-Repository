"""Single orchestration entrypoint for the canonical forecasting experiment flow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys as _sys

    _sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DEFAULT_CP_DETECTOR,
    DEFAULT_CP_EXCLUSION_RADIUS,
    DEFAULT_CP_PERIOD_LENGTH,
    DEFAULT_CP_SCORE_THRESHOLD,
    DEFAULT_DATASET_PATH,
    DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM,
    DEFAULT_DEEPAR_LIKELIHOOD,
    DEFAULT_DEEPAR_NUM_SAMPLES,
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    SUPPORTED_DEEPAR_LIKELIHOODS,
    TRAINABLE_BASELINE_MODELS,
    parse_quantiles,
)
from src.device import AUTO_DEVICE, AUTO_DEVICE_HELP, resolve_device_name
from src.experiment import build_experiment_manifest, default_experiment_dir
from src.reporting import (
    build_report_bundle,
    build_cp_sweep_report,
    build_example_window_report,
    build_forecast_eval_report,
    build_prepare_data_report,
    build_system_eval_report,
    build_tau_calibration_report,
    build_training_report,
    publish_report_plots,
)

STAGE_ORDER = (
    "prepare_data",
    "train",
    "forecast_eval",
    "cp_sweep",
    "system_eval",
    "tau_calibration",
    "calibrated_system_eval",
    "energy_eval",
)

DEFAULT_STAGE_ORDER = (
    "prepare_data",
    "train",
    "forecast_eval",
    "system_eval",
    "energy_eval",
)

OPTIONAL_TAU_STAGE_ORDER = (
    "tau_calibration",
    "calibrated_system_eval",
)


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_stages(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_STAGE_ORDER)
    stages = _parse_csv(raw)
    invalid = [stage for stage in stages if stage not in STAGE_ORDER]
    if invalid:
        raise ValueError(
            f"Unsupported stage(s): {', '.join(invalid)}. "
            f"Supported stages: {', '.join(STAGE_ORDER)}."
        )
    return stages


def _filter_train_models(raw: str) -> str:
    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    train_models = [model_name for model_name in requested if model_name in TRAINABLE_BASELINE_MODELS]
    if not train_models:
        raise ValueError(
            "No trainable models were requested for the train stage. "
            f"Supported trainable models: {', '.join(TRAINABLE_BASELINE_MODELS)}."
        )
    return ",".join(train_models)


def _resolve_requested_stages(stages: list[str], resume_from: str | None) -> list[str]:
    if resume_from is None:
        return stages
    if resume_from not in STAGE_ORDER:
        raise ValueError(f"Unsupported --resume-from stage: {resume_from}")
    resume_idx = STAGE_ORDER.index(resume_from)
    return [stage for stage in STAGE_ORDER[resume_idx:] if stage in stages or stages == list(STAGE_ORDER)]


def _status_path(output_dir: Path) -> Path:
    return output_dir / "reports" / "stage_status.json"


def _load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text())


def _save_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _print_banner(stage: str, index: int, total: int, output_dir: Path) -> None:
    print("")
    print("=" * 80)
    print(f"Stage {index}/{total}: {stage}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _stage_metrics(output_dir: Path, stage: str) -> tuple[dict[str, Any], dict[str, str]]:
    if stage == "prepare_data":
        manifest = _read_json_if_exists(output_dir / "manifest.json") or {}
        return {
            "dataset_path": manifest.get("dataset_path"),
            "cadence": manifest.get("cadence"),
            "context_length": manifest.get("context_length"),
            "horizon": manifest.get("horizon"),
            "num_rows": manifest.get("num_rows"),
            "num_series": manifest.get("num_series"),
        }, build_prepare_data_report(output_dir)

    if stage == "train":
        summary = _read_json_if_exists(output_dir / "training" / "training_summary.json") or {}
        metrics: dict[str, Any] = {}
        for model_name, payload in summary.get("models", {}).items():
            monitor_metrics = payload.get("monitor_metrics", {})
            val_metrics = monitor_metrics.get("val", {})
            metrics[model_name] = {
                "monitor_val_loss": monitor_metrics.get("monitor_val_loss"),
                "monitor_val_rmse": monitor_metrics.get("monitor_val_rmse"),
                "val_pinball_q50": val_metrics.get("pinball_q50"),
            }
        return metrics, build_training_report(output_dir)

    if stage == "forecast_eval":
        summary = _read_json_if_exists(output_dir / "forecast_eval" / "forecast_eval_summary.json") or {}
        metrics: dict[str, Any] = {}
        for split in summary.get("splits", {}):
            split_dir = output_dir / "forecast_eval" / split
            split_metrics = {}
            for path in sorted(split_dir.glob("*_forecast_metrics.json")):
                split_metrics[path.stem.replace("_forecast_metrics", "")] = json.loads(path.read_text())
            metrics[split] = split_metrics
        plots = build_forecast_eval_report(output_dir)
        plots.update(build_example_window_report(output_dir))
        return metrics, plots

    if stage == "system_eval":
        summary = _read_json_if_exists(output_dir / "system_eval" / "system_eval_summary.json") or {}
        metrics: dict[str, Any] = {}
        for split in summary.get("splits", {}):
            split_dir = output_dir / "system_eval" / split
            split_metrics = {}
            for path in sorted(split_dir.glob("*_system_metrics.json")):
                split_metrics[path.stem.replace("_system_metrics", "")] = json.loads(path.read_text())
            metrics[split] = split_metrics
        return metrics, build_system_eval_report(output_dir)

    if stage == "cp_sweep":
        summary_path = output_dir / "cp_sweep" / "train" / "cp_sweep_best_configs.csv"
        metrics: dict[str, Any] = {}
        if summary_path.exists():
            import pandas as pd

            df = pd.read_csv(summary_path)
            if not df.empty:
                metrics["best_break_f1_rows"] = df.to_dict("records")
        return metrics, build_cp_sweep_report(output_dir, split="train")

    if stage == "tau_calibration":
        summary_path = output_dir / "tau_calibration" / "tau_calibration_best_test_rows.csv"
        metrics: dict[str, Any] = {}
        if summary_path.exists():
            import pandas as pd

            metrics["best_test_rows"] = pd.read_csv(summary_path).to_dict("records")
        return metrics, build_tau_calibration_report(output_dir)

    if stage == "calibrated_system_eval":
        summary_path = output_dir / "calibrated_system_eval" / "calibrated_system_eval_comparison.csv"
        metrics: dict[str, Any] = {}
        if summary_path.exists():
            import pandas as pd

            metrics["comparison_rows"] = pd.read_csv(summary_path).to_dict("records")
        return metrics, build_report_bundle(output_dir)

    if stage == "energy_eval":
        metrics: dict[str, Any] = {}
        energy_root = output_dir / "energy"
        if energy_root.exists():
            for split_dir in sorted(p for p in energy_root.iterdir() if p.is_dir()):
                split_metrics: dict[str, Any] = {}
                for path in sorted(split_dir.glob("*_energy.json")):
                    split_metrics[path.stem.replace("_energy", "")] = json.loads(path.read_text())
                metrics[split_dir.name] = split_metrics
        return metrics, {}

    return {}, {}


def _build_stage_command(
    stage: str,
    *,
    output_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    base = [sys.executable]
    if stage == "prepare_data":
        return base + [
            "src/prepare_data.py",
            "--data-path",
            str(args.data_path),
            "--timestamp-col",
            args.timestamp_col,
            "--context-length",
            str(args.context_length),
            "--horizon",
            str(args.horizon),
            "--quantiles",
            args.quantiles,
            "--random-seed",
            str(args.random_seed),
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--output-dir",
            str(output_dir),
        ]
    if stage == "train":
        command = base + [
            "src/train.py",
            "--manifest-path",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--models",
            _filter_train_models(args.models),
            "--batch-size",
            str(args.batch_size),
            "--train-window-step",
            str(args.train_window_step),
            "--learning-rate",
            str(args.learning_rate),
            "--hidden-size",
            str(args.hidden_size),
            "--num-layers",
            str(args.num_layers),
            "--tft-num-heads",
            str(args.tft_num_heads),
            "--deepar-likelihood",
            args.deepar_likelihood,
            "--deepar-num-samples",
            str(args.deepar_num_samples),
            "--deepar-item-embedding-dim",
            str(args.deepar_item_embedding_dim),
            "--device",
            args.device,
        ]
        if args.log_every is not None:
            command.extend(["--log-every", str(args.log_every)])
        if args.max_iterations is not None:
            command.extend(["--max-iterations", str(args.max_iterations)])
        if args.patience_iterations is not None:
            command.extend(["--patience-iterations", str(args.patience_iterations)])
        if args.validate_epochs is not None:
            command.extend(["--validate-epochs", str(args.validate_epochs)])
        if args.max_epochs is not None:
            command.extend(["--max-epochs", str(args.max_epochs)])
        if args.patience is not None:
            command.extend(["--patience", str(args.patience)])
        return command
    if stage == "forecast_eval":
        return base + [
            "src/forecast_eval.py",
            "--manifest-path",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--models",
            args.models,
            "--splits",
            args.splits,
            "--deepar-num-samples",
            str(args.deepar_num_samples),
            "--device",
            args.device,
        ] + (
            ["--max-windows-per-series", str(args.max_windows_per_series)]
            if args.max_windows_per_series is not None
            else []
        ) + (
            ["--paired-windows"] if args.paired_windows else ["--independent-windows"]
        )
    if stage == "system_eval":
        cp_manifest = output_dir / "cp_sweep" / "train" / "cp_sweep_manifest.json"
        command = base + [
            "src/system_eval.py",
            "--forecast-dir",
            str(output_dir / "forecast_eval"),
            "--models",
            args.models,
            "--splits",
            args.system_splits,
            "--output-dir",
            str(output_dir / "system_eval"),
            "--tolerance",
            str(args.tolerance),
        ]
        if cp_manifest.exists():
            command += ["--cp-config-manifest", str(cp_manifest)]
        else:
            command += [
                "--cp-detector",
                args.cp_detector,
                "--cp-period-length",
                str(args.cp_period_length),
                "--cp-exclusion-radius",
                str(args.cp_exclusion_radius),
                "--cp-score-threshold",
                str(args.cp_score_threshold),
                "--cp-model",
                args.cp_model,
                "--cp-penalty",
                str(args.cp_penalty),
                "--cp-min-size",
                str(args.cp_min_size),
                "--cp-jump",
                str(args.cp_jump),
            ]
        return command
    if stage == "cp_sweep":
        return base + [
            "src/cp_sweep.py",
            "--forecast-dir",
            str(output_dir / "forecast_eval"),
            "--models",
            args.models,
            "--split",
            "train",
            "--output-dir",
            str(output_dir / "cp_sweep" / "train"),
            "--train-window-step",
            str(args.train_window_step),
            "--cp-detector",
            args.cp_detector,
            "--cp-period-lengths",
            args.cp_period_lengths,
            "--cp-exclusion-radii",
            args.cp_exclusion_radii,
            "--cp-score-thresholds",
            args.cp_score_thresholds,
            "--cp-models",
            args.cp_models,
            "--cp-penalties",
            args.cp_penalties,
            "--cp-min-sizes",
            args.cp_min_sizes,
            "--cp-jumps",
            args.cp_jumps,
            "--tolerance",
            str(args.tolerance),
            "--coverage-target",
            str(args.coverage_target),
        ] + (
            ["--cp-max-windows-per-series", str(args.cp_max_windows_per_series)]
            if args.cp_max_windows_per_series is not None
            else []
        )
    if stage == "tau_calibration":
        cp_manifest = output_dir / "cp_sweep" / "train" / "cp_sweep_manifest.json"
        command = base + [
            "src/tau_calibration.py",
            "--forecast-dir",
            str(output_dir / "forecast_eval"),
            "--models",
            args.models,
            "--data-path",
            str(args.data_path),
            "--output-dir",
            str(output_dir / "tau_calibration"),
            "--tolerance",
            str(args.tolerance),
            "--coverage-target",
            str(args.coverage_target),
        ]
        if cp_manifest.exists():
            command += ["--cp-config-manifest", str(cp_manifest)]
        command += [
            "--cp-detector",
            args.cp_detector,
            "--cp-period-length",
            str(args.cp_period_length),
            "--cp-exclusion-radius",
            str(args.cp_exclusion_radius),
            "--cp-score-threshold",
            str(args.cp_score_threshold),
            "--cp-model",
            args.cp_model,
            "--cp-penalty",
            str(args.cp_penalty),
            "--cp-min-size",
            str(args.cp_min_size),
            "--cp-jump",
            str(args.cp_jump),
            "--random-seed",
            str(args.random_seed),
        ]
        return command
    if stage == "calibrated_system_eval":
        return base + [
            "src/calibrated_system_eval.py",
            "--forecast-dir",
            str(output_dir / "forecast_eval"),
            "--tau-calibration-dir",
            str(output_dir / "tau_calibration"),
            "--raw-system-eval-dir",
            str(output_dir / "system_eval"),
            "--models",
            args.models,
            "--splits",
            args.system_splits,
            "--output-dir",
            str(output_dir / "calibrated_system_eval"),
            "--selection-rule",
            args.tau_selection_rule,
        ]
    if stage == "energy_eval":
        return base + [
            "src/energy_eval.py",
            "--manifest-path",
            str(manifest_path),
            "--forecast-dir",
            str(output_dir / "forecast_eval"),
            "--training-dir",
            str(output_dir / "training"),
            "--output-dir",
            str(output_dir / "energy"),
            "--models",
            args.models,
            "--splits",
            args.system_splits,
            "--context-length",
            str(args.context_length),
            "--horizon",
            str(args.horizon),
            "--batch-size",
            str(args.batch_size),
            "--hidden-size",
            str(args.hidden_size),
            "--num-layers",
            str(args.num_layers),
            "--tft-num-heads",
            str(args.tft_num_heads),
            "--deepar-num-samples",
            str(args.deepar_num_samples),
            "--n-quantiles",
            str(len(parse_quantiles(args.quantiles))),
            "--market-scenarios",
            args.market_scenarios,
            "--headline-markets",
            str(args.headline_markets),
            "--retrain-cycles-per-year",
            str(args.retrain_cycles_per_year),
            "--deployment-days",
            str(args.deployment_days),
        ]
    raise ValueError(f"Unsupported stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full split-aware experiment flow on any regular wide time-series dataset."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--stages", default=None, help="Comma-separated subset of stages to run.")
    parser.add_argument("--resume-from", default=None, help="Resume from this stage onward.")
    parser.add_argument("--overwrite", action="store_true", help="Allow rerunning stages in an existing output directory.")
    parser.add_argument("--with-tau-calibration", action="store_true", help="Append tau calibration and calibrated system eval to the default stage flow.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--context-length", type=int, default=144)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--quantiles", default="0.5,0.99")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--models", default="lstm,deepar,tft,chronos2,seasonal_naive,sarima")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--system-splits", default="test")
    parser.add_argument("--sampling-mode", choices=("random", "rolling"), default="random")
    parser.add_argument("--random-windows-per-series", type=int, default=50)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--max-windows-per-series", type=int, default=None)
    parser.add_argument("--paired-windows", action="store_true")
    parser.add_argument("--independent-windows", dest="paired_windows", action="store_false")
    parser.set_defaults(paired_windows=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-window-step", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--patience-iterations", type=int, default=1000)
    parser.add_argument("--validate-epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--patience", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tft-num-heads", type=int, default=4)
    parser.add_argument("--deepar-likelihood", default=DEFAULT_DEEPAR_LIKELIHOOD, choices=SUPPORTED_DEEPAR_LIKELIHOODS)
    parser.add_argument("--deepar-num-samples", type=int, default=DEFAULT_DEEPAR_NUM_SAMPLES)
    parser.add_argument("--deepar-item-embedding-dim", type=int, default=DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM)
    parser.add_argument("--market-scenarios", default="1,10,100")
    parser.add_argument("--headline-markets", type=int, default=10)
    parser.add_argument("--retrain-cycles-per-year", type=int, default=4)
    parser.add_argument("--deployment-days", type=int, default=365)
    parser.add_argument("--device", default=AUTO_DEVICE, help=AUTO_DEVICE_HELP)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--cp-detector", default=DEFAULT_CP_DETECTOR)
    parser.add_argument("--cp-period-length", default=DEFAULT_CP_PERIOD_LENGTH)
    parser.add_argument("--cp-exclusion-radius", type=float, default=DEFAULT_CP_EXCLUSION_RADIUS)
    parser.add_argument("--cp-score-threshold", type=float, default=DEFAULT_CP_SCORE_THRESHOLD)
    parser.add_argument("--cp-model", default="normal")
    parser.add_argument("--cp-penalty", type=float, default=10.0)
    parser.add_argument("--cp-min-size", type=int, default=8)
    parser.add_argument("--cp-jump", type=int, default=1)
    parser.add_argument("--cp-period-lengths", default="auto,8,16")
    parser.add_argument("--cp-exclusion-radii", default="0.05")
    parser.add_argument("--cp-score-thresholds", default="0.5,0.6,0.7,0.75")
    parser.add_argument("--cp-models", default="normal")
    parser.add_argument("--cp-penalties", default="10,15,20")
    parser.add_argument("--cp-min-sizes", default="8,10,12")
    parser.add_argument("--cp-jumps", default="1")
    parser.add_argument("--cp-max-windows-per-series", type=int, default=None)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    parser.add_argument("--tau-selection-rule", default="best_val_mae_cp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest_path is None and args.data_path is None:
        print("Error: --data-path is required when --manifest-path is not provided.", file=sys.stderr)
        sys.exit(1)
    args.device = resolve_device_name(args.device)
    if args.stages is None:
        requested_stages = list(DEFAULT_STAGE_ORDER)
        if args.with_tau_calibration:
            requested_stages.extend(OPTIONAL_TAU_STAGE_ORDER)
    else:
        requested_stages = _normalize_stages(args.stages)
    stages = _resolve_requested_stages(requested_stages, args.resume_from)

    if args.manifest_path is not None:
        output_dir = args.output_dir or args.manifest_path.parent
        manifest_path = args.manifest_path
    else:
        manifest = build_experiment_manifest(
            data_path=args.data_path,
            timestamp_col=args.timestamp_col,
            context_length=args.context_length,
            horizon=args.horizon,
            quantiles=parse_quantiles(args.quantiles) if args.quantiles else DEFAULT_QUANTILES,
            random_seed=args.random_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        output_dir = args.output_dir or default_experiment_dir(manifest)
        manifest_path = output_dir / "manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = _status_path(output_dir)
    status_payload = _load_status(status_path)
    status_payload.setdefault("experiment_dir", str(output_dir))
    status_payload.setdefault("stages", {})

    if not args.overwrite and args.resume_from is None:
        existing = [
            stage
            for stage in stages
            if status_payload["stages"].get(stage, {}).get("status") == "success"
        ]
        if existing:
            next_stage = None
            for candidate in stages:
                payload = status_payload["stages"].get(candidate, {})
                if payload.get("status") != "success":
                    next_stage = candidate
                    break
            resume_hint = (
                f" Use --resume-from {next_stage} to continue from the next incomplete stage."
                if next_stage is not None
                else ""
            )
            raise RuntimeError(
                "Existing successful stage results were found for: "
                f"{', '.join(existing)}. Use --resume-from or --overwrite."
                f"{resume_hint}"
            )

    total = len(stages)
    for index, stage in enumerate(stages, start=1):
        _print_banner(stage, index, total, output_dir)
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        command = _build_stage_command(
            stage,
            output_dir=output_dir,
            manifest_path=manifest_path,
            args=args,
        )
        status_payload["stages"][stage] = {
            "status": "running",
            "started_at_utc": started_at,
            "command": command,
        }
        _save_status(status_path, status_payload)
        try:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
            metrics, plots = _stage_metrics(output_dir, stage)
            report_bundle = build_report_bundle(output_dir)
            published_plots = publish_report_plots(output_dir)
            elapsed = time.perf_counter() - start_time
            status_payload["stages"][stage] = {
                "status": "success",
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_sec": round(elapsed, 3),
                "command": command,
                "metrics": metrics,
                "plots": plots,
                "report_index": report_bundle.get("report_index"),
                "published_plots": published_plots,
            }
            _save_status(status_path, status_payload)
            print(f"Completed {stage} in {elapsed:.1f}s")
            if metrics:
                print(f"Headline metrics: {json.dumps(metrics, default=str)[:600]}")
            if published_plots.get("published_report_index"):
                print(f"Published README plots: {published_plots['published_report_index']}")
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - start_time
            status_payload["stages"][stage] = {
                "status": "failed",
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_sec": round(elapsed, 3),
                "command": command,
                "returncode": int(exc.returncode),
            }
            _save_status(status_path, status_payload)
            raise RuntimeError(f"Stage '{stage}' failed with exit code {exc.returncode}.") from exc

    print("")
    print(f"Experiment completed. Stage status: {status_path}")
    print(f"Report index: {output_dir / 'reports' / 'report_index.json'}")


if __name__ == "__main__":
    main()
