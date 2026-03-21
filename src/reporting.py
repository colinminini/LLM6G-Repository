"""Reusable plots and lightweight summaries for experiment artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "llm6g_mplconfig"
_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "llm6g_xdg_cache"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PLOTS_DIR = REPO_ROOT / "results" / "plots" / "readme"
PREFERRED_EXAMPLE_MODEL = "chronos2"
PUBLISHED_FILENAME_OVERRIDES = {
    "example_windows_test_plot": "forecast_example_chronos2_test.png",
    "example_windows_system_eval_test_plot": "pipeline_example_chronos2_test.png",
}


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _save_figure(fig: plt.Figure, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _preferred_model_order(
    models: Sequence[str],
    *,
    preferred_model: str = PREFERRED_EXAMPLE_MODEL,
) -> list[str]:
    ordered: list[str] = []
    for model_name in (preferred_model, *models):
        normalized = str(model_name)
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _resolve_repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _select_example_window_row(
    window_dir: str | Path,
    *,
    window_suffix: str,
    models: Sequence[str] = ("lstm", "deepar", "chronos2"),
    preferred_model: str = PREFERRED_EXAMPLE_MODEL,
) -> tuple[str, pd.Series] | None:
    window_dir = Path(window_dir)
    checked_models: set[str] = set()
    for model_name in _preferred_model_order(models, preferred_model=preferred_model):
        checked_models.add(model_name)
        path = window_dir / f"{model_name}_{window_suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not df.empty:
            return model_name, df.iloc[0]

    for path in sorted(window_dir.glob(f"*_{window_suffix}.csv")):
        model_name = path.name[: -len(f"_{window_suffix}.csv")]
        if model_name in checked_models:
            continue
        df = pd.read_csv(path)
        if not df.empty:
            return model_name, df.iloc[0]
    return None


def _parse_json_series(value: Any) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _published_plot_name(artifact_name: str, source_path: Path) -> str:
    return PUBLISHED_FILENAME_OVERRIDES.get(artifact_name, source_path.name)


def publish_report_plots(
    experiment_dir: str | Path,
    *,
    publish_dir: str | Path | None = None,
) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    report_index_path = experiment_dir / "reports" / "report_index.json"
    if not report_index_path.exists():
        return {}

    report_index = _read_json(report_index_path)
    published_dir = _ensure_dir(publish_dir or README_PLOTS_DIR)
    published_outputs: dict[str, str] = {}
    published_artifacts: dict[str, dict[str, str]] = {}
    for artifact_name, raw_source_path in sorted(report_index.items()):
        source_path = Path(raw_source_path)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        if source_path.suffix.lower() != ".png" or not source_path.exists():
            continue

        published_name = _published_plot_name(artifact_name, source_path)
        published_path = published_dir / published_name
        shutil.copy2(source_path, published_path)
        published_outputs[artifact_name] = str(published_path)
        published_artifacts[artifact_name] = {
            "source_path": _resolve_repo_relative(source_path),
            "published_path": _resolve_repo_relative(published_path),
        }

    manifest_path = published_dir / "published_report_index.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_experiment_dir": _resolve_repo_relative(experiment_dir),
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": published_artifacts,
            },
            indent=2,
        )
    )
    published_outputs["published_report_index"] = str(manifest_path)
    return published_outputs


def build_prepare_data_report(experiment_dir: str | Path) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    split_summary_path = experiment_dir / "split_summary.csv"
    if not split_summary_path.exists():
        return {}

    df = pd.read_csv(split_summary_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["split"], df["num_rows"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Chronological Split Sizes")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.3)
    plot_path = experiment_dir / "reports" / "prepare_data_split_summary.png"
    _save_figure(fig, plot_path)
    return {"split_summary_plot": str(plot_path)}


def build_training_report(experiment_dir: str | Path) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    training_dir = experiment_dir / "training"
    if not training_dir.exists():
        return {}

    history_paths = sorted(training_dir.glob("*_history.json"))
    if not history_paths:
        return {}

    metrics = [("loss", "Loss"), ("rmse", "RMSE"), ("coverage", "Coverage"), ("interval_width", "Interval Width")]
    outputs: dict[str, str] = {}
    for metric_key, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for history_path in history_paths:
            payload = _read_json(history_path)
            model_name = history_path.name.replace("_history.json", "")
            x_values = payload.get("iteration")
            for split in ("train", "val", "test"):
                key = f"{split}_{metric_key}"
                if key in payload:
                    values = payload[key]
                    if not values:
                        continue
                    if isinstance(x_values, list) and len(x_values) == len(values):
                        x_axis = np.asarray(x_values, dtype=float)
                    else:
                        x_axis = np.arange(1, len(values) + 1)
                    ax.plot(
                        x_axis,
                        values,
                        label=f"{model_name}:{split}",
                        marker="o",
                        markersize=4,
                        linewidth=1.5,
                    )
                    if x_axis.size == 1:
                        x0 = float(x_axis[0])
                        ax.set_xlim(x0 - 1.0, x0 + 1.0)
                    plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Training {title}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        plot_path = experiment_dir / "reports" / f"training_{metric_key}.png"
        _save_figure(fig, plot_path)
        outputs[f"training_{metric_key}_plot"] = str(plot_path)
    return outputs


def build_forecast_eval_report(experiment_dir: str | Path, splits: Sequence[str] | None = None) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    forecast_dir = experiment_dir / "forecast_eval"
    if not forecast_dir.exists():
        return {}

    splits = list(splits) if splits is not None else sorted(p.name for p in forecast_dir.iterdir() if p.is_dir())
    outputs: dict[str, str] = {}
    for split in splits:
        metrics_paths = sorted((forecast_dir / split).glob("*_forecast_metrics.json"))
        if not metrics_paths:
            continue
        rows = [json.loads(path.read_text()) for path in metrics_paths]
        df = pd.DataFrame(rows).sort_values("model")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        metrics = [
            ("rmse_q50", "RMSE q50"),
            ("coverage_q95", "Coverage q95"),
            ("interval_width_mean", "Mean q95 - q50"),
        ]
        for ax, (key, title) in zip(axes, metrics):
            ax.bar(df["model"], df[key], color="#1f77b4")
            ax.set_title(f"{title} ({split})")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.3)
        plot_path = experiment_dir / "reports" / f"forecast_eval_{split}.png"
        _save_figure(fig, plot_path)
        outputs[f"forecast_eval_{split}_plot"] = str(plot_path)
    return outputs


def build_system_eval_report(experiment_dir: str | Path, splits: Sequence[str] | None = None) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    system_dir = experiment_dir / "system_eval"
    if not system_dir.exists():
        return {}

    splits = list(splits) if splits is not None else sorted(p.name for p in system_dir.iterdir() if p.is_dir())
    outputs: dict[str, str] = {}
    for split in splits:
        metrics_paths = sorted((system_dir / split).glob("*_system_metrics.json"))
        if not metrics_paths:
            continue
        rows = [json.loads(path.read_text()) for path in metrics_paths]
        df = pd.DataFrame(rows).sort_values("model")
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        metrics = [
            ("MAE_CP", "MAE CP"),
            ("tolerance_hit_rate", "Tolerance Hit Rate"),
            ("coverage_rate", "Coverage Rate"),
            ("sharpness", "Sharpness"),
        ]
        for ax, (key, title) in zip(axes.reshape(-1), metrics):
            ax.bar(df["model"], df[key], color="#ff7f0e")
            ax.set_title(f"{title} ({split})")
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.3)
        plot_path = experiment_dir / "reports" / f"system_eval_{split}.png"
        _save_figure(fig, plot_path)
        outputs[f"system_eval_{split}_plot"] = str(plot_path)
    return outputs


def build_system_example_window_report(
    experiment_dir: str | Path,
    *,
    split: str = "test",
    models: Sequence[str] = ("lstm", "deepar", "chronos2"),
) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    system_dir = experiment_dir / "system_eval" / split
    selected = _select_example_window_row(
        system_dir,
        window_suffix="system_windows",
        models=models,
    )
    if selected is None:
        return {}

    model_name, row = selected
    history = _parse_json_series(row["history"])
    future_true = _parse_json_series(row["future_true"])
    y50 = _parse_json_series(row["y_pred_median"])
    y95 = _parse_json_series(row["y_pred_95"])
    tau_pred = int(row["tau_pred"])
    tau_true = int(row["tau_true"])
    safe_ceiling = float(row["safe_ceiling"])

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x_hist = np.arange(history.size)
    x_future = np.arange(history.size, history.size + future_true.size)
    ax.plot(x_hist, history, label="history", color="#444444")
    ax.plot(x_future, future_true, label="future_true", color="#000000")
    ax.plot(x_future, y50, label="q50", color="#1f77b4")
    ax.plot(x_future, y95, label="q95", color="#ff7f0e")
    ax.hlines(
        safe_ceiling,
        xmin=x_future[0],
        xmax=x_future[-1],
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="safe_ceiling",
    )
    ax.axvline(history.size + tau_pred, color="#d62728", linestyle="--", linewidth=1.2, label="tau_pred")
    ax.axvline(history.size + tau_true, color="#9467bd", linestyle=":", linewidth=1.2, label="tau_true")
    ax.set_title(f"{model_name} | series={row['series']} | start={int(row['start_index'])}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

    plot_path = experiment_dir / "reports" / f"example_windows_system_eval_{split}.png"
    _save_figure(fig, plot_path)
    return {f"example_windows_system_eval_{split}_plot": str(plot_path)}


def build_cp_sweep_report(experiment_dir: str | Path, split: str = "val") -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    summary_path = experiment_dir / "cp_sweep" / split / "cp_sweep_summary.csv"
    if not summary_path.exists():
        return {}

    df = pd.read_csv(summary_path)
    if df.empty:
        return {}
    models = list(dict.fromkeys(df["model"].tolist()))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        subset = df[df["model"] == model_name].copy()
        pivot = subset.pivot_table(
            index="cp_min_size",
            columns="cp_penalty",
            values="MAE_CP",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_title(f"CP Sweep MAE ({model_name})")
        ax.set_xlabel("cp_penalty")
        ax.set_ylabel("cp_min_size")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index])
        values = pivot.to_numpy(dtype=float)
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isnan(value):
                    continue
                text_color = "white" if value >= np.nanmedian(values) else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plot_path = experiment_dir / "reports" / f"cp_sweep_{split}.png"
    _save_figure(fig, plot_path)
    return {f"cp_sweep_{split}_plot": str(plot_path)}


def build_tau_calibration_report(experiment_dir: str | Path) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    summary_path = experiment_dir / "tau_calibration" / "tau_calibration_summary.csv"
    outputs: dict[str, str] = {}
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        test_df = df[df["split"] == "test"].copy()
        if not test_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            pivot_mae = test_df.pivot_table(index="calibrator", columns="model", values="MAE_CP")
            pivot_tol = test_df.pivot_table(index="calibrator", columns="model", values="tolerance_hit_rate")
            pivot_mae.plot(kind="bar", ax=axes[0], rot=25)
            axes[0].set_title("Tau Calibration Test MAE_CP")
            axes[0].grid(axis="y", alpha=0.3)
            pivot_tol.plot(kind="bar", ax=axes[1], rot=25)
            axes[1].set_title("Tau Calibration Test Tolerance Hit Rate")
            axes[1].grid(axis="y", alpha=0.3)
            plot_path = experiment_dir / "reports" / "tau_calibration_test.png"
            _save_figure(fig, plot_path)
            outputs["tau_calibration_test_plot"] = str(plot_path)

    feat_path = experiment_dir / "tau_calibration" / "tau_calibration_feature_importances.csv"
    if feat_path.exists():
        try:
            feat_df = pd.read_csv(feat_path)
        except pd.errors.EmptyDataError:
            feat_df = pd.DataFrame()
        if not feat_df.empty:
            top = feat_df.head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top["feature"], top["importance"], color="#2ca02c")
            ax.set_title("Top Tau Calibration Feature Importances")
            ax.grid(axis="x", alpha=0.3)
            plot_path = experiment_dir / "reports" / "tau_calibration_feature_importance.png"
            _save_figure(fig, plot_path)
            outputs["tau_calibration_feature_importance_plot"] = str(plot_path)
    return outputs


def build_calibrated_system_eval_report(
    experiment_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    comparison_path = experiment_dir / "calibrated_system_eval" / "calibrated_system_eval_comparison.csv"
    if not comparison_path.exists():
        return {}

    try:
        df = pd.read_csv(comparison_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        return {}
    subset = df[df["split"] == split].copy()
    if subset.empty:
        return {}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    metric_specs = [
        ("MAE_CP", "MAE CP"),
        ("tolerance_hit_rate", "Tolerance Hit Rate"),
        ("coverage_rate", "Coverage Rate"),
        ("sharpness", "Sharpness"),
    ]
    x = np.arange(len(subset))
    width = 0.36
    for ax, (metric_key, title) in zip(axes.reshape(-1), metric_specs):
        raw_values = subset[f"raw_{metric_key}"].to_numpy(dtype=float)
        calibrated_values = subset[f"calibrated_{metric_key}"].to_numpy(dtype=float)
        ax.bar(x - width / 2, raw_values, width=width, label="raw", color="#7f7f7f")
        ax.bar(x + width / 2, calibrated_values, width=width, label="calibrated", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(subset["model"], rotation=20)
        ax.set_title(f"{title} ({split})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    plot_path = experiment_dir / "reports" / f"calibrated_system_eval_{split}.png"
    _save_figure(fig, plot_path)
    return {f"calibrated_system_eval_{split}_plot": str(plot_path)}


def build_example_window_report(
    experiment_dir: str | Path,
    *,
    split: str = "test",
    models: Sequence[str] = ("lstm", "deepar", "chronos2"),
) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    forecast_dir = experiment_dir / "forecast_eval" / split
    selected = _select_example_window_row(
        forecast_dir,
        window_suffix="forecast_windows",
        models=models,
    )
    if selected is None:
        return {}

    model_name, row = selected
    history = _parse_json_series(row["history"])
    future_true = _parse_json_series(row["future_true"])
    y50 = _parse_json_series(row["y_pred_median"])
    y95 = _parse_json_series(row["y_pred_95"])

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x_hist = np.arange(history.size)
    x_future = np.arange(history.size, history.size + future_true.size)
    ax.plot(x_hist, history, label="history", color="#444444")
    ax.plot(x_future, future_true, label="future_true", color="#000000")
    ax.plot(x_future, y50, label="q50", color="#1f77b4")
    ax.plot(x_future, y95, label="q95", color="#ff7f0e")
    ax.set_title(f"{model_name} | series={row['series']} | start={int(row['start_index'])}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plot_path = experiment_dir / "reports" / f"example_windows_{split}.png"
    _save_figure(fig, plot_path)
    return {f"example_windows_{split}_plot": str(plot_path)}


def build_report_metadata(experiment_dir: str | Path) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    metadata: dict[str, Any] = {
        "forecast_metrics": {
            "interval_width_mean": {
                "label": "Forecast Interval Width",
                "definition": "Mean one-sided forecast margin mean(q95 - q50) over all forecast horizon steps and windows.",
            }
        }
    }

    comparison_path = experiment_dir / "calibrated_system_eval" / "calibrated_system_eval_comparison.csv"
    if comparison_path.exists():
        try:
            comparison_df = pd.read_csv(comparison_path)
        except pd.errors.EmptyDataError:
            comparison_df = pd.DataFrame()
        metadata["calibrated_system_eval"] = {
            "comparison_csv": str(comparison_path),
            "rows": comparison_df.to_dict("records"),
        }

    metadata_path = experiment_dir / "reports" / "report_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return {"report_metadata": str(metadata_path)}


def build_report_bundle(experiment_dir: str | Path) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    outputs: dict[str, str] = {}
    outputs.update(build_prepare_data_report(experiment_dir))
    outputs.update(build_training_report(experiment_dir))
    outputs.update(build_forecast_eval_report(experiment_dir))
    outputs.update(build_system_eval_report(experiment_dir))
    outputs.update(build_system_example_window_report(experiment_dir))
    outputs.update(build_cp_sweep_report(experiment_dir))
    outputs.update(build_tau_calibration_report(experiment_dir))
    outputs.update(build_calibrated_system_eval_report(experiment_dir))
    outputs.update(build_example_window_report(experiment_dir))
    outputs.update(build_report_metadata(experiment_dir))
    index_path = experiment_dir / "reports" / "report_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(outputs, indent=2))
    outputs["report_index"] = str(index_path)
    return outputs
