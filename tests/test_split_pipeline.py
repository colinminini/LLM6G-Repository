from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset import TrafficWindowDataset
from src.experiment import (
    build_experiment_manifest,
    build_sampled_starts_map,
    cadence_from_timestamps,
    daily_seasonal_period,
    validate_regular_timestamps,
    valid_window_start_indices,
)
from src.models import DeepARForecast, LSTMForecast
from src.pipeline import SeasonalNaiveForecaster, TorchCheckpointForecaster


def _write_dataset(
    path: Path,
    rows: int = 120,
    series_count: int = 3,
    *,
    freq: str = "15min",
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq=freq)
    frame = {"timestamp": timestamps.astype(str)}
    x = np.arange(rows, dtype=float)
    for idx in range(series_count):
        frame[f"S{idx}"] = (
            np.sin(x / (4.0 + idx))
            + 0.2 * np.cos(x / (7.0 + idx))
            + 0.01 * x
            + idx
        )
    pd.DataFrame(frame).to_csv(path, index=False)

class SplitPipelineTests(unittest.TestCase):
    def test_cadence_inference_supports_multiple_regular_intervals(self) -> None:
        ten_min = pd.date_range("2024-01-01", periods=6, freq="10min")
        fifteen_min = pd.date_range("2024-01-01", periods=6, freq="15min")
        forty_five_min = pd.date_range("2024-01-01", periods=6, freq="45min")

        self.assertEqual(cadence_from_timestamps(ten_min)[1], "10min")
        self.assertEqual(cadence_from_timestamps(fifteen_min)[1], "15min")
        self.assertEqual(cadence_from_timestamps(forty_five_min)[1], "45min")
        self.assertEqual(daily_seasonal_period("10min"), 144)
        self.assertEqual(daily_seasonal_period("15min"), 96)

    def test_regular_timestamp_validation_rejects_malformed_and_irregular_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "malformed timestamps"):
            validate_regular_timestamps(
                pd.Series(["2024-01-01 00:00:00", "not-a-timestamp", "2024-01-01 00:20:00"]),
                data_path="toy.csv",
            )

        with self.assertRaisesRegex(ValueError, "irregular timestamps"):
            validate_regular_timestamps(
                pd.Series(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:10:00",
                        "2024-01-01 00:25:00",
                    ]
                ),
                data_path="toy.csv",
            )

    def test_manifest_split_bounds_and_window_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            _write_dataset(data_path, rows=100, series_count=2)
            manifest = build_experiment_manifest(
                data_path=data_path,
                context_length=8,
                horizon=4,
                train_ratio=0.70,
                val_ratio=0.10,
                test_ratio=0.20,
            )

            self.assertEqual(manifest.train_split.end_idx, 70)
            self.assertEqual(manifest.val_split.start_idx, 70)
            self.assertEqual(manifest.val_split.end_idx, 80)
            self.assertEqual(manifest.test_split.start_idx, 80)

            val_starts = valid_window_start_indices(manifest, "val")
            self.assertEqual(int(val_starts[0]), 70)
            self.assertEqual(int(val_starts[-1]), 76)

            dataset = TrafficWindowDataset(
                data_path,
                split="test",
                context_length=8,
                forecast_length=4,
                manifest=manifest,
            )
            context, target = dataset[0]
            self.assertEqual(tuple(context.shape), (8,))
            self.assertEqual(tuple(target.shape), (4,))
            self.assertEqual(
                len(dataset),
                manifest.num_series * len(valid_window_start_indices(manifest, "test")),
            )

    def test_sampled_starts_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            _write_dataset(data_path, rows=96, series_count=2)
            manifest = build_experiment_manifest(
                data_path=data_path,
                context_length=8,
                horizon=4,
            )
            self.assertEqual(manifest.cadence, "15min")

            first = build_sampled_starts_map(
                manifest=manifest,
                split="test",
                sampling_mode="random",
                random_windows_per_series=5,
                random_seed=123,
            )
            second = build_sampled_starts_map(
                manifest=manifest,
                split="test",
                sampling_mode="random",
                random_windows_per_series=5,
                random_seed=123,
            )
            third = build_sampled_starts_map(
                manifest=manifest,
                split="test",
                sampling_mode="random",
                random_windows_per_series=5,
                random_seed=456,
            )

            self.assertEqual(first, second)
            self.assertNotEqual(first, third)

    def test_torch_checkpoint_forecasters_respect_quantile_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            context_length = 8
            horizon = 4
            history = np.linspace(0.0, 1.0, context_length, dtype=float)

            lstm = LSTMForecast(
                context_length=context_length,
                forecast_length=horizon,
                hidden_size=4,
                num_layers=1,
                quantiles=(0.5, 0.95),
            )
            lstm_path = tmp_path / "lstm.pt"
            torch.save(lstm.state_dict(), lstm_path)
            lstm_forecaster = TorchCheckpointForecaster(
                model_type="lstm",
                checkpoint_path=lstm_path,
                context_length=context_length,
                forecast_length=horizon,
                quantiles=(0.5, 0.95),
            )
            lstm_pred = lstm_forecaster.predict_quantiles(history, horizon)
            self.assertEqual(tuple(lstm_pred.y_pred_median.shape), (horizon,))
            self.assertEqual(tuple(lstm_pred.y_pred_95.shape), (horizon,))
            self.assertTrue(np.all(lstm_pred.y_pred_95 >= lstm_pred.y_pred_median))

            deepar = DeepARForecast(
                context_length=context_length,
                forecast_length=horizon,
                hidden_size=4,
                num_layers=1,
            )
            deepar_path = tmp_path / "deepar.pt"
            torch.save(deepar.state_dict(), deepar_path)
            deepar_forecaster = TorchCheckpointForecaster(
                model_type="deepar",
                checkpoint_path=deepar_path,
                context_length=context_length,
                forecast_length=horizon,
                quantiles=(0.5, 0.95),
            )
            deepar_pred = deepar_forecaster.predict_quantiles(history, horizon)
            self.assertEqual(tuple(deepar_pred.y_pred_median.shape), (horizon,))
            self.assertEqual(tuple(deepar_pred.y_pred_95.shape), (horizon,))
            self.assertTrue(np.all(deepar_pred.y_pred_95 >= deepar_pred.y_pred_median))

    def test_seasonal_naive_forecaster_uses_daily_lag_and_fallback(self) -> None:
        forecaster = SeasonalNaiveForecaster(season_length=4)
        history = np.asarray([1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
        pred = forecaster.predict_quantiles(history, horizon=4)
        np.testing.assert_allclose(pred.y_pred_median, np.asarray([11.0, 12.0, 13.0, 14.0]))
        self.assertTrue(np.all(pred.y_pred_95 >= pred.y_pred_median))

        short_history = np.asarray([5.0, 6.0], dtype=float)
        short_pred = forecaster.predict_quantiles(short_history, horizon=3)
        np.testing.assert_allclose(short_pred.y_pred_median, np.asarray([6.0, 6.0, 6.0]))
        self.assertTrue(np.all(short_pred.y_pred_95 >= short_pred.y_pred_median))

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the integration smoke test",
    )
    def test_end_to_end_split_aware_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
            _write_dataset(data_path, rows=144, series_count=2)

            def run_cmd(*args: str) -> None:
                subprocess.run([sys.executable, *args], cwd=Path(__file__).resolve().parents[1], check=True)

            run_cmd(
                "src/prepare_data.py",
                "--data-path",
                str(data_path),
                "--context-length",
                "8",
                "--horizon",
                "4",
                "--output-dir",
                str(run_dir),
            )
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())

            run_cmd(
                "src/train.py",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(run_dir),
                "--models",
                "lstm",
                "--max-epochs",
                "2",
                "--patience",
                "2",
                "--log-every",
                "2",
                "--batch-size",
                "8",
                "--hidden-size",
                "8",
                "--num-layers",
                "1",
                "--device",
                "cpu",
            )
            self.assertTrue((run_dir / "checkpoints" / "lstm_toy_best.pt").exists())
            history = json.loads((run_dir / "training" / "lstm_history.json").read_text())
            self.assertEqual(history["epoch"], [1, 2])

            run_cmd(
                "src/forecast_eval.py",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(run_dir),
                "--models",
                "lstm",
                "--splits",
                "train,val,test",
                "--sampling-mode",
                "rolling",
                "--max-windows-per-series",
                "3",
                "--device",
                "cpu",
            )
            self.assertTrue((run_dir / "forecast_eval" / "test" / "lstm_forecast_windows.csv").exists())

            run_cmd(
                "src/system_eval.py",
                "--forecast-dir",
                str(run_dir / "forecast_eval"),
                "--models",
                "lstm",
                "--splits",
                "test",
                "--output-dir",
                str(run_dir / "system_eval"),
                "--cp-penalty",
                "1",
                "--cp-min-size",
                "2",
            )
            self.assertTrue((run_dir / "system_eval" / "test" / "lstm_system_metrics.json").exists())

            run_cmd(
                "src/cp_sweep.py",
                "--forecast-dir",
                str(run_dir / "forecast_eval"),
                "--models",
                "lstm",
                "--split",
                "val",
                "--cp-penalties",
                "1",
                "--cp-min-sizes",
                "2",
                "--cp-jumps",
                "1",
                "--output-dir",
                str(run_dir / "cp_sweep" / "val"),
            )
            sweep_manifest = json.loads((run_dir / "cp_sweep" / "val" / "cp_sweep_manifest.json").read_text())
            self.assertEqual(sweep_manifest["split"], "val")

            run_cmd(
                "src/tau_calibration.py",
                "--forecast-dir",
                str(run_dir / "forecast_eval"),
                "--models",
                "lstm",
                "--data-path",
                str(data_path),
                "--cp-penalty",
                "1",
                "--cp-min-size",
                "2",
                "--output-dir",
                str(run_dir / "tau_calibration"),
            )
            summary_df = pd.read_csv(run_dir / "tau_calibration" / "tau_calibration_summary.csv")
            self.assertEqual(set(summary_df["split"]), {"train", "val", "test"})
            best_df = pd.read_csv(run_dir / "tau_calibration" / "tau_calibration_best_test_rows.csv")
            self.assertTrue(any(str(value).startswith("best_val_") for value in best_df["selected_on"]))

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the cadence integration smoke test",
    )
    def test_end_to_end_split_aware_scripts_support_10min(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy_10min.csv"
            run_dir = tmp_path / "run_10min"
            _write_dataset(data_path, rows=144, series_count=2, freq="10min")

            def run_cmd(*args: str) -> None:
                subprocess.run([sys.executable, *args], cwd=Path(__file__).resolve().parents[1], check=True)

            run_cmd(
                "src/prepare_data.py",
                "--data-path",
                str(data_path),
                "--context-length",
                "8",
                "--horizon",
                "4",
                "--output-dir",
                str(run_dir),
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["cadence"], "10min")

            run_cmd(
                "src/train.py",
                "--manifest-path",
                str(run_dir / "manifest.json"),
                "--output-dir",
                str(run_dir),
                "--models",
                "lstm",
                "--max-epochs",
                "2",
                "--patience",
                "2",
                "--log-every",
                "2",
                "--batch-size",
                "8",
                "--hidden-size",
                "8",
                "--num-layers",
                "1",
                "--device",
                "cpu",
            )
            run_cmd(
                "src/forecast_eval.py",
                "--manifest-path",
                str(run_dir / "manifest.json"),
                "--output-dir",
                str(run_dir),
                "--models",
                "lstm",
                "--splits",
                "test",
                "--sampling-mode",
                "rolling",
                "--max-windows-per-series",
                "3",
                "--device",
                "cpu",
            )
            run_cmd(
                "src/system_eval.py",
                "--forecast-dir",
                str(run_dir / "forecast_eval"),
                "--models",
                "lstm",
                "--splits",
                "test",
                "--output-dir",
                str(run_dir / "system_eval"),
                "--cp-penalty",
                "1",
                "--cp-min-size",
                "2",
            )
            metrics = json.loads(
                (run_dir / "system_eval" / "test" / "lstm_system_metrics.json").read_text()
            )
            self.assertEqual(metrics["num_windows_total"], 6)


if __name__ == "__main__":
    unittest.main()
