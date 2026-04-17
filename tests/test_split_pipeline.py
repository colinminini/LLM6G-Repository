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

from src.dataset import DeepARWindowDataset, TFTWindowDataset, TrafficWindowDataset
from src.deepar_support import build_feature_spec, build_time_feature_matrix
from src.experiment import (
    build_experiment_manifest,
    build_canonical_eval_starts_map,
    build_sampled_starts_map,
    canonical_eval_start_indices,
    cadence_from_timestamps,
    daily_seasonal_period,
    validate_regular_timestamps,
    valid_window_start_indices,
)
from src.loader import DataLoaderConfig, build_dataloaders, build_datasets
from src.models import DeepARForecast, LSTMForecast, TFTForecast
from src.pipeline import Chronos2ZeroShotForecaster, SeasonalNaiveForecaster, TorchCheckpointForecaster


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

    def test_canonical_eval_starts_are_non_overlapping_and_anchored(self) -> None:
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

            starts = canonical_eval_start_indices(manifest, "test")
            self.assertEqual(starts.tolist(), [80, 84, 88, 92, 96])

            starts_map = build_canonical_eval_starts_map(manifest=manifest, split="test")
            self.assertEqual(starts_map["S0"], [80, 84, 88, 92, 96])
            self.assertEqual(starts_map["S1"], [80, 84, 88, 92, 96])

    def test_training_window_step_reduces_train_density_only(self) -> None:
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

            dense_train = TrafficWindowDataset(
                data_path,
                split="train",
                context_length=8,
                forecast_length=4,
                manifest=manifest,
            )
            stepped_train = TrafficWindowDataset(
                data_path,
                split="train",
                context_length=8,
                forecast_length=4,
                manifest=manifest,
                window_step=4,
            )
            dense_val = TrafficWindowDataset(
                data_path,
                split="val",
                context_length=8,
                forecast_length=4,
                manifest=manifest,
            )

            self.assertEqual(dense_train.start_indices[:5], [8, 9, 10, 11, 12])
            self.assertEqual(stepped_train.start_indices[:5], [8, 12, 16, 20, 24])
            self.assertLess(len(stepped_train), len(dense_train))
            self.assertEqual(
                len(dense_val),
                manifest.num_series * len(valid_window_start_indices(manifest, "val")),
            )

    def test_deepar_feature_generation_supports_subdaily_covariates(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=24, freq="10min")
        spec = build_feature_spec(timestamps, cadence="10min", train_end_idx=12)
        matrix = build_time_feature_matrix(timestamps, cadence="10min", feature_spec=spec)

        self.assertEqual(spec.feature_names, ("age", "time_of_day_step", "day_of_week"))
        self.assertEqual(matrix.shape, (24, 3))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertLess(abs(float(matrix[:12, 0].mean())), 1e-6)

    def test_deepar_dataset_exposes_item_ids_padding_and_sampling_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            timestamps = pd.date_range("2024-01-01", periods=40, freq="15min")
            pd.DataFrame(
                {
                    "timestamp": timestamps.astype(str),
                    "S0": np.linspace(1.0, 2.0, 40),
                    "S1": np.linspace(10.0, 20.0, 40),
                }
            ).to_csv(data_path, index=False)
            manifest = build_experiment_manifest(
                data_path=data_path,
                context_length=8,
                horizon=4,
            )
            dataset = DeepARWindowDataset(
                data_path,
                split="train",
                context_length=8,
                forecast_length=4,
                manifest=manifest,
            )

            first = dataset[0]
            self.assertEqual(tuple(first["context"].shape), (8,))
            self.assertEqual(tuple(first["target"].shape), (4,))
            self.assertEqual(tuple(first["time_features"].shape), (12, 3))
            self.assertEqual(tuple(first["observed_mask"].shape), (12,))
            self.assertEqual(int(first["start_index"]), 0)
            self.assertEqual(float(first["observed_mask"][:8].sum()), 0.0)
            self.assertEqual(len(dataset.sampling_weights), len(dataset))

            weight_idx = min(dataset.num_windows - 1, 10)
            s0_weight = float(dataset.sampling_weights[weight_idx])
            s1_weight = float(dataset.sampling_weights[dataset.num_windows + weight_idx])
            self.assertGreater(s1_weight, s0_weight)

    def test_tft_dataset_exposes_static_and_known_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            _write_dataset(data_path, rows=64, series_count=2, freq="10min")
            manifest = build_experiment_manifest(
                data_path=data_path,
                context_length=8,
                horizon=4,
            )
            spec = build_feature_spec(
                pd.read_csv(data_path)["timestamp"],
                cadence=manifest.cadence,
                train_end_idx=manifest.train_split.end_idx,
            )
            dataset = TFTWindowDataset(
                data_path,
                split="val",
                context_length=8,
                forecast_length=4,
                manifest=manifest,
                feature_spec=spec,
            )

            first = dataset[0]
            self.assertEqual(tuple(first["context"].shape), (8,))
            self.assertEqual(tuple(first["target"].shape), (4,))
            self.assertEqual(tuple(first["past_inputs"].shape), (8, 1 + spec.num_features))
            self.assertEqual(tuple(first["future_inputs"].shape), (4, spec.num_features))
            self.assertEqual(tuple(first["static_categorical"].shape), (1,))

    def test_torch_checkpoint_forecasters_respect_quantile_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            context_length = 8
            horizon = 4
            history = np.linspace(0.0, 1.0, context_length, dtype=float)
            history_batch = np.stack([history, history + 1.0], axis=0)

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
            lstm_batch_pred = lstm_forecaster.predict_quantiles_batch(history_batch, horizon)
            self.assertEqual(tuple(lstm_batch_pred.y_pred_median.shape), (2, horizon))
            self.assertEqual(tuple(lstm_batch_pred.y_pred_95.shape), (2, horizon))
            self.assertTrue(np.all(lstm_batch_pred.y_pred_95 >= lstm_batch_pred.y_pred_median))
            np.testing.assert_allclose(
                lstm_batch_pred.y_pred_median[0],
                lstm_pred.y_pred_median,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                lstm_batch_pred.y_pred_95[0],
                lstm_pred.y_pred_95,
                atol=1e-6,
            )

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
            deepar_batch_pred = deepar_forecaster.predict_quantiles_batch(history_batch, horizon)
            self.assertEqual(tuple(deepar_batch_pred.y_pred_median.shape), (2, horizon))
            self.assertEqual(tuple(deepar_batch_pred.y_pred_95.shape), (2, horizon))
            self.assertTrue(np.all(deepar_batch_pred.y_pred_95 >= deepar_batch_pred.y_pred_median))

            tft = TFTForecast(
                context_length=context_length,
                forecast_length=horizon,
                hidden_size=8,
                num_lstm_layers=1,
                num_heads=2,
                quantiles=(0.5, 0.95),
                num_past_features=1,
                num_future_features=0,
            )
            tft_path = tmp_path / "tft.pt"
            torch.save(tft.state_dict(), tft_path)
            tft_forecaster = TorchCheckpointForecaster(
                model_type="tft",
                checkpoint_path=tft_path,
                context_length=context_length,
                forecast_length=horizon,
                quantiles=(0.5, 0.95),
            )
            tft_pred = tft_forecaster.predict_quantiles(history, horizon)
            self.assertEqual(tuple(tft_pred.y_pred_median.shape), (horizon,))
            self.assertEqual(tuple(tft_pred.y_pred_95.shape), (horizon,))
            self.assertTrue(np.all(tft_pred.y_pred_95 >= tft_pred.y_pred_median))
            tft_batch_pred = tft_forecaster.predict_quantiles_batch(history_batch, horizon)
            self.assertEqual(tuple(tft_batch_pred.y_pred_median.shape), (2, horizon))
            self.assertEqual(tuple(tft_batch_pred.y_pred_95.shape), (2, horizon))
            self.assertTrue(np.all(tft_batch_pred.y_pred_95 >= tft_batch_pred.y_pred_median))
            np.testing.assert_allclose(
                tft_batch_pred.y_pred_median[0],
                tft_pred.y_pred_median,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                tft_batch_pred.y_pred_95[0],
                tft_pred.y_pred_95,
                atol=1e-6,
            )

    def test_deepar_checkpoint_forecaster_honors_requested_upper_quantile(self) -> None:
        # DeepAR Gaussian inference uses the analytic path: y50 = mean,
        # y_upper = mean + z(q) * sigma. Stub analytic_gaussian_forecast to
        # verify the requested upper quantile is honored end-to-end.
        from scipy.stats import norm

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            context_length = 8
            horizon = 4
            history = np.linspace(0.0, 1.0, context_length, dtype=float)

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
                quantiles=(0.5, 0.99),
                deepar_num_samples=100,
                device="cpu",
            )
            self.assertEqual(deepar_forecaster.output_quantiles, (0.5, 0.99))

            stub_mean = torch.arange(1, 1 + horizon, dtype=torch.float32).reshape(1, horizon)
            stub_sigma = torch.full((1, horizon), 2.0, dtype=torch.float32)

            def _fixed_analytic(**kwargs: object) -> dict[str, torch.Tensor]:
                return {"mean": stub_mean, "sigma": stub_sigma}

            deepar_forecaster.model.analytic_gaussian_forecast = _fixed_analytic  # type: ignore[method-assign]
            pred = deepar_forecaster.predict_quantiles(history, horizon)
            z = float(norm.ppf(0.99))
            expected_q50 = stub_mean[0].numpy()
            expected_q99 = (stub_mean + z * stub_sigma)[0].numpy()
            np.testing.assert_allclose(pred.y_pred_median, expected_q50, atol=1e-6)
            np.testing.assert_allclose(pred.y_pred_95, expected_q99, atol=1e-6)

    def test_seasonal_naive_forecaster_uses_daily_lag_and_fallback(self) -> None:
        forecaster = SeasonalNaiveForecaster(season_length=4)
        history = np.asarray([1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
        pred = forecaster.predict_quantiles(history, horizon=4)
        np.testing.assert_allclose(pred.y_pred_median, np.asarray([11.0, 12.0, 13.0, 14.0]))
        self.assertTrue(np.all(pred.y_pred_95 >= pred.y_pred_median))
        batch_pred = forecaster.predict_quantiles_batch(
            np.stack([history, history + 10.0], axis=0),
            horizon=4,
        )
        self.assertEqual(tuple(batch_pred.y_pred_median.shape), (2, 4))
        self.assertEqual(tuple(batch_pred.y_pred_95.shape), (2, 4))
        np.testing.assert_allclose(batch_pred.y_pred_median[0], pred.y_pred_median)
        self.assertTrue(np.all(batch_pred.y_pred_95 >= batch_pred.y_pred_median))

        short_history = np.asarray([5.0, 6.0], dtype=float)
        short_pred = forecaster.predict_quantiles(short_history, horizon=3)
        np.testing.assert_allclose(short_pred.y_pred_median, np.asarray([6.0, 6.0, 6.0]))
        self.assertTrue(np.all(short_pred.y_pred_95 >= short_pred.y_pred_median))

    def test_non_torch_forecasters_ignore_batch_metadata_kwargs(self) -> None:
        seasonal = SeasonalNaiveForecaster(season_length=4)
        seasonal_batch = seasonal.predict_quantiles_batch(
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=float),
            horizon=2,
            series_names=["S0"],
            start_indices=[10],
        )
        self.assertEqual(tuple(seasonal_batch.y_pred_median.shape), (1, 2))

        class _FakeChronosPipeline:
            quantiles = [round(step / 20, 2) for step in range(21)]

            def predict(self, inputs, prediction_length):
                batch = len(inputs)
                grid = np.linspace(0.0, 1.0, prediction_length, dtype=float)
                quantiles = np.stack([grid + 0.1 * idx for idx in range(21)], axis=0)
                return np.stack([quantiles for _ in range(batch)], axis=0)

        chronos = Chronos2ZeroShotForecaster.__new__(Chronos2ZeroShotForecaster)
        chronos.pipeline = _FakeChronosPipeline()
        chronos_batch = chronos.predict_quantiles_batch(
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=float),
            horizon=2,
            series_names=["S0"],
            start_indices=[10],
        )
        self.assertEqual(tuple(chronos_batch.y_pred_median.shape), (1, 2))
        self.assertTrue(np.all(chronos_batch.y_pred_95 >= chronos_batch.y_pred_median))

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
                "--max-iterations",
                "4",
                "--patience-iterations",
                "4",
                "--validate-every",
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
            self.assertEqual(history["iteration"], [2, 4])

            run_cmd(
                "src/forecast_eval.py",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(run_dir),
                "--models",
                "lstm",
                "--splits",
                "val,test",
                "--max-windows-per-series",
                "3",
                "--device",
                "cpu",
            )
            training_summary = json.loads((run_dir / "training" / "training_summary.json").read_text())
            self.assertIn("monitor_metrics", training_summary["models"]["lstm"])
            self.assertEqual(training_summary["models"]["lstm"]["canonical_metrics"], {})
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
                "train",
                "--train-window-step",
                "2",
                "--cp-detector",
                "ruptures_pelt",
                "--cp-penalties",
                "1",
                "--cp-min-sizes",
                "2",
                "--cp-jumps",
                "1",
                "--output-dir",
                str(run_dir / "cp_sweep" / "train"),
            )
            sweep_manifest = json.loads((run_dir / "cp_sweep" / "train" / "cp_sweep_manifest.json").read_text())
            self.assertEqual(sweep_manifest["split"], "train")
            self.assertEqual(sweep_manifest["train_window_step"], 2)

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
            self.assertTrue(
                (run_dir / "tau_calibration" / "forecast_cache" / "train" / "lstm_forecast_windows.csv").exists()
            )
            summary_df = pd.read_csv(run_dir / "tau_calibration" / "tau_calibration_summary.csv")
            self.assertEqual(set(summary_df["split"]), {"train", "val", "test"})
            best_df = pd.read_csv(run_dir / "tau_calibration" / "tau_calibration_best_test_rows.csv")
            self.assertTrue(any(str(value).startswith("best_val_") for value in best_df["selected_on"]))

    def test_deepar_training_summary_uses_monitor_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
            _write_dataset(data_path, rows=120, series_count=2)

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
            run_cmd(
                "src/train.py",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(run_dir),
                "--models",
                "deepar",
                "--max-iterations",
                "2",
                "--patience-iterations",
                "2",
                "--validate-every",
                "1",
                "--batch-size",
                "8",
                "--hidden-size",
                "8",
                "--num-layers",
                "1",
                "--device",
                "cpu",
            )
            training_summary = json.loads((run_dir / "training" / "training_summary.json").read_text())
            self.assertIn("monitor_metrics", training_summary["models"]["deepar"])
            self.assertEqual(training_summary["models"]["deepar"]["canonical_metrics"], {})
            self.assertIn(
                "monitor_val_rmse",
                training_summary["models"]["deepar"]["monitor_metrics"],
            )

    def test_deepar_loader_uses_same_train_windows_as_other_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            _write_dataset(data_path, rows=120, series_count=2)
            manifest = build_experiment_manifest(
                data_path=data_path,
                context_length=8,
                horizon=4,
            )
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
            spec = build_feature_spec(
                pd.read_csv(data_path)["timestamp"],
                cadence=manifest.cadence,
                train_end_idx=manifest.train_split.end_idx,
            )

            deepar_train, _, _ = build_datasets(
                DataLoaderConfig(
                    data_path=data_path,
                    manifest_path=manifest_path,
                    model_name="deepar",
                    context_length=8,
                    forecast_length=4,
                    train_window_step=2,
                    deepar_feature_spec=spec,
                )
            )
            traffic_train, _, _ = build_datasets(
                DataLoaderConfig(
                    data_path=data_path,
                    manifest_path=manifest_path,
                    model_name="lstm",
                    context_length=8,
                    forecast_length=4,
                    train_window_step=2,
                )
            )
            self.assertEqual(deepar_train.start_indices, traffic_train.start_indices)

            train_loader, _, _ = build_dataloaders(
                DataLoaderConfig(
                    data_path=data_path,
                    manifest_path=manifest_path,
                    model_name="deepar",
                    context_length=8,
                    forecast_length=4,
                    train_window_step=2,
                    deepar_feature_spec=spec,
                    batch_size=4,
                )
            )
            self.assertEqual(train_loader.sampler.__class__.__name__, "RandomSampler")

    def test_tft_training_summary_uses_monitor_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
            _write_dataset(data_path, rows=120, series_count=2)

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
            run_cmd(
                "src/train.py",
                "--manifest-path",
                str(manifest_path),
                "--output-dir",
                str(run_dir),
                "--models",
                "tft",
                "--max-iterations",
                "2",
                "--patience-iterations",
                "2",
                "--validate-every",
                "1",
                "--batch-size",
                "8",
                "--hidden-size",
                "8",
                "--num-layers",
                "1",
                "--tft-num-heads",
                "2",
                "--device",
                "cpu",
            )
            training_summary = json.loads((run_dir / "training" / "training_summary.json").read_text())
            self.assertIn("monitor_metrics", training_summary["models"]["tft"])
            self.assertEqual(training_summary["models"]["tft"]["canonical_metrics"], {})
            self.assertIn(
                "monitor_val_rmse",
                training_summary["models"]["tft"]["monitor_metrics"],
            )

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
                "--max-iterations",
                "4",
                "--patience-iterations",
                "4",
                "--validate-every",
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
