from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import src.forecast_eval as forecast_eval_module
import src.progress as progress_module
import src.run_experiment as run_experiment_module
import src.system_eval as system_eval_module
import src.train as train_module
import src.train_utils as train_utils_module
from src.device import default_device_type, resolve_device_name
from src.reporting import (
    _select_example_window_row,
    build_cp_sweep_report,
    build_example_window_report,
    build_system_example_window_report,
    build_training_report,
    publish_report_plots,
)
from src.run_experiment import _filter_train_models


def _write_dataset(path: Path, rows: int = 120, series_count: int = 3) -> None:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="15min")
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


def _write_example_windows_csv(path: Path, *, include_system_fields: bool = False) -> None:
    row = {
        "series": "S0",
        "start_index": 5,
        "history": json.dumps([0.0, 1.0, 2.0, 3.0]),
        "future_true": json.dumps([4.0, 5.0]),
        "y_pred_median": json.dumps([4.5, 5.5]),
        "y_pred_95": json.dumps([5.0, 6.0]),
    }
    if include_system_fields:
        row.update(
            {
                "tau_pred": 1,
                "tau_true": 1,
                "safe_ceiling": 6.5,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_example_windows_csv_with_rows(
    path: Path,
    rows: list[dict[str, object]],
    *,
    include_system_fields: bool = False,
) -> None:
    normalized_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        base_row = {
            "series": str(row.get("series", f"S{idx}")),
            "start_index": int(row.get("start_index", idx)),
            "history": json.dumps(row.get("history", [0.0, 1.0, 2.0, 3.0])),
            "future_true": json.dumps(row.get("future_true", [4.0, 5.0])),
            "y_pred_median": json.dumps(row.get("y_pred_median", [4.5, 5.5])),
            "y_pred_95": json.dumps(row.get("y_pred_95", [5.0, 6.0])),
        }
        if include_system_fields:
            base_row.update(
                {
                    "tau_pred": int(row.get("tau_pred", 1)),
                    "tau_true": int(row.get("tau_true", 1)),
                    "safe_ceiling": float(row.get("safe_ceiling", 6.5)),
                }
            )
        normalized_rows.append(base_row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(normalized_rows).to_csv(path, index=False)


class RunnerWorkflowTests(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[1]

    def _preserve_paths(self, *paths: Path) -> None:
        snapshots = {path: path.read_bytes() if path.exists() else None for path in paths}

        def restore() -> None:
            for path, payload in snapshots.items():
                if payload is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)

        self.addCleanup(restore)

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the runner smoke test",
    )
    def test_run_experiment_smoke_writes_stage_status_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
            published_dir = self._repo_root() / "results" / "plots" / "readme"
            published_paths = [
                published_dir / "forecast_eval_test.png",
                published_dir / "forecast_example_chronos2_test.png",
                published_dir / "system_eval_test.png",
                published_dir / "pipeline_example_chronos2_test.png",
                published_dir / "published_report_index.json",
            ]
            self._preserve_paths(*published_paths)
            _write_dataset(data_path, rows=144, series_count=2)

            subprocess.run(
                [
                    sys.executable,
                    "src/run_experiment.py",
                    "--data-path",
                    str(data_path),
                    "--output-dir",
                    str(run_dir),
                    "--context-length",
                    "8",
                    "--horizon",
                    "4",
                    "--models",
                    "lstm",
                    "--splits",
                    "val,test",
                    "--system-splits",
                    "test",
                    "--max-windows-per-series",
                    "3",
                    "--batch-size",
                    "8",
                    "--max-iterations",
                    "4",
                    "--patience-iterations",
                    "4",
                    "--validate-every",
                    "2",
                    "--log-every",
                    "2",
                    "--hidden-size",
                    "8",
                    "--num-layers",
                    "1",
                    "--device",
                    "cpu",
                    "--cp-penalties",
                    "1",
                    "--cp-min-sizes",
                    "2",
                    "--cp-jumps",
                    "1",
                    "--cp-penalty",
                    "1",
                    "--cp-min-size",
                    "2",
                    "--cp-jump",
                    "1",
                ],
                cwd=self._repo_root(),
                check=True,
            )

            status_path = run_dir / "reports" / "stage_status.json"
            report_index_path = run_dir / "reports" / "report_index.json"
            self.assertTrue(status_path.exists())
            self.assertTrue(report_index_path.exists())

            status = json.loads(status_path.read_text())
            self.assertEqual(
                list(status["stages"].keys()),
                [
                    "prepare_data",
                    "train",
                    "forecast_eval",
                    "cp_sweep",
                    "system_eval",
                ],
            )
            self.assertTrue(all(stage["status"] == "success" for stage in status["stages"].values()))

            expected_reports = [
                run_dir / "reports" / "prepare_data_split_summary.png",
                run_dir / "reports" / "training_loss.png",
                run_dir / "reports" / "forecast_eval_test.png",
                run_dir / "reports" / "example_windows_test.png",
                run_dir / "reports" / "system_eval_test.png",
                run_dir / "reports" / "example_windows_system_eval_test.png",
                run_dir / "reports" / "cp_sweep_train.png",
                run_dir / "reports" / "report_metadata.json",
            ]
            for path in expected_reports:
                self.assertTrue(path.exists(), msg=f"Missing report artifact: {path}")

            metadata = json.loads((run_dir / "reports" / "report_metadata.json").read_text())
            self.assertIn("forecast_metrics", metadata)
            self.assertIn("interval_width_mean", metadata["forecast_metrics"])
            self.assertNotIn("tau_calibration", status["stages"])
            for path in published_paths:
                self.assertTrue(path.exists(), msg=f"Missing published README plot artifact: {path}")

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the tau runner smoke test",
    )
    def test_run_experiment_with_tau_writes_calibrated_system_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
            published_dir = self._repo_root() / "results" / "plots" / "readme"
            published_paths = [
                published_dir / "tau_calibration_test.png",
                published_dir / "calibrated_system_eval_test.png",
                published_dir / "published_report_index.json",
            ]
            self._preserve_paths(*published_paths)
            _write_dataset(data_path, rows=144, series_count=2)

            subprocess.run(
                [
                    sys.executable,
                    "src/run_experiment.py",
                    "--data-path",
                    str(data_path),
                    "--output-dir",
                    str(run_dir),
                    "--context-length",
                    "8",
                    "--horizon",
                    "4",
                    "--models",
                    "lstm",
                    "--splits",
                    "val,test",
                    "--system-splits",
                    "test",
                    "--max-windows-per-series",
                    "3",
                    "--batch-size",
                    "8",
                    "--max-iterations",
                    "4",
                    "--patience-iterations",
                    "4",
                    "--validate-every",
                    "2",
                    "--log-every",
                    "2",
                    "--hidden-size",
                    "8",
                    "--num-layers",
                    "1",
                    "--device",
                    "cpu",
                    "--cp-penalties",
                    "1",
                    "--cp-min-sizes",
                    "2",
                    "--cp-jumps",
                    "1",
                    "--cp-penalty",
                    "1",
                    "--cp-min-size",
                    "2",
                    "--cp-jump",
                    "1",
                    "--with-tau-calibration",
                ],
                cwd=self._repo_root(),
                check=True,
            )

            status = json.loads((run_dir / "reports" / "stage_status.json").read_text())
            self.assertEqual(
                list(status["stages"].keys()),
                [
                    "prepare_data",
                    "train",
                    "forecast_eval",
                    "cp_sweep",
                    "system_eval",
                    "tau_calibration",
                    "calibrated_system_eval",
                ],
            )
            comparison_path = run_dir / "calibrated_system_eval" / "calibrated_system_eval_comparison.csv"
            self.assertTrue(comparison_path.exists())
            comparison_df = pd.read_csv(comparison_path)
            self.assertIn("delta_MAE_CP", comparison_df.columns)
            self.assertTrue((run_dir / "reports" / "calibrated_system_eval_test.png").exists())
            for path in published_paths:
                self.assertTrue(path.exists(), msg=f"Missing published tau README artifact: {path}")

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the runner resume test",
    )
    def test_run_experiment_resume_and_overwrite_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
            _write_dataset(data_path, rows=128, series_count=2)

            base_cmd = [
                sys.executable,
                "src/run_experiment.py",
                "--data-path",
                str(data_path),
                "--output-dir",
                str(run_dir),
                "--context-length",
                "8",
                "--horizon",
                "4",
                "--models",
                "lstm",
                "--batch-size",
                "8",
                "--max-iterations",
                "4",
                "--patience-iterations",
                "4",
                "--validate-every",
                "2",
                "--log-every",
                "2",
                "--hidden-size",
                "8",
                "--num-layers",
                "1",
                "--device",
                "cpu",
            ]

            subprocess.run(
                base_cmd + ["--stages", "prepare_data,train"],
                cwd=self._repo_root(),
                check=True,
            )

            rerun = subprocess.run(
                base_cmd + ["--stages", "prepare_data,train"],
                cwd=self._repo_root(),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("--resume-from or --overwrite", rerun.stderr)

            subprocess.run(
                base_cmd
                + [
                    "--stages",
                    "prepare_data,train,forecast_eval",
                    "--resume-from",
                    "forecast_eval",
                    "--sampling-mode",
                    "rolling",
                    "--max-windows-per-series",
                    "2",
                ],
                cwd=self._repo_root(),
                check=True,
            )
            status = json.loads((run_dir / "reports" / "stage_status.json").read_text())
            self.assertEqual(status["stages"]["forecast_eval"]["status"], "success")

            subprocess.run(
                base_cmd + ["--stages", "prepare_data", "--overwrite"],
                cwd=self._repo_root(),
                check=True,
            )
            status = json.loads((run_dir / "reports" / "stage_status.json").read_text())
            self.assertEqual(status["stages"]["prepare_data"]["status"], "success")

    def test_experiment_report_notebook_is_artifact_reader(self) -> None:
        notebook_path = self._repo_root() / "notebooks" / "experiment_report.ipynb"
        payload = json.loads(notebook_path.read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))

        self.assertIn("EXPERIMENT_DIR", source)
        self.assertIn("report_index.json", source)
        self.assertIn("report_metadata.json", source)
        self.assertIn("stage_status.json", source)
        self.assertIn("example_windows_system_eval_test.png", source)
        self.assertIn("calibrated_system_eval_comparison.csv", source)
        self.assertIn("('chronos2', 'lstm', 'deepar', 'tft', 'seasonal_naive')", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("src/train.py", source)
        self.assertNotIn("src/forecast_eval.py", source)
        self.assertNotIn("src/system_eval.py", source)

    def test_runner_filters_zero_shot_models_out_of_train_stage(self) -> None:
        self.assertEqual(
            _filter_train_models("lstm,deepar,tft,chronos2,seasonal_naive"),
            "lstm,deepar,tft",
        )
        self.assertEqual(_filter_train_models("chronos2,lstm"), "lstm")
        self.assertEqual(_filter_train_models("seasonal_naive,lstm"), "lstm")
        with self.assertRaises(ValueError):
            _filter_train_models("chronos2")

    def test_forecast_eval_resolves_no_checkpoint_for_non_trainable_models(self) -> None:
        self.assertIsNone(
            forecast_eval_module._resolve_checkpoint(
                explicit_path=None,
                output_dir=Path("results/run"),
                model_name="chronos2",
                dataset_name="toy",
            )
        )
        self.assertIsNone(
            forecast_eval_module._resolve_checkpoint(
                explicit_path=None,
                output_dir=Path("results/run"),
                model_name="seasonal_naive",
                dataset_name="toy",
            )
        )

    def test_default_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        with mock.patch("src.device.torch.cuda.is_available", return_value=True), mock.patch(
            "src.device.is_mps_available", return_value=True
        ):
            self.assertEqual(default_device_type(), "cuda")

        with mock.patch("src.device.torch.cuda.is_available", return_value=False), mock.patch(
            "src.device.is_mps_available", return_value=True
        ):
            self.assertEqual(default_device_type(), "mps")

        with mock.patch("src.device.torch.cuda.is_available", return_value=False), mock.patch(
            "src.device.is_mps_available", return_value=False
        ):
            self.assertEqual(default_device_type(), "cpu")

    def test_training_progress_updates_during_iteration(self) -> None:
        class _FakeTqdm:
            def __init__(self, *args, **kwargs) -> None:
                self.total = kwargs["total"]
                self.bar_format = kwargs.get("bar_format")
                self.n = 0.0
                self.postfix: dict[str, str] | None = None
                self.closed = False

            def update(self, delta: float) -> None:
                self.n += delta

            def set_postfix(self, postfix: dict[str, str], refresh: bool = False) -> None:
                self.postfix = dict(postfix)

            def write(self, message: str) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        with mock.patch.object(train_utils_module, "tqdm", side_effect=_FakeTqdm):
            progress = train_utils_module._TrainingProgressDisplay(
                "demo",
                total_iterations=20,
                total_train_batches=20,
            )
            fake_bar = progress._bar
            assert fake_bar is not None
            progress.update(
                10,
                phase="train",
                iteration=10,
                epoch=1,
                batch_idx=10,
                batch_total=20,
                batch_loss=0.123,
            )
            self.assertAlmostEqual(fake_bar.n, 10.0)
            self.assertIsNotNone(fake_bar.postfix)
            assert fake_bar.postfix is not None
            self.assertEqual(
                fake_bar.bar_format,
                "{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            self.assertEqual(fake_bar.postfix["iter"], "10/20")
            self.assertEqual(fake_bar.postfix["epoch"], "1")
            self.assertEqual(fake_bar.postfix["batch"], "10/20")
            self.assertEqual(fake_bar.postfix["phase"], "train")
            self.assertEqual(fake_bar.postfix["batch_loss"], "0.123")
            self.assertNotEqual(fake_bar.postfix["eta"], "--:--")
            progress.close()
            self.assertTrue(fake_bar.closed)

    def test_stage_progress_updates_with_eta(self) -> None:
        class _FakeTqdm:
            def __init__(self, *args, **kwargs) -> None:
                self.total = kwargs["total"]
                self.bar_format = kwargs.get("bar_format")
                self.position = kwargs.get("position")
                self.leave = kwargs.get("leave")
                self.n = 0.0
                self.postfix: dict[str, str] | None = None
                self.closed = False

            def update(self, delta: float) -> None:
                self.n += delta

            def set_postfix(self, postfix: dict[str, str], refresh: bool = False) -> None:
                self.postfix = dict(postfix)

            def write(self, message: str) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        with mock.patch.object(progress_module, "tqdm", side_effect=_FakeTqdm):
            progress = progress_module.StageProgressDisplay(
                "forecast_eval:test:lstm",
                total_items=10,
                unit="window",
            )
            fake_bar = progress._bar
            assert fake_bar is not None
            progress.advance(3, phase="predict", series="S0")
            self.assertAlmostEqual(fake_bar.n, 3.0)
            self.assertIsNotNone(fake_bar.postfix)
            assert fake_bar.postfix is not None
            self.assertEqual(
                fake_bar.bar_format,
                "{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            self.assertEqual(fake_bar.postfix["phase"], "predict")
            self.assertEqual(fake_bar.postfix["series"], "S0")
            self.assertNotEqual(fake_bar.postfix["eta"], "--:--")
            progress.close()
            self.assertTrue(fake_bar.closed)
            self.assertEqual(fake_bar.position, 0)
            self.assertTrue(fake_bar.leave)

    def test_stage_progress_supports_nested_positioning(self) -> None:
        class _FakeTqdm:
            def __init__(self, *args, **kwargs) -> None:
                self.total = kwargs["total"]
                self.position = kwargs.get("position")
                self.leave = kwargs.get("leave")
                self.closed = False

            def update(self, delta: float) -> None:
                return None

            def set_postfix(self, postfix: dict[str, str], refresh: bool = False) -> None:
                return None

            def write(self, message: str) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        with mock.patch.object(progress_module, "tqdm", side_effect=_FakeTqdm):
            progress = progress_module.StageProgressDisplay(
                "nested",
                total_items=5,
                unit="window",
                position=1,
                leave=False,
            )
            fake_bar = progress._bar
            assert fake_bar is not None
            self.assertEqual(fake_bar.position, 1)
            self.assertFalse(fake_bar.leave)
            progress.close()
            self.assertTrue(fake_bar.closed)

    def test_resolve_device_name_uses_auto_priority(self) -> None:
        with mock.patch("src.device.torch.cuda.is_available", return_value=False), mock.patch(
            "src.device.is_mps_available", return_value=True
        ):
            self.assertEqual(resolve_device_name("auto"), "mps")

    def test_entrypoint_parsers_default_device_to_auto(self) -> None:
        with mock.patch.object(sys, "argv", ["run_experiment.py"]):
            self.assertEqual(run_experiment_module.parse_args().device, "auto")
            self.assertEqual(run_experiment_module.parse_args().splits, "val,test")
        with mock.patch.object(sys, "argv", ["train.py"]):
            self.assertEqual(train_module.parse_args().device, "auto")
        with mock.patch.object(sys, "argv", ["forecast_eval.py"]):
            self.assertEqual(forecast_eval_module.parse_args().device, "auto")
            self.assertEqual(forecast_eval_module.parse_args().splits, "val,test")
        if importlib.util.find_spec("ruptures") is not None:
            import src.evaluate as evaluate_module

            with mock.patch.object(sys, "argv", ["evaluate.py"]):
                self.assertEqual(evaluate_module.parse_args().device, "auto")

    def test_readme_and_notebook_layout_document_canonical_runner(self) -> None:
        repo_root = self._repo_root()
        readme = (repo_root / "README.md").read_text()
        self.assertIn("src/run_experiment.py", readme)
        self.assertIn("notebooks/experiment_report.ipynb", readme)
        self.assertIn("notebooks/legacy/", readme)
        self.assertIn("results/plots/readme/", readme)

        top_level_notebooks = sorted(path.name for path in (repo_root / "notebooks").glob("*.ipynb"))
        self.assertEqual(top_level_notebooks, ["experiment_report.ipynb"])

    def test_example_window_selection_prefers_chronos2_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forecast_dir = tmp_path / "forecast_eval" / "test"
            system_dir = tmp_path / "system_eval" / "test"
            _write_example_windows_csv(forecast_dir / "lstm_forecast_windows.csv")
            _write_example_windows_csv(forecast_dir / "chronos2_forecast_windows.csv")
            _write_example_windows_csv(forecast_dir / "seasonal_naive_forecast_windows.csv")
            _write_example_windows_csv(system_dir / "deepar_system_windows.csv", include_system_fields=True)

            selected_forecast = _select_example_window_row(
                forecast_dir,
                window_suffix="forecast_windows",
            )
            self.assertIsNotNone(selected_forecast)
            self.assertEqual(selected_forecast[0], "chronos2")

            selected_system = _select_example_window_row(
                system_dir,
                window_suffix="system_windows",
            )
            self.assertIsNotNone(selected_system)
            self.assertEqual(selected_system[0], "deepar")

            forecast_outputs = build_example_window_report(tmp_path, split="test")
            system_outputs = build_system_example_window_report(tmp_path, split="test")
            self.assertIn("example_windows_test_plot", forecast_outputs)
            self.assertTrue((tmp_path / "reports" / "example_windows_test.png").exists())
            self.assertIn("example_windows_chronos2_test_plot", forecast_outputs)
            self.assertTrue((tmp_path / "reports" / "example_windows_chronos2_test.png").exists())
            self.assertIn("example_windows_lstm_test_plot", forecast_outputs)
            self.assertTrue((tmp_path / "reports" / "example_windows_lstm_test.png").exists())
            self.assertIn("example_windows_seasonal_naive_test_plot", forecast_outputs)
            self.assertTrue((tmp_path / "reports" / "example_windows_seasonal_naive_test.png").exists())
            self.assertIn("example_windows_system_eval_test_plot", system_outputs)
            self.assertTrue((tmp_path / "reports" / "example_windows_system_eval_test.png").exists())
            self.assertIn("example_windows_system_eval_deepar_test_plot", system_outputs)
            self.assertTrue((tmp_path / "reports" / "example_windows_system_eval_deepar_test.png").exists())

    def test_render_example_windows_script_supports_per_model_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            experiment_dir = tmp_path / "experiment"
            forecast_dir = experiment_dir / "forecast_eval" / "test"
            system_dir = experiment_dir / "system_eval" / "test"
            reports_dir = experiment_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "report_index.json").write_text(json.dumps({"existing_plot": "keep-me"}))

            _write_example_windows_csv_with_rows(
                forecast_dir / "seasonal_naive_forecast_windows.csv",
                [
                    {"series": "bad", "start_index": 3, "history": [0.0, 0.0], "future_true": [0.5, 0.5]},
                    {
                        "series": "paper",
                        "start_index": 77,
                        "history": [10.0, 11.0, 12.0],
                        "future_true": [13.0, 14.0],
                        "y_pred_median": [13.2, 14.1],
                        "y_pred_95": [14.0, 15.0],
                    },
                ],
            )
            _write_example_windows_csv_with_rows(
                system_dir / "seasonal_naive_system_windows.csv",
                [
                    {
                        "series": "bad",
                        "start_index": 3,
                        "history": [0.0, 0.0],
                        "future_true": [0.5, 0.5],
                        "y_pred_median": [0.6, 0.6],
                        "y_pred_95": [0.8, 0.8],
                        "tau_pred": 0,
                        "tau_true": 1,
                        "safe_ceiling": 0.9,
                    },
                    {
                        "series": "paper",
                        "start_index": 77,
                        "history": [10.0, 11.0, 12.0],
                        "future_true": [13.0, 14.0],
                        "y_pred_median": [13.2, 14.1],
                        "y_pred_95": [14.0, 15.0],
                        "tau_pred": 1,
                        "tau_true": 1,
                        "safe_ceiling": 15.0,
                    }
                ],
                include_system_fields=True,
            )

            subprocess.run(
                [
                    sys.executable,
                    "src/render_example_windows.py",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--split",
                    "test",
                    "--models",
                    "seasonal_naive",
                    "--kinds",
                    "forecast,system",
                    "--model-row-indexes",
                    "seasonal_naive:1",
                    "--strict",
                ],
                cwd=self._repo_root(),
                check=True,
            )

            self.assertTrue((reports_dir / "example_windows_seasonal_naive_test.png").exists())
            self.assertTrue((reports_dir / "example_windows_system_eval_seasonal_naive_test.png").exists())
            report_index = json.loads((reports_dir / "report_index.json").read_text())
            self.assertEqual(report_index["existing_plot"], "keep-me")
            self.assertIn("example_windows_seasonal_naive_test_plot", report_index)
            self.assertIn("example_windows_system_eval_seasonal_naive_test_plot", report_index)

    def test_publish_report_plots_copies_pngs_and_normalizes_example_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            experiment_dir = tmp_path / "experiment"
            reports_dir = experiment_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "forecast_eval_test.png").write_bytes(b"forecast")
            (reports_dir / "example_windows_test.png").write_bytes(b"forecast-example")
            (reports_dir / "example_windows_system_eval_test.png").write_bytes(b"pipeline-example")
            (reports_dir / "report_metadata.json").write_text("{}")
            (reports_dir / "report_index.json").write_text(
                json.dumps(
                    {
                        "forecast_eval_test_plot": str(reports_dir / "forecast_eval_test.png"),
                        "example_windows_test_plot": str(reports_dir / "example_windows_test.png"),
                        "example_windows_system_eval_test_plot": str(
                            reports_dir / "example_windows_system_eval_test.png"
                        ),
                        "report_metadata": str(reports_dir / "report_metadata.json"),
                    }
                )
            )
            publish_dir = tmp_path / "published"

            published = publish_report_plots(experiment_dir, publish_dir=publish_dir)

            self.assertTrue((publish_dir / "forecast_eval_test.png").exists())
            self.assertEqual(
                (publish_dir / "forecast_example_chronos2_test.png").read_bytes(),
                b"forecast-example",
            )
            self.assertEqual(
                (publish_dir / "pipeline_example_chronos2_test.png").read_bytes(),
                b"pipeline-example",
            )
            self.assertNotIn("report_metadata", published)
            manifest = json.loads((publish_dir / "published_report_index.json").read_text())
            self.assertEqual(
                manifest["artifacts"]["example_windows_test_plot"]["published_path"],
                str(publish_dir / "forecast_example_chronos2_test.png"),
            )
            self.assertIn("published_report_index", published)

    def test_training_report_handles_single_iteration_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            training_dir = tmp_path / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "iteration": [5000],
                "train_loss": [1.0],
                "val_loss": [1.2],
                "test_loss": [1.1],
                "train_rmse": [0.9],
                "val_rmse": [1.0],
                "test_rmse": [0.95],
                "train_coverage": [0.7],
                "val_coverage": [0.65],
                "test_coverage": [0.66],
                "train_interval_width": [0.2],
                "val_interval_width": [0.25],
                "test_interval_width": [0.22],
            }
            (training_dir / "lstm_history.json").write_text(json.dumps(payload))
            outputs = build_training_report(tmp_path)
            self.assertIn("training_loss_plot", outputs)
            self.assertTrue((tmp_path / "reports" / "training_loss.png").exists())

    def test_cp_sweep_report_handles_single_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sweep_dir = tmp_path / "cp_sweep" / "train"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"model": "lstm", "cp_detector": "clasp", "cp_min_size": 8, "break_f1": 0.72, "tau_mae_when_both_break": 1.2, "coverage_gap": 0.02, "sharpness": 0.4},
                    {"model": "lstm", "cp_detector": "clasp", "cp_min_size": 10, "break_f1": 0.81, "tau_mae_when_both_break": 1.0, "coverage_gap": 0.03, "sharpness": 0.3},
                    {"model": "lstm", "cp_detector": "clasp", "cp_min_size": 12, "break_f1": 0.77, "tau_mae_when_both_break": 1.5, "coverage_gap": 0.01, "sharpness": 0.5},
                ]
            ).to_csv(sweep_dir / "cp_sweep_summary.csv", index=False)
            outputs = build_cp_sweep_report(tmp_path, split="train")
            self.assertIn("cp_sweep_train_plot", outputs)
            self.assertTrue((tmp_path / "reports" / "cp_sweep_train.png").exists())


if __name__ == "__main__":
    unittest.main()
