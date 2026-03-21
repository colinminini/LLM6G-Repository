from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.reporting import build_cp_sweep_report, build_training_report
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


class RunnerWorkflowTests(unittest.TestCase):
    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the runner smoke test",
    )
    def test_run_experiment_smoke_writes_stage_status_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
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
                    "train,val,test",
                    "--system-splits",
                    "test",
                    "--sampling-mode",
                    "rolling",
                    "--max-windows-per-series",
                    "3",
                    "--batch-size",
                    "8",
                    "--max-iterations",
                    "6",
                    "--patience-iterations",
                    "6",
                    "--validate-every",
                    "3",
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
                    "system_eval",
                    "cp_sweep",
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
                run_dir / "reports" / "cp_sweep_val.png",
                run_dir / "reports" / "report_metadata.json",
            ]
            for path in expected_reports:
                self.assertTrue(path.exists(), msg=f"Missing report artifact: {path}")

            metadata = json.loads((run_dir / "reports" / "report_metadata.json").read_text())
            self.assertIn("forecast_metrics", metadata)
            self.assertIn("interval_width_mean", metadata["forecast_metrics"])
            self.assertNotIn("tau_calibration", status["stages"])

    @unittest.skipUnless(
        importlib.util.find_spec("ruptures") and importlib.util.find_spec("sklearn"),
        "ruptures and scikit-learn are required for the tau runner smoke test",
    )
    def test_run_experiment_with_tau_writes_calibrated_system_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "toy.csv"
            run_dir = tmp_path / "run"
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
                    "train,val,test",
                    "--system-splits",
                    "test",
                    "--sampling-mode",
                    "rolling",
                    "--max-windows-per-series",
                    "3",
                    "--batch-size",
                    "8",
                    "--max-iterations",
                    "6",
                    "--patience-iterations",
                    "6",
                    "--validate-every",
                    "3",
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
                    "system_eval",
                    "cp_sweep",
                    "tau_calibration",
                    "calibrated_system_eval",
                ],
            )
            comparison_path = run_dir / "calibrated_system_eval" / "calibrated_system_eval_comparison.csv"
            self.assertTrue(comparison_path.exists())
            comparison_df = pd.read_csv(comparison_path)
            self.assertIn("delta_MAE_CP", comparison_df.columns)
            self.assertTrue((run_dir / "reports" / "calibrated_system_eval_test.png").exists())

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
        self.assertNotIn("subprocess", source)
        self.assertNotIn("src/train.py", source)
        self.assertNotIn("src/forecast_eval.py", source)
        self.assertNotIn("src/system_eval.py", source)

    def test_runner_filters_zero_shot_models_out_of_train_stage(self) -> None:
        self.assertEqual(_filter_train_models("lstm,deepar,chronos2"), "lstm,deepar")
        self.assertEqual(_filter_train_models("chronos2,lstm"), "lstm")
        with self.assertRaises(ValueError):
            _filter_train_models("chronos2")

    def test_readme_and_notebook_layout_document_canonical_runner(self) -> None:
        repo_root = self._repo_root()
        readme = (repo_root / "README.md").read_text()
        self.assertIn("src/run_experiment.py", readme)
        self.assertIn("notebooks/experiment_report.ipynb", readme)
        self.assertIn("notebooks/legacy/", readme)

        top_level_notebooks = sorted(path.name for path in (repo_root / "notebooks").glob("*.ipynb"))
        self.assertEqual(top_level_notebooks, ["experiment_report.ipynb"])

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
            sweep_dir = tmp_path / "cp_sweep" / "val"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"model": "lstm", "cp_penalty": 10.0, "cp_min_size": 8, "MAE_CP": 1.23},
                    {"model": "lstm", "cp_penalty": 15.0, "cp_min_size": 8, "MAE_CP": 1.11},
                    {"model": "lstm", "cp_penalty": 10.0, "cp_min_size": 10, "MAE_CP": 1.09},
                    {"model": "lstm", "cp_penalty": 15.0, "cp_min_size": 10, "MAE_CP": 0.98},
                ]
            ).to_csv(sweep_dir / "cp_sweep_summary.csv", index=False)
            outputs = build_cp_sweep_report(tmp_path, split="val")
            self.assertIn("cp_sweep_val_plot", outputs)
            self.assertTrue((tmp_path / "reports" / "cp_sweep_val.png").exists())


if __name__ == "__main__":
    unittest.main()
