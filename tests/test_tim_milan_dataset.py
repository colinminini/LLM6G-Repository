from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_archive(path: Path, frame: pd.DataFrame) -> None:
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    info = tarfile.TarInfo(name="tim_milan.csv")
    info.size = len(csv_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(csv_bytes))


class TimMilanDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tim_module = _load_module(
            "build_tim_milan_dataset",
            "data/build_tim_milan_dataset.py",
        )
        cls.trainable_subset_module = _load_module(
            "build_tim_milan_trainable_subset",
            "data/build_tim_milan_trainable_subset.py",
        )

    def test_build_tim_milan_dataset_writes_full_wide_csv_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            output_path = tmp_path / "data_tim_milan_10min.csv"
            metadata_path = tmp_path / "data_tim_milan_10min_metadata.csv"
            dropped_metadata_path = tmp_path / "data_tim_milan_10min_dropped_cells.csv"

            timestamps = pd.date_range("2013-11-01 00:00:00", periods=6, freq="10min")
            epoch_ms = (timestamps.view("int64") // 10**6).astype(int)
            rows_day_1 = []
            rows_day_2 = []
            square_ids = [4259, 4456, 5060, 6000, 7000]
            for step, ts_ms in enumerate(epoch_ms):
                for square_id in square_ids:
                    value = {
                        4259: 10 + step,
                        4456: 12 + 2 * step,
                        5060: 14 + step,
                        6000: 30 + 3 * step,
                        7000: 5,
                    }[square_id]
                    rows_day_1.append(
                        {
                            "SquareID": square_id,
                            "Timestamp": ts_ms,
                            "InternetTraffic": value,
                        }
                    )
                    rows_day_2.append(
                        {
                            "SquareID": square_id,
                            "Timestamp": ts_ms + 60 * 60 * 1000,
                            "InternetTraffic": value + 1,
                        }
                    )

            _write_archive(
                raw_dir / "sms-call-internet-mi-2013-11-01_parsed.tar.gz",
                pd.DataFrame(rows_day_1),
            )
            _write_archive(
                raw_dir / "sms-call-internet-mi-2013-11-02_parsed.tar.gz",
                pd.DataFrame(rows_day_2),
            )

            self.tim_module.build_tim_milan_dataset(
                output_path=output_path,
                metadata_path=metadata_path,
                dropped_metadata_path=dropped_metadata_path,
                raw_dir=raw_dir,
                start_date=self.tim_module.DEFAULT_START_DATE,
                end_date=self.tim_module.DEFAULT_START_DATE + timedelta(days=1),
                download=False,
            )

            dataset_df = pd.read_csv(output_path)
            metadata_df = pd.read_csv(metadata_path)
            dropped_metadata_df = pd.read_csv(dropped_metadata_path)

            self.assertEqual(
                dataset_df.columns.tolist(),
                ["timestamp", "MI4259", "MI4456", "MI5060", "MI6000", "MI7000"],
            )
            self.assertFalse(dataset_df.drop(columns=["timestamp"]).isna().any().any())
            parsed = pd.to_datetime(dataset_df["timestamp"])
            self.assertTrue(bool(parsed.is_monotonic_increasing))
            diffs = parsed.diff().dropna().unique()
            self.assertEqual(len(diffs), 1)
            self.assertEqual(pd.Timedelta(diffs[0]), pd.Timedelta(minutes=10))

            self.assertEqual(
                metadata_df["column_name"].tolist(),
                ["MI4259", "MI4456", "MI5060", "MI6000", "MI7000"],
            )
            self.assertTrue((metadata_df["coverage"] == 1.0).all())
            self.assertEqual(set(metadata_df["cadence"]), {"10min"})
            self.assertTrue(dropped_metadata_df.empty)

    def test_build_tim_milan_dataset_drops_incomplete_cells_and_writes_drop_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            output_path = tmp_path / "data_tim_milan_10min.csv"
            metadata_path = tmp_path / "data_tim_milan_10min_metadata.csv"
            dropped_metadata_path = tmp_path / "data_tim_milan_10min_dropped_cells.csv"

            timestamps = pd.date_range("2013-11-01 00:00:00", periods=4, freq="10min")
            epoch_ms = (timestamps.view("int64") // 10**6).astype(int)
            rows = []
            for step, ts_ms in enumerate(epoch_ms):
                for square_id in (4259, 4456):
                    if square_id == 4456 and step == 2:
                        continue
                    rows.append(
                        {
                            "SquareID": square_id,
                            "Timestamp": ts_ms,
                            "InternetTraffic": 10 + square_id + step,
                        }
                    )

            _write_archive(
                raw_dir / "sms-call-internet-mi-2013-11-01_parsed.tar.gz",
                pd.DataFrame(rows),
            )

            self.tim_module.build_tim_milan_dataset(
                output_path=output_path,
                metadata_path=metadata_path,
                dropped_metadata_path=dropped_metadata_path,
                raw_dir=raw_dir,
                start_date=self.tim_module.DEFAULT_START_DATE,
                end_date=self.tim_module.DEFAULT_START_DATE,
                download=False,
            )

            dataset_df = pd.read_csv(output_path)
            metadata_df = pd.read_csv(metadata_path)
            dropped_metadata_df = pd.read_csv(dropped_metadata_path)

            self.assertEqual(dataset_df.columns.tolist(), ["timestamp", "MI4259"])
            self.assertEqual(metadata_df["square_id"].tolist(), [4259])
            self.assertEqual(dropped_metadata_df["square_id"].tolist(), [4456])
            self.assertLess(float(dropped_metadata_df.loc[0, "coverage"]), 1.0)

    def test_build_trainable_subset_selects_first_complete_cells_by_square_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "data_tim_milan_10min.csv"
            metadata_path = tmp_path / "data_tim_milan_10min_metadata.csv"
            output_path = tmp_path / "data_tim_milan_10min_trainable_200.csv"
            output_metadata_path = tmp_path / "data_tim_milan_10min_trainable_200_metadata.csv"

            pd.DataFrame(
                {
                    "timestamp": [
                        "2013-11-01 00:00:00",
                        "2013-11-01 00:10:00",
                    ],
                    "MI25": [1.0, 2.0],
                    "MI10": [3.0, 4.0],
                    "MI30": [5.0, 6.0],
                }
            ).to_csv(input_path, index=False)
            pd.DataFrame(
                {
                    "square_id": [25, 10, 30],
                    "column_name": ["MI25", "MI10", "MI30"],
                    "coverage": [1.0, 1.0, 1.0],
                    "cadence": ["10min", "10min", "10min"],
                }
            ).to_csv(metadata_path, index=False)

            self.trainable_subset_module.build_trainable_subset(
                input_path=input_path,
                metadata_path=metadata_path,
                output_path=output_path,
                output_metadata_path=output_metadata_path,
                num_cells=2,
            )

            subset_df = pd.read_csv(output_path)
            subset_metadata_df = pd.read_csv(output_metadata_path)

            self.assertEqual(subset_df.columns.tolist(), ["timestamp", "MI10", "MI25"])
            self.assertEqual(subset_metadata_df["square_id"].tolist(), [10, 25])
            self.assertEqual(
                subset_metadata_df["selection_rule"].unique().tolist(),
                ["first_2_complete_cells_by_square_id"],
            )
            self.assertEqual(subset_metadata_df["selection_rank"].tolist(), [1, 2])

    def test_build_trainable_subset_fails_when_request_exceeds_complete_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "data_tim_milan_10min.csv"
            metadata_path = tmp_path / "data_tim_milan_10min_metadata.csv"

            pd.DataFrame(
                {
                    "timestamp": ["2013-11-01 00:00:00"],
                    "MI10": [3.0],
                }
            ).to_csv(input_path, index=False)
            pd.DataFrame(
                {
                    "square_id": [10],
                    "column_name": ["MI10"],
                    "coverage": [1.0],
                }
            ).to_csv(metadata_path, index=False)

            with self.assertRaisesRegex(ValueError, "Requested 2 cells"):
                self.trainable_subset_module.build_trainable_subset(
                    input_path=input_path,
                    metadata_path=metadata_path,
                    output_path=tmp_path / "subset.csv",
                    output_metadata_path=tmp_path / "subset_metadata.csv",
                    num_cells=2,
                )


if __name__ == "__main__":
    unittest.main()
