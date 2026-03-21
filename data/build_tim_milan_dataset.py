"""Download and build a full wide TIM Milan internet-traffic benchmark."""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ORAN_TIM_URL_TEMPLATE = (
    "https://nexus3.o-ran-sc.org/repository/datasets/"
    "sms-call-internet-mi-{day}_parsed.tar.gz"
)
DEFAULT_OUTPUT_PATH = Path("data/data_tim_milan_10min.csv")
DEFAULT_METADATA_PATH = Path("data/data_tim_milan_10min_metadata.csv")
DEFAULT_DROPPED_METADATA_PATH = Path("data/data_tim_milan_10min_dropped_cells.csv")
DEFAULT_RAW_DIR = Path("data/raw/tim_milan")
DEFAULT_START_DATE = date(2013, 11, 1)
DEFAULT_END_DATE = date(2014, 1, 1)


def _progress(message: str) -> None:
    print(message, flush=True)


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be on or after start date.")
    values: list[date] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(days=1)
    return values


def _archive_name(day: date) -> str:
    return f"sms-call-internet-mi-{day.isoformat()}_parsed.tar.gz"


def _archive_path(raw_dir: Path, day: date) -> Path:
    return raw_dir / _archive_name(day)


def _partial_archive_path(raw_dir: Path, day: date) -> Path:
    archive_path = _archive_path(raw_dir, day)
    return archive_path.with_name(f"{archive_path.name}.part")


def _download_reporthook(day: date):
    state = {"last_percent": -1}

    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = int((100 * downloaded) / total_size)
        if percent != state["last_percent"] and (percent == 100 or percent % 10 == 0):
            state["last_percent"] = percent
            _progress(f"[download {day.isoformat()}] {percent}%")

    return _hook


def _download_archive(day: date, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _archive_path(raw_dir, day)
    partial_path = _partial_archive_path(raw_dir, day)
    if archive_path.exists():
        _progress(f"[download {day.isoformat()}] cached -> {archive_path}")
        return archive_path
    if partial_path.exists():
        _progress(f"[download {day.isoformat()}] removing stale partial -> {partial_path}")
        partial_path.unlink()
    url = ORAN_TIM_URL_TEMPLATE.format(day=day.isoformat())
    _progress(f"[download {day.isoformat()}] starting -> {url}")
    try:
        urllib.request.urlretrieve(url, partial_path, reporthook=_download_reporthook(day))
        partial_path.replace(archive_path)
    except KeyboardInterrupt:
        if partial_path.exists():
            partial_path.unlink()
        _progress(f"[download {day.isoformat()}] interrupted, partial file removed")
        raise
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        _progress(f"[download {day.isoformat()}] failed, partial file removed")
        raise
    _progress(f"[download {day.isoformat()}] saved -> {archive_path}")
    return archive_path


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _resolve_column(columns: Sequence[str], candidates: Iterable[str]) -> str:
    normalized_map = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        resolved = normalized_map.get(_normalize_column_name(candidate))
        if resolved is not None:
            return resolved
    raise ValueError(
        "Unable to resolve required column from archive. "
        f"Available columns: {list(columns)}"
    )


def _parse_epoch_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("TIM Milan timestamps must be parseable as numeric epoch values.")
    max_abs = float(np.nanmax(np.abs(numeric.to_numpy(dtype=float))))
    if max_abs >= 1e14:
        unit = "us"
    elif max_abs >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    parsed = pd.to_datetime(numeric, unit=unit, utc=True)
    return parsed.dt.tz_convert("Europe/Rome").dt.tz_localize(None)


def _normalize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    square_col = _resolve_column(
        df.columns,
        ("square_id", "squareid", "cell_id", "cellid", "grid_id", "id"),
    )
    timestamp_col = _resolve_column(
        df.columns,
        ("timestamp", "time_interval", "timeinterval", "start_time", "time"),
    )
    value_col = _resolve_column(
        df.columns,
        ("internet_traffic", "internettraffic", "internet", "value"),
    )

    out = df[[square_col, timestamp_col, value_col]].copy()
    out.columns = ["square_id", "timestamp", "internet_traffic"]
    out["square_id"] = pd.to_numeric(out["square_id"], errors="coerce").astype("Int64")
    out["internet_traffic"] = pd.to_numeric(out["internet_traffic"], errors="coerce")
    out = out.dropna(subset=["square_id", "timestamp", "internet_traffic"]).copy()
    out["square_id"] = out["square_id"].astype(int)
    out["timestamp"] = _parse_epoch_timestamp(out["timestamp"])
    return out.sort_values(["timestamp", "square_id"]).reset_index(drop=True)


def _load_daily_frame(archive_path: Path) -> pd.DataFrame:
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        csv_member = next(
            (member for member in members if member.name.lower().endswith((".csv", ".txt"))),
            None,
        )
        if csv_member is None:
            raise ValueError(f"{archive_path} does not contain a CSV-like payload.")
        handle = tar.extractfile(csv_member)
        if handle is None:
            raise ValueError(f"Unable to extract {csv_member.name} from {archive_path}.")
        payload = handle.read()
    frame = pd.read_csv(io.BytesIO(payload), sep=None, engine="python")
    return _normalize_raw_frame(frame)


def _complete_wide_frame(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timedelta]:
    if long_df.empty:
        raise ValueError("No TIM Milan rows were loaded.")
    timestamps = long_df["timestamp"].sort_values().drop_duplicates()
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        raise ValueError("At least two timestamps are required to infer cadence.")
    cadence = diffs.mode().iloc[0]
    if pd.isna(cadence) or cadence <= pd.Timedelta(0):
        raise ValueError("TIM Milan cadence must be a positive regular interval.")
    expected_index = pd.date_range(
        start=timestamps.iloc[0],
        end=timestamps.iloc[-1],
        freq=cadence,
    )
    pivot = long_df.pivot_table(
        index="timestamp",
        columns="square_id",
        values="internet_traffic",
        aggfunc="mean",
    ).sort_index()
    pivot = pivot.reindex(expected_index)
    return pivot, cadence


def build_metadata(wide_df: pd.DataFrame) -> pd.DataFrame:
    value_df = wide_df.drop(columns=["timestamp"])
    coverage = 1.0 - value_df.isna().mean(axis=0)
    daily_totals = value_df.groupby(pd.to_datetime(wide_df["timestamp"]).dt.date).sum(min_count=1)
    weekday_mask = pd.to_datetime(wide_df["timestamp"]).dt.dayofweek < 5

    metadata = pd.DataFrame(
        {
            "square_id": [int(col) for col in value_df.columns],
            "coverage": coverage.to_numpy(dtype=float),
            "mean_traffic": value_df.mean(axis=0).to_numpy(dtype=float),
            "daily_std": daily_totals.std(axis=0, ddof=0).to_numpy(dtype=float),
            "weekday_weekend_gap": (
                value_df.loc[weekday_mask].mean(axis=0) - value_df.loc[~weekday_mask].mean(axis=0)
            ).abs().to_numpy(dtype=float),
        }
    )
    metadata[["mean_traffic", "daily_std", "weekday_weekend_gap"]] = metadata[
        ["mean_traffic", "daily_std", "weekday_weekend_gap"]
    ].fillna(0.0)
    metadata["column_name"] = metadata["square_id"].map(lambda square_id: f"MI{square_id}")
    return metadata.sort_values("square_id").reset_index(drop=True)


def build_tim_milan_dataset(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    dropped_metadata_path: Path = DEFAULT_DROPPED_METADATA_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    download: bool = True,
) -> tuple[Path, Path, Path]:
    days = _date_range(start_date, end_date)
    _progress(
        f"[tim_milan] preparing {len(days)} day(s) from {start_date.isoformat()} "
        f"to {end_date.isoformat()}"
    )
    archive_paths: list[Path] = []
    for index, day in enumerate(days, start=1):
        _progress(f"[tim_milan] archive {index}/{len(days)} -> {day.isoformat()}")
        archive_path = _archive_path(raw_dir, day)
        if download:
            archive_path = _download_archive(day, raw_dir)
        elif not archive_path.exists():
            raise FileNotFoundError(
                f"Missing cached TIM Milan archive: {archive_path}. "
                "Either download the files first or omit --skip-download."
            )
        else:
            _progress(f"[download {day.isoformat()}] cached -> {archive_path}")
        archive_paths.append(archive_path)

    daily_frames: list[pd.DataFrame] = []
    for index, path in enumerate(archive_paths, start=1):
        _progress(f"[process {index}/{len(archive_paths)}] loading {path.name}")
        frame = _load_daily_frame(path)
        _progress(
            f"[process {index}/{len(archive_paths)}] loaded {len(frame)} rows, "
            f"{frame['square_id'].nunique()} cells"
        )
        daily_frames.append(frame)

    _progress("[tim_milan] concatenating daily frames")
    long_df = pd.concat(daily_frames, ignore_index=True)
    _progress(
        f"[tim_milan] concatenated {len(long_df)} rows across "
        f"{long_df['square_id'].nunique()} unique cells"
    )
    _progress("[tim_milan] building full wide matrix")
    full_wide, cadence = _complete_wide_frame(long_df)
    full_wide = full_wide.reset_index().rename(columns={"index": "timestamp"})
    metadata_df = build_metadata(full_wide)
    kept_metadata_df = metadata_df.loc[metadata_df["coverage"] == 1.0].copy()
    dropped_metadata_df = metadata_df.loc[metadata_df["coverage"] < 1.0].copy()
    if kept_metadata_df.empty:
        raise ValueError("No TIM Milan cells have complete coverage over the aligned full horizon.")
    kept_square_ids = kept_metadata_df["square_id"].astype(int).tolist()
    dropped_count = len(dropped_metadata_df)
    _progress(
        f"[tim_milan] wide matrix shape={full_wide.shape} cadence="
        f"{int(cadence.total_seconds() // 60)}min"
    )
    _progress(
        f"[tim_milan] keeping {len(kept_square_ids)} complete cells and dropping "
        f"{dropped_count} incomplete cells"
    )
    kept_wide = full_wide[["timestamp", *kept_square_ids]].copy()
    rename_map = {int(square_id): f"MI{int(square_id)}" for square_id in kept_square_ids}
    kept_wide = kept_wide.rename(columns=rename_map)
    kept_wide["timestamp"] = pd.to_datetime(kept_wide["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    if kept_wide.drop(columns=["timestamp"]).isna().any().any():
        raise ValueError("Complete-cell filtering failed; kept TIM Milan matrix still contains NaN values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f"[tim_milan] writing dataset -> {output_path}")
    kept_wide.to_csv(output_path, index=False)
    kept_metadata_df = kept_metadata_df.copy()
    kept_metadata_df["column_name"] = kept_metadata_df["square_id"].map(lambda square_id: f"MI{int(square_id)}")
    kept_metadata_df["cadence"] = f"{int(cadence.total_seconds() // 60)}min"
    dropped_metadata_df = dropped_metadata_df.copy()
    dropped_metadata_df["column_name"] = dropped_metadata_df["square_id"].map(lambda square_id: f"MI{int(square_id)}")
    dropped_metadata_df["cadence"] = f"{int(cadence.total_seconds() // 60)}min"
    _progress(f"[tim_milan] writing metadata -> {metadata_path}")
    kept_metadata_df.to_csv(metadata_path, index=False)
    _progress(f"[tim_milan] writing dropped-cell metadata -> {dropped_metadata_path}")
    dropped_metadata_df.to_csv(dropped_metadata_path, index=False)
    _progress(
        f"[tim_milan] done rows={len(kept_wide)} kept_series={kept_wide.shape[1] - 1} "
        f"dropped_series={dropped_count} cadence={kept_metadata_df['cadence'].iloc[0]}"
    )
    return output_path, metadata_path, dropped_metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and build the complete-cell native-cadence TIM Milan benchmark."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--dropped-metadata-path", type=Path, default=DEFAULT_DROPPED_METADATA_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--start-date", type=_parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=_parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, metadata_path, dropped_metadata_path = build_tim_milan_dataset(
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        dropped_metadata_path=args.dropped_metadata_path,
        raw_dir=args.raw_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        download=not args.skip_download,
    )
    print(f"Saved TIM Milan dataset to {output_path}")
    print(f"Saved TIM Milan metadata to {metadata_path}")
    print(f"Saved dropped TIM Milan cell metadata to {dropped_metadata_path}")


if __name__ == "__main__":
    main()
