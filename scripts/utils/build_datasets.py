"""
Create wide dataset CSVs from the histo_* traffic files.

Each output CSV has a timestamp column followed by one column per sector,
with traffic values (Mbps) as the cell values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable
from unicodedata import normalize

import pandas as pd


DEFAULT_JOBS: Dict[str, str] = {
    "histo_trafic_instant.csv": "data_instant.csv",
    "histo_trafic_original.csv": "data_original.csv",
    "histo_1to7.csv": "data_1to7.csv",
}

NO_FREQUENCY_ENFORCEMENT = {"histo_1to7.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset CSVs from histo_* traffic files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/histo"),
        help="Directory containing histo_* CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write dataset CSV files.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="latin-1",
        help="Encoding used by the CSV files.",
    )
    return parser.parse_args()


def load_histo(path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")

    col_map = {}
    if "secteur" in df.columns:
        col_map["secteur"] = "sector"
    if "sector" in df.columns:
        col_map["sector"] = "sector"
    if "tstamp" in df.columns:
        col_map["tstamp"] = "timestamp"
    if "timestamp" in df.columns:
        col_map["timestamp"] = "timestamp"
    if "trafic_mbps" in df.columns:
        col_map["trafic_mbps"] = "value"
    if "psi_instant" in df.columns:
        col_map["psi_instant"] = "value"
    if "value" in df.columns:
        col_map["value"] = "value"

    df = df.rename(columns=col_map)

    expected = {"sector", "timestamp", "value"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns for sector, timestamp, value. Found: {df.columns.tolist()}"
        )

    df = df[["sector", "timestamp", "value"]]
    df["sector"] = df["sector"].astype(str).str.strip()
    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["sector", "timestamp", "value"])
    return df


MONTH_MAP: Dict[str, int] = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}


def normalize_month(name: str) -> int | None:
    key = (
        normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )
    return MONTH_MAP.get(key)


def french_date_to_iso(value: str) -> str:
    tokens = value.strip().split()
    if len(tokens) < 4:
        return value
    try:
        day = int(tokens[-3])
        month = normalize_month(tokens[-2])
        year = int(tokens[-1])
        if month is None:
            return value
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (TypeError, ValueError):
        return value


def parse_timestamps(values: Iterable[str]) -> pd.Series:
    parsed = []
    for ts in values:
        if isinstance(ts, str) and any(ch.isalpha() for ch in ts):
            ts = french_date_to_iso(ts)
        parsed.append(pd.to_datetime(ts, errors="coerce"))
    return pd.Series(parsed)


def infer_frequency(timestamps: pd.Series) -> pd.Timedelta | None:
    diffs = timestamps.diff().dropna()
    if diffs.empty:
        return None
    return diffs.value_counts().idxmax() # type: ignore


def pivot_dataset(df: pd.DataFrame) -> pd.DataFrame:
    timestamp_order = pd.Categorical(
        df["timestamp"], categories=pd.unique(df["timestamp"]), ordered=True
    )
    sector_order = pd.unique(df["sector"])

    df = df.assign(timestamp_order=timestamp_order)
    pivot = df.pivot_table(
        index="timestamp_order",
        columns="sector",
        values="value",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=sector_order).sort_index()
    pivot = pivot.reset_index().rename(columns={"timestamp_order": "timestamp"})
    pivot.columns.name = None
    return pivot


def select_longest_complete_segment(
    dataset: pd.DataFrame, enforce_frequency: bool = True
) -> pd.DataFrame:
    if "timestamp" not in dataset.columns:
        raise ValueError("Dataset must contain a timestamp column.")

    value_df = dataset.drop(columns=["timestamp"])
    complete = ~value_df.isna().any(axis=1)
    timestamps = parse_timestamps(dataset["timestamp"].astype(str).tolist())
    expected_delta = infer_frequency(timestamps) if enforce_frequency else None

    best_start = 0
    best_len = 0
    current_start = None
    current_len = 0

    for idx, is_complete in enumerate(complete):
        if not is_complete:
            if current_len > best_len:
                best_start = current_start if current_start is not None else 0
                best_len = current_len
            current_start = None
            current_len = 0
            continue

        if current_start is None:
            current_start = idx
            current_len = 1
            continue

        if expected_delta is None:
            current_len += 1
            continue

        prev_ts = timestamps.iloc[idx - 1]
        curr_ts = timestamps.iloc[idx]
        if pd.isna(prev_ts) or pd.isna(curr_ts):
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
            current_start = idx
            current_len = 1
            continue

        if curr_ts - prev_ts == expected_delta:
            current_len += 1
        else:
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
            current_start = idx
            current_len = 1

    if current_len > best_len:
        best_start = current_start if current_start is not None else 0
        best_len = current_len

    if best_len == 0:
        raise RuntimeError("No complete continuous segment found.")

    trimmed = dataset.iloc[best_start : best_start + best_len].reset_index(drop=True)
    if trimmed.drop(columns=["timestamp"]).isna().any().any():
        raise RuntimeError("Selected segment still contains missing values.")
    return trimmed


def split_dataset(
    dataset: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    if train_end == 0 or val_end <= train_end or val_end >= total:
        raise ValueError("Dataset is too small to split with the given ratios.")

    train_df = dataset.iloc[:train_end].reset_index(drop=True)
    val_df = dataset.iloc[train_end:val_end].reset_index(drop=True)
    test_df = dataset.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def write_splits(
    dataset: pd.DataFrame, dataset_dir: Path, base_name: str
) -> None:
    train_df, val_df, test_df = split_dataset(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(dataset_dir / f"{base_name}_train.csv", index=False)
    val_df.to_csv(dataset_dir / f"{base_name}_val.csv", index=False)
    test_df.to_csv(dataset_dir / f"{base_name}_test.csv", index=False)


def build_dataset(input_path: Path, output_path: Path, encoding: str) -> None:
    df = load_histo(input_path, encoding)
    dataset = pivot_dataset(df)
    enforce_frequency = input_path.name not in NO_FREQUENCY_ENFORCEMENT
    dataset = select_longest_complete_segment(dataset, enforce_frequency=enforce_frequency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    write_splits(dataset, output_path.parent / "datasets", output_path.stem)


def main() -> None:
    args = parse_args()
    for input_name, output_name in DEFAULT_JOBS.items():
        input_path = args.input_dir / input_name
        output_path = args.output_dir / output_name
        build_dataset(input_path, output_path, args.encoding)
        print(f"Saved dataset to {output_path}.")


if __name__ == "__main__":
    main()
