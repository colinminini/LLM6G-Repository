"""
Preprocess a traffic CSV dataset into context/target pairs per sector.

Each sector is turned into a single example where the context is all historical
measurements except the final one, and the target is the final measurement. The
result is written as JSONL to feed the single-step evaluation script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from unicodedata import normalize

import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare single-step forecasting examples from traffic CSV data. "
            "Outputs a JSONL file with one entry per sector."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/histo_trafic_original.csv"),
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed_trafic_original.jsonl"),
        help="Destination JSONL path (directories are created as needed).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="latin-1",
        help="Encoding used by the CSV file.",
    )
    return parser.parse_args()


def normalize_month(name: str) -> int | None:
    """Return month number (1-12) from a French month name."""
    key = (
        normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )
    return MONTH_MAP.get(key)


def french_date_to_iso(value: str) -> str:
    """
    Convert strings like 'lundi 18 juin 2018' into '2018-06-18'.
    Falls back to the original string when parsing fails.
    """
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


def prepare_examples(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Group by sector and build context/target pairs."""
    # Preserve the chronological order present in the file.
    timestamp_order = pd.Categorical(
        df["timestamp_iso"], categories=pd.unique(df["timestamp_iso"]), ordered=True
    )
    df = df.assign(timestamp_order=timestamp_order).sort_values(
        ["sector", "timestamp_order"]
    )

    examples: List[Dict[str, object]] = []
    for sector, group in df.groupby("sector"):
        values = group["trafic_mbps"].tolist()
        if len(values) < 2:
            continue
        context = values[:-1]
        target = float(values[-1])
        examples.append(
            {
                "sector": sector,
                "site": group["site"].iloc[0],
                "timestamps": group["timestamp_iso"].tolist(),
                "context": context,
                "target": target,
                "context_length": len(context),
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    # Autodetect delimiter to support both ';' (legacy) and ',' (instant) files.
    df = pd.read_csv(
        args.input_path,
        sep=None,
        engine="python",
        encoding=args.encoding,
    )

    # Normalize column names
    col_map = {}
    if "secteur" in df.columns:
        col_map["secteur"] = "sector"
    if "site" in df.columns:
        col_map["site"] = "site"
    if "tstamp" in df.columns:
        col_map["tstamp"] = "timestamp"
    if "timestamp" in df.columns:
        col_map["timestamp"] = "timestamp"
    if "trafic_mbps" in df.columns:
        col_map["trafic_mbps"] = "value"
    if "psi_instant" in df.columns:
        col_map["psi_instant"] = "value"

    df = df.rename(columns=col_map)

    expected_cols = {"sector", "timestamp", "value"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Input file must contain columns for sector, timestamp, value. Found: {df.columns.tolist()}"
        )

    if "site" not in df.columns:
        df["site"] = df["sector"]

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["sector", "timestamp", "value"])
    df["timestamp"] = df["timestamp"].astype(str).str.strip()

    # Normalize timestamp: handle both French dates and ISO timestamps.
    def normalize_ts(ts: str) -> str:
        if any(ch.isalpha() for ch in ts):
            return french_date_to_iso(ts)
        try:
            return pd.to_datetime(ts).isoformat()
        except Exception:
            return ts

    df["timestamp_iso"] = df["timestamp"].apply(normalize_ts)
    df = df.rename(columns={"value": "trafic_mbps"})

    examples = prepare_examples(df)
    if not examples:
        raise RuntimeError("No examples were generated from the input file.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fout:
        for item in examples:
            json.dump(item, fout)
            fout.write("\n")

    min_context = min(x["context_length"] for x in examples) # type: ignore
    max_context = max(x["context_length"] for x in examples) # type: ignore
    avg_context = sum(x["context_length"] for x in examples) / len(examples) # type: ignore
    print(
        f"Saved {len(examples)} examples to {args.output_path} "
        f"(min context {min_context}, max context {max_context}, avg context {avg_context:.2f})."
    )


if __name__ == "__main__":
    main()
