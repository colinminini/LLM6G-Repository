"""Create the canonical experiment manifest for split-aware training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_QUANTILES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TIMESTAMP_COL,
    parse_quantiles,
)
from src.experiment import build_experiment_manifest, default_experiment_dir, save_manifest, split_label_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the canonical experiment manifest for data/data_1to672.csv."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--quantiles", default="0.5,0.95")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = save_manifest(manifest, output_dir / "manifest.json")
    split_summary = split_label_series(manifest)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    (output_dir / "README.prepare_data.json").write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "context_length": manifest.context_length,
                "horizon": manifest.horizon,
                "quantiles": list(manifest.quantiles),
            },
            indent=2,
        )
    )

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved split summary: {output_dir / 'split_summary.csv'}")


if __name__ == "__main__":
    main()
