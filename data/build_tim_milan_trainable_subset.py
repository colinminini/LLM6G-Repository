"""Build a smaller trainable subset from the cleaned TIM Milan benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_PATH = Path("data/data_tim_milan_10min.csv")
DEFAULT_METADATA_PATH = Path("data/data_tim_milan_10min_metadata.csv")
DEFAULT_OUTPUT_PATH = Path("data/data_tim_milan_10min_trainable_200.csv")
DEFAULT_OUTPUT_METADATA_PATH = Path("data/data_tim_milan_10min_trainable_200_metadata.csv")
DEFAULT_NUM_CELLS = 200


def build_trainable_subset(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    output_metadata_path: Path = DEFAULT_OUTPUT_METADATA_PATH,
    num_cells: int = DEFAULT_NUM_CELLS,
) -> tuple[Path, Path]:
    if num_cells <= 0:
        raise ValueError("num_cells must be positive.")

    full_df = pd.read_csv(input_path)
    metadata_df = pd.read_csv(metadata_path)
    required_cols = {"square_id", "column_name", "coverage"}
    missing = required_cols.difference(metadata_df.columns)
    if missing:
        raise ValueError(
            f"{metadata_path} is missing required metadata columns: {sorted(missing)}"
        )

    eligible = metadata_df.loc[metadata_df["coverage"] == 1.0].copy()
    if len(eligible) < num_cells:
        raise ValueError(
            f"Requested {num_cells} cells, but only {len(eligible)} fully observed cells are available."
        )

    selected = eligible.sort_values("square_id").head(num_cells).copy()
    selected_columns = ["timestamp", *selected["column_name"].tolist()]
    missing_columns = [col for col in selected_columns if col not in full_df.columns]
    if missing_columns:
        raise ValueError(
            f"{input_path} is missing expected columns from metadata: {missing_columns[:10]}"
        )

    subset_df = full_df[selected_columns].copy()
    if subset_df.drop(columns=["timestamp"]).isna().any().any():
        raise ValueError("Trainable subset contains NaN values.")

    selected["selection_rule"] = f"first_{num_cells}_complete_cells_by_square_id"
    selected["selection_rank"] = range(1, len(selected) + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "[tim_milan_subset] selected "
        f"{len(selected)} / {len(eligible)} complete cells "
        "using ascending square_id"
    )
    print(f"[tim_milan_subset] writing dataset -> {output_path}")
    print(f"[tim_milan_subset] writing metadata -> {output_metadata_path}")
    subset_df.to_csv(output_path, index=False)
    selected.to_csv(output_metadata_path, index=False)
    return output_path, output_metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a neutral deterministic trainable subset from the cleaned TIM Milan benchmark."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output-metadata-path", type=Path, default=DEFAULT_OUTPUT_METADATA_PATH)
    parser.add_argument("--num-cells", type=int, default=DEFAULT_NUM_CELLS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, output_metadata_path = build_trainable_subset(
        input_path=args.input_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        output_metadata_path=args.output_metadata_path,
        num_cells=args.num_cells,
    )
    print(f"Saved TIM Milan trainable subset to {output_path}")
    print(f"Saved TIM Milan trainable subset metadata to {output_metadata_path}")


if __name__ == "__main__":
    main()
