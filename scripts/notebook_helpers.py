"""
Helper utilities shared with the notebook for preparing data and running evaluations.
"""

import json
import pathlib
import subprocess
import sys
from typing import Dict, Tuple

import torch

RESULTS_DIR = pathlib.Path("results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DatasetConfig = Dict[str, pathlib.Path]

DATASETS: Dict[str, DatasetConfig] = {
    # Keep key spelling aligned with the datasets mentioned in the notebook.
    "orignal": {
        "input": pathlib.Path("data/histo_trafic_origial.csv"),
        "processed": pathlib.Path("data/processed_trafic_original.jsonl"),
    },
    "instant": {
        "input": pathlib.Path("data/histo_trafic_instant.csv"),
        "processed": pathlib.Path("data/processed_trafic_instant.jsonl"),
    },
    "intant_short": {
        "input": pathlib.Path("data/histo_trafic_instant_short.csv"),
        "processed": pathlib.Path("data/processed_trafic_instant_short.jsonl"),
    },
}

DATASET_ALIASES: Dict[str, str] = {
    # Accept common spellings but keep the user-facing names unchanged.
    "original": "orignal",
    "instant_short": "intant_short",
}


def normalize_dataset(dataset: str) -> Tuple[str, DatasetConfig]:
    dataset_key = dataset.lower()
    dataset_key = DATASET_ALIASES.get(dataset_key, dataset_key)
    if dataset_key not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Expected one of {sorted(DATASETS)}."
        )
    return dataset_key, DATASETS[dataset_key]


def ensure_processed_data(dataset: str = "instant") -> pathlib.Path:
    """Generate the processed dataset if it is missing."""
    dataset_key, cfg = normalize_dataset(dataset)
    data_path = cfg["processed"]
    if data_path.exists():
        return data_path
    cmd = [
        sys.executable,
        "scripts/dataprocessing.py",
        "--input-path",
        str(cfg["input"]),
        "--output-path",
        str(data_path),
    ]
    print(f"Preparing dataset '{dataset_key}':", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return data_path


def evaluate_model(model_id: str, num_samples: int = 32, dataset: str = "instant"):
    """Run the evaluation script for a single model and return the loaded results."""
    dataset_key, cfg = normalize_dataset(dataset)
    data_path = ensure_processed_data(dataset_key)
    slug = model_id.replace("/", "__")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "evals" / f"eval_{slug}_{dataset_key}.json"
    cmd = [
        sys.executable,
        "scripts/single_eval.py",
        "--model-id",
        model_id,
        "--data-path",
        str(data_path),
        "--num-samples",
        str(num_samples),
        "--device",
        str(device),
        "--output-path",
        str(out_path),
    ]
    print(f"Evaluating on '{dataset_key}':", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with out_path.open() as fin:
        return json.load(fin)
