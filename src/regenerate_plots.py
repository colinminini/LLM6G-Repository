"""Regenerate all report plots from existing experiment data without re-running compute.

Usage:
    python src/regenerate_plots.py                  # default windows
    python src/regenerate_plots.py --window-seed 7  # random window shared across all models
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reporting import build_report_bundle

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "results/experiments/data_tim_milan_10min_trainable_200_ctx144_h48"

parser = argparse.ArgumentParser()
parser.add_argument("--window-seed", type=int, default=None, help="Seed for random shared example window selection")
parser.add_argument(
    "--experiment-dir",
    type=Path,
    default=None,
    help="Override the experiment dir. If not set, picks q50_q99_seed42 then q50_q95_seed42.",
)
args = parser.parse_args()

if args.experiment_dir is not None:
    EXPERIMENT_DIR = args.experiment_dir
else:
    candidates = [
        EXPERIMENT_ROOT / "q50_q99_seed42",
        EXPERIMENT_ROOT / "q50_q95_seed42",
    ]
    EXPERIMENT_DIR = next((p for p in candidates if p.exists()), candidates[0])

build_report_bundle(EXPERIMENT_DIR, window_seed=args.window_seed)

print("Done.")
