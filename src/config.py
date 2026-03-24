"""Shared defaults and validation helpers for the published forecasting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATASET_PATH = Path("data/data_tim_milan_10min_trainable_200.csv")
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_QUANTILES = (0.5, 0.95)
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.20
DEFAULT_RANDOM_SEED = 42
DEFAULT_CADENCE = "10min"
DEFAULT_RESULTS_ROOT = Path("results/experiments")
DEFAULT_SEASONAL_NAIVE_PERIOD = 144
DEFAULT_DEEPAR_LIKELIHOOD = "gaussian"
SUPPORTED_DEEPAR_LIKELIHOODS = ("gaussian", "negative_binomial")
DEFAULT_DEEPAR_NUM_SAMPLES = 200
DEFAULT_DEEPAR_ITEM_EMBEDDING_DIM = 20

DEFAULT_CP_DETECTOR = "clasp"
SUPPORTED_CP_DETECTORS = ("clasp", "ruptures_pelt")
DEFAULT_CP_PERIOD_LENGTH = "auto"
DEFAULT_CP_EXCLUSION_RADIUS = 0.05
DEFAULT_CP_SCORE_THRESHOLD = 0.5
DEFAULT_CP_MODEL = "normal"
DEFAULT_CP_PENALTY = 10.0
DEFAULT_CP_MIN_SIZE = 8
DEFAULT_CP_JUMP = 1
DEFAULT_TOLERANCE = 3

SUPPORTED_MAIN_MODELS = ("lstm", "deepar", "tft", "chronos2", "seasonal_naive")
TRAINABLE_BASELINE_MODELS = ("lstm", "deepar", "tft")
LEGACY_MODELS = ("chronos2_finetune",)


def normalize_quantiles(quantiles: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(q) for q in quantiles)
    if not values:
        raise ValueError("quantiles must contain at least one value.")
    for q in values:
        if q <= 0.0 or q >= 1.0:
            raise ValueError("quantiles must be in the open interval (0, 1).")
    return values


def parse_quantiles(raw: str) -> tuple[float, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("quantiles must be a comma-separated list like 0.5,0.95")
    return normalize_quantiles(float(part) for part in parts)


def quantiles_tag(quantiles: Sequence[float]) -> str:
    values = normalize_quantiles(quantiles)
    return "_".join(f"q{int(round(100 * q))}" for q in values)
