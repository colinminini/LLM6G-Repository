"""Helpers for breakpoint detector configuration resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DEFAULT_CP_DETECTOR,
    DEFAULT_CP_EXCLUSION_RADIUS,
    DEFAULT_CP_JUMP,
    DEFAULT_CP_MIN_SIZE,
    DEFAULT_CP_MODEL,
    DEFAULT_CP_PENALTY,
    DEFAULT_CP_PERIOD_LENGTH,
    DEFAULT_CP_SCORE_THRESHOLD,
)


def parse_csv_list(raw: str, cast: Any = str) -> list[Any]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return [cast(item) for item in values]


def _legacy_overrides_used(
    *,
    cp_model: str,
    cp_penalty: float,
    cp_min_size: int,
    cp_jump: int,
) -> bool:
    return (
        str(cp_model) != DEFAULT_CP_MODEL
        or float(cp_penalty) != float(DEFAULT_CP_PENALTY)
        or int(cp_min_size) != int(DEFAULT_CP_MIN_SIZE)
        or int(cp_jump) != int(DEFAULT_CP_JUMP)
    )


def resolve_cli_cp_detector(
    *,
    cp_detector: str | None,
    cp_model: str,
    cp_penalty: float,
    cp_min_size: int,
    cp_jump: int,
) -> str:
    if cp_detector not in (None, ""):
        return str(cp_detector).strip().lower()
    if _legacy_overrides_used(
        cp_model=cp_model,
        cp_penalty=cp_penalty,
        cp_min_size=cp_min_size,
        cp_jump=cp_jump,
    ):
        return "ruptures_pelt"
    return DEFAULT_CP_DETECTOR


def default_cp_config(
    *,
    cp_detector: str | None = None,
    cp_model: str = DEFAULT_CP_MODEL,
    cp_penalty: float = DEFAULT_CP_PENALTY,
    cp_min_size: int = DEFAULT_CP_MIN_SIZE,
    cp_jump: int = DEFAULT_CP_JUMP,
    cp_period_length: int | str | None = DEFAULT_CP_PERIOD_LENGTH,
    cp_exclusion_radius: float = DEFAULT_CP_EXCLUSION_RADIUS,
    cp_score_threshold: float = DEFAULT_CP_SCORE_THRESHOLD,
) -> dict[str, Any]:
    resolved_detector = resolve_cli_cp_detector(
        cp_detector=cp_detector,
        cp_model=cp_model,
        cp_penalty=cp_penalty,
        cp_min_size=cp_min_size,
        cp_jump=cp_jump,
    )
    return {
        "cp_detector": resolved_detector,
        "cp_model": str(cp_model),
        "cp_penalty": float(cp_penalty),
        "cp_min_size": int(cp_min_size),
        "cp_jump": int(cp_jump),
        "cp_period_length": cp_period_length,
        "cp_exclusion_radius": float(cp_exclusion_radius),
        "cp_score_threshold": float(cp_score_threshold),
    }


def load_best_cp_configs(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text())
    best_csv = Path(payload["best_configs_csv"])
    if not best_csv.is_absolute():
        best_csv = (manifest_file.parent / best_csv).resolve()
    best_df = pd.read_csv(best_csv)
    configs: dict[str, dict[str, Any]] = {}
    for row in best_df.to_dict("records"):
        model_name = str(row["model"])
        configs[model_name] = dict(row)
    return configs
