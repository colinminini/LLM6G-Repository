"""Shared helpers for the paper-faithful TFT implementation."""

from __future__ import annotations

from typing import Any

from src.config import DEFAULT_RANDOM_SEED
from src.deepar_support import (
    DeepARFeatureSpec,
    build_feature_spec as build_tft_feature_spec,
    build_series_item_index,
    build_time_feature_matrix as build_tft_time_feature_matrix,
    feature_spec_from_payload,
)
from src.experiment import ExperimentManifest


def resolve_tft_model_config(
    *,
    manifest: ExperimentManifest,
    hidden_size: int,
    num_heads: int,
    num_lstm_layers: int,
    feature_spec: DeepARFeatureSpec,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    return {
        "model_type": "tft",
        "dataset_path": manifest.dataset_path,
        "timestamp_col": manifest.timestamp_col,
        "cadence": manifest.cadence,
        "context_length": int(manifest.context_length),
        "forecast_length": int(manifest.horizon),
        "quantiles": list(manifest.quantiles),
        "series_columns": list(manifest.series_columns),
        "num_items": int(manifest.num_series),
        "tft_hidden_size": int(hidden_size),
        "tft_num_heads": int(num_heads),
        "tft_num_lstm_layers": int(num_lstm_layers),
        "tft_num_static_categorical": 1,
        "tft_static_categorical_cardinalities": [int(manifest.num_series)],
        "tft_num_static_continuous": 0,
        "tft_num_known_features": int(feature_spec.num_features),
        "tft_num_past_features": int(1 + feature_spec.num_features),
        "tft_num_future_features": int(feature_spec.num_features),
        "tft_random_seed": int(random_seed),
        "tft_known_feature_spec": feature_spec.to_dict(),
    }


__all__ = [
    "DeepARFeatureSpec",
    "build_series_item_index",
    "build_tft_feature_spec",
    "build_tft_time_feature_matrix",
    "feature_spec_from_payload",
    "resolve_tft_model_config",
]
