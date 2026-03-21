"""Shared helpers for system-level evaluation on probabilistic forecast windows."""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np


def resolve_tau(tau: int | None, horizon: int) -> int:
    if tau is None:
        return int(horizon)
    return int(np.clip(int(tau), 0, horizon))


def extract_pre_change_interval(values: np.ndarray, tau: int, horizon: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if tau >= horizon:
        segment = arr[:horizon]
    else:
        segment = arr[: max(1, tau)]
    if segment.size == 0:
        return arr[:1]
    return segment


def safe_ceiling_from_tau(y95: np.ndarray, tau_pred: int, horizon: int) -> float:
    arr = np.asarray(y95, dtype=float).reshape(-1)
    if tau_pred >= horizon:
        stationary_95 = arr[:horizon]
    else:
        stationary_95 = arr[: max(1, tau_pred)]
    return float(np.max(stationary_95))


def pinball_loss(target: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    y_true = np.asarray(target, dtype=float).reshape(-1)
    y_pred = np.asarray(pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("pinball_loss expects target and prediction with matching shape.")
    q = float(quantile)
    errors = y_true - y_pred
    loss = np.maximum(q * errors, (q - 1.0) * errors)
    return float(np.mean(loss))


def json_array(raw: str) -> np.ndarray:
    return np.asarray(json.loads(raw), dtype=float).reshape(-1)


def clamp_upper_quantile(y50: Sequence[float], y95: Sequence[float]) -> np.ndarray:
    median = np.asarray(y50, dtype=float).reshape(-1)
    upper = np.asarray(y95, dtype=float).reshape(-1)
    if median.shape != upper.shape:
        raise ValueError("q50 and q95 must have matching shapes.")
    return np.maximum(upper, median)
