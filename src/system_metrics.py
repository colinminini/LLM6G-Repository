"""Shared helpers for system-level evaluation on probabilistic forecast windows."""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np


def resolve_tau(tau: int | None, horizon: int) -> int:
    if tau is None:
        return int(horizon)
    return int(np.clip(int(tau), 0, horizon))


def extract_pre_change_interval(values: np.ndarray, tau: int | None, horizon: int) -> np.ndarray:
    """Slice ``values`` up to ``tau`` (the pre-break interval).

    In the LLM6G control loop the controller refreshes at the *predicted*
    break ``tau_pred``; coverage and sharpness must therefore be evaluated on
    ``[0, tau_pred)``, the interval the safe ceiling is asked to honor in
    deployment. Callers should pass ``tau=tau_pred`` (or its calibrated
    counterpart) for both the truth slice and the ceiling.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    resolved_tau = resolve_tau(tau, horizon)
    if resolved_tau >= horizon:
        segment = arr[:horizon]
    else:
        segment = arr[: max(1, resolved_tau)]
    if segment.size == 0:
        return arr[:1]
    return segment


def safe_ceiling_from_tau(y95: np.ndarray, tau_pred: int | None, horizon: int) -> float:
    arr = np.asarray(y95, dtype=float).reshape(-1)
    resolved_tau = resolve_tau(tau_pred, horizon)
    if resolved_tau >= horizon:
        stationary_95 = arr[:horizon]
    else:
        stationary_95 = arr[: max(1, resolved_tau)]
    return float(np.max(stationary_95))


def binary_classification_metrics(
    truth: Sequence[bool],
    pred: Sequence[bool],
) -> dict[str, float]:
    truth_arr = np.asarray(truth, dtype=bool).reshape(-1)
    pred_arr = np.asarray(pred, dtype=bool).reshape(-1)
    if truth_arr.shape != pred_arr.shape:
        raise ValueError("truth and pred must have matching shapes.")
    tp = int(np.sum(truth_arr & pred_arr))
    fp = int(np.sum(~truth_arr & pred_arr))
    fn = int(np.sum(truth_arr & ~pred_arr))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


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
    median = np.asarray(y50, dtype=float)
    upper = np.asarray(y95, dtype=float)
    if median.shape != upper.shape:
        raise ValueError("q50 and q95 must have matching shapes.")
    return np.maximum(upper, median)


def ratio_of_means(numerators: Sequence[float], denominators: Sequence[float]) -> float:
    """Aggregate a per-window ratio as Σnumerator / Σdenominator.

    Scale-robust (one ratio per population, not a mean of per-window ratios) and
    aggregation-consistent with the coverage rate, which is also a ratio-of-sums.
    Returns NaN when the denominator sums to zero.
    """
    num_arr = np.asarray(list(numerators), dtype=float)
    den_arr = np.asarray(list(denominators), dtype=float)
    if num_arr.size == 0 or den_arr.size == 0:
        return float("nan")
    den_sum = float(np.sum(den_arr))
    if den_sum <= 0.0:
        return float("nan")
    return float(np.sum(num_arr) / den_sum)


def covered_sharpness_summary(
    sharpness_values: Sequence[float],
    actual_max_values: Sequence[float],
) -> dict[str, float]:
    """Sharpness restricted to windows where safe_ceiling >= actual_max (no undercoverage)."""
    sharp = np.asarray(list(sharpness_values), dtype=float)
    actual = np.asarray(list(actual_max_values), dtype=float)
    if sharp.size == 0:
        return {
            "sharpness_covered": float("nan"),
            "relative_sharpness_covered": float("nan"),
            "covered_window_frac": float("nan"),
        }
    mask = sharp >= 0.0
    if not bool(mask.any()):
        return {
            "sharpness_covered": float("nan"),
            "relative_sharpness_covered": float("nan"),
            "covered_window_frac": 0.0,
        }
    return {
        "sharpness_covered": float(sharp[mask].mean()),
        "relative_sharpness_covered": ratio_of_means(sharp[mask], actual[mask]),
        "covered_window_frac": float(mask.mean()),
    }


def undercoverage_summary(
    sharpness_values: Sequence[float],
    actual_max_values: Sequence[float],
) -> dict[str, float]:
    """Fraction and severity of windows where safe_ceiling < actual_max (undercovered)."""
    sharp = np.asarray(list(sharpness_values), dtype=float)
    actual = np.asarray(list(actual_max_values), dtype=float)
    if sharp.size == 0:
        return {
            "undercoverage_window_frac": float("nan"),
            "undercoverage_severity": float("nan"),
        }
    mask = sharp < 0.0
    if not bool(mask.any()):
        return {
            "undercoverage_window_frac": 0.0,
            "undercoverage_severity": 0.0,
        }
    return {
        "undercoverage_window_frac": float(mask.mean()),
        # Mean shortfall as a fraction of realized peak, reported as a positive number.
        "undercoverage_severity": ratio_of_means(-sharp[mask], actual[mask]),
    }
