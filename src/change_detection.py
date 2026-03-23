"""Change-point detection utilities for forecasted traffic series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_CP_DETECTOR,
    DEFAULT_CP_EXCLUSION_RADIUS,
    DEFAULT_CP_JUMP,
    DEFAULT_CP_MIN_SIZE,
    DEFAULT_CP_MODEL,
    DEFAULT_CP_PENALTY,
    DEFAULT_CP_PERIOD_LENGTH,
    DEFAULT_CP_SCORE_THRESHOLD,
    SUPPORTED_CP_DETECTORS,
)


@dataclass(frozen=True)
class ChangePointResult:
    tau: int | None
    has_break: bool
    score: float | None
    score_threshold: float | None
    detector: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_series(timeseries_array: np.ndarray | list[float]) -> np.ndarray:
    series = np.asarray(timeseries_array, dtype=float).reshape(-1)
    if series.size <= 1:
        return series
    if not np.isfinite(series).all():
        raise ValueError("timeseries_array contains NaN/Inf values.")
    return series


def _z_normalize_windows(windows: np.ndarray) -> np.ndarray:
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    safe_stds = np.where(stds < 1e-8, 1.0, stds)
    return (windows - means) / safe_stds


def _pairwise_sq_dists(windows: np.ndarray) -> np.ndarray:
    gram = windows @ windows.T
    norms = np.sum(windows * windows, axis=1, keepdims=True)
    dists = norms + norms.T - (2.0 * gram)
    dists = np.maximum(dists, 0.0)
    np.fill_diagonal(dists, np.inf)
    return dists


def _resolve_period_length(
    raw_period_length: int | str | None,
    series_size: int,
    min_size: int,
) -> int:
    if raw_period_length in (None, "", "auto"):
        return max(2, min(series_size - 1, max(int(min_size), series_size // 4)))
    return max(2, min(series_size - 1, int(raw_period_length)))


class ChangePointDetector(ABC):
    """Abstract change-point detector interface."""

    @abstractmethod
    def detect_change_point(
        self,
        timeseries_array: np.ndarray | list[float],
        **kwargs: Any,
    ) -> ChangePointResult:
        """Return the main change-point decision for the given series."""

    def detect_first_change_point(
        self,
        timeseries_array: np.ndarray | list[float],
        **kwargs: Any,
    ) -> int | None:
        return self.detect_change_point(timeseries_array, **kwargs).tau

    @abstractmethod
    def config_dict(self) -> dict[str, Any]:
        """Return the serializable detector configuration."""


class ClaspDetector(ChangePointDetector):
    """Single-break classification-score profile detector with explicit no-break threshold."""

    def __init__(
        self,
        *,
        period_length: int | str = DEFAULT_CP_PERIOD_LENGTH,
        exclusion_radius: float = DEFAULT_CP_EXCLUSION_RADIUS,
        score_threshold: float = DEFAULT_CP_SCORE_THRESHOLD,
        min_size: int = DEFAULT_CP_MIN_SIZE,
    ) -> None:
        self.period_length = period_length
        self.exclusion_radius = float(exclusion_radius)
        self.score_threshold = float(score_threshold)
        self.min_size = int(min_size)

    def config_dict(self) -> dict[str, Any]:
        return {
            "cp_detector": DEFAULT_CP_DETECTOR,
            "cp_period_length": self.period_length,
            "cp_exclusion_radius": float(self.exclusion_radius),
            "cp_score_threshold": float(self.score_threshold),
            "cp_min_size": int(self.min_size),
        }

    def detect_change_point(
        self,
        timeseries_array: np.ndarray | list[float],
        **kwargs: Any,
    ) -> ChangePointResult:
        series = _normalize_series(timeseries_array)
        score_threshold = float(kwargs.get("score_threshold", self.score_threshold))
        if series.size <= 1:
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=score_threshold,
                detector=DEFAULT_CP_DETECTOR,
                metadata={"reason": "series_too_short"},
            )

        min_size = max(1, int(kwargs.get("min_size", self.min_size)))
        if series.size < (2 * min_size):
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=score_threshold,
                detector=DEFAULT_CP_DETECTOR,
                metadata={"reason": "series_too_short_for_min_size"},
            )

        exclusion_radius = max(
            0.0,
            float(kwargs.get("exclusion_radius", self.exclusion_radius)),
        )
        period_length = _resolve_period_length(
            kwargs.get("period_length", self.period_length),
            series.size,
            min_size,
        )
        if series.size <= period_length:
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=score_threshold,
                detector=DEFAULT_CP_DETECTOR,
                metadata={"reason": "series_too_short_for_period_length"},
            )

        windows = np.lib.stride_tricks.sliding_window_view(series, period_length)
        normalized = _z_normalize_windows(np.asarray(windows, dtype=float))
        dists = _pairwise_sq_dists(normalized)

        edge_guard = max(min_size, int(np.ceil(exclusion_radius * series.size)))
        candidate_range = range(edge_guard, series.size - edge_guard + 1)
        best_tau: int | None = None
        best_score = float("-inf")
        profile: list[tuple[int, float]] = []
        window_starts = np.arange(normalized.shape[0], dtype=int)
        for tau in candidate_range:
            left_mask = np.where(window_starts + period_length <= tau)[0]
            right_mask = np.where(window_starts >= tau)[0]
            if left_mask.size == 0 or right_mask.size == 0:
                continue

            selected = np.concatenate([left_mask, right_mask])
            labels = np.concatenate(
                [
                    np.zeros(left_mask.size, dtype=int),
                    np.ones(right_mask.size, dtype=int),
                ]
            )
            if selected.size < 2:
                continue
            sub_dists = dists[np.ix_(selected, selected)]
            nearest = np.argmin(sub_dists, axis=1)
            predicted = labels[nearest]
            score = float(np.mean(predicted == labels))
            profile.append((int(tau), score))
            if score > best_score:
                best_score = score
                best_tau = int(tau)

        if best_tau is None:
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=score_threshold,
                detector=DEFAULT_CP_DETECTOR,
                metadata={"reason": "no_valid_candidates"},
            )

        has_break = bool(best_score >= score_threshold)
        return ChangePointResult(
            tau=best_tau if has_break else None,
            has_break=has_break,
            score=float(best_score),
            score_threshold=score_threshold,
            detector=DEFAULT_CP_DETECTOR,
            metadata={
                "period_length": int(period_length),
                "exclusion_radius": float(exclusion_radius),
                "profile": profile,
                "implementation": "internal_profile",
            },
        )


class RupturesPeltDetector(ChangePointDetector):
    """PELT-based change-point detector powered by ruptures."""

    def __init__(
        self,
        model: str = DEFAULT_CP_MODEL,
        penalty: float = DEFAULT_CP_PENALTY,
        min_size: int = DEFAULT_CP_MIN_SIZE,
        jump: int = DEFAULT_CP_JUMP,
    ) -> None:
        self.model = model
        self.penalty = float(penalty)
        self.min_size = int(min_size)
        self.jump = int(jump)

    def config_dict(self) -> dict[str, Any]:
        return {
            "cp_detector": "ruptures_pelt",
            "cp_model": self.model,
            "cp_penalty": float(self.penalty),
            "cp_min_size": int(self.min_size),
            "cp_jump": int(self.jump),
        }

    def detect_change_point(
        self,
        timeseries_array: np.ndarray | list[float],
        **kwargs: Any,
    ) -> ChangePointResult:
        try:
            import ruptures as rpt
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "ruptures is required for change-point detection. "
                "Install it with `pip install ruptures`."
            ) from exc

        series = _normalize_series(timeseries_array)
        if series.size <= 1:
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=None,
                detector="ruptures_pelt",
                metadata={"reason": "series_too_short"},
            )

        min_size = max(1, int(kwargs.get("min_size", self.min_size)))
        if series.size < (2 * min_size):
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=None,
                detector="ruptures_pelt",
                metadata={"reason": "series_too_short_for_min_size"},
            )

        model = str(kwargs.get("model", self.model))
        penalty = float(kwargs.get("penalty", self.penalty))
        jump = int(kwargs.get("jump", self.jump))

        signal = series.reshape(-1, 1)
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
        change_points = algo.fit(signal).predict(pen=penalty)

        valid_cps = [cp for cp in change_points if 0 < cp < series.size]
        if not valid_cps:
            return ChangePointResult(
                tau=None,
                has_break=False,
                score=None,
                score_threshold=None,
                detector="ruptures_pelt",
                metadata={
                    "model": model,
                    "penalty": penalty,
                    "min_size": min_size,
                    "jump": jump,
                },
            )
        return ChangePointResult(
            tau=int(valid_cps[0]),
            has_break=True,
            score=None,
            score_threshold=None,
            detector="ruptures_pelt",
            metadata={
                "model": model,
                "penalty": penalty,
                "min_size": min_size,
                "jump": jump,
                "all_change_points": [int(cp) for cp in valid_cps],
            },
        )


def build_change_point_detector(
    *,
    cp_detector: str = DEFAULT_CP_DETECTOR,
    cp_period_length: int | str | None = DEFAULT_CP_PERIOD_LENGTH,
    cp_exclusion_radius: float = DEFAULT_CP_EXCLUSION_RADIUS,
    cp_score_threshold: float = DEFAULT_CP_SCORE_THRESHOLD,
    cp_model: str = DEFAULT_CP_MODEL,
    cp_penalty: float = DEFAULT_CP_PENALTY,
    cp_min_size: int = DEFAULT_CP_MIN_SIZE,
    cp_jump: int = DEFAULT_CP_JUMP,
) -> ChangePointDetector:
    normalized = str(cp_detector).strip().lower()
    if normalized not in SUPPORTED_CP_DETECTORS:
        raise ValueError(
            f"Unsupported cp detector '{cp_detector}'. "
            f"Supported detectors: {', '.join(SUPPORTED_CP_DETECTORS)}."
        )
    if normalized == "clasp":
        return ClaspDetector(
            period_length=cp_period_length,
            exclusion_radius=cp_exclusion_radius,
            score_threshold=cp_score_threshold,
            min_size=cp_min_size,
        )
    return RupturesPeltDetector(
        model=cp_model,
        penalty=cp_penalty,
        min_size=cp_min_size,
        jump=cp_jump,
    )


def detect_first_change_point(
    timeseries_array: np.ndarray | list[float],
    **kwargs: Any,
) -> int | None:
    """Convenience wrapper for first change-point detection."""
    detector = build_change_point_detector(
        cp_detector=str(kwargs.pop("cp_detector", DEFAULT_CP_DETECTOR)),
        cp_period_length=kwargs.pop("cp_period_length", DEFAULT_CP_PERIOD_LENGTH),
        cp_exclusion_radius=float(
            kwargs.pop("cp_exclusion_radius", DEFAULT_CP_EXCLUSION_RADIUS)
        ),
        cp_score_threshold=float(
            kwargs.pop("cp_score_threshold", DEFAULT_CP_SCORE_THRESHOLD)
        ),
        cp_model=str(kwargs.pop("model", DEFAULT_CP_MODEL)),
        cp_penalty=float(kwargs.pop("penalty", DEFAULT_CP_PENALTY)),
        cp_min_size=int(kwargs.pop("min_size", DEFAULT_CP_MIN_SIZE)),
        cp_jump=int(kwargs.pop("jump", DEFAULT_CP_JUMP)),
    )
    return detector.detect_first_change_point(timeseries_array, **kwargs)
