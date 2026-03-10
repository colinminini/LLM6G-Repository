"""Change-point detection utilities for forecasted traffic series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ChangePointDetector(ABC):
    """Abstract change-point detector interface."""

    @abstractmethod
    def detect_first_change_point(
        self,
        timeseries_array: np.ndarray | list[float],
        **kwargs: Any,
    ) -> int | None:
        """Return the first change-point index in [0, len(series)-1], or None."""


class RupturesPeltDetector(ChangePointDetector):
    """PELT-based change-point detector powered by ruptures."""

    def __init__(
        self,
        model: str = "normal",
        penalty: float = 15.0,
        min_size: int = 10,
        jump: int = 1,
    ) -> None:
        self.model = model
        self.penalty = float(penalty)
        self.min_size = int(min_size)
        self.jump = int(jump)

    def detect_first_change_point(
        self,
        timeseries_array: np.ndarray | list[float],
        **kwargs: Any,
    ) -> int | None:
        try:
            import ruptures as rpt
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "ruptures is required for change-point detection. "
                "Install it with `pip install ruptures`."
            ) from exc

        series = np.asarray(timeseries_array, dtype=float).reshape(-1)
        if series.size == 0:
            return None
        if series.size == 1:
            return None
        if not np.isfinite(series).all():
            raise ValueError("timeseries_array contains NaN/Inf values.")

        # Minimum size guard: PELT cannot split tiny windows meaningfully.
        min_size = max(1, int(kwargs.get("min_size", self.min_size)))
        if series.size < (2 * min_size):
            return None

        model = str(kwargs.get("model", self.model))
        penalty = float(kwargs.get("penalty", self.penalty))
        jump = int(kwargs.get("jump", self.jump))

        signal = series.reshape(-1, 1)
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
        change_points = algo.fit(signal).predict(pen=penalty)

        # ruptures includes the end-of-signal index by construction.
        valid_cps = [cp for cp in change_points if 0 < cp < series.size]
        if not valid_cps:
            return None
        return int(valid_cps[0])


def detect_first_change_point(
    timeseries_array: np.ndarray | list[float],
    **kwargs: Any,
) -> int | None:
    """Convenience wrapper for first change-point detection using PELT."""
    detector = RupturesPeltDetector(
        model=str(kwargs.pop("model", "normal")),
        penalty=float(kwargs.pop("penalty", 15.0)),
        min_size=int(kwargs.pop("min_size", 10)),
        jump=int(kwargs.pop("jump", 1)),
    )
    return detector.detect_first_change_point(timeseries_array, **kwargs)
