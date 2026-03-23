"""Shared lightweight progress displays for long-running experiment stages."""

from __future__ import annotations

import math
import time
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class StageProgressDisplay:
    def __init__(
        self,
        run_name: str,
        total_items: int,
        *,
        unit: str = "item",
        position: int = 0,
        leave: bool = True,
    ) -> None:
        self.run_name = run_name
        self.total_items = max(1, int(total_items))
        self.unit = unit
        self.position = max(0, int(position))
        self.leave = bool(leave)
        self.started_at = time.perf_counter()
        self.completed_items = 0
        self._bar = (
            tqdm(
                total=self.total_items,
                desc=run_name,
                unit=unit,
                dynamic_ncols=True,
                mininterval=0.5,
                smoothing=0.15,
                position=self.position,
                leave=self.leave,
                bar_format=(
                    "{l_bar}{bar}| {n:.0f}/{total:.0f} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
            )
            if tqdm is not None
            else None
        )
        self.update(0, phase="warmup")

    def _timings(self) -> tuple[float, float | None, float | None]:
        elapsed = time.perf_counter() - self.started_at
        if self.completed_items <= 0:
            return elapsed, None, None
        estimated_total = elapsed * self.total_items / self.completed_items
        remaining = max(0.0, estimated_total - elapsed)
        return elapsed, estimated_total, remaining

    def update(self, completed_items: int, *, phase: str, **fields: Any) -> None:
        target = max(0, min(int(completed_items), self.total_items))
        delta = target - self.completed_items
        if self._bar is not None and delta > 0:
            self._bar.update(delta)
        self.completed_items = target

        elapsed, estimated_total, remaining = self._timings()
        postfix: dict[str, str] = {
            "phase": phase,
            "elapsed": format_duration(elapsed),
            "eta": format_duration(remaining),
            "est_total": format_duration(estimated_total),
        }
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, float):
                postfix[str(key)] = f"{value:.3f}"
            else:
                postfix[str(key)] = str(value)

        if self._bar is not None:
            self._bar.set_postfix(postfix, refresh=False)
        elif target > 0:
            print(
                f"[{self.run_name}] progress "
                f"{target}/{self.total_items} phase={phase} "
                f"elapsed={postfix['elapsed']} eta={postfix['eta']} "
                f"est_total={postfix['est_total']}"
            )

    def advance(self, delta: int = 1, *, phase: str, **fields: Any) -> None:
        self.update(self.completed_items + int(delta), phase=phase, **fields)

    def write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message)
        else:
            print(message)

    def close(self) -> None:
        total_elapsed = time.perf_counter() - self.started_at
        self.update(self.completed_items, phase="done")
        self.write(
            f"[{self.run_name}] wall_time={format_duration(total_elapsed)} "
            f"completed={self.completed_items}/{self.total_items}"
        )
        if self._bar is not None:
            self._bar.close()
