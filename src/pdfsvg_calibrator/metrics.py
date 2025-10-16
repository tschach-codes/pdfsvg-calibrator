"""Utilities for collecting timing and counter metrics across the pipeline."""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator, Optional


_TRACKER_VAR: ContextVar["MetricsTracker | None"] = ContextVar(
    "pdfsvg_calibrator_metrics_tracker", default=None
)


@dataclass
class MetricsTracker:
    """Collects duration measurements and arbitrary counters."""

    timings: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, float] = field(default_factory=dict)

    def add_time(self, key: str, duration: float) -> None:
        if duration < 0.0:
            return
        self.timings[key] = self.timings.get(key, 0.0) + duration

    def increment(self, key: str, value: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + value

    def get_time(self, key: str) -> float:
        return self.timings.get(key, 0.0)

    def get_count(self, key: str) -> float:
        return self.counters.get(key, 0.0)


def get_tracker() -> "MetricsTracker | None":
    """Return the currently active metrics tracker, if any."""

    return _TRACKER_VAR.get()


@contextmanager
def use_tracker(tracker: MetricsTracker) -> Iterator[MetricsTracker]:
    """Activate *tracker* for the duration of the context."""

    token = _TRACKER_VAR.set(tracker)
    try:
        yield tracker
    finally:
        _TRACKER_VAR.reset(token)


class Timer(AbstractContextManager["Timer"]):
    """Context manager that records elapsed wall-clock time."""

    def __init__(
        self,
        key: str,
        *,
        tracker: Optional[MetricsTracker] = None,
        logger: Optional[logging.Logger] = None,
        level: int = logging.DEBUG,
        message: str | None = None,
    ) -> None:
        self.key = key
        self._tracker = tracker
        self._logger = logger
        self._level = level
        self._message = message
        self.duration: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":  # type: ignore[override]
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._start is None:
            return None
        self.duration = perf_counter() - self._start
        tracker = self._tracker or get_tracker()
        if tracker is not None:
            tracker.add_time(self.key, self.duration)
        if self._logger is not None and self._logger.isEnabledFor(self._level):
            msg = self._message or "%s took %.3f s"
            self._logger.log(self._level, msg, self.key, self.duration)
        return None


__all__ = ["MetricsTracker", "Timer", "get_tracker", "use_tracker"]
