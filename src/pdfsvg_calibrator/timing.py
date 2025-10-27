from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, List, Tuple


class Timings:
    def __init__(self) -> None:
        self.items: List[Tuple[str, float, bool]] = []

    @contextmanager
    def section(self, name: str, include_in_compute: bool = True) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.items.append((name, dt, include_in_compute))

    def summarize(self) -> str:
        total = sum(dt for _, dt, inc in self.items if inc)
        lines = ["\n=== Timing Summary ==="]
        for name, dt, inc in self.items:
            flag = "" if inc else " (excluded)"
            lines.append(f"{name:30s} {dt * 1000:8.1f} ms{flag}")
        lines.append("-" * 44)
        lines.append(f"{'Pure compute total':30s} {total * 1000:8.1f} ms")
        return "\n".join(lines)
