from __future__ import annotations

from time import perf_counter

from pdfsvg_calibrator.match_verify import _prepare_segment_infos
from pdfsvg_calibrator.types import Segment


def _grid_segments(rows: int, cols: int, spacing: float, length: float) -> list[Segment]:
    segments: list[Segment] = []
    for r in range(rows):
        y = r * spacing
        for c in range(cols):
            x = c * spacing
            segments.append(Segment(x, y, x + length, y))
    for c in range(cols):
        x = c * spacing
        for r in range(rows):
            y = r * spacing
            segments.append(Segment(x, y, x, y + length))
    return segments


def _measure_runtime(segments: list[Segment], radius: float, angle_tol_deg: float, use_grid: bool) -> float:
    start = perf_counter()
    _prepare_segment_infos(segments, radius, angle_tol_deg, use_grid=use_grid)
    return perf_counter() - start


def test_prepare_segment_infos_fast_neighbors_is_significantly_faster() -> None:
    rows = 32
    cols = 32
    spacing = 6.0
    seg_length = 5.0
    radius = 10.0
    angle_tol_deg = 6.0
    segments = _grid_segments(rows, cols, spacing, seg_length)

    # Warm-up to reduce noise from initial allocations or imports.
    _prepare_segment_infos(segments[: min(50, len(segments))], radius, angle_tol_deg, use_grid=False)
    _prepare_segment_infos(segments[: min(50, len(segments))], radius, angle_tol_deg, use_grid=True)

    slow = min(_measure_runtime(segments, radius, angle_tol_deg, use_grid=False) for _ in range(2))
    fast = min(_measure_runtime(segments, radius, angle_tol_deg, use_grid=True) for _ in range(2))

    assert slow > 0.0
    # Expect at least a 2x speedup to ensure the optimized path is effective.
    assert fast * 0.5 <= slow, f"fast path {fast:.4f}s vs legacy {slow:.4f}s"
