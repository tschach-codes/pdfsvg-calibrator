import os
import sys
from time import perf_counter
from typing import List

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.calibrate import calibrate
from pdfsvg_calibrator.metrics import MetricsTracker, use_tracker
from pdfsvg_calibrator.types import Segment


def _dense_segments(width: float, height: float, count_x: int = 14, count_y: int = 12) -> List[Segment]:
    segs: List[Segment] = []
    xs: List[float] = []
    ys: List[float] = []
    for i in range(count_x):
        base = width * (i + 1) / (count_x + 1)
        offset = (i % 3 - 1) * width * 0.015
        xs.append(min(width - 1.0, max(1.0, base + offset)))
    for j in range(count_y):
        base = height * (j + 1) / (count_y + 1)
        offset = (j % 4 - 1.5) * height * 0.012
        ys.append(min(height - 1.0, max(1.0, base + offset)))
    for idx, y in enumerate(ys):
        span = width * (0.9 if idx % 2 == 0 else 0.8)
        start = width * 0.05
        segs.append(Segment(start, y, start + span, y))
    for idx, x in enumerate(xs):
        span = height * (0.88 if idx % 2 == 0 else 0.76)
        start = height * 0.06
        segs.append(Segment(x, start, x, start + span))
    return segs


def _affine_transform(segments: List[Segment], sx: float, sy: float, tx: float, ty: float) -> List[Segment]:
    transformed: List[Segment] = []
    for seg in segments:
        pts = [(seg.x1, seg.y1), (seg.x2, seg.y2)]
        mapped = [(sx * x + tx, sy * y + ty) for x, y in pts]
        (x1, y1), (x2, y2) = mapped
        transformed.append(Segment(x1=x1, y1=y1, x2=x2, y2=y2))
    return transformed


@pytest.mark.slow
def test_runtime_budget_keeps_within_limits() -> None:
    pdf_size = (612.0, 792.0)
    pdf_segments = _dense_segments(*pdf_size)

    sx_true = 1.012
    sy_true = 0.993
    tx_true = 4.2
    ty_true = -5.6

    svg_segments = _affine_transform(pdf_segments, sx_true, sy_true, tx_true, ty_true)
    svg_size = (pdf_size[0] * sx_true, pdf_size[1] * sy_true)

    cfg = {
        "refine": {
            "scale_max_dev_rel": 0.02,
            "trans_max_dev_px": 12.0,
            "max_iters": 120,
            "max_samples": 1500,
        },
        "sampling": {"step_rel": 0.025, "max_points": 1800},
        "grid": {"initial_cell_rel": 0.05, "final_cell_rel": 0.02},
        "ransac": {"iters": 400, "patience": 120},
    }

    tracker = MetricsTracker()
    start = perf_counter()
    with use_tracker(tracker):
        calibrate(pdf_segments, svg_segments, pdf_size, svg_size, cfg)
    elapsed = perf_counter() - start

    assert elapsed < 6.0
    assert tracker.get_count("chamfer.calls") < 400
