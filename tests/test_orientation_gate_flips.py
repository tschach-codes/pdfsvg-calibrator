import math
import os
import sys
from itertools import product
from typing import Iterable, List, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pdfsvg_calibrator.orientation as orientation_module
from pdfsvg_calibrator.orientation import pick_flip_and_rot
from pdfsvg_calibrator.types import Segment


def _base_segments() -> List[Segment]:
    """Return a non-symmetric set of segments within the page."""

    return [
        Segment(25.0, 30.0, 180.0, 45.0),
        Segment(60.0, 40.0, 95.0, 150.0),
        Segment(40.0, 120.0, 190.0, 95.0),
        Segment(150.0, 60.0, 158.0, 145.0),
        Segment(80.0, 22.0, 125.0, 72.0),
        Segment(45.0, 55.0, 140.0, 52.0),
        Segment(58.0, 70.0, 68.0, 155.0),
        Segment(120.0, 110.0, 200.0, 160.0),
    ]


def _transform_segments(
    segments: Iterable[Segment],
    flip: Tuple[float, float],
    rot_deg: int,
    page_size: Tuple[float, float],
) -> List[Segment]:
    cx, cy = page_size[0] / 2.0, page_size[1] / 2.0
    ang = math.radians(rot_deg % 360)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    transformed: List[Segment] = []
    for seg in segments:
        coords = [(seg.x1, seg.y1), (seg.x2, seg.y2)]
        new_coords = []
        for x, y in coords:
            fx = cx + flip[0] * (x - cx)
            fy = cy + flip[1] * (y - cy)
            tx = fx - cx
            ty = fy - cy
            rx = cos_a * tx - sin_a * ty
            ry = sin_a * tx + cos_a * ty
            new_coords.append((rx + cx, ry + cy))
        (x1, y1), (x2, y2) = new_coords
        transformed.append(Segment(x1=x1, y1=y1, x2=x2, y2=y2))
    return transformed


def _matrix_for(flip: Tuple[float, float], rot_deg: int) -> Tuple[float, float]:
    rot_norm = rot_deg % 360
    if rot_norm == 0:
        return flip
    if rot_norm == 180:
        return (-flip[0], -flip[1])
    raise ValueError("rotation must be 0 or 180 degrees")


@pytest.mark.parametrize(
    "flip, rot_deg",
    list(product(((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)), (0, 180))),
)
def test_orientation_gate_matches_expected_transform(
    flip: Tuple[float, float], rot_deg: int
) -> None:
    pdf_size = (220.0, 170.0)
    pdf_segments = _base_segments()
    svg_segments = _transform_segments(pdf_segments, flip, rot_deg, pdf_size)

    result_flip, result_rot, tx, ty = pick_flip_and_rot(
        pdf_segments,
        svg_segments,
        pdf_size,
        pdf_size,
    )

    expected_matrix = _matrix_for(flip, rot_deg)
    result_matrix = _matrix_for(result_flip, result_rot)

    assert result_matrix == expected_matrix
    assert abs(tx) < 1e-2
    assert abs(ty) < 1e-2


def test_orientation_fallback_prefers_default_when_better(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_size = (220.0, 170.0)
    pdf_segments = _base_segments()
    svg_segments = list(pdf_segments)

    monkeypatch.setattr(orientation_module, "DEFAULT_USE_PHASE_CORRELATION", False)
    monkeypatch.setattr(orientation_module, "DEFAULT_MIN_ACCEPT_SCORE", 0.05)

    overlap_values = iter([0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.01, 0.2])

    def fake_overlap(_img_a, _img_b):
        return next(overlap_values)

    monkeypatch.setattr(orientation_module, "_normalized_overlap", fake_overlap)

    result = orientation_module.pick_flip_and_rot(
        pdf_segments,
        svg_segments,
        pdf_size,
        pdf_size,
    )

    assert result.path == "fallback_orientation"
    assert result.widen_trans_window is False
    assert result.trans_window_hint_px is None
    assert result.primary_score == pytest.approx(0.01)
    assert result.fallback_score == pytest.approx(0.2)
    assert result.score == pytest.approx(0.2)


def test_orientation_fallback_requests_wide_window(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_size = (220.0, 170.0)
    pdf_segments = _base_segments()
    svg_segments = list(pdf_segments)

    monkeypatch.setattr(orientation_module, "DEFAULT_USE_PHASE_CORRELATION", False)
    monkeypatch.setattr(orientation_module, "DEFAULT_MIN_ACCEPT_SCORE", 0.05)

    overlap_values = iter([0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.01, 0.005])

    def fake_overlap(_img_a, _img_b):
        return next(overlap_values)

    monkeypatch.setattr(orientation_module, "_normalized_overlap", fake_overlap)

    result = orientation_module.pick_flip_and_rot(
        pdf_segments,
        svg_segments,
        pdf_size,
        pdf_size,
    )

    assert result.path == "fallback_wide_window"
    assert result.widen_trans_window is True
    assert result.trans_window_hint_px == pytest.approx(orientation_module.WIDE_TRANS_WINDOW_PX)
    assert result.primary_score == pytest.approx(0.01)
    assert result.fallback_score == pytest.approx(0.005)
    assert result.score == pytest.approx(0.01)
