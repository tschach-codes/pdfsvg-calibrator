import math

import pytest

from pdfsvg_calibrator.geom import classify_hv, fit_straight_segment
from pdfsvg_calibrator.io_svg_pdf import load_pdf_segments
from pdfsvg_calibrator.types import Segment

def test_fit_straight_segment_returns_best_fit() -> None:
    points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.1)]

    seg, delta_max, delta_rms, angle = fit_straight_segment(points)

    assert delta_max == pytest.approx(0.03333, abs=1e-4)
    assert delta_rms == pytest.approx(0.02357, abs=1e-4)
    endpoints = sorted(
        [(seg.x1, seg.y1), (seg.x2, seg.y2)], key=lambda pt: pt[0]
    )
    (x_start, y_start), (x_end, y_end) = endpoints
    assert x_start == pytest.approx(0.0001667, abs=1e-4)
    assert y_start == pytest.approx(-0.01667, abs=1e-3)
    assert x_end == pytest.approx(10.00017, abs=1e-4)
    assert y_end == pytest.approx(0.08334, abs=1e-3)
    assert angle == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize(
    "segment, expected",
    [
        (Segment(0.0, 0.0, 10.0, 0.05), "h"),
        (Segment(0.0, 0.0, 0.05, 10.0), "v"),
        (Segment(0.0, 0.0, 5.0, 5.0), "d"),
    ],
)
def test_classify_hv(segment: Segment, expected: str) -> None:
    horiz, vert = classify_hv([segment], angle_tol_deg=2.0)
    classification = "d"
    if horiz:
        classification = "h"
    elif vert:
        classification = "v"
    assert classification == expected


def test_load_pdf_segments_uses_raster_path(monkeypatch) -> None:
    recorded = {}

    def fake_raster(path: str, page: int, cfg: dict):
        recorded["args"] = (path, page, cfg)
        return [Segment(0.0, 0.0, 1.0, 1.0)], (100.0, 200.0)

    monkeypatch.setattr(
        "pdfsvg_calibrator.io_svg_pdf._load_pdf_segments_raster", fake_raster
    )

    segments, size = load_pdf_segments("dummy.pdf", 0, {"foo": "bar"}, use_pdfium=False)

    assert len(segments) == 1
    assert size == (100.0, 200.0)
    assert recorded["args"] == ("dummy.pdf", 0, {"foo": "bar", "_pdf_segment_source": "raster"})


@pytest.mark.parametrize(
    "seg, expected",
    [
        ({"x1": 0, "y1": 0, "x2": 1, "y2": 1}, Segment(0.0, 0.0, 1.0, 1.0)),
        ({"x1": 2.5, "y1": 4.5, "x2": -1.25, "y2": 3.0}, Segment(2.5, 4.5, -1.25, 3.0)),
    ],
)
def test_segment_type_conversions(seg: dict, expected: Segment) -> None:
    instance = Segment(**{k: float(v) for k, v in seg.items()})
    assert instance == expected
    assert math.hypot(instance.x2 - instance.x1, instance.y2 - instance.y1) >= 0.0
