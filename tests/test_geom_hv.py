import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdfsvg_calibrator import io_svg_pdf
from pdfsvg_calibrator.geom import classify_hv, fit_straight_segment
from pdfsvg_calibrator.io_svg_pdf import load_pdf_segments, load_svg_segments
from pdfsvg_calibrator.types import Segment


class DummyRect:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height


class DummyPage:
    def __init__(self, drawings, width: float, height: float) -> None:
        self._drawings = drawings
        self.rect = DummyRect(width, height)

    def get_drawings(self):
        return self._drawings


class DummyDoc:
    def __init__(self, page: DummyPage) -> None:
        self._page = page
        self.closed = False

    def load_page(self, index: int) -> DummyPage:
        assert index == 0
        return self._page

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def straight_cfg():
    return {
        "curve_tol_rel": 0.001,
        "straight_max_dev_rel": 0.002,
        "straight_max_angle_spread_deg": 4.0,
    }


def test_fit_straight_segment_perfect_line():
    points = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    seg, delta_max, delta_rms, angle = fit_straight_segment(points)
    assert delta_max == pytest.approx(0.0)
    assert delta_rms == pytest.approx(0.0)
    assert seg.x1 == pytest.approx(0.0)
    assert seg.y1 == pytest.approx(0.0)
    assert seg.x2 == pytest.approx(10.0)
    assert seg.y2 == pytest.approx(0.0)
    assert angle == pytest.approx(0.0)


def test_fit_straight_segment_noisy_line():
    points = [(0.0, 0.0), (5.0, 0.01), (10.0, -0.02), (15.0, 0.0)]
    seg, delta_max, delta_rms, angle = fit_straight_segment(points)
    assert delta_max == pytest.approx(0.016, abs=1e-4)
    assert delta_rms < 0.02
    assert min(angle, abs(angle - 180.0)) == pytest.approx(0.0, abs=0.1)
    xs = sorted([seg.x1, seg.x2])
    assert xs[0] == pytest.approx(0.0, abs=1e-5)
    assert xs[1] == pytest.approx(15.0, abs=1e-5)


def test_fit_straight_segment_curved():
    points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    seg, delta_max, delta_rms, angle = fit_straight_segment(points)
    assert delta_max > 3.0
    assert delta_rms > 2.0
    assert angle == pytest.approx(0.0, abs=1.0)
    assert seg.x1 == pytest.approx(0.0)
    assert seg.x2 == pytest.approx(10.0)


def test_classify_hv_angle_tolerance():
    segments = [
        Segment(0.0, 0.0, 10.0, 0.0),
        Segment(0.0, 0.0, 0.5, 10.0),
        Segment(0.0, 0.0, 10.0, 1.0),
        Segment(0.0, 0.0, -1.0, 10.0),
        Segment(0.0, 0.0, 10.0, 10.0),
    ]
    horiz, vert = classify_hv(segments, angle_tol_deg=6.0)
    assert segments[0] in horiz
    assert segments[2] in horiz
    assert segments[1] in vert
    assert segments[3] in vert
    assert segments[4] not in horiz and segments[4] not in vert


def test_load_svg_segments_viewbox_only(tmp_path):
    svg = tmp_path / "viewbox.svg"
    svg.write_text(
        """
        <svg viewBox=\"0 0 200 100\">
            <line x1=\"0\" y1=\"0\" x2=\"200\" y2=\"0\" />
        </svg>
        """
    )
    segments, size = load_svg_segments(str(svg), {})
    assert len(segments) == 1
    assert size == (200.0, 100.0)


def test_load_svg_segments_width_height_units(tmp_path):
    svg = tmp_path / "units.svg"
    svg.write_text(
        """
        <svg width=\"20mm\" height=\"10mm\">
            <line x1=\"0\" y1=\"0\" x2=\"1\" y2=\"0\" />
        </svg>
        """
    )
    segments, size = load_svg_segments(str(svg), {})
    assert len(segments) == 1
    assert size[0] == pytest.approx(75.590551, rel=1e-6)
    assert size[1] == pytest.approx(37.795275, rel=1e-6)


def test_load_svg_segments_viewbox_fallback(tmp_path):
    svg = tmp_path / "fallback.svg"
    svg.write_text(
        """
        <svg width=\"0\" height=\"0\" viewBox=\"0 0 50 25\">
            <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"25\" />
        </svg>
        """
    )
    segments, size = load_svg_segments(str(svg), {})
    assert len(segments) == 1
    assert size == (50.0, 25.0)


def _patch_fitz(monkeypatch, doc):
    monkeypatch.setattr(io_svg_pdf.fitz, "open", lambda *args, **kwargs: doc)


def test_load_pdf_segments_straight_path(monkeypatch, straight_cfg):
    drawings = [
        {
            "path": [
                [("m", 0.0, 0.0), ("l", 50.0, 0.05), ("l", 100.0, -0.02)],
            ]
        }
    ]
    page = DummyPage(drawings, width=100.0, height=200.0)
    doc = DummyDoc(page)
    _patch_fitz(monkeypatch, doc)
    segments, size = load_pdf_segments("dummy.pdf", 0, straight_cfg)
    assert doc.closed is True
    assert size == (100.0, 200.0)
    assert len(segments) == 1
    seg = segments[0]
    xs = sorted([seg.x1, seg.x2])
    ys = [seg.y1, seg.y2]
    assert xs[0] == pytest.approx(0.0, abs=1e-5)
    assert xs[1] == pytest.approx(100.0, abs=1e-5)
    for val in ys:
        assert val == pytest.approx(0.0, abs=3e-2)


def test_load_pdf_segments_curved_rejected(monkeypatch, straight_cfg):
    drawings = [
        {
            "path": [
                [("m", 0.0, 0.0), ("l", 50.0, 0.0), ("l", 50.0, 50.0)],
            ]
        }
    ]
    page = DummyPage(drawings, width=100.0, height=200.0)
    doc = DummyDoc(page)
    _patch_fitz(monkeypatch, doc)
    segments, _ = load_pdf_segments("dummy.pdf", 0, straight_cfg)
    assert len(segments) == 0


def test_load_pdf_segments_curve_items(monkeypatch, straight_cfg):
    drawings = [
        {
            "items": [
                ("m", (0.0, 0.0)),
                ("c", (33.0, 0.01), (66.0, -0.02), (100.0, 0.0)),
            ]
        }
    ]
    page = DummyPage(drawings, width=100.0, height=200.0)
    doc = DummyDoc(page)
    _patch_fitz(monkeypatch, doc)
    segments, _ = load_pdf_segments("dummy.pdf", 0, straight_cfg)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.x1 == pytest.approx(0.0, abs=1e-6)
    assert seg.x2 == pytest.approx(100.0, abs=1e-6)


def test_load_pdf_segments_rectangle(monkeypatch, straight_cfg):
    drawings = [
        {"path": [[("re", 10.0, 20.0, 30.0, 40.0)]]}
    ]
    page = DummyPage(drawings, width=100.0, height=200.0)
    doc = DummyDoc(page)
    _patch_fitz(monkeypatch, doc)
    segments, _ = load_pdf_segments("dummy.pdf", 0, straight_cfg)
    assert len(segments) == 4
    lengths = [math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1) for seg in segments]
    assert all(length > 0 for length in lengths)
