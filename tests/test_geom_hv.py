import math
import sys
import types
from typing import Iterable, List

import pytest
from pdfsvg_calibrator.geom import classify_hv, fit_straight_segment
from pdfsvg_calibrator.io_svg_pdf import load_pdf_segments, load_svg_segments
from pdfsvg_calibrator.pdfium_extract import extract_segments
from pdfsvg_calibrator.types import Segment


class FakePathSegment:
    def __init__(
        self,
        seg_type: str,
        pos: Iterable[float] | None = None,
        ctrl1: Iterable[float] | None = None,
        ctrl2: Iterable[float] | None = None,
    ) -> None:
        self.type = seg_type
        self.pos = tuple(pos) if pos is not None else None
        self.ctrl1 = tuple(ctrl1) if ctrl1 is not None else None
        self.ctrl2 = tuple(ctrl2) if ctrl2 is not None else None


class FakePathObject:
    type = "path"

    def __init__(self, segments: List[FakePathSegment], matrix=None) -> None:
        self._segments = segments
        self._matrix = matrix or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def get_path(self):
        return self._segments

    def get_matrix(self):
        return self._matrix


class FakeFormObject:
    type = "form"

    def __init__(self, objects: List, matrix=None) -> None:
        self._objects = list(objects)
        self._matrix = matrix or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def get_form(self):
        return self

    def get_matrix(self):
        return self._matrix

    def get_objects_count(self):
        return len(self._objects)

    def get_object(self, idx: int):
        return self._objects[idx]

    @property
    def objects(self):
        return list(self._objects)


class FakePage:
    def __init__(self, objects: List, size: tuple[float, float]) -> None:
        self._objects = list(objects)
        self._size = size
        self.closed = False

    def get_objects_count(self):
        return len(self._objects)

    def get_object(self, idx: int):
        return self._objects[idx]

    def get_size(self):
        return self._size

    def close(self):
        self.closed = True


class FakeDoc:
    def __init__(self, pages: List[FakePage]) -> None:
        self._pages = list(pages)
        self.closed = False

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, idx: int):
        return self._pages[idx]

    def close(self):
        self.closed = True


@pytest.fixture
def straight_cfg():
    return {
        "curve_tol_rel": 0.001,
        "straight_max_dev_rel": 0.002,
        "straight_max_angle_spread_deg": 4.0,
    }


@pytest.fixture
def pdfium_stub(monkeypatch):
    state: dict[str, List] = {"docs": [], "pages": []}

    def install(objects: List, size: tuple[float, float] = (100.0, 200.0)):
        def factory():
            page = FakePage(objects, size)
            doc = FakeDoc([page])
            state["pages"].append(page)
            state["docs"].append(doc)
            return doc

        module = types.SimpleNamespace(PdfDocument=lambda *args, **kwargs: factory())
        monkeypatch.setitem(sys.modules, "pypdfium2", module)
        return state

    return install


def _assert_segments_plain(segments: List[Segment]) -> None:
    for seg in segments:
        assert isinstance(seg, Segment)
        for value in (seg.x1, seg.y1, seg.x2, seg.y2):
            assert not hasattr(value, "__array__")
            assert not isinstance(value, (list, tuple))


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
    assert size == (150.0, 75.0)
    _assert_segments_plain(segments)


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
    assert size[0] == pytest.approx(56.692913, rel=1e-6)
    assert size[1] == pytest.approx(28.346457, rel=1e-6)
    _assert_segments_plain(segments)


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
    assert size == (37.5, 18.75)
    _assert_segments_plain(segments)


def test_extract_segments_simple_line(pdfium_stub):
    pdfium_stub([
        FakePathObject(
            [
                FakePathSegment("move", (0.0, 0.0)),
                FakePathSegment("line", (100.0, 0.0)),
            ]
        )
    ])
    segments = extract_segments("dummy.pdf", 0, curve_tol_pt=0.01)
    assert segments == [{"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 0.0}]


def test_extract_segments_bezier(pdfium_stub):
    pdfium_stub([
        FakePathObject(
            [
                FakePathSegment("move", (0.0, 0.0)),
                FakePathSegment(
                    "bezier",
                    (100.0, 0.0),
                    ctrl1=(33.0, 0.0),
                    ctrl2=(66.0, 0.0),
                ),
            ]
        )
    ])
    segments = extract_segments("dummy.pdf", 0, curve_tol_pt=0.01)
    assert len(segments) >= 1
    assert segments[0]["x1"] == pytest.approx(0.0)
    assert segments[-1]["x2"] == pytest.approx(100.0)


def test_extract_segments_form_matrix(pdfium_stub):
    child = FakePathObject(
        [
            FakePathSegment("move", (0.0, 0.0)),
            FakePathSegment("line", (0.0, 10.0)),
        ]
    )
    form = FakeFormObject([child], matrix=(1.0, 0.0, 5.0, 0.0, 1.0, 7.0))
    pdfium_stub([form])
    segments = extract_segments("dummy.pdf", 0, curve_tol_pt=0.01)
    assert segments == [{"x1": 5.0, "y1": 7.0, "x2": 5.0, "y2": 17.0}]


def test_load_pdf_segments_uses_pdfium(pdfium_stub, straight_cfg):
    state = pdfium_stub(
        [
            FakePathObject(
                [
                    FakePathSegment("move", (0.0, 0.0)),
                    FakePathSegment("line", (0.0, 10.0)),
                ]
            )
        ],
        size=(200.0, 100.0),
    )
    segments, size = load_pdf_segments("dummy.pdf", 0, straight_cfg)
    assert size == (200.0, 100.0)
    assert len(segments) == 1
    _assert_segments_plain(segments)
    assert all(doc.closed for doc in state["docs"])
    assert all(page.closed for page in state["pages"])


def test_public_api_no_tuple_outputs(pdfium_stub, tmp_path, straight_cfg):
    pdfium_stub(
        [
            FakePathObject(
                [
                    FakePathSegment("move", (0.0, 0.0)),
                    FakePathSegment("line", (10.0, 0.0)),
                ]
            )
        ]
    )
    pdf_segments, _ = load_pdf_segments("dummy.pdf", 0, straight_cfg)
    _assert_segments_plain(pdf_segments)

    svg = tmp_path / "segments.svg"
    svg.write_text(
        """
        <svg width=\"10\" height=\"10\">
            <line x1=\"0\" y1=\"0\" x2=\"10\" y2=\"0\" />
        </svg>
        """
    )
    svg_segments, _ = load_svg_segments(str(svg), {})
    _assert_segments_plain(svg_segments)
