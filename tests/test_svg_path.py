from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pdfsvg_calibrator.svg_path import parse_svg_segments


BASE_CFG = {
    "curve_tol_rel": 0.0005,
    "straight_max_dev_rel": 0.005,
    "straight_max_angle_spread_deg": 4.0,
}


def _write_svg(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "test.svg"
    path.write_text(content.strip())
    return path


def _segment_coords(segments):
    return {(round(s.x1, 3), round(s.y1, 3), round(s.x2, 3), round(s.y2, 3)) for s in segments}


def test_line_and_rect_with_group_transform(tmp_path: Path):
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
      <g transform="translate(10,20)">
        <line x1="0" y1="0" x2="30" y2="0" />
        <rect x="5" y="5" width="20" height="10" />
      </g>
    </svg>
    """
    path = _write_svg(tmp_path, svg)
    segs = parse_svg_segments(str(path), BASE_CFG)

    assert len(segs) == 5
    coords = _segment_coords(segs)
    expected = {
        (10.0, 20.0, 40.0, 20.0),
        (15.0, 25.0, 35.0, 25.0),
        (35.0, 25.0, 35.0, 35.0),
        (35.0, 35.0, 15.0, 35.0),
        (15.0, 35.0, 15.0, 25.0),
    }
    assert coords == expected


def test_cubic_curve_is_reduced_to_segment(tmp_path: Path):
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 10" width="120" height="10">
      <path d="M 0 5 C 30 6 90 4 120 5" />
    </svg>
    """
    path = _write_svg(tmp_path, svg)
    segs = parse_svg_segments(str(path), BASE_CFG)

    assert len(segs) == 1
    seg = segs[0]
    assert pytest.approx(min(seg.x1, seg.x2), abs=0.5) == 0.0
    assert pytest.approx(max(seg.x1, seg.x2), abs=0.5) == 120.0
    assert abs(seg.y1 - 5.0) < 0.5
    assert abs(seg.y2 - 5.0) < 0.5


def test_use_and_symbol_with_viewbox(tmp_path: Path):
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100" width="400" height="200">
      <defs>
        <symbol id="tick" viewBox="0 0 10 10">
          <polyline points="0,10 5,0 10,10" />
        </symbol>
      </defs>
      <use href="#tick" x="50" y="40" transform="scale(2)" />
    </svg>
    """
    path = _write_svg(tmp_path, svg)
    segs = parse_svg_segments(str(path), BASE_CFG)

    # Polyline should produce two segments after scaling and translation.
    assert len(segs) == 2
    xs = [seg.x1 for seg in segs] + [seg.x2 for seg in segs]
    ys = [seg.y1 for seg in segs] + [seg.y2 for seg in segs]

    assert min(xs) == pytest.approx(100.0, rel=1e-3)
    assert max(xs) == pytest.approx(140.0, rel=1e-3)
    assert min(ys) == pytest.approx(80.0, rel=1e-3)
    assert max(ys) == pytest.approx(120.0, rel=1e-3)
