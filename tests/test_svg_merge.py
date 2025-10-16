import pytest

from pdfsvg_calibrator.svg_path import _merge_collinear_segments
from pdfsvg_calibrator.types import Segment


def _merge_cfg(**overrides):
    cfg = {
        "merge": {
            "enable": True,
            "collinear_angle_tol_deg": 3.0,
            "gap_max_rel": 0.01,
            "offset_tol_rel": 0.01,
        }
    }
    cfg["merge"].update(overrides)
    return cfg


def test_merge_collinear_segments_combines_chain():
    segments = [
        Segment(0.0, 0.0, 10.0, 0.0),
        Segment(10.0, 0.0, 20.0, 0.0),
        Segment(20.5, 0.0, 30.5, 0.0),
    ]

    merged = _merge_collinear_segments(segments, diag=100.0, cfg=_merge_cfg())

    assert len(merged) == 1
    merged_seg = merged[0]
    endpoints = sorted([(merged_seg.x1, merged_seg.y1), (merged_seg.x2, merged_seg.y2)])
    assert endpoints[0][0] == pytest.approx(0.0)
    assert endpoints[0][1] == pytest.approx(0.0)
    assert endpoints[1][0] == pytest.approx(30.5)
    assert endpoints[1][1] == pytest.approx(0.0)


def test_merge_segments_respects_angle_and_offset():
    base = Segment(0.0, 0.0, 10.0, 0.0)
    angled = Segment(10.0, 0.0, 20.0, 1.0)
    offset = Segment(10.0, 0.8, 20.0, 0.8)

    # Angle above tolerance prevents merge
    merged_angle = _merge_collinear_segments(
        [base, angled], diag=100.0, cfg=_merge_cfg(collinear_angle_tol_deg=2.0)
    )
    assert len(merged_angle) == 2

    # Offset above tolerance prevents merge
    merged_offset = _merge_collinear_segments(
        [base, offset], diag=100.0, cfg=_merge_cfg(offset_tol_rel=0.001)
    )
    assert len(merged_offset) == 2


def test_merge_disabled_returns_original():
    segments = [Segment(0.0, 0.0, 10.0, 0.0), Segment(10.0, 0.0, 20.0, 0.0)]
    merged = _merge_collinear_segments(segments, diag=100.0, cfg={"merge": {"enable": False}})
    assert merged == segments
