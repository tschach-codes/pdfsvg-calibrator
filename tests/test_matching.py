from __future__ import annotations

from typing import List

from pdfsvg_calibrator.match_verify import match_lines
from pdfsvg_calibrator.types import Match, Model, Segment


def make_segment(x1: float, y1: float, x2: float, y2: float) -> Segment:
    return Segment(x1, y1, x2, y2)


def make_model() -> Model:
    return Model(rot_deg=0, sx=2.0, sy=3.0, tx=10.0, ty=-5.0, score=1.0, rmse=0.0, p95=0.0, median=0.0)


def make_config(max_cost: float | None = None) -> dict:
    cfg = {
        "verify": {
            "dir_tol_deg": 1.5,
            "radius_px": 6.0,
            "tol_rel": 0.01,
        },
        "neighbors": {
            "use": True,
            "radius_rel": 0.6,
            "dt": 0.2,
            "dtheta_deg": 5.0,
            "rho_soft": 0.2,
            "penalty_empty": 6.0,
            "penalty_miss": 2.0,
        },
        "cost_weights": {
            "endpoint": 1.5,
            "midpoint": 0.6,
            "direction": 0.4,
            "neighbors": 1.0,
        },
    }
    if max_cost is not None:
        cfg["verify"]["max_cost"] = max_cost
    return cfg


def reference_pdf_segments() -> List[Segment]:
    return [
        make_segment(0.0, 0.0, 6.0, 0.0),
        make_segment(0.0, 0.0, 0.0, 4.0),
        make_segment(0.0, 4.0, 6.0, 4.0),
        make_segment(6.0, 0.0, 6.0, 4.0),
        make_segment(0.0, 2.0, 6.0, 2.0),
    ]


def transformed_svg_segments() -> List[Segment]:
    return [
        make_segment(10.0, -5.0, 22.0, -5.0),
        make_segment(10.0, -5.0, 10.0, 7.0),
        make_segment(10.0, 7.0, 22.0, 7.0),
        make_segment(22.0, -5.0, 22.0, 7.0),
        make_segment(10.0, 1.0, 22.0, 1.0),
    ]


def test_match_lines_clean_scene() -> None:
    pdf_lines = reference_pdf_segments()
    svg_lines = transformed_svg_segments()
    matches = match_lines(pdf_lines, svg_lines, make_model(), make_config())
    assert len(matches) == 5
    for match in matches:
        assert match.svg_seg is not None
        assert match.pass01 == 1
        assert match.rel_error is not None and match.rel_error <= 1e-9
        assert match.confidence >= 0.4


def test_neighbor_signature_disambiguates_candidates() -> None:
    pdf_lines = reference_pdf_segments()
    svg_lines = transformed_svg_segments()
    decoy = make_segment(13.0, -5.0, 25.0, -5.0)
    svg_lines.append(decoy)
    matches = match_lines(pdf_lines, svg_lines, make_model(), make_config())
    first_match = matches[0]
    assert first_match.svg_seg is not None
    assert first_match.svg_seg.x1 == 10.0 and first_match.svg_seg.x2 == 22.0
    assert first_match.cost < 10.0
    assert matches[0].svg_seg != decoy


def test_partial_missing_segments_produce_no_match() -> None:
    pdf_lines = reference_pdf_segments()
    svg_lines = transformed_svg_segments()
    removed = svg_lines[:3]
    matches = match_lines(pdf_lines, removed, make_model(), make_config(max_cost=120000.0))
    assert len(matches) == 5
    matched_svg_ids = [id(m.svg_seg) for m in matches if m.svg_seg is not None]
    assert len(set(matched_svg_ids)) == len(matched_svg_ids)
    no_match_indices = [i for i, m in enumerate(matches) if m.svg_seg is None]
    assert len(no_match_indices) >= 2
    for idx in no_match_indices:
        assert matches[idx].pass01 is None
        assert matches[idx].confidence == 0.0
