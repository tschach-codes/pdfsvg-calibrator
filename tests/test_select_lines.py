import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdfsvg_calibrator.match_verify import select_lines
from pdfsvg_calibrator.geom import classify_hv
from pdfsvg_calibrator.types import Model, Segment


def _base_cfg():
    return {
        "verify": {"pick_k": 5, "diversity_rel": 0.1, "dir_tol_deg": 6.0},
        "sampling": {"step_rel": 0.02, "max_points": 2048},
        "chamfer": {"sigma_rel": 0.01, "hard_mul": 3.0},
        "grid_cell_rel": 0.05,
    }


def _diag(segments):
    xs = [seg.x1 for seg in segments] + [seg.x2 for seg in segments]
    ys = [seg.y1 for seg in segments] + [seg.y2 for seg in segments]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _model():
    return Model(rot_deg=0, sx=1.0, sy=1.0, tx=0.0, ty=0.0, score=1.0, rmse=0.0, p95=0.0, median=0.0)


def test_select_lines_balanced_diversity():
    horizontals = [
        Segment(0.0, 0.0, 200.0, 0.0),
        Segment(0.0, 150.0, 200.0, 150.0),
        Segment(0.0, 300.0, 200.0, 300.0),
    ]
    verticals = [
        Segment(0.0, 0.0, 0.0, 320.0),
        Segment(200.0, 0.0, 200.0, 320.0),
        Segment(100.0, 0.0, 100.0, 320.0),
    ]
    pdf_segments = horizontals + verticals
    svg_segments = [Segment(s.x1, s.y1, s.x2, s.y2) for s in pdf_segments]

    selected, info = select_lines(pdf_segments, _model(), svg_segments, _base_cfg())

    assert len(selected) == 5
    h_sel, v_sel = classify_hv(selected)
    assert len(h_sel) >= 2
    assert len(v_sel) >= 2
    diag = _diag(pdf_segments)
    threshold = diag * _base_cfg()["verify"]["diversity_rel"]
    centers = [((s.x1 + s.x2) * 0.5, (s.y1 + s.y2) * 0.5) for s in selected]
    for i, (cx, cy) in enumerate(centers):
        for ox, oy in centers[i + 1 :]:
            assert math.hypot(cx - ox, cy - oy) >= threshold - 1e-6
    assert info["diversity_enforced"] is True
    assert info["had_enough_H"] is True
    assert info["had_enough_V"] is True


def test_select_lines_fill_shortage():
    horizontals = [
        Segment(0.0, y, 220.0, y) for y in (0.0, 120.0, 240.0, 360.0, 480.0)
    ]
    verticals = [Segment(0.0, 0.0, 0.0, 500.0)]
    pdf_segments = horizontals + verticals
    svg_segments = [Segment(s.x1, s.y1, s.x2, s.y2) for s in pdf_segments]

    selected, info = select_lines(pdf_segments, _model(), svg_segments, _base_cfg())

    assert len(selected) == 5
    h_sel, v_sel = classify_hv(selected)
    assert len(v_sel) == 1
    assert info["had_enough_H"] is True
    assert info["had_enough_V"] is False
    assert any("vertical" in note.lower() for note in info["notes"])


def test_select_lines_support_prefers_stronger_matches():
    pdf_segments = [
        Segment(0.0, y, 200.0, y) for y in (0.0, 120.0, 240.0, 360.0, 480.0, 600.0)
    ]
    svg_segments = [Segment(s.x1, s.y1, s.x2, s.y2) for s in pdf_segments[:-1]]

    selected, info = select_lines(pdf_segments, _model(), svg_segments, _base_cfg())

    assert len(selected) == 5
    unsupported = pdf_segments[-1]
    assert unsupported not in selected
    assert info["had_enough_V"] is False
    assert any("vertical" in note.lower() for note in info["notes"])


def test_select_lines_prefilter_limits_scoring():
    horizontals = [
        Segment(0.0, float(idx) * 20.0, length, float(idx) * 20.0)
        for idx, length in enumerate((120.0, 160.0, 200.0, 240.0, 280.0, 320.0))
    ]
    verticals = [
        Segment(float(idx) * 20.0, 0.0, float(idx) * 20.0, length)
        for idx, length in enumerate((120.0, 160.0, 200.0, 240.0, 280.0, 320.0))
    ]
    pdf_segments = horizontals + verticals
    svg_segments = [Segment(s.x1, s.y1, s.x2, s.y2) for s in pdf_segments]

    cfg = _base_cfg()
    cfg["verify"]["pick_k"] = 3
    cfg["verify"]["max_candidates_per_axis"] = 3
    cfg["rng_seed"] = 1234

    selected, info = select_lines(pdf_segments, _model(), svg_segments, cfg)

    assert len(selected) == 3
    assert info["scored_candidates"] == {"H": 3, "V": 3}
    assert info["prefilter_dropped"] == {"H": 3, "V": 3}
    # Ensure the selected segments respect the requested pick count and contain
    # candidates from both axes after the prefiltering.
    h_sel, v_sel = classify_hv(selected)
    assert len(h_sel) >= 1
    assert len(v_sel) >= 1
