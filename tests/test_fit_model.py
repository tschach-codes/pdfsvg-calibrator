import logging
import math
import os
import re
import sys
from typing import List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.fit_model import calibrate, _filter_by_length
from pdfsvg_calibrator.types import Segment


def _base_segments(width: float, height: float) -> List[Segment]:
    segments: List[Segment] = []
    margin_x = width * 0.1
    margin_y = height * 0.1
    xs = [margin_x, width * 0.4, width * 0.7, width - margin_x]
    ys = [margin_y, height * 0.35, height * 0.65, height - margin_y]
    for y in ys:
        segments.append(Segment(x1=xs[0], y1=y, x2=xs[-1], y2=y))
    for x in xs:
        segments.append(Segment(x1=x, y1=ys[0], x2=x, y2=ys[-1]))
    return segments


def _dense_segments(width: float, height: float, count_x: int, count_y: int) -> List[Segment]:
    segments: List[Segment] = []
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
        segments.append(Segment(x1=start, y1=y, x2=start + span, y2=y))
    for idx, x in enumerate(xs):
        span = height * (0.88 if idx % 2 == 0 else 0.76)
        start = height * 0.06
        segments.append(Segment(x1=x, y1=start, x2=x, y2=start + span))
    return segments


def _rotate_point(x: float, y: float, angle_deg: float, center: Tuple[float, float]) -> Tuple[float, float]:
    cx, cy = center
    rad = math.radians(angle_deg % 360.0)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    tx = x - cx
    ty = y - cy
    rx = cos_a * tx - sin_a * ty
    ry = sin_a * tx + cos_a * ty
    return rx + cx, ry + cy


def _transform_segments(
    segs: Sequence[Segment],
    pdf_size: Tuple[float, float],
    rot_deg: float,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
) -> List[Segment]:
    center = (pdf_size[0] * 0.5, pdf_size[1] * 0.5)
    out: List[Segment] = []
    for seg in segs:
        x1, y1 = _rotate_point(seg.x1, seg.y1, rot_deg, center)
        x2, y2 = _rotate_point(seg.x2, seg.y2, rot_deg, center)
        out.append(
            Segment(
                x1=sx * x1 + tx,
                y1=sy * y1 + ty,
                x2=sx * x2 + tx,
                y2=sy * y2 + ty,
            )
        )
    return out


def _transformed_page_size(
    pdf_size: Tuple[float, float],
    rot_deg: float,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
) -> Tuple[float, float]:
    corners = [
        (0.0, 0.0),
        (pdf_size[0], 0.0),
        (0.0, pdf_size[1]),
        (pdf_size[0], pdf_size[1]),
    ]
    transformed = []
    center = (pdf_size[0] * 0.5, pdf_size[1] * 0.5)
    for x, y in corners:
        rx, ry = _rotate_point(x, y, rot_deg, center)
        transformed.append((sx * rx + tx, sy * ry + ty))
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    return max(xs) - min(xs), max(ys) - min(ys)


def _config(rot_degrees: Sequence[int], seed: int = 123) -> dict:
    return {
        "rot_degrees": list(rot_degrees),
        "angle_tol_deg": 6.0,
        "grid_cell_rel": 0.02,
        "chamfer": {"sigma_rel": 0.004, "hard_mul": 3.0},
        "ransac": {
            "iters": 800,
            "refine_scale_step": 0.004,
            "refine_trans_px": 3.0,
            "max_no_improve": 250,
        },
        "sampling": {"step_rel": 0.03, "max_points": 3000},
        "rng_seed": seed,
    }


def _assert_model_close(model, expected, cfg, svg_size):
    tol_scale = 0.01
    tol_trans = 2 * cfg["ransac"]["refine_trans_px"]
    assert model.rot_deg == expected["rot_deg"]
    assert abs(model.sx - expected["sx"]) / expected["sx"] <= tol_scale
    assert abs(model.sy - expected["sy"]) / expected["sy"] <= tol_scale
    assert abs(model.tx - expected["tx"]) <= tol_trans
    assert abs(model.ty - expected["ty"]) <= tol_trans
    diag_svg = math.hypot(*svg_size)
    hard = cfg["chamfer"]["sigma_rel"] * diag_svg * cfg["chamfer"]["hard_mul"]
    assert model.rmse < hard
    assert model.p95 < hard
    assert model.median < hard


def test_calibrate_affine_no_flip():
    pdf_size = (220.0, 160.0)
    pdf_segments = _base_segments(*pdf_size)
    true_params = {"rot_deg": 0, "sx": 1.18, "sy": 0.92, "tx": 40.0, "ty": 26.0}
    svg_segments = _transform_segments(pdf_segments, pdf_size, **true_params)
    svg_size = _transformed_page_size(pdf_size, **true_params)
    cfg = _config([0, 180], seed=321)

    model = calibrate(pdf_segments, svg_segments, pdf_size, svg_size, cfg)

    _assert_model_close(model, true_params, cfg, svg_size)


def test_calibrate_affine_with_rotation_180():
    pdf_size = (200.0, 150.0)
    pdf_segments = _base_segments(*pdf_size)
    true_params = {"rot_deg": 180, "sx": 0.85, "sy": 1.05, "tx": 60.0, "ty": 35.0}
    svg_segments = _transform_segments(pdf_segments, pdf_size, **true_params)
    svg_size = _transformed_page_size(pdf_size, **true_params)
    cfg = _config([180, 0], seed=999)

    model = calibrate(pdf_segments, svg_segments, pdf_size, svg_size, cfg)

    _assert_model_close(model, true_params, cfg, svg_size)


def test_filter_by_length_uses_absolute_threshold():
    segs = [
        Segment(x1=0.0, y1=0.0, x2=30.0, y2=0.0),  # long enough
        Segment(x1=0.0, y1=10.0, x2=15.0, y2=10.0),  # too short
        Segment(x1=0.0, y1=20.0, x2=0.0, y2=45.0),  # long enough vertical
    ]

    filtered = _filter_by_length(segs, min_length=20.0)

    assert filtered == [segs[0], segs[2]]


def test_filter_by_length_keeps_original_for_zero_threshold():
    segs = [
        Segment(x1=0.0, y1=0.0, x2=5.0, y2=0.0),
        Segment(x1=0.0, y1=5.0, x2=0.0, y2=9.0),
    ]

    filtered = _filter_by_length(segs, min_length=0.0)

    assert filtered == segs


def test_ransac_early_stop_large_scene(caplog):
    pdf_size = (980.0, 720.0)
    pdf_segments = _dense_segments(*pdf_size, count_x=14, count_y=12)
    true_params = {"rot_deg": 0, "sx": 1.08, "sy": 0.95, "tx": 45.0, "ty": 32.0}
    svg_segments = _transform_segments(pdf_segments, pdf_size, **true_params)
    svg_size = _transformed_page_size(pdf_size, **true_params)

    cfg = _config([0], seed=4242)
    cfg["ransac"]["iters"] = 600
    cfg["ransac"]["max_no_improve"] = 40
    cfg["sampling"]["max_points"] = 2500

    caplog.set_level(logging.DEBUG, "pdfsvg_calibrator")

    model = calibrate(pdf_segments, svg_segments, pdf_size, svg_size, cfg)

    _assert_model_close(model, true_params, cfg, svg_size)

    iter_log = None
    abort_logged = False
    for record in caplog.records:
        message = record.getMessage()
        if "Abbruch" in message and "max_no_improve" in message:
            abort_logged = True
        if "[calib] rot=0 fertig" in message and "Iterationen" in message:
            iter_log = message
            break

    assert iter_log is not None, "Erwartete Abschlussmeldung wurde nicht geloggt"
    match = re.search(r"(\d+) Iterationen", iter_log)
    assert match is not None, f"Iterationen konnten nicht aus Log extrahiert werden: {iter_log}"
    executed_iters = int(match.group(1))

    assert executed_iters < cfg["ransac"]["iters"]
    assert abort_logged, "RANSAC sollte vorzeitig wegen max_no_improve abbrechen"

    assert model.score > 0.0
