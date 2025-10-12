import math
import os
import sys
from typing import List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.fit_model import calibrate
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
        "ransac": {"iters": 800, "refine_scale_step": 0.004, "refine_trans_px": 3.0},
        "sampling": {"step_rel": 0.02, "max_points": 4000},
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
