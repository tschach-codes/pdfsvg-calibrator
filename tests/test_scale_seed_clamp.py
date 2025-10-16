import os
import sys
from typing import List, Tuple

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pdfsvg_calibrator.calibrate as calibrate_module
import pdfsvg_calibrator.fit_model as fit_model_module
from pdfsvg_calibrator.calibrate import calibrate
from pdfsvg_calibrator.types import Model, Segment


def _grid_segments(width: float, height: float) -> List[Segment]:
    segs: List[Segment] = []
    xs = [width * f for f in (0.1, 0.25, 0.55, 0.82)]
    ys = [height * f for f in (0.12, 0.38, 0.6, 0.86)]
    for y in ys:
        segs.append(Segment(xs[0], y, xs[-1], y))
    for x in xs:
        segs.append(Segment(x, ys[0], x, ys[-1]))
    return segs


def _transform_segments(
    segments: List[Segment],
    sx: float,
    sy: float,
    tx: float,
    ty: float,
) -> List[Segment]:
    transformed: List[Segment] = []
    for seg in segments:
        pts = [(seg.x1, seg.y1), (seg.x2, seg.y2)]
        mapped = [(sx * x + tx, sy * y + ty) for x, y in pts]
        (x1, y1), (x2, y2) = mapped
        transformed.append(Segment(x1=x1, y1=y1, x2=x2, y2=y2))
    return transformed


def test_scale_seed_and_bounds_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_size = (180.0, 120.0)
    pdf_segments = _grid_segments(*pdf_size)

    sx_true = 1.015
    sy_true = 0.992
    tx_true = 1.5
    ty_true = -2.4

    svg_segments = _transform_segments(pdf_segments, sx_true, sy_true, tx_true, ty_true)
    svg_size = (pdf_size[0] * sx_true, pdf_size[1] * sy_true)

    refine_scale_window = 0.012
    cfg = {
        "orientation": {"enabled": True},
        "refine": {
            "scale_max_dev_rel": refine_scale_window,
            "trans_max_dev_px": 4.0,
            "max_iters": 24,
            "max_samples": 260,
        },
        "sampling": {"step_rel": 0.05, "max_points": 320},
        "grid": {"initial_cell_rel": 0.08, "final_cell_rel": 0.04},
        "ransac": {"patience": 20, "iters": 60},
    }

    expected_bounds = (
        (sx_true * (1.0 - refine_scale_window), sx_true * (1.0 + refine_scale_window)),
        (sy_true * (1.0 - refine_scale_window), sy_true * (1.0 + refine_scale_window)),
    )

    captured: dict[str, Tuple[Tuple[float, float], Tuple[float, float]] | Tuple[float, float]] = {}

    def fake_pick(*_args, **_kwargs):
        return (1.0, 1.0), 0, 0.0, 0.0

    monkeypatch.setattr(calibrate_module, "pick_flip_and_rot", fake_pick)

    def stub_ransac(*args, **kwargs):
        cfg_local = kwargs.get("cfg") if "cfg" in kwargs else args[4]
        refine_cfg = cfg_local.get("refine", {})
        captured["scale_seed"] = refine_cfg.get("scale_seed")
        captured["scale_abs_bounds"] = refine_cfg.get("scale_abs_bounds")
        return Model(rot_deg=0, sx=sx_true, sy=sy_true, tx=tx_true, ty=ty_true, score=1.0, rmse=0.0, p95=0.0, median=0.0)

    monkeypatch.setattr(calibrate_module, "_calibrate_ransac", stub_ransac)

    calibrate(pdf_segments, svg_segments, pdf_size, svg_size, cfg)

    scale_seed = captured.get("scale_seed")
    assert scale_seed is not None
    assert scale_seed == pytest.approx((sx_true, sy_true))

    bounds = captured.get("scale_abs_bounds")
    assert bounds is not None
    sx_bounds, sy_bounds = bounds  # type: ignore[misc]
    assert sx_bounds == pytest.approx(expected_bounds[0])
    assert sy_bounds == pytest.approx(expected_bounds[1])

    chamfer_calls: list[Tuple[float, float]] = []
    real_chamfer = fit_model_module.chamfer_score

    def recording_chamfer(*args, **kwargs):
        sx_eval = abs(args[1])
        sy_eval = abs(args[2])
        chamfer_calls.append((sx_eval, sy_eval))
        return real_chamfer(*args, **kwargs)

    monkeypatch.setattr(fit_model_module, "chamfer_score", recording_chamfer)

    refine_cfg = {
        "scale_seed": scale_seed,
        "trans_seed": (tx_true, ty_true),
        "scale_abs_bounds": bounds,
        "scale_max_dev_rel": refine_scale_window,
        "trans_max_dev_px": cfg["refine"]["trans_max_dev_px"],
        "max_iters": cfg["refine"]["max_iters"],
        "max_samples": cfg["refine"]["max_samples"],
    }
    cfg_fit = {
        "refine": refine_cfg,
        "sampling": cfg["sampling"],
        "grid": cfg["grid"],
        "ransac": cfg["ransac"],
        "orientation": {"flip_xy": (1.0, 1.0)},
    }

    fit_model_module.calibrate(pdf_segments, svg_segments, pdf_size, svg_size, cfg_fit)

    for sx_eval, sy_eval in chamfer_calls:
        assert expected_bounds[0][0] - 1e-6 <= sx_eval <= expected_bounds[0][1] + 1e-6
        assert expected_bounds[1][0] - 1e-6 <= sy_eval <= expected_bounds[1][1] + 1e-6
