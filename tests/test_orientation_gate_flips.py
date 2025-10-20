import math
import os
import sys
from types import SimpleNamespace
from itertools import product

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.orientation import pick_flip_rot_and_shift
from pdfsvg_calibrator.types import Segment


def _base_segments() -> list[Segment]:
    return [
        Segment(25.0, 30.0, 180.0, 45.0),
        Segment(60.0, 40.0, 95.0, 150.0),
        Segment(40.0, 120.0, 190.0, 95.0),
        Segment(150.0, 60.0, 158.0, 145.0),
        Segment(80.0, 22.0, 125.0, 72.0),
        Segment(45.0, 55.0, 140.0, 52.0),
        Segment(58.0, 70.0, 68.0, 155.0),
        Segment(120.0, 110.0, 200.0, 160.0),
    ]


def _transform_segments(
    segments: list[Segment],
    flip: tuple[float, float],
    rot_deg: int,
    page_size: tuple[float, float],
    tx: float = 0.0,
    ty: float = 0.0,
) -> list[Segment]:
    cx, cy = page_size[0] / 2.0, page_size[1] / 2.0
    ang = math.radians(rot_deg % 360)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    transformed: list[Segment] = []
    for seg in segments:
        coords = [(seg.x1, seg.y1), (seg.x2, seg.y2)]
        new_coords = []
        for x, y in coords:
            fx = cx + flip[0] * (x - cx)
            fy = cy + flip[1] * (y - cy)
            tx_rel = fx - cx
            ty_rel = fy - cy
            rx = cos_a * tx_rel - sin_a * ty_rel
            ry = sin_a * tx_rel + cos_a * ty_rel
            new_coords.append((rx + cx + tx, ry + cy + ty))
        (x1, y1), (x2, y2) = new_coords
        transformed.append(Segment(x1=x1, y1=y1, x2=x2, y2=y2))
    return transformed


def _matrix_for(flip: tuple[float, float], rot_deg: int) -> tuple[int, int]:
    rot_norm = rot_deg % 360
    if rot_norm == 0:
        return (int(math.copysign(1, flip[0])), int(math.copysign(1, flip[1])))
    if rot_norm == 180:
        return (-int(math.copysign(1, flip[0])), -int(math.copysign(1, flip[1])))
    raise ValueError("rotation must be 0 or 180 degrees")


@pytest.mark.parametrize(
    "flip, rot_deg",
    list(product(((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)), (0, 180))),
)
def test_pick_flip_rot_and_shift_detects_orientation(
    flip: tuple[float, float], rot_deg: int
) -> None:
    pdf_size = (220.0, 170.0)
    pdf_segments = _base_segments()
    svg_segments = _transform_segments(pdf_segments, flip, rot_deg, pdf_size)

    cfg = {"orientation": {"sample_topk_rel": 1.0, "raster_size": 128, "min_accept_score": 0.01}}
    page = SimpleNamespace(w=pdf_size[0], h=pdf_size[1])

    result = pick_flip_rot_and_shift(pdf_segments, svg_segments, page, page, cfg)

    result_flip = tuple(result["flip"])
    result_rot = int(result["rot_deg"])

    expected = _matrix_for(flip, rot_deg)
    actual = _matrix_for(result_flip, result_rot)
    assert actual == expected
    assert abs(result.get("dx_doc", 0.0)) < 1e-3
    assert abs(result.get("dy_doc", 0.0)) < 1e-3


def test_pick_flip_rot_and_shift_recovers_translation() -> None:
    pdf_size = (220.0, 170.0)
    pdf_segments = _base_segments()
    tx_true = 6.5
    ty_true = -4.0
    svg_segments = _transform_segments(pdf_segments, (1.0, 1.0), 0, pdf_size, tx=tx_true, ty=ty_true)

    cfg = {"orientation": {"sample_topk_rel": 1.0, "raster_size": 128, "min_accept_score": 0.01}}
    page = SimpleNamespace(w=pdf_size[0], h=pdf_size[1])

    result = pick_flip_rot_and_shift(pdf_segments, svg_segments, page, page, cfg)

    assert result["flip"] == (1, 1)
    assert result["rot_deg"] == 0
    tx_seed, ty_seed = result["t_seed"]
    assert pytest.approx(tx_seed, abs=1.0) == -tx_true
    assert pytest.approx(ty_seed, abs=1.0) == -ty_true
