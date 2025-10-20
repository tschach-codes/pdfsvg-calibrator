import math
import os
import sys
import time
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coarse.pipeline import coarse_align  # type: ignore
from coarse.corr import Orientation  # type: ignore
from coarse.geom import apply_orientation_pts, transform_segments  # type: ignore


_DEF_CFG = {
    "coarse": {
        "enabled": True,
        "bins": 512,
        "blur_sigma_bins": 1.0,
        "orientations": [0, 180],
        "flips": ["none", "x", "y"],
        "scale_quantiles": [0.05, 0.95],
        "fallback_use_heatmap": False,
        "kdtree_match": {
            "angle_tol_deg": 5.0,
            "len_tol_rel": 0.2,
            "dist_tol_px": 10.0,
            "max_pairs": 5000,
        },
        "score_weights": {"corr_x": 0.4, "corr_y": 0.4, "inliers": 0.2},
        "adapt": {
            "min_long_segments": 1,
            "relax_topk_rel": [1.0],
            "relax_hv_angle": [10.0],
        },
    }
}


def _synthetic_grid_segments(
    width: float,
    height: float,
    n_vertical: int,
    n_horizontal: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    segments = []
    for _ in range(n_vertical):
        x = rng.uniform(width * 0.1, width * 0.9)
        y0 = rng.uniform(height * 0.1, height * 0.4)
        y1 = rng.uniform(height * 0.6, height * 0.9)
        segments.append((x, y0, x, y1))
    for _ in range(n_horizontal):
        y = rng.uniform(height * 0.1, height * 0.9)
        x0 = rng.uniform(width * 0.1, width * 0.5)
        x1 = rng.uniform(width * 0.6, width * 0.9)
        segments.append((x0, y, x1, y))
    return np.asarray(segments, dtype=float)


def _apply_scale(segments: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = segments.copy()
    scaled[:, [0, 2]] *= sx
    scaled[:, [1, 3]] *= sy
    return scaled


def _bbox_from_segments(segments: np.ndarray) -> Tuple[float, float, float, float]:
    xmin = float(np.min(segments[:, [0, 2]]))
    xmax = float(np.max(segments[:, [0, 2]]))
    ymin = float(np.min(segments[:, [1, 3]]))
    ymax = float(np.max(segments[:, [1, 3]]))
    return (xmin, ymin, xmax, ymax)


def test_coarse_align_recovers_orientation_and_scale() -> None:
    width, height = 240.0, 180.0
    base_svg = _synthetic_grid_segments(width, height, 140, 110, seed=123)
    orientation = Orientation(rot_deg=0, flip="x")
    scale = (1.035, 0.965)
    shift = (0.0, 0.0)

    bbox_svg = _bbox_from_segments(base_svg)
    pdf_segments = transform_segments(
        base_svg, lambda pts: apply_orientation_pts(pts, orientation, bbox_svg)
    )
    pdf_segments = _apply_scale(pdf_segments, scale[0], scale[1])

    bbox_pdf = _bbox_from_segments(pdf_segments)

    result = coarse_align(pdf_segments, base_svg, bbox_pdf, bbox_svg, _DEF_CFG)

    assert result.ok, "coarse alignment should succeed"
    assert result.orientation is not None
    assert result.scale is not None
    assert result.shift is not None

    assert result.orientation.rot_deg == orientation.rot_deg
    assert result.orientation.flip == orientation.flip

    assert math.isclose(result.scale[0], scale[0], rel_tol=0.05)
    assert math.isclose(result.scale[1], scale[1], rel_tol=0.05)

    assert abs(result.shift[0] - shift[0]) <= 5.0
    assert abs(result.shift[1] - shift[1]) <= 5.0


@pytest.mark.slow
def test_coarse_align_runtime_stays_under_budget() -> None:
    rng = np.random.default_rng(1234)
    n_short = 80_000
    n_long = 5_000

    short_start = rng.uniform(0.0, 1000.0, size=(n_short, 2))
    angles = rng.choice([0.0, np.pi / 2], size=n_short)
    lengths = rng.uniform(2.0, 8.0, size=n_short)
    dx = np.cos(angles) * lengths
    dy = np.sin(angles) * lengths
    short_segments = np.column_stack((short_start, short_start[:, 0] + dx, short_start[:, 1] + dy))

    grid_long = []
    xs = np.linspace(20.0, 980.0, int(np.sqrt(n_long)))
    ys = np.linspace(30.0, 970.0, int(np.sqrt(n_long)))
    for x in xs:
        grid_long.append((x, 30.0, x, 970.0))
    for y in ys:
        grid_long.append((20.0, y, 980.0, y))
    long_segments = np.asarray(grid_long[:n_long], dtype=float)

    pdf_segments = np.vstack((short_segments, long_segments))
    svg_segments = pdf_segments.copy()

    bbox = _bbox_from_segments(pdf_segments)

    start = time.perf_counter()
    result = coarse_align(pdf_segments, svg_segments, bbox, bbox, _DEF_CFG)
    duration = time.perf_counter() - start

    assert result.ok, "coarse alignment should return ok for identical inputs"
    assert duration < 1.5, f"coarse_align took {duration:.3f}s"
