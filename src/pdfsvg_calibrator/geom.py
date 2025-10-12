from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple, List

import numpy as np

from .types import Segment


def _normalize_angle_deg(angle: float) -> float:
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def fit_straight_segment(
    points: Sequence[Tuple[float, float]]
) -> Tuple[Segment, float, float, float]:
    if len(points) < 2:
        raise ValueError("At least two points required for straightness fit")

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Points must be a sequence of 2D coordinates")

    unique = np.unique(arr, axis=0)
    if unique.shape[0] < 2:
        raise ValueError("Straightness fit requires at least two distinct points")

    centroid = arr.mean(axis=0)
    centered = arr - centroid
    scatter = centered.T @ centered

    vals, vecs = np.linalg.eigh(scatter)
    if vals.shape != (2,):
        raise RuntimeError("Unexpected eigen decomposition result")

    direction = vecs[:, int(np.argmax(vals))]
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = direction / norm

    projections = centered @ direction
    start = centroid + direction * projections.min()
    end = centroid + direction * projections.max()

    perpendicular = centered - np.outer(projections, direction)
    distances = np.linalg.norm(perpendicular, axis=1)
    delta_max = float(distances.max())
    delta_rms = float(np.sqrt(np.mean(distances ** 2)))

    angle_deg = math.degrees(math.atan2(direction[1], direction[0]))
    angle_deg = _normalize_angle_deg(angle_deg)

    segment = Segment(float(start[0]), float(start[1]), float(end[0]), float(end[1]))
    return segment, delta_max, delta_rms, angle_deg


def classify_hv(
    segments: Iterable[Segment], angle_tol_deg: float = 6.0
) -> Tuple[List[Segment], List[Segment]]:
    horizontal: List[Segment] = []
    vertical: List[Segment] = []
    tol = max(angle_tol_deg, 0.0)

    for seg in segments:
        dx = seg.x2 - seg.x1
        dy = seg.y2 - seg.y1
        if dx == 0 and dy == 0:
            continue

        angle = math.degrees(math.atan2(dy, dx))
        angle = _normalize_angle_deg(angle)

        delta_h = min(angle, 180.0 - angle)
        delta_v = abs(angle - 90.0)

        if delta_h <= tol and delta_h <= delta_v:
            horizontal.append(seg)
        elif delta_v <= tol:
            vertical.append(seg)

    return horizontal, vertical


def merge_collinear(segments: List[Segment]) -> List[Segment]:
    return segments
