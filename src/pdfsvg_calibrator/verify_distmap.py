from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .core.grid_safety import ensure_ndarray2d, zeros2d

from .types import Segment
from .transform import Transform2D

_LARGE_DISTANCE = 1e12


def _draw_line(mask: np.ndarray, u1: int, v1: int, u2: int, v2: int) -> None:
    mask = ensure_ndarray2d("mask", mask)
    height, width = mask.shape
    du = abs(u2 - u1)
    dv = abs(v2 - v1)
    steps = int(max(du, dv))
    if steps == 0:
        if 0 <= v1 < height and 0 <= u1 < width:
            mask[v1, u1] = 255
        return
    for step in range(steps + 1):
        t = step / steps
        u = int(round(u1 + (u2 - u1) * t))
        v = int(round(v1 + (v2 - v1) * t))
        if 0 <= v < height and 0 <= u < width:
            mask[v, u] = 255


def _edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    result = np.empty(n, dtype=float)
    v = np.zeros(n, dtype=int)
    z = np.zeros(n + 1, dtype=float)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        diff = q - v[k]
        result[q] = diff * diff + f[v[k]]
    return result


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    mask = ensure_ndarray2d("mask", mask)
    height, width = mask.shape
    features = mask > 0
    if not np.any(features):
        return np.full((height, width), np.inf, dtype=float)
    f = np.where(features, 0.0, _LARGE_DISTANCE)
    tmp = np.empty_like(f, dtype=float)
    for row in range(height):
        tmp[row, :] = _edt_1d(f[row, :])
    dist_sq = np.empty_like(tmp, dtype=float)
    for col in range(width):
        dist_sq[:, col] = _edt_1d(tmp[:, col])
    return np.sqrt(dist_sq, out=dist_sq)


def _segment_points(seg: Segment) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (seg.x1, seg.y1), (seg.x2, seg.y2)


def svg_mask_from_segments(
    svg_segments: Iterable[Segment],
    page_w: float,
    page_h: float,
    *,
    W: int = 768,
    H: int = 768,
) -> np.ndarray:
    mask = zeros2d(H, W, dtype=np.uint8)
    if page_w <= 0 or page_h <= 0:
        return mask
    sx, sy = W / page_w, H / page_h
    for seg in svg_segments:
        (x1, y1), (x2, y2) = _segment_points(seg)
        u1 = int(round(x1 * sx))
        v1 = int(round((page_h - y1) * sy))
        u2 = int(round(x2 * sx))
        v2 = int(round((page_h - y2) * sy))
        u1 = int(np.clip(u1, 0, W - 1))
        v1 = int(np.clip(v1, 0, H - 1))
        u2 = int(np.clip(u2, 0, W - 1))
        v2 = int(np.clip(v2, 0, H - 1))
        _draw_line(mask, u1, v1, u2, v2)
    return mask


def sample_points_on_pdf(
    pdf_segments: Sequence[Segment],
    *,
    n_per_seg: int = 16,
) -> np.ndarray:
    if n_per_seg <= 0:
        n_per_seg = 1
    pts = []
    for seg in pdf_segments:
        (x1, y1), (x2, y2) = _segment_points(seg)
        for i in range(1, n_per_seg + 1):
            t = i / (n_per_seg + 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            pts.append((x, y))
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def evaluate_rmse(
    transform: Transform2D,
    pdf_segments: Sequence[Segment],
    svg_segments: Sequence[Segment],
    page_w: float,
    page_h: float,
    *,
    W: int = 768,
    H: int = 768,
    n_per_seg: int = 16,
) -> dict[str, float | int]:
    mask = svg_mask_from_segments(svg_segments, page_w, page_h, W=W, H=H)
    dt = _distance_transform(mask)
    pts_pdf = sample_points_on_pdf(pdf_segments, n_per_seg=n_per_seg)
    if pts_pdf.size == 0:
        return {"rmse": float("nan"), "p95": float("nan"), "median": float("nan"), "score": 0.0, "n": 0}
    pts_svg = transform.apply(pts_pdf)
    sx, sy = W / page_w if page_w > 0 else 1.0, H / page_h if page_h > 0 else 1.0
    u = np.clip(np.round(pts_svg[:, 0] * sx).astype(int), 0, W - 1)
    v = np.clip(np.round((page_h - pts_svg[:, 1]) * sy).astype(int), 0, H - 1)
    d = dt[v, u]
    avg_scale = 0.5 * ((page_w / W) + (page_h / H)) if W > 0 and H > 0 else 1.0
    d_doc = d * avg_scale
    rmse = float(np.sqrt(np.mean(d_doc**2)))
    p95 = float(np.percentile(d_doc, 95))
    median = float(np.median(d_doc))
    score = 1.0 / (1.0 + rmse)
    return {"rmse": rmse, "p95": p95, "median": median, "score": score, "n": int(d_doc.size)}

