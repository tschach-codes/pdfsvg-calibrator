"""Orientation gating utilities for pdfsvg-calibrator."""
from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .types import Segment

log = logging.getLogger(__name__)

DEFAULT_RASTER_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_USE_PHASE_CORRELATION: bool = True

_FOUR_ROTATIONS = (0, 90, 180, 270)
_FLIPS = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def _ensure_size_tuple(size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    seq = list(size)
    if len(seq) != 2:
        raise ValueError("size must be an int or a sequence of two ints")
    return int(seq[0]), int(seq[1])


def _apply_flip_and_rotation(
    x: float, y: float, flip: Tuple[float, float], rot_deg: int
) -> Tuple[float, float]:
    cx = 0.5
    cy = 0.5
    # flip around center
    x = cx + flip[0] * (x - cx)
    y = cy + flip[1] * (y - cy)
    if rot_deg % 360 == 0:
        return x, y
    rad = math.radians(rot_deg % 360)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    tx = x - cx
    ty = y - cy
    rx = cos_a * tx - sin_a * ty
    ry = sin_a * tx + cos_a * ty
    return rx + cx, ry + cy


def _normalize_segments(
    segments: Sequence[Segment],
    page_size: Tuple[float, float],
) -> Iterable[Segment]:
    width, height = page_size
    if width <= 0 or height <= 0:
        raise ValueError("page dimensions must be positive")
    inv_w = 1.0 / width
    inv_h = 1.0 / height
    for seg in segments:
        yield Segment(
            x1=seg.x1 * inv_w,
            y1=seg.y1 * inv_h,
            x2=seg.x2 * inv_w,
            y2=seg.y2 * inv_h,
        )


def rasterize_segments(
    segments: Sequence[Segment],
    size: Tuple[int, int] | int = (512, 512),
    flip: Tuple[float, float] = (1.0, 1.0),
    rot_deg: int = 0,
) -> np.ndarray:
    """Rasterize ``segments`` into a binary image.

    Parameters
    ----------
    segments:
        Segment list normalised to the unit square (0..1).
    size:
        Raster size either as a scalar or (width, height) tuple.
    flip:
        Flip applied around the image centre for x/y axes (1 or -1).
    rot_deg:
        Rotation in degrees applied after flipping.
    """

    width, height = _ensure_size_tuple(size)
    img = np.zeros((height, width), dtype=np.float32)
    if not segments:
        return img

    for seg in segments:
        x1 = float(seg.x1)
        y1 = float(seg.y1)
        x2 = float(seg.x2)
        y2 = float(seg.y2)
        x1, y1 = _apply_flip_and_rotation(x1, y1, flip, rot_deg)
        x2, y2 = _apply_flip_and_rotation(x2, y2, flip, rot_deg)
        num = max(int(math.hypot((x2 - x1) * width, (y2 - y1) * height) * 2), 1)
        t_values = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
        xs_line = x1 + (x2 - x1) * t_values
        ys_line = y1 + (y2 - y1) * t_values
        ix = np.clip(np.round(xs_line * (width - 1)), 0, width - 1).astype(int)
        iy = np.clip(np.round(ys_line * (height - 1)), 0, height - 1).astype(int)
        img[iy, ix] = 1.0

    return img


def _phase_correlation_fft(img_a: np.ndarray, img_b: np.ndarray) -> Tuple[float, float, float]:
    if img_a.shape != img_b.shape:
        raise ValueError("images must share the same shape")
    if img_a.ndim != 2:
        raise ValueError("images must be 2-D arrays")
    fa = np.fft.fft2(img_a)
    fb = np.fft.fft2(img_b)
    product = fa * np.conj(fb)
    magnitude = np.abs(product)
    magnitude[magnitude == 0] = 1.0
    cross_power = product / magnitude
    corr = np.fft.ifft2(cross_power)
    corr_mag = np.abs(corr)
    peak_index = np.unravel_index(np.argmax(corr_mag), corr_mag.shape)
    peak_value = float(corr_mag[peak_index])
    shift_y = float(peak_index[0])
    shift_x = float(peak_index[1])
    height, width = img_a.shape
    if shift_x > width / 2:
        shift_x -= width
    if shift_y > height / 2:
        shift_y -= height
    return shift_x, shift_y, peak_value


def phase_correlation(img_a: np.ndarray, img_b: np.ndarray) -> Tuple[float, float]:
    """Compute phase correlation shift between two 2-D images."""

    shift_x, shift_y, _ = _phase_correlation_fft(img_a, img_b)
    return shift_x, shift_y


def pick_flip_and_rot(
    pdf_segments: Sequence[Segment],
    svg_segments: Sequence[Segment],
    page_size_pdf: Tuple[float, float],
    page_size_svg: Tuple[float, float],
) -> Tuple[Tuple[float, float], int, float, float]:
    """Pick flip and rotation candidate via raster correlation."""

    if not pdf_segments or not svg_segments:
        raise ValueError("Both PDF and SVG segments are required")

    raster_size = DEFAULT_RASTER_SIZE
    pdf_norm = list(_normalize_segments(pdf_segments, page_size_pdf))
    svg_norm = list(_normalize_segments(svg_segments, page_size_svg))
    pdf_image = rasterize_segments(pdf_norm, raster_size, (1.0, 1.0), 0)

    best_score = -math.inf
    best_flip = (1.0, 1.0)
    best_rot = 0
    best_shift = (0.0, 0.0)

    for rot_deg in _FOUR_ROTATIONS:
        for flip in _FLIPS:
            svg_image = rasterize_segments(svg_norm, raster_size, flip, rot_deg)
            if DEFAULT_USE_PHASE_CORRELATION:
                dx, dy, score = _phase_correlation_fft(pdf_image, svg_image)
            else:
                diff = np.sum(np.abs(pdf_image - svg_image))
                score = -float(diff)
                dx = dy = 0.0
            if score > best_score:
                best_score = score
                best_flip = flip
                best_rot = rot_deg
                best_shift = (dx, dy)

    width, height = _ensure_size_tuple(raster_size)
    oriented_width = page_size_svg[0] if best_rot % 180 == 0 else page_size_svg[1]
    oriented_height = page_size_svg[1] if best_rot % 180 == 0 else page_size_svg[0]
    dx_norm = best_shift[0] / float(max(width, 1))
    dy_norm = best_shift[1] / float(max(height, 1))
    tx0 = -dx_norm * oriented_width
    ty0 = -dy_norm * oriented_height

    log.debug(
        "[orient] pick flip=%s rot=%d score=%.3f shift_px=(%.1f, %.1f)",
        best_flip,
        best_rot,
        best_score,
        best_shift[0],
        best_shift[1],
    )

    return best_flip, best_rot, tx0, ty0
