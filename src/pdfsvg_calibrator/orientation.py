"""Orientation gating utilities for pdfsvg-calibrator."""
from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from .types import Segment
from .metrics import Timer

log = logging.getLogger(__name__)

DEFAULT_RASTER_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_USE_PHASE_CORRELATION: bool = True

_FLIPS = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))
_ROTATIONS = (0, 180)


def _ensure_size_tuple(size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    seq = list(size)
    if len(seq) != 2:
        raise ValueError("size must be an int or a sequence of two ints")
    return int(seq[0]), int(seq[1])


def _segments_to_array(segments: Sequence[Segment | Sequence[float]]) -> np.ndarray:
    if isinstance(segments, np.ndarray):
        arr = np.asarray(segments, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("segments must be Nx4 array-like entries")
        return arr

    if not segments:
        return np.zeros((0, 4), dtype=np.float32)

    first = segments[0]
    if isinstance(first, Segment):
        arr = np.array(
            [[seg.x1, seg.y1, seg.x2, seg.y2] for seg in segments],
            dtype=np.float32,
        )
    else:
        arr = np.asarray(segments, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("segments must be Nx4 array-like entries")
    return arr.astype(np.float32, copy=False)


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


def _augment_with_unit_frame(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
    frame = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    return np.vstack([arr, frame])


def _apply_flip_and_rotation(
    coords: np.ndarray,
    flip: Tuple[float, float],
    rot_deg: int,
) -> np.ndarray:
    if coords.size == 0:
        return coords
    cx = 0.5
    cy = 0.5
    arr = coords.copy()
    arr[:, [0, 2]] = cx + flip[0] * (arr[:, [0, 2]] - cx)
    arr[:, [1, 3]] = cy + flip[1] * (arr[:, [1, 3]] - cy)
    angle = rot_deg % 360
    if angle == 0:
        return arr
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    for idx in range(arr.shape[0]):
        for j in (0, 2):
            tx = arr[idx, j] - cx
            ty = arr[idx, j + 1] - cy
            rx = cos_a * tx - sin_a * ty
            ry = sin_a * tx + cos_a * ty
            arr[idx, j] = rx + cx
            arr[idx, j + 1] = ry + cy
    return arr


def rasterize_segments(
    segments: Sequence[Segment | Sequence[float]],
    size: Tuple[int, int] | int = (512, 512),
    flip: Tuple[float, float] = (1.0, 1.0),
    rot_deg: int = 0,
) -> np.ndarray:
    """Rasterize segments into a float32 mask using anti-aliased drawing."""

    with Timer("orientation.rasterize"):
        width, height = _ensure_size_tuple(size)
        img = np.zeros((height, width), dtype=np.float32)
        seg_arr = _segments_to_array(segments)
        if seg_arr.size == 0:
            return img

        xs = seg_arr[:, [0, 2]]
        ys = seg_arr[:, [1, 3]]
        min_x = float(xs.min())
        max_x = float(xs.max())
        min_y = float(ys.min())
        max_y = float(ys.max())
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)

        norm = seg_arr.copy()
        norm[:, [0, 2]] = (norm[:, [0, 2]] - min_x) / span_x
        norm[:, [1, 3]] = (norm[:, [1, 3]] - min_y) / span_y
        norm = np.clip(norm, 0.0, 1.0)
        norm = _apply_flip_and_rotation(norm, flip, rot_deg)

        for x1, y1, x2, y2 in norm:
            length = math.hypot((x2 - x1) * width, (y2 - y1) * height)
            samples = max(int(length * 3), 1)
            ts = np.linspace(0.0, 1.0, samples, dtype=np.float64)
            xs_line = x1 + (x2 - x1) * ts
            ys_line = y1 + (y2 - y1) * ts
            xs_line = np.clip(xs_line, 0.0, 1.0)
            ys_line = np.clip(ys_line, 0.0, 1.0)
            px = xs_line * (width - 1)
            py = ys_line * (height - 1)
            ix0 = np.floor(px).astype(int)
            iy0 = np.floor(py).astype(int)
            ix1 = np.clip(ix0 + 1, 0, width - 1)
            iy1 = np.clip(iy0 + 1, 0, height - 1)
            dx = px - ix0
            dy = py - iy0

            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy

            np.add.at(img, (iy0, ix0), w00)
            np.add.at(img, (iy0, ix1), w10)
            np.add.at(img, (iy1, ix0), w01)
            np.add.at(img, (iy1, ix1), w11)

        np.clip(img, 0.0, 1.0, out=img)
        return img


def _parabolic_offset(f_m1: float, f_0: float, f_p1: float) -> float:
    denom = f_m1 - 2.0 * f_0 + f_p1
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (f_m1 - f_p1) / denom


def phase_correlation(img_moving: np.ndarray, img_fixed: np.ndarray) -> Tuple[float, float, float]:
    """Return the shift (du, dv, response) that aligns ``img_moving`` to ``img_fixed``."""

    with Timer("orientation.phase"):
        if img_moving.shape != img_fixed.shape:
            raise ValueError("images must share the same shape")
        if img_moving.ndim != 2:
            raise ValueError("images must be 2-D arrays")

        fa = np.fft.fft2(img_moving)
        fb = np.fft.fft2(img_fixed)
        product = fb * np.conj(fa)
        magnitude = np.abs(product)
        magnitude[magnitude == 0.0] = 1.0
        cross_power = product / magnitude
        corr = np.fft.ifft2(cross_power)
        corr_mag = np.abs(corr)
        peak_index = np.unravel_index(np.argmax(corr_mag), corr_mag.shape)
        peak_value = float(corr_mag[peak_index])

        height, width = img_moving.shape
        py, px = peak_index

        px_m1 = (px - 1) % width
        px_p1 = (px + 1) % width
        py_m1 = (py - 1) % height
        py_p1 = (py + 1) % height

        offset_x = _parabolic_offset(
            float(corr_mag[py, px_m1]),
            float(corr_mag[py, px]),
            float(corr_mag[py, px_p1]),
        )
        offset_y = _parabolic_offset(
            float(corr_mag[py_m1, px]),
            float(corr_mag[py, px]),
            float(corr_mag[py_p1, px]),
        )

        shift_x = float(px + offset_x)
        shift_y = float(py + offset_y)

        if shift_x > width / 2.0:
            shift_x -= width
        if shift_y > height / 2.0:
            shift_y -= height

        return shift_x, shift_y, peak_value


def _fourier_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if img.size == 0:
        return img
    height, width = img.shape
    ky = np.fft.fftfreq(height)
    kx = np.fft.fftfreq(width)
    phase = np.exp(-2j * np.pi * (ky[:, None] * dy + kx[None, :] * dx))
    shifted = np.fft.ifft2(np.fft.fft2(img) * phase)
    return shifted.real.astype(np.float32, copy=False)


def _normalized_overlap(img_a: np.ndarray, img_b: np.ndarray) -> float:
    num = float(np.sum(img_a * img_b))
    denom = math.sqrt(float(np.sum(img_a**2) * np.sum(img_b**2))) + 1e-9
    return num / denom


def pick_flip_and_rot(
    pdf_segments: Sequence[Segment],
    svg_segments: Sequence[Segment],
    page_size_pdf: Tuple[float, float],
    page_size_svg: Tuple[float, float],
) -> Tuple[Tuple[float, float], int, float, float]:
    """Evaluate hypotheses and return the best (flip, rot, tx, ty)."""

    if not pdf_segments:
        raise ValueError("PDF segments are required")
    if not svg_segments:
        raise ValueError("SVG segments are required")

    with Timer("orientation.total"):
        pdf_norm = _augment_with_unit_frame(
            _segments_to_array(list(_normalize_segments(pdf_segments, page_size_pdf)))
        )
        svg_norm = _augment_with_unit_frame(
            _segments_to_array(list(_normalize_segments(svg_segments, page_size_svg)))
        )

        raster_size = DEFAULT_RASTER_SIZE
        pdf_image = rasterize_segments(pdf_norm, raster_size, (1.0, 1.0), 0)

        best_score = -math.inf
        best_flip = (1.0, 1.0)
        best_rot = 0
        best_shift = (0.0, 0.0)
        best_response = 0.0

        for rot_deg in _ROTATIONS:
            for flip in _FLIPS:
                svg_image = rasterize_segments(svg_norm, raster_size, flip, rot_deg)
                if DEFAULT_USE_PHASE_CORRELATION:
                    du, dv, response = phase_correlation(svg_image, pdf_image)
                    shifted_svg = _fourier_shift(svg_image, du, dv)
                    score = _normalized_overlap(pdf_image, shifted_svg) * response
                else:
                    du = dv = 0.0
                    response = _normalized_overlap(pdf_image, svg_image)
                    score = response

                if score > best_score:
                    best_score = score
                    best_flip = flip
                    best_rot = rot_deg
                    best_shift = (du, dv)
                    best_response = response

        if DEFAULT_USE_PHASE_CORRELATION:
            svg_best = rasterize_segments(svg_norm, raster_size, best_flip, best_rot)
            du, dv, best_response = phase_correlation(svg_best, pdf_image)
            best_shift = (du, dv)
        else:
            svg_best = rasterize_segments(svg_norm, raster_size, best_flip, best_rot)
            best_shift = (0.0, 0.0)
            best_response = _normalized_overlap(pdf_image, svg_best)

        width, height = _ensure_size_tuple(raster_size)
        page_w, page_h = page_size_svg
        scale_x = page_w / float(width) if width > 0 else 0.0
        scale_y = page_h / float(height) if height > 0 else 0.0
        dx_doc = best_shift[0] * scale_x
        dy_doc = -best_shift[1] * scale_y
        doc_shift = np.array([dx_doc, dy_doc], dtype=np.float64)

        angle_rad = math.radians(best_rot % 360)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        flip_matrix = np.array(
            [[best_flip[0], 0.0], [0.0, best_flip[1]]], dtype=np.float64
        )
        M = flip_matrix @ rot_matrix
        t_seed = M.T @ doc_shift
        tx0 = float(t_seed[0])
        ty0 = float(t_seed[1])

        log.debug(
            "[orient] best rot=%d flip=%s score=%.4f response=%.4f shift=(%.2f, %.2f)px",
            best_rot,
            best_flip,
            best_score,
            best_response,
            best_shift[0],
            best_shift[1],
        )
        log.info(
            "[orient] phase du=%.2fpx dv=%.2fpx -> doc=(%.2f, %.2f) units -> t_seed=(%.2f, %.2f)",
            best_shift[0],
            best_shift[1],
            dx_doc,
            dy_doc,
            tx0,
            ty0,
        )

        return best_flip, best_rot, tx0, ty0
