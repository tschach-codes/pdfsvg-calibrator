from __future__ import annotations

"""Raster-based alignment fallback between PDF pages and SVG renderings."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import io

import numpy as np
import fitz  # PyMuPDF
import cairosvg
from PIL import Image


@dataclass
class RasterAlignmentResult:
    """Result of raster-based alignment estimation."""

    rotation_deg: float
    flip_y: bool
    scale: float
    shift_xy: Tuple[float, float]
    score: float
    confidence: str


def _render_pdf_gray(pdf_path: str, page_index: int, max_dim: int = 2048) -> np.ndarray:
    """Render PDF page to grayscale array with side length limited to ``max_dim``."""

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        rect = page.rect
        longer_pt = max(rect.width, rect.height, 1e-9)
        zoom = max_dim / longer_pt
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return image.astype(np.float32) / 255.0
    finally:
        doc.close()


def _render_svg_gray(
    svg_path: str,
    rotation_deg: float,
    flip_y: bool,
    scale: float,
    target_wh: Tuple[int, int],
) -> np.ndarray:
    """Rasterise the SVG and apply transformations to match the PDF raster."""

    target_w, target_h = target_wh
    out_w = max(int(scale * target_w), 1)
    out_h = max(int(scale * target_h), 1)

    png_bytes = cairosvg.svg2png(
        url=svg_path,
        output_width=out_w,
        output_height=out_h,
        background_color="white",
    )
    image = Image.open(io.BytesIO(png_bytes)).convert("L")
    array = np.array(image, dtype=np.float32) / 255.0

    if flip_y:
        array = np.flipud(array)

    rotation = int(rotation_deg) % 360
    if rotation == 90:
        array = np.rot90(array, k=1)
    elif rotation == 180:
        array = np.rot90(array, k=2)
    elif rotation == 270:
        array = np.rot90(array, k=3)

    pil_image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    pil_image = pil_image.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_image, dtype=np.float32) / 255.0


def _phase_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Compute phase correlation shift and correlation score between two images."""

    a_fft = np.fft.fft2(a - a.mean())
    b_fft = np.fft.fft2(b - b.mean())
    cross_power = a_fft * np.conj(b_fft)
    cross_power /= np.abs(cross_power) + 1e-9
    response = np.fft.ifft2(cross_power)
    response = np.abs(response)

    max_pos = np.unravel_index(np.argmax(response), response.shape)
    peak_value = response[max_pos]

    height, width = a.shape
    peak_y, peak_x = max_pos
    if peak_y > height // 2:
        peak_y -= height
    if peak_x > width // 2:
        peak_x -= width

    return (float(peak_y), float(peak_x)), float(peak_value)


def estimate_raster_alignment(
    pdf_path: str,
    page_index: int,
    svg_path: str,
    scales: Optional[List[float]] = None,
    rotations: Optional[List[float]] = None,
    flips_y: Optional[List[bool]] = None,
    max_dim: int = 2048,
) -> Optional[RasterAlignmentResult]:
    """Estimate alignment parameters by raster correlation over candidate transforms."""

    if scales is None:
        scales = [0.5, 0.66, 0.75, 1.0, 1.5, 2.0]
    if rotations is None:
        rotations = [0.0, 90.0, 180.0, 270.0]
    if flips_y is None:
        flips_y = [False, True]

    pdf_img = _render_pdf_gray(pdf_path, page_index, max_dim=max_dim)
    height, width = pdf_img.shape
    target_wh = (width, height)

    best_candidate = None

    for rotation_deg in rotations:
        for flip_y in flips_y:
            for scale in scales:
                try:
                    svg_img = _render_svg_gray(
                        svg_path,
                        rotation_deg=rotation_deg,
                        flip_y=flip_y,
                        scale=scale,
                        target_wh=target_wh,
                    )
                except Exception:
                    continue

                (shift_y, shift_x), score = _phase_correlation(pdf_img, svg_img)
                candidate = {
                    "rotation_deg": rotation_deg,
                    "flip_y": flip_y,
                    "scale": scale,
                    "shift_xy": (shift_x, shift_y),
                    "score": score,
                }
                if best_candidate is None or candidate["score"] > best_candidate["score"]:
                    best_candidate = candidate

    if best_candidate is None:
        return None

    return RasterAlignmentResult(
        rotation_deg=float(best_candidate["rotation_deg"]),
        flip_y=bool(best_candidate["flip_y"]),
        scale=float(best_candidate["scale"]),
        shift_xy=(float(best_candidate["shift_xy"][0]), float(best_candidate["shift_xy"][1])),
        score=float(best_candidate["score"]),
        confidence="medium",
    )
