"""Rendering helpers that preserve absolute scale relationships for PDF/SVG inputs."""

from __future__ import annotations

import io
import math
import re
from typing import Tuple
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
import pypdfium2
from PIL import Image


_VIEWBOX_RE = re.compile(r"viewBox\\s*=\\s*\"([^\"]+)\"|viewBox\\s*=\\s*'([^']+)'", re.IGNORECASE)


def render_pdf_page_gray(pdf_bytes: bytes, dpi: int) -> np.ndarray:
    """Render the first PDF page into an 8-bit grayscale raster at the given DPI."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")

    pdf = pypdfium2.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        page = pdf[0]
        try:
            scale = dpi / 72.0  # PDF user space is 1/72 inch → px = pts * scale
            pil = page.render(scale=scale).to_pil().convert("L")
        finally:
            page.close()
    finally:
        pdf.close()

    return np.array(pil, dtype=np.uint8)


def get_svg_viewbox(svg_bytes: bytes) -> Tuple[float, float, float, float]:
    """Extract the SVG viewBox (minx, miny, width, height)."""

    def _parse(viewbox_text: str) -> Tuple[float, float, float, float]:
        parts = re.split(r"[,\s]+", viewbox_text.strip())
        if len(parts) != 4:
            raise ValueError("viewBox must have four components")
        values = [float(part) for part in parts]
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError("viewBox dimensions must be positive")
        return values[0], values[1], values[2], values[3]

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        text = svg_bytes.decode("utf-8", errors="ignore")
        match = _VIEWBOX_RE.search(text)
        if not match:
            raise ValueError("SVG viewBox attribute not found")
        viewbox_raw = match.group(1) or match.group(2)
        return _parse(viewbox_raw)

    viewbox_attr = root.get("viewBox")
    if not viewbox_attr:
        text = svg_bytes.decode("utf-8", errors="ignore")
        match = _VIEWBOX_RE.search(text)
        if not match:
            raise ValueError("SVG viewBox attribute not found")
        viewbox_attr = match.group(1) or match.group(2)

    return _parse(viewbox_attr)


def render_svg_viewbox_gray(svg_bytes: bytes, ppu: float) -> np.ndarray:
    """Render the SVG viewBox into an 8-bit grayscale raster using pixels-per-unit ``ppu``."""

    if ppu <= 0:
        raise ValueError("ppu must be positive")

    vx, vy, vw, vh = get_svg_viewbox(svg_bytes)
    out_w = max(1, int(math.ceil(vw * ppu)))
    out_h = max(1, int(math.ceil(vh * ppu)))
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=out_w, output_height=out_h)
    image = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.array(image, dtype=np.uint8)


def pad_to_same_canvas(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad two rasters so they share the same canvas size without resampling."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("pad_to_same_canvas expects 2D arrays")

    height = max(a.shape[0], b.shape[0])
    width = max(a.shape[1], b.shape[1])
    padded_a = np.zeros((height, width), dtype=np.uint8)
    padded_b = np.zeros((height, width), dtype=np.uint8)
    padded_a[: a.shape[0], : a.shape[1]] = a
    padded_b[: b.shape[0], : b.shape[1]] = b
    return padded_a, padded_b


# ---------------------------------------------------------------------------
# Compatibility helpers (legacy API)


def render_pdf_page_to_bitmap(pdf_bytes: bytes, raster_size: int) -> np.ndarray:
    """Legacy wrapper returning a float32 raster normalized to 0..1."""

    dpi = max(int(raster_size), 1)
    gray = render_pdf_page_gray(pdf_bytes, dpi)
    pil = Image.fromarray(gray)
    if raster_size > 0:
        scale = min(raster_size / gray.shape[0], raster_size / gray.shape[1])
        new_w = max(1, int(round(gray.shape[1] * scale)))
        new_h = max(1, int(round(gray.shape[0] * scale)))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32)
    if arr.size:
        arr /= 255.0
    return arr


def render_svg_to_bitmap(svg_bytes: bytes, raster_size: int) -> np.ndarray:
    """Legacy wrapper returning a float32 SVG raster normalized to 0..1."""

    try:
        _, _, vw, vh = get_svg_viewbox(svg_bytes)
        max_side = max(vw, vh, 1e-6)
    except ValueError:
        max_side = 1.0
    if raster_size > 0:
        ppu = raster_size / max_side
    else:
        ppu = 1.0
    gray = render_svg_viewbox_gray(svg_bytes, max(ppu, 1e-3))
    pil = Image.fromarray(gray)
    if raster_size > 0:
        scale = min(raster_size / gray.shape[0], raster_size / gray.shape[1])
        new_w = max(1, int(round(gray.shape[1] * scale)))
        new_h = max(1, int(round(gray.shape[0] * scale)))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32)
    if arr.size:
        arr /= 255.0
    return arr
