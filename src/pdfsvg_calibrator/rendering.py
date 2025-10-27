"""Utility helpers for rendering PDF and SVG sources into normalized rasters."""

from __future__ import annotations

import io
import re
from typing import Tuple
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
import pypdfium2 as pdfium
from PIL import Image


def _parse_length(value: str | None) -> float | None:
    """Parse a numeric SVG length ignoring potential units."""

    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    match = re.match(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _infer_svg_dimensions(svg_bytes: bytes) -> Tuple[float | None, float | None]:
    """Infer intrinsic SVG dimensions (in arbitrary units)."""

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        return None, None

    width = _parse_length(root.get("width"))
    height = _parse_length(root.get("height"))

    view_box = root.get("viewBox")
    if view_box:
        parts = re.split(r"[,\s]+", view_box.strip())
        if len(parts) == 4:
            try:
                view_w = float(parts[2])
                view_h = float(parts[3])
            except ValueError:
                view_w = view_h = None
            else:
                if width is None:
                    width = view_w
                if height is None:
                    height = view_h

    return width, height


def render_pdf_page_to_bitmap(pdf_bytes: bytes, raster_size: int) -> np.ndarray:
    """Render PDF page 0 to a grayscale float raster limited by ``raster_size``."""

    if raster_size <= 0:
        raster_size = 512

    doc = pdfium.PdfDocument(memoryview(pdf_bytes))
    try:
        page = doc.get_page(0)
        try:
            width_pt = float(page.get_width())
            height_pt = float(page.get_height())
            max_side = max(width_pt, height_pt, 1e-6)
            scale = raster_size / max_side if raster_size else 1.0
            scale = max(scale, 1e-3)
            bitmap = page.render(scale=scale, rotation=0, grayscale=True)
            try:
                array = bitmap.to_numpy()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        doc.close()

    if array.ndim == 3:
        array = array[..., 0]
    image = np.asarray(array, dtype=np.float32)
    if image.max(initial=0.0) > 0:
        image /= 255.0
    return image


def render_svg_to_bitmap(svg_bytes: bytes, raster_size: int) -> np.ndarray:
    """Render an SVG snippet to a grayscale float raster limited by ``raster_size``."""

    if raster_size <= 0:
        raster_size = 512

    width, height = _infer_svg_dimensions(svg_bytes)
    if width and height:
        max_side = max(width, height, 1e-6)
        scale = raster_size / max_side if raster_size else 1.0
        width_px = max(int(round(width * scale)), 1)
        height_px = max(int(round(height * scale)), 1)
    else:
        width_px = height_px = max(int(raster_size), 1)

    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=width_px,
        output_height=height_px,
        background_color="white",
    )

    with Image.open(io.BytesIO(png_bytes)) as image:
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background.convert("L")
        else:
            image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32)

    if arr.max(initial=0.0) > 0:
        arr /= 255.0
    return arr
