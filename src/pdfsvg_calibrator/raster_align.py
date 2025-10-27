from __future__ import annotations

"""Raster-based coarse alignment helpers built on existing orientation logic."""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .orientation import phase_correlation
from .rendering import render_pdf_page_to_bitmap, render_svg_to_bitmap


@dataclass
class RenderedPage:
    """Container holding a rendered PDF page raster and its dimensions."""

    image: np.ndarray
    width_pt: float
    height_pt: float


def _get_pdf_page_dimensions(pdf_page_bytes: bytes, page_index: int = 0) -> Tuple[float, float]:
    """Return width/height of a PDF page in points."""

    doc = pdfium.PdfDocument(memoryview(pdf_page_bytes))
    try:
        page = doc.get_page(page_index)
        try:
            width_pt = float(page.get_width())
            height_pt = float(page.get_height())
        finally:
            page.close()
    finally:
        doc.close()
    return width_pt, height_pt


def _parse_length(value: str | None) -> float | None:
    """Parse an SVG length attribute (ignoring units if present)."""

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


def _parse_svg_metadata(svg_bytes: bytes) -> Dict[str, float]:
    """Extract basic size metadata from an SVG snippet."""

    meta: Dict[str, float] = {}
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        return meta

    view_box = root.get("viewBox")
    if view_box:
        parts = re.split(r"[,\s]+", view_box.strip())
        if len(parts) == 4:
            try:
                meta["viewbox_min_x"] = float(parts[0])
                meta["viewbox_min_y"] = float(parts[1])
                meta["viewbox_width"] = float(parts[2])
                meta["viewbox_height"] = float(parts[3])
            except ValueError:
                pass

    width = _parse_length(root.get("width"))
    height = _parse_length(root.get("height"))
    if width is not None:
        meta.setdefault("width", width)
    if height is not None:
        meta.setdefault("height", height)

    if "viewbox_width" in meta and "width" not in meta:
        meta["width"] = meta["viewbox_width"]
    if "viewbox_height" in meta and "height" not in meta:
        meta["height"] = meta["viewbox_height"]

    return meta


def _resize_like(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if image.shape == target_shape:
        return image.astype(np.float32, copy=False)
    target_w = int(target_shape[1])
    target_h = int(target_shape[0])
    with Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)) as pil_img:
        resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
    if arr.max(initial=0.0) > 0:
        arr /= 255.0
    return arr


def _transform_candidate(
    base: np.ndarray,
    rotation_deg: float,
    flip_horizontal: bool,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    arr = base
    if flip_horizontal:
        arr = np.fliplr(arr)

    rot_norm = rotation_deg % 360.0
    rot_rounded = round(rot_norm / 90.0) * 90.0
    if abs(rot_norm - rot_rounded) <= 1e-3:
        k = int((rot_rounded % 360) / 90) % 4
        if k:
            arr = np.rot90(arr, k=k)
        arr = arr.astype(np.float32, copy=False)
    else:
        with Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)) as pil_img:
            rotated = pil_img.rotate(-rotation_deg, resample=Image.BILINEAR, expand=False)
            arr = np.asarray(rotated, dtype=np.float32)
            if arr.max(initial=0.0) > 0:
                arr /= 255.0

    return _resize_like(arr, target_shape)


def _candidate_score(svg_raster: np.ndarray, pdf_raster: np.ndarray) -> Tuple[float, float, float]:
    if svg_raster.size == 0 or pdf_raster.size == 0:
        return 0.0, 0.0, 0.0
    du, dv, response = phase_correlation(svg_raster, pdf_raster)
    if not math.isfinite(response):
        response = 0.0
    return float(du), float(dv), float(response)


def coarse_raster_align(
    pdf_page_bytes: bytes,
    svg_bytes: bytes,
    config: Mapping[str, object],
) -> Dict[str, object]:
    """
    Execute the coarse raster orientation pipeline using rendered PDF/SVG rasters:
      - renders PDF page 0 via :mod:`pypdfium2` into a low-resolution bitmap,
      - renders the SVG into the same raster canvas via :mod:`cairosvg`,
      - tests rotation and flip hypotheses from ``config['rot_degrees']`` and orientation cfg,
      - uses FFT-based phase correlation to estimate tx/ty seeds,
      - derives scale seeds from PDF page size vs. SVG viewBox dimensions.
    """

    orientation_cfg: Mapping[str, object]
    orientation_raw = config.get("orientation") if isinstance(config, Mapping) else None
    if isinstance(orientation_raw, Mapping):
        orientation_cfg = orientation_raw
    else:
        orientation_cfg = {}

    raster_size = int(orientation_cfg.get("raster_size", 512))
    pdf_image = render_pdf_page_to_bitmap(pdf_page_bytes, raster_size=raster_size)
    pdf_width_pt, pdf_height_pt = _get_pdf_page_dimensions(pdf_page_bytes)
    rendered_pdf = RenderedPage(image=pdf_image, width_pt=pdf_width_pt, height_pt=pdf_height_pt)
    pdf_raster = rendered_pdf.image
    pdf_shape = pdf_raster.shape

    svg_raster_base = render_svg_to_bitmap(svg_bytes, raster_size=raster_size)
    svg_meta = _parse_svg_metadata(svg_bytes)

    pdf_w_pt = rendered_pdf.width_pt or 1.0
    pdf_h_pt = rendered_pdf.height_pt or 1.0
    svg_w = svg_meta.get("viewbox_width") or svg_meta.get("width") or pdf_w_pt
    svg_h = svg_meta.get("viewbox_height") or svg_meta.get("height") or pdf_h_pt
    sx_seed = float(pdf_w_pt / svg_w) if svg_w else 1.0
    sy_seed = float(pdf_h_pt / svg_h) if svg_h else 1.0

    rot_candidates_raw = config.get("rot_degrees") if isinstance(config, Mapping) else None
    if not rot_candidates_raw:
        rot_candidates = [0.0, 180.0]
    else:
        rot_candidates = [float(r) for r in rot_candidates_raw]  # type: ignore[not-an-iterable]

    flip_candidates_raw = orientation_cfg.get("flip_horizontal_candidates")
    if isinstance(flip_candidates_raw, Sequence):
        flip_candidates = [bool(x) for x in flip_candidates_raw]
    else:
        flip_candidates = [False, True]

    target_shape = pdf_raster.shape
    candidates: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None

    for rot in rot_candidates:
        for flip in flip_candidates:
            svg_candidate = _transform_candidate(svg_raster_base, rot, flip, target_shape)
            tx_px, ty_px, score = _candidate_score(svg_candidate, pdf_raster)
            print(
                f"[coarse_raster_align] rot={rot:.1f} flip={'H' if flip else 'none'} tx={tx_px:.2f} ty={ty_px:.2f} score={score:.6f}"
            )
            candidate_info = {
                "rotation_deg": float(rot),
                "flip_horizontal": bool(flip),
                "tx_px": float(tx_px),
                "ty_px": float(ty_px),
                "score": float(score),
            }
            candidates.append(candidate_info)
            if best is None or candidate_info["score"] > best["score"]:  # type: ignore[index]
                best = candidate_info

    if best is None:
        best = {
            "rotation_deg": 0.0,
            "flip_horizontal": False,
            "tx_px": 0.0,
            "ty_px": 0.0,
            "score": 0.0,
        }

    debug_info = {
        "pdf_shape": (int(pdf_shape[0]), int(pdf_shape[1])),
        "svg_shape": (int(svg_raster_base.shape[0]), int(svg_raster_base.shape[1])),
        "pdf_size_pt": (float(pdf_w_pt), float(pdf_h_pt)),
        "svg_meta": svg_meta,
        "raster_size": raster_size,
        "candidates": candidates,
    }

    result = {
        "rotation_deg": float(best["rotation_deg"]),
        "flip_horizontal": bool(best["flip_horizontal"]),
        "tx_px": float(best["tx_px"]),
        "ty_px": float(best["ty_px"]),
        "sx_seed": float(sx_seed),
        "sy_seed": float(sy_seed),
        "score": float(best["score"]),
        "debug": debug_info,
    }
    return result
