from __future__ import annotations

"""Raster-based coarse alignment helpers built on existing orientation logic."""

import io
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .orientation import _apply_flip_rotate_raster, phase_correlation
from .rendering import (
    get_svg_viewbox,
    render_pdf_page_to_bitmap,
    render_svg_to_bitmap,
    render_svg_viewbox_gray,
)


@dataclass
class RenderedPage:
    """Container holding a rendered PDF page raster and its dimensions."""

    image: np.ndarray
    width_pt: float
    height_pt: float


logger = logging.getLogger(__name__)


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


def _to_float_image(arr: np.ndarray) -> np.ndarray:
    image = np.asarray(arr)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.size and image.max(initial=0.0) > 1.0:
        image /= 255.0
    return image


def _crop_margin(image: np.ndarray, margin_pct: float) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("_crop_margin expects 2D arrays")
    if margin_pct <= 0:
        return image
    height, width = image.shape
    if height <= 2 or width <= 2:
        return image
    margin_y = int(round(height * margin_pct / 100.0))
    margin_x = int(round(width * margin_pct / 100.0))
    margin_y = min(margin_y, height // 2)
    margin_x = min(margin_x, width // 2)
    if margin_y == 0 and margin_x == 0:
        return image
    y0 = margin_y
    y1 = height - margin_y
    x0 = margin_x
    x1 = width - margin_x
    if y1 <= y0 or x1 <= x0:
        return image
    return image[y0:y1, x0:x1]


def _center_crop_float(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("_center_crop_float expects 2D arrays")
    h, w = image.shape
    if target_h > h or target_w > w:
        raise ValueError("target shape must be <= source shape")
    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2
    return image[start_y : start_y + target_h, start_x : start_x + target_w]


def _center_fit_uint8(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("_center_fit_uint8 expects 2D arrays")
    target_h, target_w = target_shape
    arr = image
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.shape[0] > target_h:
        start = (arr.shape[0] - target_h) // 2
        arr = arr[start : start + target_h, :]
    elif arr.shape[0] < target_h:
        pad_top = (target_h - arr.shape[0]) // 2
        pad_bottom = target_h - arr.shape[0] - pad_top
        arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    if arr.shape[1] > target_w:
        start = (arr.shape[1] - target_w) // 2
        arr = arr[:, start : start + target_w]
    elif arr.shape[1] < target_w:
        pad_left = (target_w - arr.shape[1]) // 2
        pad_right = target_w - arr.shape[1] - pad_left
        arr = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="constant")

    return arr.astype(np.uint8, copy=False)


def _rescale_canvas_uint8(image: np.ndarray, scale: float, target_shape: Tuple[int, int]) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if image.ndim != 2:
        raise ValueError("_rescale_canvas_uint8 expects 2D arrays")

    height, width = image.shape
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    if new_h == height and new_w == width:
        resized = image.astype(np.uint8, copy=False)
    else:
        pil_img = Image.fromarray(image)
        resized = np.asarray(pil_img.resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)

    return _center_fit_uint8(resized, target_shape)


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_mean = float(a.mean())
    b_mean = float(b.mean())
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 1e-9 or b_std <= 1e-9:
        return 0.0
    return float(((a - a_mean) * (b - b_mean)).mean() / (a_std * b_std))


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


def estimate_raster_alignment(
    svg_path: str,
    pdf_path: str,
    page: int = 0,
    dpi: int = 150,
    search_range: Tuple[float, float] = (0.4, 3.0),
    coarse_step: float = 0.01,
    fine_window: float = 0.02,
    fine_step: float = 0.001,
    ncc_margin_crop_pct: float = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """Estimate a global raster scale between an SVG and PDF page."""

    result: Dict[str, Any] = {
        "scale": None,
        "ncc_score": None,
        "coarse_best": None,
        "coarse_score": None,
        "fine_best": None,
        "fine_score": None,
        "search_range": [float(search_range[0]), float(search_range[1])],
        "coarse_step": float(coarse_step),
        "fine_window": float(fine_window),
        "fine_step": float(fine_step),
        "margin_crop_pct": float(ncc_margin_crop_pct),
        "dpi": int(dpi),
        "page": int(page),
    }

    try:
        with open(pdf_path, "rb") as f_pdf:
            pdf_bytes = f_pdf.read()
    except OSError as exc:
        result["error"] = f"pdf read failed: {exc}"
        return False, result

    try:
        with open(svg_path, "rb") as f_svg:
            svg_bytes = f_svg.read()
    except OSError as exc:
        result["error"] = f"svg read failed: {exc}"
        return False, result

    if dpi <= 0:
        result["error"] = "dpi must be positive"
        return False, result
    if coarse_step <= 0:
        coarse_step = 0.01
    if fine_step <= 0:
        fine_step = 0.001
    if fine_window < 0:
        fine_window = 0.0

    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = pdfium.PdfDocument(pdf_stream)
        try:
            if page < 0 or page >= len(doc):
                result["error"] = f"page index {page} out of range"
                return False, result
            pdf_page = doc[page]
            try:
                scale = dpi / 72.0
                pil_pdf = pdf_page.render(scale=scale).to_pil().convert("L")
            finally:
                pdf_page.close()
        finally:
            doc.close()
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"pdf render failed: {exc}"
        return False, result

    pdf_gray = np.asarray(pil_pdf, dtype=np.uint8)
    if pdf_gray.ndim != 2:
        result["error"] = "pdf raster is not 2D"
        return False, result

    try:
        _, _, vb_w, vb_h = get_svg_viewbox(svg_bytes)
        max_side = max(float(vb_w), float(vb_h), 1e-6)
    except Exception:
        max_side = None

    if max_side is not None and max_side > 0:
        ppu_seed = max(
            pdf_gray.shape[1] / max(float(vb_w), 1e-6),
            pdf_gray.shape[0] / max(float(vb_h), 1e-6),
        )
    else:
        ppu_seed = dpi / 72.0
    ppu_seed = max(ppu_seed, 1e-3)

    try:
        svg_gray = render_svg_viewbox_gray(svg_bytes, ppu_seed)
    except Exception as exc:  # pragma: no cover - render failure
        result["error"] = f"svg render failed: {exc}"
        return False, result

    if svg_gray.ndim != 2:
        result["error"] = "svg raster is not 2D"
        return False, result

    search_low, search_high = search_range
    try:
        search_low = float(search_low)
        search_high = float(search_high)
    except Exception:
        search_low, search_high = (0.4, 3.0)
    if search_low <= 0 or not math.isfinite(search_low):
        search_low = 0.4
    if search_high <= 0 or not math.isfinite(search_high):
        search_high = max(search_low + 0.01, 3.0)
    if search_low >= search_high:
        search_low, search_high = (min(search_low, search_high), max(search_low, search_high) + 0.01)

    target_shape = (int(pdf_gray.shape[0]), int(pdf_gray.shape[1]))
    pdf_float = _to_float_image(pdf_gray)
    pdf_crop = _crop_margin(pdf_float, float(ncc_margin_crop_pct))

    best_scale: Optional[float] = None
    best_score = float("-inf")
    coarse_best: Optional[float] = None
    coarse_best_score = float("-inf")
    fine_best: Optional[float] = None
    fine_best_score = float("-inf")

    def score_for(scale_value: float) -> float:
        try:
            scaled_uint8 = _rescale_canvas_uint8(svg_gray, scale_value, target_shape)
        except ValueError:
            return float("nan")
        scaled_float = _to_float_image(scaled_uint8)
        scaled_crop = _crop_margin(scaled_float, float(ncc_margin_crop_pct))
        pdf_aligned = pdf_crop
        if scaled_crop.shape != pdf_crop.shape:
            target_h = min(scaled_crop.shape[0], pdf_crop.shape[0])
            target_w = min(scaled_crop.shape[1], pdf_crop.shape[1])
            if target_h <= 0 or target_w <= 0:
                return float("nan")
            scaled_crop = _center_crop_float(scaled_crop, target_h, target_w)
            pdf_aligned = _center_crop_float(pdf_crop, target_h, target_w)
        return _normalized_cross_correlation(scaled_crop, pdf_aligned)

    scale_value = search_low
    while scale_value <= search_high + 1e-12:
        score = score_for(scale_value)
        if math.isfinite(score) and score > coarse_best_score:
            coarse_best_score = score
            coarse_best = scale_value
        scale_value += coarse_step

    if coarse_best is not None:
        best_scale = coarse_best
        best_score = coarse_best_score

        fine_start = max(search_low, coarse_best - fine_window)
        fine_end = min(search_high, coarse_best + fine_window)

        scale_value = fine_start
        while scale_value <= fine_end + 1e-12:
            score = score_for(scale_value)
            if math.isfinite(score) and score > fine_best_score:
                fine_best_score = score
                fine_best = scale_value
            scale_value += fine_step

        if fine_best is not None and fine_best_score >= best_score:
            best_scale = fine_best
            best_score = fine_best_score

    result.update(
        {
            "coarse_best": coarse_best,
            "coarse_score": coarse_best_score if math.isfinite(coarse_best_score) else None,
            "fine_best": fine_best,
            "fine_score": fine_best_score if math.isfinite(fine_best_score) else None,
            "scale": best_scale,
            "ncc_score": best_score if math.isfinite(best_score) else None,
            "pdf_shape": (int(pdf_gray.shape[0]), int(pdf_gray.shape[1])),
            "svg_shape": (int(svg_gray.shape[0]), int(svg_gray.shape[1])),
            "svg_ppu_seed": float(ppu_seed),
        }
    )

    ok_flag = best_scale is not None and math.isfinite(best_score)
    return ok_flag, result


def orient_svg_bitmap_no_translation(
    svg_bitmap: np.ndarray, coarse: Mapping[str, object]
) -> np.ndarray:
    """Rotate/flip an SVG raster according to coarse orientation without tx/ty."""

    rotation = int(round(float(coarse.get("rotation_deg", 0))))
    flip = bool(coarse.get("flip_horizontal", False))
    return _apply_flip_rotate_raster(svg_bitmap, rotation, flip)


def save_split_pdf_svg(
    pdf_img: np.ndarray,
    svg_img_oriented: np.ndarray,
    out_path: str,
    pdf_shape: tuple[int, int],
    svg_shape: tuple[int, int],
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Side-by-side debug: left=PDF (red), right=SVG (green).
    - Do NOT use tx/ty here.
    - Canvas: H=max(h_pdf,h_svg), W=w_pdf+w_svg
    - Place PDF at (0,0), SVG at (w_pdf,0)
    - Draw a vertical black separator at x=w_pdf
    - Annotate small white text (top-left) with sizes and scales from meta.
    """

    import cv2
    import numpy as np

    h_pdf, w_pdf = pdf_shape
    h_svg, w_svg = svg_shape
    H = max(h_pdf, h_svg)
    W = w_pdf + w_svg
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Ensure inputs match their own panes (pad if needed)
    def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
        hh, ww = img.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:hh, :ww] = img[:h, :w]
        return out

    pdf_pad = pad_to(pdf_img, h_pdf, w_pdf)
    svg_pad = pad_to(svg_img_oriented, h_svg, w_svg)

    # Colorize: PDF->red, SVG->green
    pdf_red = np.zeros_like(pdf_pad)
    pdf_red[..., 2] = cv2.cvtColor(pdf_pad, cv2.COLOR_BGR2GRAY)
    svg_green = np.zeros_like(svg_pad)
    svg_green[..., 1] = cv2.cvtColor(svg_pad, cv2.COLOR_BGR2GRAY)

    canvas[0:h_pdf, 0:w_pdf] = pdf_red
    canvas[0:h_svg, w_pdf : w_pdf + w_svg] = svg_green

    # Separator
    cv2.line(canvas, (w_pdf, 0), (w_pdf, H - 1), (0, 0, 0), 2)

    # Small label
    if meta:
        label = f"pdf {w_pdf}x{h_pdf} | svg {w_svg}x{h_svg}"
        if "scale_hint" in meta:
            label += f" | hint {meta['scale_hint']:.6f}"
        if "Sx" in meta and "Sy" in meta:
            label += f" | Sx {meta['Sx']} Sy {meta['Sy']}"
        cv2.putText(
            canvas,
            label,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    logger.info(
        "split_debug: pdf=%s svg=%s placed=(0,0) & (%s,0) use_tx_ty=False",
        pdf_shape,
        svg_shape,
        w_pdf,
    )

    cv2.imwrite(out_path, canvas)
