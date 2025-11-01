from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pypdfium2
from PIL import Image

from . import debug_utils as dbg
from .dimension_scale_integration import refine_metric_scale
from .orientation import apply_orientation_to_raster, coarse_orientation_from_rasters
from .rendering import get_svg_viewbox, render_pdf_page_gray, render_svg_viewbox_gray
from .timing import Timings


logger = logging.getLogger(__name__)


def _clone_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    cloned: Dict[str, Any] = dict(config)
    for key, value in list(cloned.items()):
        if isinstance(value, MutableMapping):
            cloned[key] = dict(value)
    return cloned


def _load_pdf_page_size_pts(pdf_bytes: bytes) -> Tuple[float, float]:
    pdf = pypdfium2.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        page = pdf[0]
        try:
            width = float(page.get_width())
            height = float(page.get_height())
        finally:
            page.close()
    finally:
        pdf.close()
    return width, height


def _percent_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    if a == 0:
        return None
    return abs((b - a) / a) * 100.0


def _extract_global_raster_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "dpi": 150,
        "search_range": (0.4, 3.0),
        "coarse_step": 0.01,
        "fine_window": 0.02,
        "fine_step": 0.001,
        "ncc_margin_crop_pct": 3.0,
    }

    if not isinstance(config, Mapping):
        return dict(defaults)

    section = config.get("global_raster")
    sources: Sequence[Mapping[str, Any]] = ()
    if isinstance(section, Mapping):
        sources = (section, config)
    else:
        sources = (config,)

    result = dict(defaults)

    for src in sources:
        value = src.get("enabled")
        if value is None:
            continue
        if isinstance(value, bool):
            result["enabled"] = value
            break
        if isinstance(value, (int, float)):
            result["enabled"] = bool(value)
            break

    # ``dpi`` is intentionally forced to the project default (150 DPI) for the
    # global raster validation.  Even if configuration files specify a
    # different value we normalise to the mandated setting so that the
    # correlation always runs on a comparable raster resolution.
    result["dpi"] = 150

    for src in sources:
        value = src.get("search_range")
        if value is None:
            continue
        if isinstance(value, Sequence) and len(value) >= 2:
            try:
                low = float(value[0])
                high = float(value[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(low) and math.isfinite(high)):
                continue
            if low == high:
                continue
            if low > high:
                low, high = high, low
            result["search_range"] = (low, high)
            break

    def _pick_float(key: str, minimum: float | None = None) -> None:
        for src in sources:
            value = src.get(key)
            if value is None:
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(candidate):
                continue
            if minimum is not None and candidate <= minimum:
                continue
            result[key] = candidate
            return

    _pick_float("coarse_step", minimum=0.0)
    _pick_float("fine_window", minimum=0.0)
    _pick_float("fine_step", minimum=0.0)
    _pick_float("ncc_margin_crop_pct", minimum=-0.1)

    return result


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

    # adjust height
    if arr.shape[0] > target_h:
        start = (arr.shape[0] - target_h) // 2
        arr = arr[start : start + target_h, :]
    elif arr.shape[0] < target_h:
        pad_top = (target_h - arr.shape[0]) // 2
        pad_bottom = target_h - arr.shape[0] - pad_top
        arr = np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    # adjust width
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


def _estimate_global_raster_scale(
    pdf_gray: np.ndarray,
    svg_oriented: np.ndarray,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if pdf_gray.ndim != 2 or svg_oriented.ndim != 2:
        raise ValueError("_estimate_global_raster_scale expects 2D rasters")

    search_low, search_high = config.get("search_range", (0.4, 3.0))
    coarse_step = float(config.get("coarse_step", 0.01)) or 0.01
    fine_window = float(config.get("fine_window", 0.02))
    fine_step = float(config.get("fine_step", 0.001)) or 0.001
    margin_pct = float(config.get("ncc_margin_crop_pct", 0.0))

    if coarse_step <= 0:
        coarse_step = 0.01
    if fine_step <= 0:
        fine_step = 0.001
    if search_low >= search_high:
        search_low, search_high = (min(search_low, search_high), max(search_low, search_high) + 0.01)

    target_shape = (int(pdf_gray.shape[0]), int(pdf_gray.shape[1]))
    pdf_float = _to_float_image(pdf_gray)
    pdf_crop = _crop_margin(pdf_float, margin_pct)

    best_scale = None
    best_score = float("-inf")
    coarse_best = None
    coarse_best_score = float("-inf")
    fine_best = None
    fine_best_score = float("-inf")

    def score_for(scale_value: float) -> float:
        try:
            scaled_uint8 = _rescale_canvas_uint8(svg_oriented, scale_value, target_shape)
        except ValueError:
            return float("nan")
        scaled_float = _to_float_image(scaled_uint8)
        scaled_crop = _crop_margin(scaled_float, margin_pct)
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

    return {
        "coarse_best": coarse_best,
        "coarse_score": coarse_best_score if math.isfinite(coarse_best_score) else None,
        "fine_best": fine_best,
        "fine_score": fine_best_score if math.isfinite(fine_best_score) else None,
        "best_scale": best_scale,
        "ncc_score": best_score if math.isfinite(best_score) else None,
        "search_range": list(config.get("search_range", (search_low, search_high))),
        "coarse_step": coarse_step,
        "fine_step": fine_step,
        "margin_crop_pct": margin_pct,
    }


def calibrate_pdf_svg_preprocess(
    pdf_bytes: bytes,
    svg_bytes: bytes,
    config: Mapping[str, Any] | None,
    *,
    svg_path: str | os.PathLike[str] | None = None,
    save_debug: bool = False,
    debug_outdir: str | os.PathLike[str] | None = None,
    debug_prefix: str = "dbg",
) -> Dict[str, Any]:
    cfg = _clone_config(config)
    global_raster_cfg = _extract_global_raster_config(cfg)
    inputs_cfg = cfg.get("inputs") if isinstance(cfg, Mapping) else {}
    if not isinstance(inputs_cfg, Mapping):
        inputs_cfg = {}
    cfg["inputs"] = dict(inputs_cfg)
    if svg_path is not None:
        cfg["inputs"]["svg_path"] = str(svg_path)

    debug_cfg = cfg.get("debug") if isinstance(cfg, Mapping) else {}
    if not isinstance(debug_cfg, Mapping):
        debug_cfg = {}
    debug_cfg = dict(debug_cfg)
    if debug_outdir is not None:
        debug_cfg.setdefault("outdir", str(debug_outdir))
    if debug_prefix:
        debug_cfg.setdefault("prefix", debug_prefix)
    cfg["debug"] = debug_cfg
    raster_cfg = cfg.get("raster", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(raster_cfg, Mapping):
        raster_cfg = {}

    dpi_global = int(global_raster_cfg.get("dpi", 150))
    dpi_coarse = dpi_global
    dpi_fallback = dpi_global
    dpi_debug = int(raster_cfg.get("dpi_debug", 300))
    svg_ppu_coarse = float(raster_cfg.get("svg_ppu_coarse", 1.0))
    svg_ppu_fallback = float(raster_cfg.get("svg_ppu_fallback", svg_ppu_coarse))
    svg_ppu_debug = float(raster_cfg.get("svg_ppu_debug", svg_ppu_coarse))

    vx, vy, vw, vh = get_svg_viewbox(svg_bytes)
    pdf_width_pt, pdf_height_pt = _load_pdf_page_size_pts(pdf_bytes)

    tm = Timings()

    def _render_pair(label: str, dpi: int, ppu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        with tm.section(f"render_pdf_{label}"):
            pdf_gray = render_pdf_page_gray(pdf_bytes, dpi)
        with tm.section(f"render_svg_{label}"):
            svg_gray = render_svg_viewbox_gray(svg_bytes, ppu)
        if vw > 0:
            pixels_per_svg_unit = svg_gray.shape[1] / float(vw)
        else:
            pixels_per_svg_unit = ppu
        return pdf_gray, svg_gray, float(pixels_per_svg_unit)

    pdf_gray, svg_gray, pixels_per_svg_unit = _render_pair("coarse", dpi_coarse, svg_ppu_coarse)

    with tm.section("coarse_orientation"):
        coarse = coarse_orientation_from_rasters(
            pdf_gray,
            svg_gray,
            cfg,
            svg_pixels_per_unit=pixels_per_svg_unit,
        )

    used_fallback = False
    coarse_score = coarse.get("score", 0.0)
    if (
        (not isinstance(coarse_score, (int, float)) or not math.isfinite(coarse_score) or coarse_score <= 0.0)
        and (dpi_fallback != dpi_coarse or svg_ppu_fallback != svg_ppu_coarse)
    ):
        used_fallback = True
        pdf_gray, svg_gray, pixels_per_svg_unit = _render_pair("fallback", dpi_fallback, svg_ppu_fallback)
        with tm.section("coarse_orientation_fallback"):
            coarse = coarse_orientation_from_rasters(
                pdf_gray,
                svg_gray,
                cfg,
                svg_pixels_per_unit=pixels_per_svg_unit,
            )

    final_dpi = dpi_fallback if used_fallback else dpi_coarse
    pdf_px_per_point = final_dpi / 72.0
    pdf_px_per_meter = final_dpi / 0.0254

    coarse["pixels_per_svg_unit"] = float(pixels_per_svg_unit)
    coarse["used_fallback"] = used_fallback

    with tm.section("apply_orient_no_scale"):
        svg_oriented = apply_orientation_to_raster(
            svg_gray,
            coarse,
            canvas_shape=(int(pdf_gray.shape[0]), int(pdf_gray.shape[1])),
            apply_scale=False,
        )

    debug_path: Path | str | None = None

    with tm.section("debug_export", include_in_compute=False):
        if save_debug and debug_outdir is not None:
            debug_path = Path(debug_outdir)
            debug_path.mkdir(parents=True, exist_ok=True)
            pdf_dbg = render_pdf_page_gray(pdf_bytes, dpi_debug)
            svg_dbg = render_svg_viewbox_gray(svg_bytes, svg_ppu_debug)
            svg_dbg_oriented = apply_orientation_to_raster(
                svg_dbg,
                coarse,
                canvas_shape=(int(pdf_dbg.shape[0]), int(pdf_dbg.shape[1])),
                apply_scale=False,
            )
            dbg.save_debug_rasters(
                pdf_dbg,
                svg_dbg,
                svg_dbg_oriented,
                str(debug_path),
                prefix=debug_prefix,
            )

    if save_debug:
        # ensure debug_path is str/Path correctly
        assert isinstance(debug_path, (str, Path))

    with tm.section("dimscale_refine"):
        metric = refine_metric_scale(svg_bytes, coarse, cfg)

    global_raster_result: Dict[str, Any] | None = None
    global_raster_ok = False
    scale_hint_raw: float | None = None
    scale_hint_value = coarse.get("scale_hint")
    if isinstance(scale_hint_value, (int, float)) and math.isfinite(scale_hint_value):
        scale_hint_raw = float(scale_hint_value)

    if global_raster_cfg.get("enabled", True):
        with tm.section("global_raster_scale"):
            try:
                global_raster_result = _estimate_global_raster_scale(
                    pdf_gray,
                    svg_oriented,
                    global_raster_cfg,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Global raster scale failed: %s", exc)
                global_raster_result = {"error": str(exc)}
    else:
        global_raster_result = {
            "enabled": False,
            "ok": False,
            "dpi": int(global_raster_cfg.get("dpi", 150)),
        }

    if isinstance(global_raster_result, dict):
        enabled_flag = bool(global_raster_cfg.get("enabled", True))
        global_raster_result.setdefault("ncc_score", None)
        global_raster_result.setdefault("best_scale", None)
        global_raster_result.setdefault("ok", False)
        global_raster_result["enabled"] = enabled_flag
        global_raster_result.setdefault("dpi", int(global_raster_cfg.get("dpi", 150)))
        global_raster_result["search_range"] = list(
            global_raster_cfg.get("search_range", (0.4, 3.0))
        )
        global_raster_result["coarse_step"] = float(global_raster_cfg.get("coarse_step", 0.01))
        global_raster_result["fine_window"] = float(global_raster_cfg.get("fine_window", 0.02))
        global_raster_result["fine_step"] = float(global_raster_cfg.get("fine_step", 0.001))
        global_raster_result["margin_crop_pct"] = float(
            global_raster_cfg.get("ncc_margin_crop_pct", 0.0)
        )

        if enabled_flag:
            best_scale_val = global_raster_result.get("best_scale")
            ncc_val = global_raster_result.get("ncc_score")
            valid_scale = isinstance(best_scale_val, (int, float)) and math.isfinite(best_scale_val)
            valid_scale = valid_scale and float(best_scale_val) > 0.0
            valid_score = isinstance(ncc_val, (int, float)) and math.isfinite(ncc_val)
            global_raster_ok = bool(valid_scale and valid_score)
            global_raster_result["ok"] = global_raster_ok

            if not global_raster_ok:
                coarse["global_raster_ok"] = False
                logger.warning("Global Raster-Check lieferte keine gültige Skalierung")
            else:
                scale_factor = float(best_scale_val)  # type: ignore[arg-type]
                coarse["global_raster_ok"] = True
                coarse["global_raster_scale"] = scale_factor
                if scale_hint_raw is not None:
                    coarse.setdefault("scale_hint_raw", scale_hint_raw)
                    coarse["scale_hint"] = scale_hint_raw * scale_factor
                coarse.setdefault("pixels_per_svg_unit", float(pixels_per_svg_unit))
                coarse["pixels_per_svg_unit_global"] = float(pixels_per_svg_unit) * scale_factor
        else:
            coarse["global_raster_ok"] = False

    if isinstance(metric, dict) and isinstance(global_raster_result, dict):
        debug_section = metric.setdefault("debug", {})
        if isinstance(debug_section, dict):
            debug_section["global_raster"] = dict(global_raster_result)

    with tm.section("verify_scales"):
        scale_hint = (
            float(coarse.get("scale_hint", 0.0))
            if coarse.get("scale_hint") is not None
            else None
        )
        scale_hint_original = coarse.get("scale_hint_raw")
        if isinstance(scale_hint_original, (int, float)) and math.isfinite(scale_hint_original):
            scale_hint_raw_value: float | None = float(scale_hint_original)
        else:
            scale_hint_raw_value = scale_hint
        scale_vbox = pixels_per_svg_unit / pdf_px_per_point if pdf_px_per_point else None
        pixels_per_meter = metric.get("pixels_per_meter") if isinstance(metric, Mapping) else None
        scale_dim = None
        if isinstance(pixels_per_meter, (int, float)) and math.isfinite(pixels_per_meter):
            scale_dim = float(pixels_per_meter) / pdf_px_per_meter if pdf_px_per_meter else None
        diff_hint_vbox = _percent_diff(scale_hint, scale_vbox)
        diff_hint_dim = _percent_diff(scale_hint, scale_dim)
        diff_vbox_dim = _percent_diff(scale_vbox, scale_dim)

        dim_pixels_x: float | None = None
        dim_pixels_y: float | None = None
        dim_inliers_x: int | None = None
        dim_inliers_y: int | None = None
        dim_notices: List[str] = []

        metric_debug = metric.get("debug") if isinstance(metric, Mapping) else {}
        if isinstance(metric_debug, Mapping):
            dimscale_section = metric_debug.get("dimscale")
            if isinstance(dimscale_section, Mapping):
                scale_x_svg = dimscale_section.get("scale_x_svg_per_unit")
                scale_y_svg = dimscale_section.get("scale_y_svg_per_unit")
                if isinstance(scale_x_svg, (int, float)) and math.isfinite(scale_x_svg):
                    dim_pixels_x = float(scale_x_svg) * pixels_per_svg_unit
                if isinstance(scale_y_svg, (int, float)) and math.isfinite(scale_y_svg):
                    dim_pixels_y = float(scale_y_svg) * pixels_per_svg_unit
                dimscale_debug = dimscale_section.get("debug")
                if isinstance(dimscale_debug, Mapping):
                    agg = dimscale_debug.get("scale_aggregation")
                    if isinstance(agg, Mapping):
                        axis_x = agg.get("x")
                        axis_y = agg.get("y")
                        if isinstance(axis_x, Mapping):
                            inlier_val = axis_x.get("inlier_count")
                            if isinstance(inlier_val, (int, float)):
                                dim_inliers_x = int(inlier_val)
                        if isinstance(axis_y, Mapping):
                            inlier_val = axis_y.get("inlier_count")
                            if isinstance(inlier_val, (int, float)):
                                dim_inliers_y = int(inlier_val)

        global_ratio: float | None = None
        global_px_per_pdf_pt: float | None = None
        global_ncc: float | None = None
        global_scale_raw: float | None = None

        if isinstance(global_raster_result, Mapping):
            best_scale_val = global_raster_result.get("best_scale")
            if isinstance(best_scale_val, (int, float)) and math.isfinite(best_scale_val):
                global_scale_raw = float(best_scale_val)
                global_ratio = float(best_scale_val)
                hint_for_global = scale_hint_raw_value
                if (
                    isinstance(hint_for_global, (int, float))
                    and math.isfinite(hint_for_global)
                    and float(hint_for_global) > 0.0
                ):
                    global_ratio *= float(hint_for_global)
                if pdf_px_per_point:
                    global_px_per_pdf_pt = global_ratio * pdf_px_per_point if global_ratio is not None else None
            ncc_val = global_raster_result.get("ncc_score")
            if isinstance(ncc_val, (int, float)) and math.isfinite(ncc_val):
                global_ncc = float(ncc_val)
            error_msg = global_raster_result.get("error")
            if isinstance(error_msg, str) and error_msg:
                dim_notices.append(f"global_raster_error: {error_msg}")

        if diff_hint_dim is not None and diff_hint_dim > 5.0:
            logger.warning(
                "DimScale vs. coarse hint weicht um %.2f%% ab", diff_hint_dim,
            )
        if diff_vbox_dim is not None and diff_vbox_dim > 5.0:
            logger.warning(
                "DimScale vs. viewBox weicht um %.2f%% ab", diff_vbox_dim,
            )
        if diff_hint_vbox is not None and diff_hint_vbox > 5.0:
            logger.warning(
                "ViewBox vs. coarse hint weicht um %.2f%% ab", diff_hint_vbox,
            )

        if dim_pixels_x and dim_pixels_y:
            try:
                ratio_xy = dim_pixels_x / dim_pixels_y
            except ZeroDivisionError:
                ratio_xy = None
            if ratio_xy is not None and math.isfinite(ratio_xy) and abs(ratio_xy - 1.0) > 0.02:
                dim_notices.append(f"anisotrope Maßlinien-Skalierung: S_dim_x/S_dim_y={ratio_xy:.4f}")

        diff_dim_global = _percent_diff(scale_dim, global_ratio)
        if diff_dim_global is not None and diff_dim_global > 5.0:
            dim_notices.append(
                f"Raster-Skalierung weicht um {diff_dim_global:.2f}% von DimScale ab"
            )

        scale_summary = {
            "scale_hint": scale_hint,
            "scale_vbox": scale_vbox,
            "scale_dim": scale_dim,
            "pdf_px_per_meter": pdf_px_per_meter,
            "svg_pixels_per_unit": pixels_per_svg_unit,
            "diff_hint_vs_vbox_pct": diff_hint_vbox,
            "diff_hint_vs_dim_pct": diff_hint_dim,
            "diff_vbox_vs_dim_pct": diff_vbox_dim,
            "dim_pixels_per_unit_x": dim_pixels_x,
            "dim_pixels_per_unit_y": dim_pixels_y,
            "dim_inliers_x": dim_inliers_x,
            "dim_inliers_y": dim_inliers_y,
            "global_raster_best_scale": global_scale_raw,
            "global_raster_ratio": global_ratio,
            "global_raster_px_per_pdf_pt": global_px_per_pdf_pt,
            "global_raster_ncc": global_ncc,
            "consistency_notices": dim_notices,
        }

    summary_text = tm.summarize()
    logger.info(summary_text)

    result = {
        "coarse_alignment": coarse,
        "metric_scale": metric,
        "scale_summary": scale_summary,
        "timings": tm.items,
        "timing_summary": summary_text,
        "render_dpi": final_dpi,
        "svg_pixels_per_unit": pixels_per_svg_unit,
        "pdf_page_size_pt": (pdf_width_pt, pdf_height_pt),
    }
    result["svg_viewbox"] = (vx, vy, vw, vh)
    if isinstance(global_raster_result, dict):
        result["global_raster"] = dict(global_raster_result)
    return result
