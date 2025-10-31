from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple
import pypdfium2

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

    dpi_coarse = int(raster_cfg.get("dpi_coarse", 150))
    dpi_fallback = int(raster_cfg.get("dpi_coarse_fallback", dpi_coarse))
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

    with tm.section("verify_scales"):
        scale_hint = float(coarse.get("scale_hint", 0.0)) if coarse.get("scale_hint") is not None else None
        scale_vbox = pixels_per_svg_unit / pdf_px_per_point if pdf_px_per_point else None
        pixels_per_meter = metric.get("pixels_per_meter") if isinstance(metric, Mapping) else None
        scale_dim = None
        if isinstance(pixels_per_meter, (int, float)) and math.isfinite(pixels_per_meter):
            scale_dim = float(pixels_per_meter) / pdf_px_per_meter if pdf_px_per_meter else None
        diff_hint_vbox = _percent_diff(scale_hint, scale_vbox)
        diff_hint_dim = _percent_diff(scale_hint, scale_dim)
        diff_vbox_dim = _percent_diff(scale_vbox, scale_dim)

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

        scale_summary = {
            "scale_hint": scale_hint,
            "scale_vbox": scale_vbox,
            "scale_dim": scale_dim,
            "pdf_px_per_meter": pdf_px_per_meter,
            "svg_pixels_per_unit": pixels_per_svg_unit,
            "diff_hint_vs_vbox_pct": diff_hint_vbox,
            "diff_hint_vs_dim_pct": diff_hint_dim,
            "diff_vbox_vs_dim_pct": diff_vbox_dim,
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
    return result
