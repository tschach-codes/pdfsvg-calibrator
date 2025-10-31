from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from .dimscale_extractor import estimate_dimline_scale
from .debug_utils import write_dim_debug_svg


def _extract_scale_candidates(*values: Optional[float]) -> Optional[float]:
    """Return the arithmetic mean of finite scale candidates or ``None``."""

    valid = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _coarse_pixels_per_svg_unit(coarse_alignment: Mapping[str, Any] | None) -> Optional[float]:
    """Best-effort conversion from SVG units to pixels derived from coarse alignment."""

    if not isinstance(coarse_alignment, Mapping):
        return None

    keys_direct = (
        "pixels_per_svg_unit",
        "px_per_svg_unit",
        "svg_px_per_unit",
        "svg_pixels_per_unit",
    )
    for key in keys_direct:
        value = coarse_alignment.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)

    # Sometimes scale is provided either as tuple/list or mapping with x/y entries
    scale_value = coarse_alignment.get("scale")
    if isinstance(scale_value, Mapping):
        sx = scale_value.get("x")
        sy = scale_value.get("y")
        candidate = _extract_scale_candidates(sx, sy)
        if candidate is not None:
            return candidate
    elif isinstance(scale_value, Sequence):
        candidate = _extract_scale_candidates(*scale_value)  # type: ignore[arg-type]
        if candidate is not None:
            return candidate

    for axis_key in ("scale_x", "scale_y"):
        axis_value = coarse_alignment.get(axis_key)
        if isinstance(axis_value, (int, float)) and math.isfinite(axis_value):
            return float(axis_value)

    matrix = coarse_alignment.get("transform_matrix")
    if (
        isinstance(matrix, Sequence)
        and len(matrix) == 2
        and all(isinstance(row, Sequence) and len(row) == 2 for row in matrix)
    ):
        # Interpret matrix columns as transformed basis vectors.
        col0 = (float(matrix[0][0]), float(matrix[1][0]))
        col1 = (float(matrix[0][1]), float(matrix[1][1]))
        norms = []
        for col in (col0, col1):
            norm = math.hypot(col[0], col[1])
            if math.isfinite(norm):
                norms.append(norm)
        if norms:
            return sum(norms) / len(norms)

    return None


def _unit_to_meter_factor(unit_name: str | None, unit_factor_cfg: Any) -> Optional[float]:
    """Resolve measurement unit to a factor expressed in meters."""

    if isinstance(unit_factor_cfg, (int, float)) and math.isfinite(unit_factor_cfg):
        if unit_factor_cfg > 0:
            return float(unit_factor_cfg)
        return None

    if not unit_name:
        return 1.0

    lookup = {
        "m": 1.0,
        "meter": 1.0,
        "metre": 1.0,
        "metres": 1.0,
        "meters": 1.0,
        "mm": 0.001,
        "millimeter": 0.001,
        "millimetre": 0.001,
        "millimeters": 0.001,
        "millimetres": 0.001,
        "cm": 0.01,
        "centimeter": 0.01,
        "centimetre": 0.01,
        "centimeters": 0.01,
        "centimetres": 0.01,
        "dm": 0.1,
        "decimeter": 0.1,
        "decimetre": 0.1,
        "inch": 0.0254,
        "in": 0.0254,
    }

    key = unit_name.strip().lower()
    if not key:
        return 1.0
    return lookup.get(key)


def _serialize_dimscale_debug(result: Any) -> Dict[str, Any]:
    """Extract serializable debug information from :class:`DimScaleResult`."""

    if result is None:
        return {}

    data = {
        "ok": getattr(result, "ok", None),
        "reason": getattr(result, "reason", None),
        "scale_x_svg_per_unit": getattr(result, "scale_x_svg_per_unit", None),
        "scale_y_svg_per_unit": getattr(result, "scale_y_svg_per_unit", None),
        "scale_x_unit_per_svg": getattr(result, "scale_x_unit_per_svg", None),
        "scale_y_unit_per_svg": getattr(result, "scale_y_unit_per_svg", None),
        "candidates_used": getattr(result, "candidates_used", None),
        "inlier_cluster_size": getattr(result, "inlier_cluster_size", None),
    }
    debug_data = getattr(result, "debug", None)
    if isinstance(debug_data, Mapping):
        data["debug"] = debug_data
    return data


def refine_metric_scale(
    svg_bytes: bytes,
    coarse_alignment: dict,
    config: dict,
) -> Dict[str, Any]:
    """
    Verwendet die bestehende Maßlinien-/Bemaßungs-Extraktion des Projekts, um einen metrischen Skalierungsfaktor
    (Pixel -> echte Längeneinheit) zu bestimmen.

    Erwartung:
      - Die bestehende DimScale-Logik kann aus SVG-Geometrie bzw. aus gerenderten Ausschnitten
        echte Längen vs. beschriftete Längenpaare extrahieren.
      - Wir rufen diese Logik hier auf und geben ihr die grobe Ausrichtung (Rotation/Flip/Scale/Translation),
        falls sie das braucht, damit sie weiß in welchem Raum sie messen soll.

    Rückgabe:
    {
      "ok": bool,
      "pixels_per_meter": float | None,
      "debug": { ... beliebige Zusatzinfos der DimScale-Routine ... }
    }
    """

    coarse_alignment = dict(coarse_alignment or {})
    cfg = dict(config or {})
    dimscale_raw = cfg.get("dimscale")
    dimscale_cfg = dimscale_raw if isinstance(dimscale_raw, Mapping) else {}

    enable_ocr = bool(dimscale_cfg.get("enable_ocr_paths", dimscale_cfg.get("enable_ocr", False)))
    unit_name = dimscale_cfg.get("unit") if isinstance(dimscale_cfg, Mapping) else None
    default_unit = dimscale_cfg.get("default_unit") if isinstance(dimscale_cfg, Mapping) else None
    if (not unit_name) and isinstance(default_unit, str):
        unit_name = default_unit
    unit_factor_cfg = dimscale_cfg.get("unit_in_meters") if isinstance(dimscale_cfg, Mapping) else None

    tmp_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    try:
        tmp_svg.write(svg_bytes)
        tmp_svg.flush()
        tmp_svg_path = tmp_svg.name
    finally:
        tmp_svg.close()

    dimscale_result = None
    try:
        dimscale_result = estimate_dimline_scale(
            svg_path=tmp_svg_path,
            enable_ocr_paths=enable_ocr,
        )
    finally:
        try:
            os.unlink(tmp_svg_path)
        except OSError:
            pass

    debug_cfg = cfg.get("debug") if isinstance(cfg, Mapping) else {}
    inputs_cfg = cfg.get("inputs") if isinstance(cfg, Mapping) else {}
    overlay_enabled = True
    if isinstance(debug_cfg, Mapping):
        overlay_enabled = bool(debug_cfg.get("dimscale_overlay", True))
    else:
        debug_cfg = {}
    if overlay_enabled:
        try:
            svg_input_path = None
            if isinstance(inputs_cfg, Mapping):
                svg_input_path = inputs_cfg.get("svg_path")
            outdir_path = debug_cfg.get("outdir") if isinstance(debug_cfg, Mapping) else None
            prefix_cfg = debug_cfg.get("prefix") if isinstance(debug_cfg, Mapping) else None
            debug_prefix = prefix_cfg if isinstance(prefix_cfg, str) and prefix_cfg else "dimscale"
            debug_data = getattr(dimscale_result, "debug", {}) if dimscale_result is not None else {}
            dimscale_debug = {}
            if isinstance(debug_data, Mapping):
                dimscale_section = debug_data.get("dimscale")
                if isinstance(dimscale_section, Mapping):
                    dimscale_debug = dimscale_section
            if svg_input_path and outdir_path:
                write_dim_debug_svg(
                    str(svg_input_path),
                    str(outdir_path),
                    debug_prefix,
                    dimscale_debug.get("segments_svg", []),
                    dimscale_debug.get("texts_svg", []),
                    dimscale_debug.get("pairs", []),
                    raster_png=True,
                )
        except Exception as e:  # pragma: no cover - debug helper
            print("[pdfsvg] write_dim_debug_svg failed:", e)

    svg_units_per_unit = _extract_scale_candidates(
        getattr(dimscale_result, "scale_x_svg_per_unit", None),
        getattr(dimscale_result, "scale_y_svg_per_unit", None),
    )
    pixels_per_svg_unit = _coarse_pixels_per_svg_unit(coarse_alignment)
    unit_to_meter = _unit_to_meter_factor(unit_name, unit_factor_cfg)

    pixels_per_unit = None
    if svg_units_per_unit is not None and pixels_per_svg_unit is not None:
        pixels_per_unit = svg_units_per_unit * pixels_per_svg_unit

    pixels_per_meter = None
    if pixels_per_unit is not None and unit_to_meter:
        pixels_per_meter = pixels_per_unit / unit_to_meter

    ok = bool(getattr(dimscale_result, "ok", False) and pixels_per_meter is not None)

    debug_info: Dict[str, Any] = {
        "dimscale": _serialize_dimscale_debug(dimscale_result),
        "coarse_alignment": coarse_alignment,
        "config": dimscale_cfg if isinstance(dimscale_cfg, Mapping) else {},
        "derived": {
            "svg_units_per_unit": svg_units_per_unit,
            "pixels_per_svg_unit": pixels_per_svg_unit,
            "unit_to_meter": unit_to_meter,
            "unit_name": unit_name,
        },
    }

    return {
        "ok": ok,
        "pixels_per_meter": pixels_per_meter if ok else None,
        "debug": debug_info,
    }
