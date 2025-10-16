"""High-level calibration entry point with orientation gate."""
from __future__ import annotations

import copy
import logging
from typing import Dict, Mapping, Sequence, Tuple

from .fit_model import calibrate as _calibrate_ransac
from .orientation import DEFAULT_RASTER_SIZE, DEFAULT_USE_PHASE_CORRELATION, pick_flip_and_rot
from .types import Model, Segment

log = logging.getLogger(__name__)


def _ensure_size_tuple(value: int | Tuple[int, int] | Sequence[int]) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    seq = list(value)
    if len(seq) != 2:
        raise ValueError("raster size must have two entries")
    return int(seq[0]), int(seq[1])


def calibrate(
    pdf_segments: Sequence[Segment],
    svg_segments: Sequence[Segment],
    pdf_size: Tuple[float, float],
    svg_size: Tuple[float, float],
    cfg: Mapping[str, object],
) -> Model:
    """Run full calibration with orientation seeding."""

    if not pdf_segments:
        raise ValueError("Keine PDF-Segmente für die Kalibrierung übergeben")
    if not svg_segments:
        raise ValueError("Keine SVG-Segmente für die Kalibrierung übergeben")

    cfg_local: Dict[str, object] = copy.deepcopy(dict(cfg))

    orientation_cfg = cfg_local.setdefault("orientation", {})  # type: ignore[assignment]
    if not isinstance(orientation_cfg, dict):
        raise ValueError("orientation config must be a mapping")

    raster_size = orientation_cfg.get("raster_size", DEFAULT_RASTER_SIZE)
    try:
        raster_tuple = _ensure_size_tuple(raster_size)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("orientation.raster_size must be an int or length-2 sequence") from exc

    use_phase = bool(orientation_cfg.get("use_phase_correlation", DEFAULT_USE_PHASE_CORRELATION))

    flip_xy = (1.0, 1.0)
    rot_deg = cfg_local.get("rot_degrees", [0])  # type: ignore[assignment]
    if isinstance(rot_deg, Sequence):
        rot_deg = rot_deg[0] if rot_deg else 0
    rot_deg = int(rot_deg)  # type: ignore[assignment]
    tx0 = 0.0
    ty0 = 0.0

    use_orientation = bool(orientation_cfg.get("enabled", True))

    if use_orientation:
        from . import orientation as orientation_module

        orientation_module.DEFAULT_RASTER_SIZE = raster_tuple
        orientation_module.DEFAULT_USE_PHASE_CORRELATION = use_phase
        flip_xy, rot_deg, tx0, ty0 = pick_flip_and_rot(
            pdf_segments,
            svg_segments,
            pdf_size,
            svg_size,
        )
        orientation_cfg["flip_xy"] = flip_xy
        orientation_cfg["rot_deg"] = rot_deg
        orientation_cfg["translation"] = (tx0, ty0)
        log.info(
            "Orientation: rot=%d, flip=%s, seed_t=(%.2f, %.2f))",
            rot_deg,
            flip_xy,
            tx0,
            ty0,
        )
    else:
        orientation_cfg["flip_xy"] = flip_xy
        orientation_cfg["rot_deg"] = rot_deg
        orientation_cfg["translation"] = (tx0, ty0)

    pdf_w, pdf_h = pdf_size
    svg_w, svg_h = svg_size
    if pdf_w == 0 or pdf_h == 0:
        raise ValueError("pdf_size must contain positive values")
    if svg_w == 0 or svg_h == 0:
        raise ValueError("svg_size must contain positive values")

    sx0 = svg_w / pdf_w
    sy0 = svg_h / pdf_h
    seed_sx = flip_xy[0] * sx0
    seed_sy = flip_xy[1] * sy0

    refine_cfg = cfg_local.setdefault("refine", {})  # type: ignore[assignment]
    if not isinstance(refine_cfg, dict):
        raise ValueError("refine config must be a mapping")
    scale_window_val = refine_cfg.get("scale_max_dev_rel", 0.02)
    scale_window = abs(float(scale_window_val))
    refine_cfg["scale_max_dev_rel"] = scale_window
    refine_cfg.setdefault("trans_max_dev_px", 10.0)
    refine_cfg.setdefault("max_iters", 120)
    refine_cfg.setdefault("max_samples", 1500)
    if use_orientation:
        refine_cfg.setdefault("scale_seed", (seed_sx, seed_sy))
        refine_cfg.setdefault("trans_seed", (tx0, ty0))
        refine_cfg.setdefault(
            "scale_abs_bounds",
            (
                (sx0 * (1.0 - scale_window), sx0 * (1.0 + scale_window)),
                (sy0 * (1.0 - scale_window), sy0 * (1.0 + scale_window)),
            ),
        )
    else:
        refine_cfg.pop("scale_seed", None)
        refine_cfg.pop("trans_seed", None)
        refine_cfg.pop("scale_abs_bounds", None)

    if use_orientation:
        log.info(
            "[orient] seed_scale=(%.6f, %.6f) trans_window=±%.1fpx scale_window=±%.2f%%",
            seed_sx,
            seed_sy,
            float(refine_cfg.get("trans_max_dev_px", 0.0)),
            float(refine_cfg.get("scale_max_dev_rel", 0.0)) * 100.0,
        )
        log.info("Seed scale sx=%.6f sy=%.6f, clamp ±%.2f%%", sx0, sy0, scale_window * 100.0)

    cfg_local["rot_degrees"] = [rot_deg]

    ransac_cfg = cfg_local.setdefault("ransac", {})  # type: ignore[assignment]
    if not isinstance(ransac_cfg, dict):
        raise ValueError("ransac config must be a mapping")
    ransac_cfg.setdefault("iters", refine_cfg.get("max_iters", 120))
    if "refine_scale_step" not in ransac_cfg:
        scale_window = float(refine_cfg.get("scale_max_dev_rel", 0.02))
        ransac_cfg["refine_scale_step"] = max(scale_window / 10.0, 1e-4)
    if "refine_trans_px" not in ransac_cfg:
        trans_window = float(refine_cfg.get("trans_max_dev_px", 10.0))
        ransac_cfg["refine_trans_px"] = max(trans_window / 3.0, 0.5)

    sampling_cfg = cfg_local.setdefault("sampling", {})  # type: ignore[assignment]
    if not isinstance(sampling_cfg, dict):
        raise ValueError("sampling config must be a mapping")
    sampling_cfg.setdefault("step_rel", 0.03)
    sampling_cfg.setdefault("max_points", refine_cfg.get("max_samples", 1500))

    grid_cfg = cfg_local.setdefault("grid", {})  # type: ignore[assignment]
    if not isinstance(grid_cfg, dict):
        raise ValueError("grid config must be a mapping")
    grid_cfg.setdefault("initial_cell_rel", 0.05)
    grid_cfg.setdefault("final_cell_rel", 0.02)

    model = _calibrate_ransac(pdf_segments, svg_segments, pdf_size, svg_size, cfg_local)
    return model
