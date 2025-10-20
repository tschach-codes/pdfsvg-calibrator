"""High-level calibration entry point with orientation gate."""
from __future__ import annotations

import copy
import logging
from typing import Dict, Mapping, Sequence, Tuple

from .fit_model import calibrate as _calibrate_ransac
from .orientation import (
    DEFAULT_MIN_ACCEPT_SCORE,
    DEFAULT_RASTER_SIZE,
    DEFAULT_USE_PHASE_CORRELATION,
    pick_flip_and_rot,
)
from .types import Model, Segment
from .metrics import Timer

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
    min_accept_score = float(orientation_cfg.get("min_accept_score", DEFAULT_MIN_ACCEPT_SCORE))
    orientation_cfg["min_accept_score"] = min_accept_score

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
        orientation_module.DEFAULT_MIN_ACCEPT_SCORE = min_accept_score
        orientation_result = pick_flip_and_rot(
            pdf_segments,
            svg_segments,
            pdf_size,
            svg_size,
        )
        flip_xy, rot_deg, tx0, ty0 = orientation_result
        orientation_cfg["flip_xy"] = flip_xy
        orientation_cfg["rot_deg"] = rot_deg
        orientation_cfg["translation"] = (tx0, ty0)
        orientation_cfg["score"] = orientation_result.score
        orientation_cfg["overlap"] = orientation_result.overlap
        orientation_cfg["path"] = orientation_result.path
        orientation_cfg["primary_score"] = orientation_result.primary_score
        orientation_cfg["primary_overlap"] = orientation_result.primary_overlap
        if orientation_result.fallback_score is not None:
            orientation_cfg["fallback_score"] = orientation_result.fallback_score
        if orientation_result.fallback_overlap is not None:
            orientation_cfg["fallback_overlap"] = orientation_result.fallback_overlap
        orientation_cfg["widen_trans_window"] = orientation_result.widen_trans_window
        if orientation_result.trans_window_hint_px is not None:
            orientation_cfg["trans_window_hint_px"] = orientation_result.trans_window_hint_px
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
    seed_sx = sx0
    seed_sy = sy0

    refine_cfg = cfg_local.setdefault("refine", {})  # type: ignore[assignment]
    if not isinstance(refine_cfg, dict):
        raise ValueError("refine config must be a mapping")
    scale_window_base_val = abs(float(refine_cfg.get("scale_max_dev_rel", 0.02)))
    trans_window_base = abs(float(refine_cfg.get("trans_max_dev_px", 10.0)))
    trans_window_initial = trans_window_base
    if use_orientation:
        hint = orientation_cfg.get("trans_window_hint_px")
        if isinstance(hint, (int, float)):
            trans_window_initial = max(trans_window_initial, float(hint))
    max_iters_base = int(refine_cfg.get("max_iters", 120))
    refine_cfg.setdefault("max_samples", 1500)

    def _apply_refine_window(
        scale_window: float,
        trans_window: float,
        *,
        max_iters: int | None = None,
    ) -> None:
        refine_cfg["scale_max_dev_rel"] = scale_window
        refine_cfg["trans_max_dev_px"] = trans_window
        refine_cfg["max_iters"] = max_iters if max_iters is not None else max_iters_base
        if use_orientation:
            refine_cfg["scale_seed"] = (seed_sx, seed_sy)
            refine_cfg["trans_seed"] = (tx0, ty0)
            refine_cfg["scale_abs_bounds"] = (
                (sx0 * (1.0 - scale_window), sx0 * (1.0 + scale_window)),
                (sy0 * (1.0 - scale_window), sy0 * (1.0 + scale_window)),
            )
        else:
            refine_cfg.pop("scale_seed", None)
            refine_cfg.pop("trans_seed", None)
            refine_cfg.pop("scale_abs_bounds", None)

    _apply_refine_window(scale_window_base_val, trans_window_initial)

    if use_orientation:
        log.info(
            "[orient] seed_scale=(%.6f, %.6f) trans_window=±%.1fpx scale_window=±%.2f%%",
            seed_sx,
            seed_sy,
            float(refine_cfg.get("trans_max_dev_px", 0.0)),
            float(refine_cfg.get("scale_max_dev_rel", 0.0)) * 100.0,
        )
        log.info(
            "Seed scale sx=%.6f sy=%.6f, clamp ±%.2f%%",
            sx0,
            sy0,
            scale_window_base_val * 100.0,
        )

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

    quality_gate_cfg = refine_cfg.get("quality_gate", {})
    if quality_gate_cfg and not isinstance(quality_gate_cfg, dict):
        raise ValueError("refine.quality_gate must be a mapping if provided")
    gate_enabled = True
    if isinstance(quality_gate_cfg, dict):
        gate_enabled = bool(quality_gate_cfg.get("enabled", True))
        rmse_gate_val = quality_gate_cfg.get("rmse_px", 12.0)
        rmse_gate = float(rmse_gate_val) if rmse_gate_val is not None else None
        score_gate_val = quality_gate_cfg.get("score_min", 0.15)
        score_gate = float(score_gate_val) if score_gate_val is not None else None
        fallback_trans = abs(float(quality_gate_cfg.get("fallback_trans_max_dev_px", 25.0)))
        fallback_scale = abs(float(quality_gate_cfg.get("fallback_scale_max_dev_rel", 0.05)))
        fallback_iters_cfg = int(quality_gate_cfg.get("fallback_max_iters", 60))
    else:  # pragma: no cover - defensive
        rmse_gate = 12.0
        score_gate = 0.15
        fallback_trans = 25.0
        fallback_scale = 0.05
        fallback_iters_cfg = 60

    def _quality_failures(result: Model) -> list[str]:
        issues: list[str] = []
        if rmse_gate is not None and rmse_gate > 0.0 and result.rmse > rmse_gate:
            issues.append(f"RMSE {result.rmse:.2f}px > {rmse_gate:.2f}px")
        if score_gate is not None and result.score < score_gate:
            issues.append(f"Score {result.score:.4f} < {score_gate:.4f}")
        return issues

    with Timer("refine.total"):
        model = _calibrate_ransac(pdf_segments, svg_segments, pdf_size, svg_size, cfg_local)

    quality_notes: list[str] = []
    if gate_enabled:
        failures = _quality_failures(model)
        if failures:
            fallback_trans_window = max(trans_window_base, fallback_trans)
            fallback_scale_window = max(scale_window_base_val, fallback_scale)
            fallback_iters = max(1, min(max_iters_base, fallback_iters_cfg))
            log.warning(
                "[calib] quality gate triggered (RMSE=%.2fpx, P95=%.2fpx, Score=%.4f; %s); "
                "expanding to trans=±%.1fpx, scale=±%.2f%% and running fallback (max %d iters)",
                model.rmse,
                model.p95,
                model.score,
                "; ".join(failures),
                fallback_trans_window,
                fallback_scale_window * 100.0,
                fallback_iters,
            )
            _apply_refine_window(fallback_scale_window, fallback_trans_window, max_iters=fallback_iters)
            ransac_cfg["iters"] = fallback_iters
            model = _calibrate_ransac(pdf_segments, svg_segments, pdf_size, svg_size, cfg_local)
            post_failures = _quality_failures(model)
            if post_failures:
                message = (
                    "Calibration quality below threshold after fallback (RMSE="
                    f"{model.rmse:.2f}px, P95={model.p95:.2f}px, Score={model.score:.4f}): "
                    + "; ".join(post_failures)
                )
                log.warning("[calib] %s", message)
                quality_notes.append(message)
            else:
                note = (
                    "Quality gate fallback succeeded after expanding bounds (RMSE="
                    f"{model.rmse:.2f}px, P95={model.p95:.2f}px, Score={model.score:.4f})."
                )
                log.info("[calib] %s", note)
                quality_notes.append(note)

    model.quality_notes = tuple(quality_notes)
    return model
