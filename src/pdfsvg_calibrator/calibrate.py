"""High-level calibration entry point with orientation gate."""
from __future__ import annotations

import copy
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

from .fit_model import calibrate as _calibrate_ransac
from .orientation import pick_flip_rot_and_shift
from .transform import Transform2D
from .types import Model, Segment
from .utils.timer import timer
from .verify_distmap import evaluate_rmse

log = logging.getLogger(__name__)


def calibrate(
    pdf_segments: Sequence[Segment],
    svg_segments: Sequence[Segment],
    pdf_size: Tuple[float, float],
    svg_size: Tuple[float, float],
    cfg: Mapping[str, object],
    *,
    stats: MutableMapping[str, float] | None = None,
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

    orientation_cfg.setdefault("sample_topk_rel", 0.2)
    orientation_cfg.setdefault("raster_size", 256)
    min_accept_score = float(orientation_cfg.get("min_accept_score", 0.05))
    orientation_cfg["min_accept_score"] = min_accept_score

    flip_xy = (1.0, 1.0)
    rot_deg = 0
    tx0 = 0.0
    ty0 = 0.0

    use_orientation = bool(orientation_cfg.get("enabled", True))

    pdf_w, pdf_h = pdf_size
    svg_w, svg_h = svg_size
    if pdf_w == 0 or pdf_h == 0:
        raise ValueError("pdf_size must contain positive values")
    if svg_w == 0 or svg_h == 0:
        raise ValueError("svg_size must contain positive values")

    page_pdf = SimpleNamespace(w=float(pdf_w), h=float(pdf_h))
    page_svg = SimpleNamespace(w=float(svg_w), h=float(svg_h))

    orientation_result = None

    def _timed(key: str):
        return timer(stats, key) if stats is not None else nullcontext()

    if use_orientation:
        seed_before = stats.get("Seed", 0.0) if stats is not None else 0.0
        with _timed("Orientation"):
            orientation_result = pick_flip_rot_and_shift(
                pdf_segments,
                svg_segments,
                page_pdf,
                page_svg,
                cfg_local,
                stats=stats,
            )
        if stats is not None:
            seed_after = stats.get("Seed", 0.0)
            stats["Orientation"] = max(
                0.0,
                stats.get("Orientation", 0.0) - (seed_after - seed_before),
            )
        flip_xy = tuple(float(v) for v in orientation_result.get("flip", (1, 1)))  # type: ignore[assignment]
        rot_deg = int(orientation_result.get("rot_deg", 0))
        tx0 = float(orientation_result.get("t_seed", (0.0, 0.0))[0])
        ty0 = float(orientation_result.get("t_seed", (0.0, 0.0))[1])

        orientation_cfg["flip_xy"] = flip_xy
        orientation_cfg["rot_deg"] = rot_deg
        orientation_cfg["translation"] = (tx0, ty0)
        orientation_cfg["overlap"] = float(orientation_result.get("overlap", 0.0))
        orientation_cfg["response"] = float(orientation_result.get("response", 0.0))
        orientation_cfg["phase_response"] = float(orientation_result.get("phase_response", 0.0))
        orientation_cfg["du_dv"] = orientation_result.get("du_dv", (0.0, 0.0))
        orientation_cfg["dx_dy_doc"] = (
            float(orientation_result.get("dx_doc", 0.0)),
            float(orientation_result.get("dy_doc", 0.0)),
        )
        orientation_cfg["t_seed"] = (tx0, ty0)

        log.info(
            "[orient] best rot=%d flip=%s overlap=%.3f resp=%.3f",
            rot_deg,
            flip_xy,
            orientation_cfg["overlap"],
            orientation_cfg["response"],
        )
        du, dv = orientation_result.get("du_dv", (0.0, 0.0))
        log.info(
            "[orient] shift_pixels=(%.2f, %.2f) dx_dy_doc=(%.2f, %.2f) -> t_seed=(%.2f, %.2f)",
            float(du),
            float(dv),
            orientation_cfg["dx_dy_doc"][0],
            orientation_cfg["dx_dy_doc"][1],
            tx0,
            ty0,
        )
    else:
        orientation_cfg["flip_xy"] = flip_xy
        orientation_cfg["rot_deg"] = rot_deg
        orientation_cfg["translation"] = (tx0, ty0)

    sx0 = svg_w / pdf_w
    sy0 = svg_h / pdf_h
    seed_sx = sx0
    seed_sy = sy0

    seed_transform = Transform2D(
        flip=flip_xy,
        rot_deg=rot_deg,
        sx=sx0,
        sy=sy0,
        tx=tx0,
        ty=ty0,
    )
    orientation_cfg["transform_seed"] = seed_transform.summary()

    refine_cfg = cfg_local.setdefault("refine", {})  # type: ignore[assignment]
    if not isinstance(refine_cfg, dict):
        raise ValueError("refine config must be a mapping")
    scale_window_base_val = abs(float(refine_cfg.get("scale_max_dev_rel", 0.02)))
    trans_window_base = abs(float(refine_cfg.get("trans_max_dev_px", 8.0)))
    trans_window_initial = trans_window_base
    if use_orientation and orientation_result is not None:
        overlap = float(orientation_result.get("overlap", 0.0))
        if overlap < min_accept_score:
            log.info(
                "[orient] overlap %.3f below %.3f – keeping minimal refine window ±%.1fpx",
                overlap,
                min_accept_score,
                trans_window_initial,
            )
    max_iters_base = int(refine_cfg.get("max_iters", 60))
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
    ransac_cfg.setdefault("iters", refine_cfg.get("max_iters", 60))
    if "refine_scale_step" not in ransac_cfg:
        scale_window = float(refine_cfg.get("scale_max_dev_rel", 0.02))
        ransac_cfg["refine_scale_step"] = max(scale_window / 10.0, 1e-4)
    if "refine_trans_px" not in ransac_cfg:
        trans_window = float(refine_cfg.get("trans_max_dev_px", 8.0))
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

    verify_cfg = cfg_local.setdefault("verify", {})  # type: ignore[assignment]
    if not isinstance(verify_cfg, dict):
        raise ValueError("verify config must be a mapping")
    verify_mode = str(verify_cfg.get("mode", "")).strip().lower()

    distmap_cfg_raw = verify_cfg.get("distmap", {})
    distmap_cfg = distmap_cfg_raw if isinstance(distmap_cfg_raw, Mapping) else {}

    def _distmap_sizes() -> tuple[int, int]:
        size_cfg = distmap_cfg.get("raster")
        if size_cfg is None:
            size_cfg = verify_cfg.get("distmap_raster")
        if size_cfg is None:
            size_cfg = distmap_cfg.get("size")
        if isinstance(size_cfg, (tuple, list)) and len(size_cfg) == 2:
            w = max(1, int(round(float(size_cfg[0]))))
            h = max(1, int(round(float(size_cfg[1]))))
            return w, h
        if isinstance(size_cfg, (int, float)) and size_cfg > 0:
            val = max(1, int(round(float(size_cfg))))
            return val, val
        return 768, 768

    def _distmap_samples() -> int:
        samples = distmap_cfg.get("samples_per_seg")
        if samples is None:
            samples = verify_cfg.get("distmap_samples")
        if samples is None:
            samples = distmap_cfg.get("samples")
        try:
            value = int(samples) if samples is not None else 16
        except (TypeError, ValueError):
            value = 16
        return max(1, value)

    def _apply_distmap_metrics(model: Model) -> Model:
        if verify_mode != "distmap":
            return model
        transform = model.transform
        if transform is None:
            transform = Transform2D(
                flip=(model.flip_x, model.flip_y),
                rot_deg=model.rot_deg,
                sx=model.sx,
                sy=model.sy,
                tx=model.tx,
                ty=model.ty,
            )
        W, H = _distmap_sizes()
        samples = _distmap_samples()
        metrics = evaluate_rmse(
            transform,
            pdf_segments,
            svg_segments,
            svg_w,
            svg_h,
            W=W,
            H=H,
            n_per_seg=samples,
        )
        model.rmse = float(metrics.get("rmse", model.rmse))
        model.p95 = float(metrics.get("p95", model.p95))
        model.median = float(metrics.get("median", model.median))
        model.score = float(metrics.get("score", model.score))
        log.info(
            "[distmap] RMSE=%.3fpx P95=%.3fpx Median=%.3fpx Score=%.5f (samples=%d, raster=%dx%d)",
            model.rmse,
            model.p95,
            model.median,
            model.score,
            int(metrics.get("n", 0)),
            W,
            H,
        )
        return model

    with _timed("Refine"):
        model = _calibrate_ransac(pdf_segments, svg_segments, pdf_size, svg_size, cfg_local)

    with _timed("Verify"):
        model = _apply_distmap_metrics(model)

    quality_notes: list[str] = []
    if gate_enabled:
        failures = _quality_failures(model)
        if failures:
            fallback_trans_window = min(trans_window_base, fallback_trans)
            fallback_scale_window = min(scale_window_base_val, fallback_scale)
            fallback_iters = max(1, min(max_iters_base, fallback_iters_cfg))
            log.warning(
                "[calib] quality gate triggered (RMSE=%.2fpx, P95=%.2fpx, Score=%.4f; %s); "
                "retrying with trans=±%.1fpx, scale=±%.2f%% (max %d iters)",
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
            with _timed("Refine"):
                model = _calibrate_ransac(
                    pdf_segments, svg_segments, pdf_size, svg_size, cfg_local
                )
            with _timed("Verify"):
                model = _apply_distmap_metrics(model)
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
    model.transform = Transform2D(
        flip=(model.flip_x, model.flip_y),
        rot_deg=model.rot_deg,
        sx=model.sx,
        sy=model.sy,
        tx=model.tx,
        ty=model.ty,
    )
    return model
