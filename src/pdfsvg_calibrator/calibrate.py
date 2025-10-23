"""High-level calibration entry point with orientation gate."""
from __future__ import annotations

import copy
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

_CoarseModules = tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]
_COARSE_MODULES: _CoarseModules | None = None


def _ensure_coarse_modules() -> _CoarseModules:
    """Lazily import coarse alignment helpers to keep cold-start fast."""

    global _COARSE_MODULES
    if _COARSE_MODULES is not None:
        return _COARSE_MODULES

    try:
        from src.coarse.pipeline import coarse_align as _coarse_align
        from src.coarse.geom import (
            apply_orientation_pts as _apply_orientation_pts,
            apply_scale_shift_pts as _apply_scale_shift_pts,
            transform_segments as _transform_segments,
        )
    except ModuleNotFoundError:  # pragma: no cover - fallback for src-layout installs
        import sys

        project_root = Path(__file__).resolve().parents[2]
        project_root_str = str(project_root)
        src_root_str = str(project_root / "src")
        if src_root_str not in sys.path:
            sys.path.insert(0, src_root_str)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        from src.coarse.pipeline import coarse_align as _coarse_align
        from src.coarse.geom import (
            apply_orientation_pts as _apply_orientation_pts,
            apply_scale_shift_pts as _apply_scale_shift_pts,
            transform_segments as _transform_segments,
        )

    _COARSE_MODULES = (
        _coarse_align,
        _apply_orientation_pts,
        _apply_scale_shift_pts,
        _transform_segments,
    )
    return _COARSE_MODULES

from .fit_model import calibrate as _calibrate_ransac
from .orientation import pick_flip_rot_and_shift
from .transform import Transform2D
from .types import Model, Segment
from .utils.timer import timer
from .verify_distmap import evaluate_rmse

log = logging.getLogger(__name__)


def _coarse_affine_matrix(
    orientation: Any,
    scale: tuple[float, float],
    shift: tuple[float, float],
    bbox_svg: tuple[float, float, float, float],
    *,
    apply_orientation_pts_fn: Callable[..., np.ndarray],
    apply_scale_shift_pts_fn: Callable[..., np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute 2×2 matrix (columns) and translation for coarse alignment."""

    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    oriented = apply_orientation_pts_fn(pts, orientation, bbox_svg)
    mapped = apply_scale_shift_pts_fn(oriented, scale[0], scale[1], shift[0], shift[1])
    origin = mapped[0]
    e1 = mapped[1] - origin
    e2 = mapped[2] - origin
    matrix = (
        (float(e1[0]), float(e2[0])),
        (float(e1[1]), float(e2[1])),
    )
    translation = (float(origin[0]), float(origin[1]))
    return matrix, translation


def calibrate(
    pdf_segments: Sequence[Segment],
    svg_segments: Sequence[Segment],
    pdf_size: Tuple[float, float],
    svg_size: Tuple[float, float],
    cfg: Mapping[str, object],
    *,
    stats: MutableMapping[str, float] | None = None,
    coarse_only: bool = False,
    coarse_outputs: MutableMapping[str, object] | None = None,
    coarse_debug_dir: str | Path | None = None,
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
    if coarse_outputs is not None:
        coarse_outputs.clear()
        coarse_outputs["orientation"] = None

    def _timed(key: str):
        return timer(stats, key) if stats is not None else nullcontext()

    coarse_cfg_raw = cfg_local.get("coarse")
    coarse_cfg = coarse_cfg_raw if isinstance(coarse_cfg_raw, Mapping) else None
    coarse_enabled = bool(coarse_cfg.get("enabled", True)) if coarse_cfg else False
    scale_window_limit: float | None = None
    trans_window_limit: float | None = None
    angle_window_limit: float | None = None
    if coarse_cfg and coarse_enabled:
        refine_windows_cfg = coarse_cfg.get("refine_windows")
        if isinstance(refine_windows_cfg, Mapping):
            try:
                scale_window_limit_val = refine_windows_cfg.get("scale_rel")
                if scale_window_limit_val is not None:
                    scale_window_limit = abs(float(scale_window_limit_val))
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                scale_window_limit = None
            try:
                trans_window_limit_val = refine_windows_cfg.get("shift_px")
                if trans_window_limit_val is not None:
                    trans_window_limit = abs(float(trans_window_limit_val))
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                trans_window_limit = None
            try:
                angle_window_limit_val = refine_windows_cfg.get("angle_deg")
                if angle_window_limit_val is not None:
                    angle_window_limit = abs(float(angle_window_limit_val))
            except (TypeError, ValueError):  # pragma: no cover - defensive parsing
                angle_window_limit = None
        if angle_window_limit is not None:
            try:
                current_angle_tol = float(cfg_local.get("angle_tol_deg", 6.0))
            except (TypeError, ValueError):
                current_angle_tol = 6.0
            cfg_local["angle_tol_deg"] = min(current_angle_tol, angle_window_limit)

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
        flip_vals = orientation_result.get("flip", (1.0, 1.0))
        flip_xy = (float(flip_vals[0]), float(flip_vals[1]))
        flip_xy = (1.0 if flip_xy[0] >= 0 else -1.0, 1.0 if flip_xy[1] >= 0 else -1.0)
        if flip_xy not in {
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        }:
            raise ValueError(f"Ungültiger Flip-Wert aus Orientation: {flip_xy}")
        rot_deg = int(orientation_result.get("rot_deg", 0))
        tx0 = float(orientation_result.get("t_seed", (0.0, 0.0))[0])
        ty0 = float(orientation_result.get("t_seed", (0.0, 0.0))[1])

        ang = math.radians(rot_deg % 360)
        ca, sa = math.cos(ang), math.sin(ang)
        R = np.array([[ca, -sa], [sa, ca]], dtype=float)
        F = np.array([[flip_xy[0], 0.0], [0.0, flip_xy[1]]], dtype=float)
        seed_matrix = R @ F

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
        orientation_cfg["seed_matrix"] = (
            (float(seed_matrix[0, 0]), float(seed_matrix[0, 1])),
            (float(seed_matrix[1, 0]), float(seed_matrix[1, 1])),
        )

        log.info(
            "[orient] Applying orientation seed: rot=%d°, flip=(%.1f, %.1f)",
            rot_deg,
            flip_xy[0],
            flip_xy[1],
        )
        log.info(
            "[orient] Seed transform T0: [[%.5f, %.5f], [%.5f, %.5f]] tx=%.3f ty=%.3f",
            float(seed_matrix[0, 0]),
            float(seed_matrix[0, 1]),
            float(seed_matrix[1, 0]),
            float(seed_matrix[1, 1]),
            tx0,
            ty0,
        )
        log.info(
            "[orient] overlap=%.3f resp=%.3f",
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
        orientation_cfg["seed_matrix"] = (
            (1.0, 0.0),
            (0.0, 1.0),
        )

    if coarse_outputs is not None:
        coarse_outputs["orientation"] = orientation_result

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
    if scale_window_limit is not None:
        scale_window_base_val = min(scale_window_base_val, scale_window_limit)
    if trans_window_limit is not None:
        trans_window_base = min(trans_window_base, trans_window_limit)
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
        if scale_window_limit is not None:
            scale_window = min(scale_window, scale_window_limit)
        if trans_window_limit is not None:
            trans_window = min(trans_window, trans_window_limit)
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

    if coarse_only and not coarse_enabled:
        raise RuntimeError("Grobausrichtung deaktiviert – coarse-only ist nicht verfügbar")

    if coarse_cfg and coarse_enabled:
        (
            coarse_align_fn,
            apply_orientation_pts_fn,
            apply_scale_shift_pts_fn,
            transform_segments_fn,
        ) = _ensure_coarse_modules()
        bbox_pdf = (0.0, 0.0, float(pdf_w), float(pdf_h))
        bbox_svg = (0.0, 0.0, float(svg_w), float(svg_h))
        coarse_pdf_segments = [
            (float(seg.x1), float(seg.y1), float(seg.x2), float(seg.y2))
            for seg in pdf_segments
        ]
        coarse_svg_segments = [
            (float(seg.x1), float(seg.y1), float(seg.x2), float(seg.y2))
            for seg in svg_segments
        ]
        coarse = None
        debug_path = Path(coarse_debug_dir) if coarse_debug_dir is not None else None
        try:
            coarse = coarse_align_fn(
                coarse_pdf_segments,
                coarse_svg_segments,
                bbox_pdf,
                bbox_svg,
                cfg_local,
                debug_dir=debug_path,
            )
        except Exception as exc:  # pragma: no cover - robustness against optional feature
            log.warning(
                "[coarse] Fehler bei Grobausrichtung (%s) – fahre mit bestehenden Defaults fort",
                exc,
            )
        if coarse_outputs is not None:
            coarse_outputs["bbox_pdf"] = bbox_pdf
            coarse_outputs["bbox_svg"] = bbox_svg
            coarse_outputs["coarse"] = coarse
        if coarse and coarse.ok and coarse.orientation and coarse.scale and coarse.shift:
            log.info(
                "[coarse] rot=%d flip=%s sx=%.6f sy=%.6f shift=(%.2f,%.2f) score=%.3f",
                coarse.orientation.rot_deg,
                coarse.orientation.flip,
                coarse.scale[0],
                coarse.scale[1],
                coarse.shift[0],
                coarse.shift[1],
                coarse.score,
            )

            def _pipe_pts(P):
                Q = apply_orientation_pts_fn(P, coarse.orientation, bbox_svg)
                Q = apply_scale_shift_pts_fn(
                    Q, coarse.scale[0], coarse.scale[1], coarse.shift[0], coarse.shift[1]
                )
                return Q

            svg_segments_np = transform_segments_fn(coarse_svg_segments, _pipe_pts)
            svg_segments = [
                Segment(
                    x1=float(values[0]),
                    y1=float(values[1]),
                    x2=float(values[2]),
                    y2=float(values[3]),
                )
                for values in svg_segments_np
            ]

            if coarse_outputs is not None:
                matrix, translation = _coarse_affine_matrix(
                    coarse.orientation,
                    (float(coarse.scale[0]), float(coarse.scale[1])),
                    (float(coarse.shift[0]), float(coarse.shift[1])),
                    bbox_svg,
                    apply_orientation_pts_fn=apply_orientation_pts_fn,
                    apply_scale_shift_pts_fn=apply_scale_shift_pts_fn,
                )
                coarse_outputs["transform_matrix"] = matrix
                coarse_outputs["translation"] = translation
                coarse_outputs["svg_segments_coarse"] = svg_segments

            refine_windows = coarse_cfg.get("refine_windows")
            if isinstance(refine_windows, Mapping):
                current_scale_step = float(ransac_cfg.get("refine_scale_step", float("inf")))
                target_scale_step = refine_windows.get("scale_rel", current_scale_step)
                try:
                    target_scale_step = float(target_scale_step)
                except (TypeError, ValueError):
                    target_scale_step = current_scale_step
                ransac_cfg["refine_scale_step"] = min(current_scale_step, target_scale_step)

                current_trans_px = float(ransac_cfg.get("refine_trans_px", float("inf")))
                target_trans_px = refine_windows.get("shift_px", current_trans_px)
                try:
                    target_trans_px = float(target_trans_px)
                except (TypeError, ValueError):
                    target_trans_px = current_trans_px
                ransac_cfg["refine_trans_px"] = min(current_trans_px, target_trans_px)

                try:
                    current_angle_tol = float(cfg_local.get("angle_tol_deg", 6.0))
                except (TypeError, ValueError):
                    current_angle_tol = 6.0
                target_angle = refine_windows.get("angle_deg", current_angle_tol)
                try:
                    target_angle = float(target_angle)
                except (TypeError, ValueError):
                    target_angle = current_angle_tol
                cfg_local["angle_tol_deg"] = min(current_angle_tol, target_angle)
        else:
            log.warning("[coarse] keine robuste Grobausrichtung – fahre mit bestehenden Defaults fort")

        if coarse_only:
            if not coarse or not coarse.ok or not coarse.orientation or not coarse.scale or not coarse.shift:
                raise RuntimeError("Grobausrichtung fehlgeschlagen – kein Ergebnis verfügbar")
            flip_x = -1.0 if coarse.orientation.flip == "x" else 1.0
            flip_y = -1.0 if coarse.orientation.flip == "y" else 1.0
            transform = Transform2D(
                flip=(flip_x, flip_y),
                rot_deg=int(coarse.orientation.rot_deg),
                sx=float(coarse.scale[0]),
                sy=float(coarse.scale[1]),
                tx=float(coarse.shift[0]),
                ty=float(coarse.shift[1]),
            )
            model = Model(
                rot_deg=int(coarse.orientation.rot_deg),
                sx=float(coarse.scale[0]),
                sy=float(coarse.scale[1]),
                tx=float(coarse.shift[0]),
                ty=float(coarse.shift[1]),
                score=float(coarse.score),
                rmse=0.0,
                p95=0.0,
                median=0.0,
                flip_x=flip_x,
                flip_y=flip_y,
                transform=transform,
                quality_notes=(),
            )
            return model

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

    def _run_refine() -> Model:
        return _calibrate_ransac(pdf_segments, svg_segments, pdf_size, svg_size, cfg_local)

    model: Model | None = None
    refine_fallback_used = False
    while True:
        try:
            with _timed("Refine"):
                model = _run_refine()
        except RuntimeError as exc:
            message = str(exc)
            if (
                refine_fallback_used
                or "Keine gültige Kalibrierung gefunden" not in message
            ):
                raise
            refine_fallback_used = True
            try:
                current_scale_window = abs(
                    float(refine_cfg.get("scale_max_dev_rel", scale_window_base_val))
                )
            except (TypeError, ValueError):
                current_scale_window = scale_window_base_val
            try:
                current_trans_window = abs(
                    float(refine_cfg.get("trans_max_dev_px", trans_window_base))
                )
            except (TypeError, ValueError):
                current_trans_window = trans_window_base
            scale_cap = 0.12
            trans_cap = 64.0
            next_scale_window = current_scale_window
            if current_scale_window < scale_cap:
                next_scale_window = min(current_scale_window * 2.0, scale_cap)
            next_trans_window = current_trans_window
            if current_trans_window < trans_cap:
                next_trans_window = min(current_trans_window * 2.0, trans_cap)
            _apply_refine_window(next_scale_window, next_trans_window)
            try:
                updated_scale_window = float(
                    refine_cfg.get("scale_max_dev_rel", next_scale_window)
                )
            except (TypeError, ValueError):
                updated_scale_window = next_scale_window
            try:
                updated_trans_window = float(
                    refine_cfg.get("trans_max_dev_px", next_trans_window)
                )
            except (TypeError, ValueError):
                updated_trans_window = next_trans_window
            log.info(
                "[refine] expand windows and retry -> trans_max_dev_px=%.1f scale_max_dev_rel=%.4f",
                updated_trans_window,
                updated_scale_window,
            )
            continue
        break

    assert model is not None

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
