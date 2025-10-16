from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geom import classify_hv
from .types import Model, Segment


log = logging.getLogger(__name__)


@dataclass
class SegGrid:
    cell_size: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    cells: Dict[Tuple[int, int], List[Segment]]

    def query(self, x: float, y: float) -> Sequence[Segment]:
        if not self.cells:
            return []
        ix = math.floor((x - self.min_x) / self.cell_size)
        iy = math.floor((y - self.min_y) / self.cell_size)
        result: List[Segment] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (ix + dx, iy + dy)
                segs = self.cells.get(key)
                if segs:
                    result.extend(segs)
        return result


def build_seg_grid(svg_segs: Sequence[Segment], cell_size: float) -> SegGrid:
    if not svg_segs:
        raise ValueError("Keine SVG-Segmente zum Aufbau des Grids verfügbar")
    if cell_size <= 0:
        raise ValueError("cell_size muss größer als 0 sein")
    min_x = min(min(seg.x1, seg.x2) for seg in svg_segs)
    max_x = max(max(seg.x1, seg.x2) for seg in svg_segs)
    min_y = min(min(seg.y1, seg.y2) for seg in svg_segs)
    max_y = max(max(seg.y1, seg.y2) for seg in svg_segs)
    cells: Dict[Tuple[int, int], List[Segment]] = {}
    inv = 1.0 / cell_size
    for seg in svg_segs:
        sx0 = math.floor((min(seg.x1, seg.x2) - min_x) * inv)
        sx1 = math.floor((max(seg.x1, seg.x2) - min_x) * inv)
        sy0 = math.floor((min(seg.y1, seg.y2) - min_y) * inv)
        sy1 = math.floor((max(seg.y1, seg.y2) - min_y) * inv)
        for ix in range(int(sx0), int(sx1) + 1):
            for iy in range(int(sy0), int(sy1) + 1):
                cells.setdefault((ix, iy), []).append(seg)
    return SegGrid(cell_size=cell_size, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, cells=cells)


def _segment_length(seg: Segment) -> float:
    return math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)


def _segment_midpoint(seg: Segment) -> Tuple[float, float]:
    return ((seg.x1 + seg.x2) * 0.5, (seg.y1 + seg.y2) * 0.5)


def _point_segment_distance(x: float, y: float, seg: Segment) -> float:
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(x - seg.x1, y - seg.y1)
    t = ((x - seg.x1) * dx + (y - seg.y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = seg.x1 + t * dx
    proj_y = seg.y1 + t * dy
    return math.hypot(x - proj_x, y - proj_y)


def _rotate_point(x: float, y: float, angle_deg: float, center: Tuple[float, float]) -> Tuple[float, float]:
    cx, cy = center
    rad = math.radians(angle_deg % 360.0)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    tx = x - cx
    ty = y - cy
    rx = cos_a * tx - sin_a * ty
    ry = sin_a * tx + cos_a * ty
    return rx + cx, ry + cy


def _rotate_segment(seg: Segment, angle_deg: float, center: Tuple[float, float]) -> Segment:
    x1, y1 = _rotate_point(seg.x1, seg.y1, angle_deg, center)
    x2, y2 = _rotate_point(seg.x2, seg.y2, angle_deg, center)
    return Segment(x1=x1, y1=y1, x2=x2, y2=y2)


def sample_points_on_segments(segs: Sequence[Segment], step: float, max_pts: int) -> List[Tuple[float, float]]:
    if step <= 0:
        raise ValueError("sampling step muss > 0 sein")
    points: List[Tuple[float, float]] = []
    for seg in segs:
        length = _segment_length(seg)
        if length == 0.0:
            continue
        intervals = max(int(math.ceil(length / step)), 1)
        for i in range(intervals + 1):
            if len(points) >= max_pts:
                return points
            t = i / intervals
            x = seg.x1 + (seg.x2 - seg.x1) * t
            y = seg.y1 + (seg.y2 - seg.y1) * t
            points.append((x, y))
        if len(points) >= max_pts:
            break
    return points


def _min_distance_to_grid(x: float, y: float, grid: SegGrid, hard: float) -> float:
    candidates = grid.query(x, y)
    best = hard
    for seg in candidates:
        d = _point_segment_distance(x, y, seg)
        if d < best:
            best = d
    return best


def chamfer_score(
    pts_pdf: Sequence[Tuple[float, float]],
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    svg_grid: SegGrid,
    sigma: float,
    hard: float,
) -> float:
    if not pts_pdf:
        return 0.0
    if sigma <= 0:
        raise ValueError("sigma muss > 0 sein")
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    total = 0.0
    for px, py in pts_pdf:
        x = sx * px + tx
        y = sy * py + ty
        d = _min_distance_to_grid(x, y, svg_grid, hard)
        if d >= hard:
            continue
        total += math.exp(-(d * d) * inv_two_sigma_sq)
    return total / len(pts_pdf)


def residual_stats(
    pts_pdf: Sequence[Tuple[float, float]],
    model: Model,
    svg_grid: SegGrid,
    hard: float,
) -> Tuple[float, float, float]:
    if not pts_pdf:
        raise ValueError("Es wurden keine Stichprobenpunkte übergeben")
    dists = []
    for px, py in pts_pdf:
        x = model.sx * px + model.tx
        y = model.sy * py + model.ty
        d = _min_distance_to_grid(x, y, svg_grid, hard)
        dists.append(d)
    arr = np.asarray(dists, dtype=float)
    rmse = float(np.sqrt(np.mean(arr * arr)))
    p95 = float(np.percentile(arr, 95))
    median = float(np.median(arr))
    return rmse, p95, median


def _ensure_hv_non_empty(side: str, h: Sequence[Segment], v: Sequence[Segment]) -> None:
    missing = []
    if not h:
        missing.append("horizontal")
    if not v:
        missing.append("vertikal")
    if missing:
        raise ValueError(f"Keine {', '.join(missing)}en Segmente auf der {side}-Seite gefunden")


def calibrate(
    pdf_segs: Sequence[Segment],
    svg_segs: Sequence[Segment],
    pdf_size: Tuple[float, float],
    svg_size: Tuple[float, float],
    cfg: dict,
) -> Model:
    start_total = perf_counter()
    if not pdf_segs:
        raise ValueError("Keine PDF-Segmente für die Kalibrierung übergeben")
    if not svg_segs:
        raise ValueError("Keine SVG-Segmente für die Kalibrierung übergeben")

    diag_pdf = math.hypot(*pdf_size)
    diag_svg = math.hypot(*svg_size)
    if diag_pdf == 0.0 or diag_svg == 0.0:
        raise ValueError("Seitendiagonalen dürfen nicht 0 sein")

    grid_cell = cfg.get("grid_cell_rel", 0.02) * diag_svg
    grid_start = perf_counter()
    svg_grid = build_seg_grid(svg_segs, grid_cell)
    grid_duration = perf_counter() - grid_start
    log.debug(
        "[calib] Chamfer-Grid gebaut: cell=%.3f (%d Segmente, %d Rasterzellen) in %.3fs",
        grid_cell,
        len(svg_segs),
        len(svg_grid.cells),
        grid_duration,
    )

    chamfer_cfg = cfg.get("chamfer", {})
    sigma = chamfer_cfg.get("sigma_rel", 0.004) * diag_svg
    hard = sigma * chamfer_cfg.get("hard_mul", 3.0)

    sampling_cfg = cfg.get("sampling", {})
    step = sampling_cfg.get("step_rel", 0.02) * diag_pdf
    max_pts = max(1, int(sampling_cfg.get("max_points", 5000)))

    ransac_cfg = cfg.get("ransac", {})
    iters = int(ransac_cfg.get("iters", 900))
    refine_scale_step = ransac_cfg.get("refine_scale_step", 0.004)
    refine_trans_px = ransac_cfg.get("refine_trans_px", 3.0)

    rng_seed = cfg.get("rng_seed")
    rng = random.Random(rng_seed)

    angle_tol = cfg.get("angle_tol_deg", 6.0)

    svg_h, svg_v = classify_hv(list(svg_segs), angle_tol)
    _ensure_hv_non_empty("SVG", svg_h, svg_v)

    debug_cfg = cfg.get("debug", {})
    enable_chamfer_stats = bool(
        debug_cfg.get("chamfer_stats", log.isEnabledFor(logging.DEBUG))
    )

    chamfer_stats = {"calls": 0, "time": 0.0}

    if enable_chamfer_stats:

        def eval_chamfer(
            pts_pdf: Sequence[Tuple[float, float]],
            sx: float,
            sy: float,
            tx: float,
            ty: float,
            svg_grid: SegGrid,
            sigma: float,
            hard: float,
        ) -> float:
            start = perf_counter()
            try:
                return chamfer_score(pts_pdf, sx, sy, tx, ty, svg_grid, sigma, hard)
            finally:
                chamfer_stats["calls"] += 1
                chamfer_stats["time"] += perf_counter() - start

    else:

        def eval_chamfer(
            pts_pdf: Sequence[Tuple[float, float]],
            sx: float,
            sy: float,
            tx: float,
            ty: float,
            svg_grid: SegGrid,
            sigma: float,
            hard: float,
        ) -> float:
            return chamfer_score(pts_pdf, sx, sy, tx, ty, svg_grid, sigma, hard)

    best_model: Model | None = None
    best_score = -math.inf
    center_pdf = (pdf_size[0] * 0.5, pdf_size[1] * 0.5)
    rot_degrees = cfg.get("rot_degrees", [0, 180])
    offsets = [0.0]
    if refine_trans_px > 0.0:
        offsets = [-refine_trans_px, 0.0, refine_trans_px]

    iter_log_interval = int(debug_cfg.get("iter_log_interval", 50))
    if iter_log_interval <= 0:
        iter_log_interval = 50

    for rot_deg in rot_degrees:
        rot_start = perf_counter()
        rotated_pdf = [_rotate_segment(seg, rot_deg, center_pdf) for seg in pdf_segs]
        pdf_h, pdf_v = classify_hv(rotated_pdf, angle_tol)
        _ensure_hv_non_empty("PDF", pdf_h, pdf_v)

        log.debug(
            "[calib] rot=%s klassifiziert (%d horizontale, %d vertikale Segmente)",
            rot_deg,
            len(pdf_h),
            len(pdf_v),
        )

        pts_pdf = sample_points_on_segments(rotated_pdf, step, max_pts)
        if not pts_pdf:
            raise ValueError("Es konnten keine Stichprobenpunkte aus den PDF-Segmenten generiert werden")

        log.debug(
            "[calib] rot=%s Sampling: %d Punkte (step=%.3f, max=%d)",
            rot_deg,
            len(pts_pdf),
            step,
            max_pts,
        )

        chamfer_calls_before = chamfer_stats["calls"]
        chamfer_time_before = chamfer_stats["time"]
        executed_iters = 0
        rot_best_score = -math.inf

        for iter_idx in range(iters):
            executed_iters += 1
            ph = rng.choice(pdf_h)
            pv = rng.choice(pdf_v)
            sh = rng.choice(svg_h)
            sv = rng.choice(svg_v)

            len_ph = _segment_length(ph)
            len_pv = _segment_length(pv)
            len_sh = _segment_length(sh)
            len_sv = _segment_length(sv)
            if len_ph == 0.0 or len_pv == 0.0:
                continue

            base_sx = len_sh / len_ph
            base_sy = len_sv / len_pv

            mph = _segment_midpoint(ph)
            mpv = _segment_midpoint(pv)
            msh = _segment_midpoint(sh)
            msv = _segment_midpoint(sv)

            for sign_x in (-1.0, 1.0):
                sx = sign_x * base_sx
                for sign_y in (-1.0, 1.0):
                    sy = sign_y * base_sy
                    tx_candidates = [msh[0] - sx * mph[0], msv[0] - sx * mpv[0]]
                    ty_candidates = [msh[1] - sy * mph[1], msv[1] - sy * mpv[1]]
                    tx = sum(tx_candidates) / len(tx_candidates)
                    ty = sum(ty_candidates) / len(ty_candidates)

                    score = eval_chamfer(pts_pdf, sx, sy, tx, ty, svg_grid, sigma, hard)
                    if score > rot_best_score:
                        rot_best_score = score
                    if score <= best_score:
                        continue

                    best_local = (sx, sy, tx, ty, score)

                    for dk in (-2, -1, 0, 1, 2):
                        sx_ref = sx * (1.0 + dk * refine_scale_step)
                        if abs(sx_ref) < 1e-9:
                            continue
                        for dl in (-2, -1, 0, 1, 2):
                            sy_ref = sy * (1.0 + dl * refine_scale_step)
                            if abs(sy_ref) < 1e-9:
                                continue
                            for dx in offsets:
                                for dy in offsets:
                                    tx_ref = tx + dx
                                    ty_ref = ty + dy
                                    score_ref = eval_chamfer(
                                        pts_pdf,
                                        sx_ref,
                                        sy_ref,
                                        tx_ref,
                                        ty_ref,
                                        svg_grid,
                                        sigma,
                                        hard,
                                    )
                                    if score_ref > best_local[4]:
                                        best_local = (sx_ref, sy_ref, tx_ref, ty_ref, score_ref)
                                    if score_ref > rot_best_score:
                                        rot_best_score = score_ref

                    if best_local[4] > best_score:
                        rmse, p95, median = residual_stats(
                            pts_pdf,
                            Model(rot_deg=rot_deg, sx=best_local[0], sy=best_local[1], tx=best_local[2], ty=best_local[3], score=best_local[4], rmse=0.0, p95=0.0, median=0.0),
                            svg_grid,
                            hard,
                        )
                        best_score = best_local[4]
                        best_model = Model(
                            rot_deg=rot_deg,
                            sx=best_local[0],
                            sy=best_local[1],
                            tx=best_local[2],
                            ty=best_local[3],
                            score=best_local[4],
                            rmse=rmse,
                            p95=p95,
                            median=median,
                        )
            if (iter_idx + 1) % iter_log_interval == 0:
                calls = chamfer_stats["calls"] - chamfer_calls_before
                duration = chamfer_stats["time"] - chamfer_time_before
                log.debug(
                    "[calib] rot=%s Iteration %d/%d – rot_best=%s, global_best=%s, Chamfer=%d Aufrufe (%.3fs)",
                    rot_deg,
                    iter_idx + 1,
                    iters,
                    "-inf" if rot_best_score == -math.inf else f"{rot_best_score:.6f}",
                    "-inf" if best_score == -math.inf else f"{best_score:.6f}",
                    calls,
                    duration,
                )

        calls = chamfer_stats["calls"] - chamfer_calls_before
        duration = chamfer_stats["time"] - chamfer_time_before
        rot_duration = perf_counter() - rot_start
        log.debug(
            "[calib] rot=%s fertig nach %.3fs – %d Iterationen, best=%s, Chamfer=%d (%.3fs)",
            rot_deg,
            rot_duration,
            executed_iters,
            "-inf" if rot_best_score == -math.inf else f"{rot_best_score:.6f}",
            calls,
            duration,
        )

    if best_model is None or best_model.score <= 0.0:
        total_duration = perf_counter() - start_total
        log.debug(
            "[calib] keine Lösung gefunden (%.3fs, Chamfer=%d Aufrufe, %.3fs)",
            total_duration,
            chamfer_stats["calls"],
            chamfer_stats["time"],
        )
        raise RuntimeError("Keine gültige Kalibrierung gefunden – zu wenig Struktur oder rot_degrees prüfen")

    total_duration = perf_counter() - start_total
    log.debug(
        "[calib] abgeschlossen in %.3fs – rot=%s, score=%.6f, Chamfer=%d Aufrufe (%.3fs)",
        total_duration,
        best_model.rot_deg,
        best_model.score,
        chamfer_stats["calls"],
        chamfer_stats["time"],
    )

    return best_model
