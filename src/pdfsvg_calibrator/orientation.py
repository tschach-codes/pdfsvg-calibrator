"""Light-weight orientation and translation seeding utilities."""

from __future__ import annotations

import contextlib
import math
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .types import Segment
from .utils.timer import timer


try:  # pragma: no cover - optional dependency
    from skimage.draw import line as _skimage_draw_line  # type: ignore
except Exception:  # pragma: no cover - fallback
    _skimage_draw_line = None

try:  # pragma: no cover - optional dependency
    from skimage.registration import phase_cross_correlation as _skimage_pcc  # type: ignore
except Exception:  # pragma: no cover - fallback
    _skimage_pcc = None


SegmentInfo = Tuple[Tuple[float, float], Tuple[float, float], float, float]


def snap_axis(theta_deg: float, tol_deg: float = 0.5) -> int:
    """Return 0 for horizontal, 1 for vertical, or -1 when not aligned."""

    a = theta_deg % 180.0
    if min(a, 180.0 - a) <= tol_deg:
        return 0
    if min(abs(a - 90.0), abs(a - 90.0 - 180.0)) <= tol_deg:
        return 1
    return -1


def select_topk_longest(
    segments: Sequence[SegmentInfo], k_rel: float = 0.2
) -> List[SegmentInfo]:
    """Pick the longest ``k_rel`` fraction (min 16) of input segments."""

    if not segments:
        return []
    arr = np.array(
        [(s[0][0], s[0][1], s[1][0], s[1][1], s[3]) for s in segments],
        dtype=float,
    )
    if arr.size == 0:
        return []
    k = max(16, int(len(arr) * k_rel))
    idx = np.argsort(arr[:, 4])[::-1][:k]
    return [segments[i] for i in idx]


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> Tuple[np.ndarray, np.ndarray]:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r_step = 1 if r0 <= r1 else -1
    c_step = 1 if c0 <= c1 else -1
    rr: List[int] = []
    cc: List[int] = []
    err = dr - dc
    r, c = r0, c0
    while True:
        rr.append(r)
        cc.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += r_step
        if e2 < dr:
            err += dr
            c += c_step
    return np.array(rr, dtype=int), np.array(cc, dtype=int)


def _apply_flip_rot(
    segment: SegmentInfo,
    page_w: float,
    page_h: float,
    flip: Tuple[float, float],
    rot_deg: int,
) -> SegmentInfo:
    """Apply flip/rotation about the page centre and return new segment info."""

    (x1, y1), (x2, y2), _, length = segment
    cx, cy = page_w * 0.5, page_h * 0.5
    ang = math.radians(rot_deg % 360)
    ca, sa = math.cos(ang), math.sin(ang)

    def _map(x: float, y: float) -> Tuple[float, float]:
        xr, yr = x - cx, y - cy
        xr *= flip[0]
        yr *= flip[1]
        xrr = xr * ca - yr * sa
        yrr = xr * sa + yr * ca
        return (xrr + cx, yrr + cy)

    p1 = _map(x1, y1)
    p2 = _map(x2, y2)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.degrees(math.atan2(dy, dx)) % 360.0
    return (p1, p2, theta, length)


def rasterize_segments(
    segments: Sequence[SegmentInfo],
    page_w: float,
    page_h: float,
    W: int = 256,
    H: int = 256,
    flip: Tuple[float, float] = (1, 1),
    rot_deg: int = 0,
) -> np.ndarray:
    """Draw thin lines into a binary mask using Bresenham."""

    mask = np.zeros((H, W), dtype=np.uint8)
    if not segments:
        return mask

    cx, cy = page_w * 0.5, page_h * 0.5
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)

    def xform(x: float, y: float) -> Tuple[int, int]:
        xr, yr = x - cx, y - cy
        xr, yr = xr * flip[0], yr * flip[1]
        xrr = xr * ca - yr * sa
        yrr = xr * sa + yr * ca
        xm, ym = xrr + cx, yrr + cy
        u = int(round(xm * (W / page_w)))
        v = int(round((page_h - ym) * (H / page_h)))
        return u, v

    for (p1, p2, _, L) in segments:
        u1, v1 = xform(p1[0], p1[1])
        u2, v2 = xform(p2[0], p2[1])
        if _skimage_draw_line is not None:
            rr, cc = _skimage_draw_line(v1, u1, v2, u2)
        else:
            rr, cc = _bresenham_line(v1, u1, v2, u2)
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        mask[rr, cc] = 255
    return mask


def _histogram(
    values: Sequence[float],
    page_extent: float,
    bin_px: float,
) -> np.ndarray:
    if page_extent <= 0 or not values:
        return np.zeros(1, dtype=float)
    bins = max(16, int(math.ceil(page_extent / max(bin_px, 1e-6))))
    hist, _ = np.histogram(values, bins=bins, range=(0.0, page_extent))
    return hist.astype(float, copy=False)


def x_histogram_of_verticals(
    segments: Sequence[SegmentInfo], bin_w_px: float, page_w: float, tol_deg: float = 2.0
) -> np.ndarray:
    """Histogram X positions of vertical-aligned segments."""

    xs: List[float] = []
    for p1, p2, theta_deg, _ in segments:
        if snap_axis(theta_deg, tol_deg) != 1:
            continue
        xs.append(0.5 * (p1[0] + p2[0]))
    return _histogram(xs, page_w, bin_w_px)


def y_histogram_of_horizontals(
    segments: Sequence[SegmentInfo], bin_h_px: float, page_h: float, tol_deg: float = 2.0
) -> np.ndarray:
    """Histogram Y positions of horizontal-aligned segments."""

    ys: List[float] = []
    for p1, p2, theta_deg, _ in segments:
        if snap_axis(theta_deg, tol_deg) != 0:
            continue
        ys.append(0.5 * (p1[1] + p2[1]))
    return _histogram(ys, page_h, bin_h_px)


def argmax_xcorr(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    """Return shift and normalized response to align ``b`` onto ``a``."""

    if len(a) == 0 or len(b) == 0:
        return 0, 0.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0, 0.0
    c = np.correlate(a - a.mean(), b - b.mean(), mode="full")
    shift = int(np.argmax(c) - (len(b) - 1))
    denom = (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-9)
    return shift, float(c.max() / denom)


def phase_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Wrapper around :func:`skimage.registration.phase_cross_correlation`."""
    if _skimage_pcc is not None:
        shift, response, _ = _skimage_pcc(a, b, upsample_factor=1)
        dv, du = float(shift[0]), float(shift[1])
        return float(-du), float(-dv), float(response)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    axes = tuple(range(a.ndim))
    Fa = np.fft.rfftn(a, axes=axes)
    Fb = np.fft.rfftn(b, axes=axes)
    R = Fa * np.conj(Fb)
    denom = np.abs(R)
    denom[denom == 0] = 1.0
    R /= denom
    corr = np.fft.irfftn(R, s=a.shape, axes=axes)
    max_pos = np.unravel_index(np.argmax(corr), corr.shape)
    shifts = np.array(max_pos, dtype=float)
    for dim, size in enumerate(a.shape):
        if shifts[dim] > size // 2:
            shifts[dim] -= size
    response = float(corr.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    dv, du = shifts[0], shifts[1]
    return float(-du), float(-dv), response


def _segments_to_info(
    segments: Iterable[Segment | Sequence[float] | SegmentInfo],
) -> List[SegmentInfo]:
    result: List[SegmentInfo] = []
    for seg in segments:
        if isinstance(seg, Segment):
            x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            theta_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        elif len(seg) == 4 and isinstance(seg[0], tuple):
            (x1, y1), (x2, y2), theta_deg, length = seg  # type: ignore[misc]
        else:
            x1, y1, x2, y2 = seg  # type: ignore[misc]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            theta_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        result.append(((float(x1), float(y1)), (float(x2), float(y2)), float(theta_deg), float(length)))
    return result


def _scale_segments(
    segments: Sequence[SegmentInfo],
    scale_x: float,
    scale_y: float,
) -> List[SegmentInfo]:
    scaled: List[SegmentInfo] = []
    for (p1, p2, _, _) in segments:
        x1, y1 = p1[0] * scale_x, p1[1] * scale_y
        x2, y2 = p2[0] * scale_x, p2[1] * scale_y
        dx = x2 - x1
        dy = y2 - y1
        theta = math.degrees(math.atan2(dy, dx)) % 360.0
        length = math.hypot(dx, dy)
        scaled.append(((x1, y1), (x2, y2), theta, length))
    return scaled


def _transform_segments(
    segments: Sequence[SegmentInfo],
    page_w: float,
    page_h: float,
    flip: Tuple[float, float],
    rot_deg: int,
) -> List[SegmentInfo]:
    return [_apply_flip_rot(seg, page_w, page_h, flip, rot_deg) for seg in segments]


def pick_flip_rot_and_shift(
    pdf_segs: Iterable[Sequence[float] | SegmentInfo],
    svg_segs: Iterable[Sequence[float] | SegmentInfo],
    page_pdf,
    page_svg,
    cfg,
    *,
    stats: MutableMapping[str, float] | None = None,
):
    """Pick best orientation and translation seed between PDF and SVG segments."""

    orientation_cfg: Mapping[str, float | int | bool] | dict
    orientation_cfg = getattr(cfg, "orientation", None)
    if orientation_cfg is None:
        orientation_cfg = cfg.get("orientation", {}) if isinstance(cfg, dict) else {}

    sample_topk_rel = orientation_cfg.get("sample_topk_rel", 0.2)
    raster_size = int(orientation_cfg.get("raster_size", 256))
    min_accept = float(orientation_cfg.get("min_accept_score", 0.05))
    hist_bin_px = float(orientation_cfg.get("hist_bin_px", 4.0))
    axis_tol = float(orientation_cfg.get("axis_snap_tol_deg", 2.0))

    pdf_infos = _segments_to_info(pdf_segs)
    svg_infos_raw = _segments_to_info(svg_segs)

    scale_x = page_pdf.w / page_svg.w if getattr(page_svg, "w", 0.0) else 1.0
    scale_y = page_pdf.h / page_svg.h if getattr(page_svg, "h", 0.0) else 1.0
    svg_infos = _scale_segments(svg_infos_raw, scale_x, scale_y)

    pdf_top = select_topk_longest(pdf_infos, sample_topk_rel)
    svg_top = select_topk_longest(svg_infos, sample_topk_rel)

    pdf_hist_x = x_histogram_of_verticals(pdf_top, hist_bin_px, page_pdf.w, axis_tol)
    pdf_hist_y = y_histogram_of_horizontals(pdf_top, hist_bin_px, page_pdf.h, axis_tol)

    candidates = []
    for rot in (0, 180):
        for flip in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            dx_doc = 0.0
            dy_doc = 0.0
            used_hist_x = False
            used_hist_y = False
            responses: List[float] = []

            svg_trans = _transform_segments(svg_top, page_pdf.w, page_pdf.h, flip, rot)
            svg_hist_x = x_histogram_of_verticals(svg_trans, hist_bin_px, page_pdf.w, axis_tol)
            svg_hist_y = y_histogram_of_horizontals(svg_trans, hist_bin_px, page_pdf.h, axis_tol)

            if pdf_hist_x.size > 1 and svg_hist_x.size > 1:
                shift_x, resp_x = argmax_xcorr(pdf_hist_x, svg_hist_x)
                bin_w = page_pdf.w / max(len(pdf_hist_x), 1)
                dx_doc = shift_x * bin_w
                responses.append(resp_x)
                used_hist_x = True

            if pdf_hist_y.size > 1 and svg_hist_y.size > 1:
                shift_y, resp_y = argmax_xcorr(pdf_hist_y, svg_hist_y)
                bin_h = page_pdf.h / max(len(pdf_hist_y), 1)
                dy_doc = shift_y * bin_h
                responses.append(resp_y)
                used_hist_y = True

            seed_ctx = timer(stats, "Seed") if stats is not None else contextlib.nullcontext()
            with seed_ctx:
                A = rasterize_segments(
                    pdf_top, page_pdf.w, page_pdf.h, raster_size, raster_size, (1, 1), 0
                )
                B = rasterize_segments(
                    svg_top, page_pdf.w, page_pdf.h, raster_size, raster_size, flip, rot
                )

                sx_canvas = raster_size / page_pdf.w if page_pdf.w else 1.0
                sy_canvas = raster_size / page_pdf.h if page_pdf.h else 1.0

                du = dx_doc * sx_canvas
                dv = -dy_doc * sy_canvas

                phase_resp = 0.0
                if not (used_hist_x and used_hist_y):
                    du_pc, dv_pc, phase_resp = phase_correlation(B, A)
                    if not used_hist_x:
                        dx_doc = du_pc / sx_canvas
                        du = du_pc
                    if not used_hist_y:
                        dy_doc = -dv_pc / sy_canvas
                        dv = dv_pc
                    responses.append(phase_resp)

                overlap = 0.0
                if raster_size > 0:
                    shifted = np.roll(np.roll(B, int(round(du)), axis=1), int(round(dv)), axis=0)
                    overlap = float((A & shifted).mean() / 255.0)

            response_score = float(np.mean(responses)) if responses else 0.0

            F = np.array([[flip[0], 0.0], [0.0, flip[1]]], dtype=float)
            a = math.radians(rot)
            ca, sa = math.cos(a), math.sin(a)
            R = np.array([[ca, -sa], [sa, ca]], dtype=float)
            M = F @ R
            t_seed = M.T @ np.array([dx_doc, dy_doc], dtype=float)

            norm_shift = (abs(dx_doc) / (page_pdf.w or 1.0)) + (
                abs(dy_doc) / (page_pdf.h or 1.0)
            )

            candidates.append(
                {
                    "overlap": overlap,
                    "response": response_score,
                    "rot_deg": rot,
                    "flip": flip,
                    "dx_doc": float(dx_doc),
                    "dy_doc": float(dy_doc),
                    "du_dv": (float(du), float(dv)),
                    "t_seed": (float(t_seed[0]), float(t_seed[1])),
                    "phase_response": phase_resp,
                    "norm_shift": float(norm_shift),
                }
            )

    candidates.sort(
        key=lambda item: (item["overlap"] - 0.25 * item.get("norm_shift", 0.0), item["response"]),
        reverse=True,
    )
    best = candidates[0] if candidates else {
        "flip": (1, 1),
        "rot_deg": 0,
        "dx_doc": 0.0,
        "dy_doc": 0.0,
        "du_dv": (0.0, 0.0),
        "t_seed": (0.0, 0.0),
        "overlap": 0.0,
        "response": 0.0,
    }

    best["min_accept_score"] = min_accept
    best["widen_window"] = best.get("overlap", 0.0) < min_accept
    best.setdefault("phase_response", 0.0)
    return best

