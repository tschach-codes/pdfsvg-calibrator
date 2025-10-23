from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from pdfsvg_calibrator.core.grid_safety import ensure_ndarray2d, zeros2d


@dataclass
class Projections:
    Hx: np.ndarray  # horizontale Segmente auf x-bins integriert
    Hy: np.ndarray  # vertikale Segmente auf y-bins integriert
    bins: int


@dataclass
class Features:
    centers: np.ndarray  # (N,2) Mittelpunkte
    lengths: np.ndarray  # (N,)
    angles: np.ndarray   # (N,) in rad
    hv_mask: np.ndarray  # (N,2) bool: (is_h, is_v)


def segment_features(segments, hv_angle_tol_deg: float) -> Features:
    # segments: Iterable of (x1,y1,x2,y2)
    segs = ensure_ndarray2d("segments", segments).astype(float, copy=False)
    if segs.shape[1] != 4:
        raise ValueError(f"segments must have shape (N,4), got {segs.shape}")
    v = segs[:, 2:4] - segs[:, 0:2]
    lengths = np.linalg.norm(v, axis=1) + 1e-9
    angles = np.arctan2(v[:, 1], v[:, 0])  # rad
    centers = 0.5 * (segs[:, 0:2] + segs[:, 2:4])

    # hv mask
    ang = np.mod(angles, np.pi)  # in [0,pi)
    tol = np.deg2rad(hv_angle_tol_deg)
    is_h = np.minimum(ang, np.pi - ang) <= tol  # nahe 0 bzw. pi
    is_v = np.minimum(np.abs(ang - np.pi / 2), np.abs(ang + np.pi / 2)) <= tol
    hv = np.stack([is_h, is_v], axis=1)
    return Features(centers=centers, lengths=lengths, angles=angles, hv_mask=hv)


def make_projections(
    feats_pdf: Features,
    feats_svg: Features,
    bins: int,
    blur_sigma_bins: float,
    bbox_pdf,
    bbox_svg,
):
    # bbox = (xmin,ymin,xmax,ymax)
    bins = int(bins)
    def _proj(feats: Features, bbox, axis: int, is_h: bool):
        # axis: 0->x, 1->y; is_h True => horizontale summieren entlang x, sonst vertikale entlang y
        mask = feats.hv_mask[:, 0] if is_h else feats.hv_mask[:, 1]
        if not np.any(mask):
            return np.zeros(int(bins))
        pts = feats.centers[mask, axis]
        lengths = feats.lengths[mask]
        lo, hi = bbox[axis], bbox[2 + axis]
        t = np.clip((pts - lo) / max(hi - lo, 1e-6), 0, 1 - 1e-9)
        idx = (t * bins).astype(int)
        H = np.bincount(idx, weights=lengths, minlength=bins).astype(float)
        if blur_sigma_bins and blur_sigma_bins > 0:
            # simple gaussian blur via FFT
            freqs = np.fft.rfftfreq(bins)
            sigma = blur_sigma_bins / bins
            G = np.exp(-0.5 * (freqs / (sigma + 1e-9)) ** 2)
            H = np.fft.irfft(np.fft.rfft(H) * G, n=bins).real
        return H

    Hx_pdf = _proj(feats_pdf, bbox_pdf, axis=0, is_h=True)
    Hy_pdf = _proj(feats_pdf, bbox_pdf, axis=1, is_h=False)
    Hx_svg = _proj(feats_svg, bbox_svg, axis=0, is_h=True)
    Hy_svg = _proj(feats_svg, bbox_svg, axis=1, is_h=False)

    return Projections(Hx_pdf, Hy_pdf, bins), Projections(Hx_svg, Hy_svg, bins)


def quantile_scale(feats_pdf: Features, feats_svg: Features, q_lo: float, q_hi: float):
    # skaliert anhand quantiler Spannweiten der Mittelpunktverteilung
    def _span(pts):
        x, y = pts[:, 0], pts[:, 1]

        def qspan(v):
            q = np.quantile(v, [q_lo, q_hi])
            return max(q[1] - q[0], 1e-6)

        return qspan(x), qspan(y)

    sx_pdf, sy_pdf = _span(feats_pdf.centers)
    sx_svg, sy_svg = _span(feats_svg.centers)
    return (sx_pdf / sx_svg, sy_pdf / sy_svg)


def raster_heatmap(centers: np.ndarray, bbox, raster: int, blur_sigma_px: float):
    # returns heatmap normalized to [0,1]
    centers = ensure_ndarray2d("centers", centers).astype(float, copy=False)
    if centers.shape[1] != 2:
        raise ValueError(f"centers must have shape (N,2), got {centers.shape}")
    xmin, ymin, xmax, ymax = bbox
    xs = np.clip((centers[:, 0] - xmin) / (xmax - xmin + 1e-6), 0, 1 - 1e-9)
    ys = np.clip((centers[:, 1] - ymin) / (ymax - ymin + 1e-6), 0, 1 - 1e-9)
    ix = (xs * raster).astype(int)
    iy = (ys * raster).astype(int)
    H = zeros2d(raster, raster, dtype=float)
    np.add.at(H, (iy, ix), 1.0)
    if blur_sigma_px and blur_sigma_px > 0:
        fy = np.fft.fftfreq(raster)
        fx = np.fft.rfftfreq(raster)
        sig = blur_sigma_px / raster
        Gy = np.exp(-0.5 * (fy / (sig + 1e-9)) ** 2)[:, None]
        Gx = np.exp(-0.5 * (fx / (sig + 1e-9)) ** 2)[None, :]
        H = np.fft.irfft2(np.fft.rfft2(H) * Gy * Gx, s=H.shape).real
    H /= H.max() + 1e-9
    return H
