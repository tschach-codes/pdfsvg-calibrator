from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .profiles import segment_features, make_projections, quantile_scale, raster_heatmap
from .corr import xcorr_1d, Orientation, adjust_profile_for_orientation
from .geom import apply_orientation_pts, apply_scale_shift_pts, transform_segments
from .kdmatch import gate_matches


@dataclass
class CoarseResult:
    ok: bool
    orientation: Orientation|None
    scale: tuple[float,float]|None
    shift: tuple[float,float]|None
    score: float


def coarse_align(pdf_segments, svg_segments, bbox_pdf, bbox_svg, cfg) -> CoarseResult:
    if not cfg['coarse']['enabled']:
        return CoarseResult(False, None, None, None, 0.0)

    # Eskalation: anzahl langer Segmente prüfen und ggf. relaxen
    topk_levels = cfg['coarse']['adapt']['relax_topk_rel']
    hv_levels   = cfg['coarse']['adapt']['relax_hv_angle']
    min_long    = cfg['coarse']['adapt']['min_long_segments']
    chosen = None
    best = CoarseResult(False, None, None, None, -1e9)

    for topk_rel in topk_levels:
        for hv_tol in hv_levels:
            feats_pdf = segment_features(pdf_segments, hv_tol)
            feats_svg = segment_features(svg_segments, hv_tol)

            # Top-k längste
            def _topk(feats):
                k = max(1, int(len(feats.lengths)*topk_rel))
                idx = np.argsort(feats.lengths)[-k:]
                out = feats
                out.centers = feats.centers[idx]; out.lengths = feats.lengths[idx]; out.angles = feats.angles[idx]; out.hv_mask = feats.hv_mask[idx]
                return out
            fpdf = _topk(feats_pdf); fsvg = _topk(feats_svg)

            if len(fpdf.lengths) < min_long or len(fsvg.lengths) < min_long:
                continue

            Ppdf, Psvg = make_projections(
                fpdf, fsvg, cfg['coarse']['bins'], cfg['coarse']['blur_sigma_bins'],
                bbox_pdf, bbox_svg
            )

            for rot in cfg['coarse']['orientations']:
                for flip in cfg['coarse']['flips']:
                    O = Orientation(rot_deg=int(rot), flip=str(flip))
                    Hx_svg, Hy_svg = adjust_profile_for_orientation(Psvg.Hx, Psvg.Hy, O)

                    # 1D-Translation über Profil-Korrelation
                    dx, cx, _ = xcorr_1d(Ppdf.Hx, Hx_svg)
                    dy, cy, _ = xcorr_1d(Ppdf.Hy, Hy_svg)

                    # quantile-scale
                    qlo, qhi = cfg['coarse']['scale_quantiles']
                    sx, sy = quantile_scale(fpdf, fsvg, qlo, qhi)

                    # Score: Kombi aus Korrelationen + spätem Inlier-Score
                    score = cfg['coarse']['score_weights']['corr_x']*cx + cfg['coarse']['score_weights']['corr_y']*cy

                    # Grobe Transformation auf Segmente anwenden (für Inlier-Test)
                    def _pipe_pts(P):
                        P2 = apply_orientation_pts(P, O, bbox_svg)
                        P3 = apply_scale_shift_pts(P2, sx, sy, dx, dy)
                        return P3
                    svgT = transform_segments(svg_segments, _pipe_pts)

                    inlier_score, _pairs = gate_matches(
                        pdf_segments, svgT,
                        cfg['coarse']['kdtree_match']['angle_tol_deg'],
                        cfg['coarse']['kdtree_match']['len_tol_rel'],
                        cfg['coarse']['kdtree_match']['dist_tol_px'],
                        cfg['coarse']['kdtree_match']['max_pairs'],
                    )
                    score_total = score + cfg['coarse']['score_weights']['inliers']*inlier_score

                    if score_total > best.score:
                        best = CoarseResult(True, O, (sx,sy), (dx,dy), score_total)
                        chosen = dict(topk_rel=topk_rel, hv_tol=hv_tol)

    # Optionaler Heatmap-Fallback
    if not best.ok and cfg['coarse']['fallback_use_heatmap']:
        # Phase-Korrelation über 2D-Heatmaps (vereinfachter Ansatz)
        feats_pdf = segment_features(pdf_segments, cfg['coarse']['hv_angle_tol_deg'])
        feats_svg = segment_features(svg_segments, cfg['coarse']['hv_angle_tol_deg'])
        Hp = raster_heatmap(feats_pdf.centers, bbox_pdf, cfg['coarse']['heatmap']['raster'], cfg['coarse']['heatmap']['blur_sigma_px'])
        Hs = raster_heatmap(feats_svg.centers, bbox_svg, cfg['coarse']['heatmap']['raster'], cfg['coarse']['heatmap']['blur_sigma_px'])
        # einfache max-Korrelation via FFT (keine Subpixel)
        C = np.fft.ifft2(np.fft.fft2(Hp)*np.conj(np.fft.fft2(Hs))).real
        iy, ix = np.unravel_index(np.argmax(C), C.shape)
        dy = iy if iy < C.shape[0]//2 else iy - C.shape[0]
        dx = ix if ix < C.shape[1]//2 else ix - C.shape[1]
        best = CoarseResult(True, Orientation(0,'none'), (1.0,1.0), (dx,dy), float(C.max()))

    return best
