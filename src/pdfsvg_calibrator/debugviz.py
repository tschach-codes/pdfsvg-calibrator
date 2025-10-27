from __future__ import annotations

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

import numpy as np


def edges(x: np.ndarray) -> np.ndarray:
    if cv2 is None:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV (cv2) ist für Debug-Kanten erforderlich")
    e = cv2.Canny(x, 50, 150)
    return (e > 0).astype(np.uint8) * 255


def save_debug_rasters(
    pdf_gray: np.ndarray,
    svg_gray: np.ndarray,
    svg_oriented_no_scale: np.ndarray,
    outdir: str,
    prefix: str = "dbg",
) -> None:
    if cv2 is None:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV (cv2) ist für Debug-Exports erforderlich")
    cv2.imwrite(f"{outdir}/{prefix}_pdf_coarse.png", pdf_gray)
    cv2.imwrite(f"{outdir}/{prefix}_svg_coarse_raw.png", svg_gray)
    cv2.imwrite(f"{outdir}/{prefix}_svg_coarse_oriented.png", svg_oriented_no_scale)

    pe, se = edges(pdf_gray), edges(svg_oriented_no_scale)
    overlay = np.zeros((pe.shape[0], pe.shape[1], 3), np.uint8)
    overlay[..., 2] = pe
    overlay[..., 1] = se
    cv2.imwrite(f"{outdir}/{prefix}_overlay_edges.png", overlay)

    diff = cv2.absdiff(pe, se)
    cv2.imwrite(f"{outdir}/{prefix}_edges_diff.png", diff)
