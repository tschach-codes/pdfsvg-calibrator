"""Utility helpers for debug raster exports."""

from __future__ import annotations

from typing import Final

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import imageio.v2 as iio
except Exception:  # pragma: no cover - optional dependency
    iio = None  # type: ignore[assignment]

import numpy as np

__all__ = [
    "edges",
    "make_split_red_left_pdf_green_right_svg",
    "save_debug_rasters",
]


def edges(x: np.ndarray) -> np.ndarray:
    """Compute canny edges as white-on-black uint8 raster."""
    if cv2 is None:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV (cv2) ist für Debug-Kanten erforderlich")
    e = cv2.Canny(x, 50, 150)
    return (e > 0).astype(np.uint8) * 255


def _pad_to(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """Pad a gray image to (H, W) with zeros (top-left anchored)."""
    h, w = img.shape[:2]
    canvas = np.zeros((H, W), dtype=np.uint8)
    canvas[:h, :w] = img
    return canvas


def make_split_red_left_pdf_green_right_svg(
    pdf_gray: np.ndarray, svg_gray: np.ndarray
) -> np.ndarray:
    """Build a split-view RGB raster from coarse-aligned grayscale inputs."""
    assert pdf_gray.ndim == 2 and svg_gray.ndim == 2, "expected grayscale arrays"
    H = max(pdf_gray.shape[0], svg_gray.shape[0])
    W = max(pdf_gray.shape[1], svg_gray.shape[1])
    pdf_p = _pad_to(pdf_gray, H, W)
    svg_p = _pad_to(svg_gray, H, W)
    half = W // 2
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Links: PDF→rot
    rgb[:, :half, 0] = pdf_p[:, :half]
    # Rechts: SVG→grün
    rgb[:, half:, 1] = svg_p[:, half:]
    return rgb


def save_debug_rasters(
    pdf_gray: np.ndarray,
    svg_gray: np.ndarray,
    svg_oriented_no_scale: np.ndarray,
    outdir: str,
    prefix: str = "dbg",
) -> None:
    """Persist helper rasters and overlays for coarse alignment debugging."""
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

    # Split-View: links PDF (rot), rechts SVG (grün) – NACH Skalierung/Orientierung
    try:
        split = make_split_red_left_pdf_green_right_svg(pdf_gray, svg_oriented_no_scale)
        filename: Final[str] = f"{outdir}/{prefix}_split_red_left_pdf_green_right_svg.png"
        if iio is not None:
            iio.imwrite(filename, split)
        else:  # pragma: no cover - optional dependency
            cv2.imwrite(filename, cv2.cvtColor(split, cv2.COLOR_RGB2BGR))
    except Exception as e:  # pragma: no cover - debug helper
        print("[pdfsvg] split-debug failed:", e)
