"""Utility helpers for debug raster exports."""

from __future__ import annotations

import copy
import os
from typing import Final

from lxml import etree

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
    "write_dim_debug_svg",
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


def write_dim_debug_svg(
    src_svg_path: str,
    outdir: str,
    prefix: str,
    segments_svg,
    texts_svg,
    pairs,
    *,
    raster_png: bool = True,
):
    """
    Create an overlay SVG showing:
      - all segments (thin gray),
      - text boxes (thin white outlines),
      - candidate links (yellow),
      - INLIER segments/texts (thick green),
      - label each inlier with parsed value+unit.
    Coordinates are in SVG units (same as the source SVG).
    """

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(src_svg_path, parser)
    root = tree.getroot()

    # pick viewBox / width / height to anchor overlay group
    g_overlay = etree.Element(
        "g", id="dim-debug-overlay", style="mix-blend-mode:normal"
    )

    # draw segments (all)
    for (x1, y1, x2, y2) in segments_svg:
        line = etree.Element(
            "line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2)
        )
        line.set("stroke", "#bbbbbb")
        line.set("stroke-width", "0.6")
        line.set("vector-effect", "non-scaling-stroke")
        g_overlay.append(line)

    # draw texts (boxes)
    for t in texts_svg:
        (x, y, w, h) = t.get("bbox", (0, 0, 0, 0))
        rect = etree.Element(
            "rect", x=str(x), y=str(y), width=str(w), height=str(h)
        )
        rect.set("fill", "none")
        rect.set("stroke", "#ffffff")
        rect.set("stroke-width", "0.5")
        rect.set("vector-effect", "non-scaling-stroke")
        g_overlay.append(rect)

    # mark pairs and inliers
    for p in pairs:
        i = p["seg"]
        j = p["txt"]
        inl = bool(p.get("inlier", False))
        (x1, y1, x2, y2) = segments_svg[i]
        cx, cy = texts_svg[j].get("center", (None, None))
        if cx is not None and cy is not None:
            link = etree.Element(
                "line",
                x1=str((x1 + x2) / 2.0),
                y1=str((y1 + y2) / 2.0),
                x2=str(cx),
                y2=str(cy),
            )
            link.set("stroke", "#ffd500")  # yellow
            link.set("stroke-width", "0.5")
            link.set("stroke-dasharray", "2,2")
            link.set("vector-effect", "non-scaling-stroke")
            g_overlay.append(link)
        if inl:
            seg = etree.Element(
                "line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2)
            )
            seg.set("stroke", "#00ff88")
            seg.set("stroke-width", "1.2")
            seg.set("vector-effect", "non-scaling-stroke")
            g_overlay.append(seg)
            # label
            val = p.get("value")
            unit = p.get("unit", "m")
            lx = (x1 + x2) / 2.0
            ly = (y1 + y2) / 2.0
            txt = etree.Element("text", x=str(lx), y=str(ly))
            if val is not None:
                try:
                    txt.text = f"{float(val):g} {unit}"
                except (TypeError, ValueError):
                    txt.text = f"{val} {unit}" if unit else f"{val}"
            else:
                txt.text = unit
            txt.set("fill", "#00ff88")
            txt.set("font-size", "6")
            txt.set("text-anchor", "middle")
            g_overlay.append(txt)

    root.append(copy.deepcopy(g_overlay))

    os.makedirs(outdir, exist_ok=True)
    out_svg = os.path.join(outdir, f"{prefix}_dim_debug.svg")
    tree.write(out_svg, pretty_print=False, xml_declaration=True, encoding="utf-8")

    # optional raster
    if raster_png:
        try:
            import cairosvg

            out_png = os.path.join(outdir, f"{prefix}_dim_debug.png")
            cairosvg.svg2png(url=out_svg, write_to=out_png, dpi=144)
        except Exception as e:  # pragma: no cover - optional
            print("[pdfsvg] dim-debug raster failed:", e)
