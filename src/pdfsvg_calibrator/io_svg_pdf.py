from __future__ import annotations

import logging
import math
import os
import xml.etree.ElementTree as ET
from time import perf_counter
from typing import List, Sequence, Tuple

import fitz

from .geom import fit_straight_segment
from .pdfium_extract import extract_segments as pdfium_extract_segments
from .svg_path import parse_svg_segments
from .types import Segment


log = logging.getLogger(__name__)

PT_PER_PX = 72.0 / 96.0


def _angle_spread_deg(points: Sequence[Tuple[float, float]]) -> float:
    angles: List[float] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        if angle < 0:
            angle += 180.0
        angles.append(angle)

    if not angles:
        return 0.0

    angles.sort()
    n = len(angles)
    extended = angles + [a + 180.0 for a in angles]
    min_spread = 180.0
    for i in range(n):
        spread = extended[i + n - 1] - extended[i]
        if spread < min_spread:
            min_spread = spread
    return float(min_spread)


def _filter_duplicate_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    filtered: List[Tuple[float, float]] = []
    for pt in points:
        if filtered and filtered[-1] == pt:
            continue
        filtered.append((float(pt[0]), float(pt[1])))
    return filtered


def _emit_polyline(
    points: Sequence[Tuple[float, float]],
    segments: List[Segment],
    max_dev: float,
    angle_spread_max: float,
):
    filtered = _filter_duplicate_points(points)
    if len(filtered) < 2:
        return

    if _angle_spread_deg(filtered) > angle_spread_max:
        return

    try:
        segment, delta_max, _, _ = fit_straight_segment(filtered)
    except ValueError:
        return

    if delta_max <= max_dev:
        segments.append(segment)


def _point_line_distance(
    pt: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]
) -> float:
    x0, y0 = pt
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom == 0:
        return math.hypot(x0 - x1, y0 - y1)
    return abs((x0 - x1) * dy - (y0 - y1) * dx) / denom


def _approximate_cubic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    tol: float,
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []

    def subdivide(
        a0: Tuple[float, float],
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        a3: Tuple[float, float],
    ):
        p01 = ((a0[0] + a1[0]) * 0.5, (a0[1] + a1[1]) * 0.5)
        p12 = ((a1[0] + a2[0]) * 0.5, (a1[1] + a2[1]) * 0.5)
        p23 = ((a2[0] + a3[0]) * 0.5, (a2[1] + a3[1]) * 0.5)
        p012 = ((p01[0] + p12[0]) * 0.5, (p01[1] + p12[1]) * 0.5)
        p123 = ((p12[0] + p23[0]) * 0.5, (p12[1] + p23[1]) * 0.5)
        p0123 = ((p012[0] + p123[0]) * 0.5, (p012[1] + p123[1]) * 0.5)
        return (a0, p01, p012, p0123), (p0123, p123, p23, a3)

    def recurse(
        a0: Tuple[float, float],
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        a3: Tuple[float, float],
    ):
        dist1 = _point_line_distance(a1, a0, a3)
        dist2 = _point_line_distance(a2, a0, a3)
        if max(dist1, dist2) <= tol:
            result.append(a0)
            return
        left, right = subdivide(a0, a1, a2, a3)
        recurse(*left)
        recurse(*right)

    recurse(p0, p1, p2, p3)
    result.append(p3)
    return result


def _approximate_quadratic(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    tol: float,
) -> List[Tuple[float, float]]:
    c1 = (p0[0] + (p1[0] - p0[0]) * 2.0 / 3.0, p0[1] + (p1[1] - p0[1]) * 2.0 / 3.0)
    c2 = (p2[0] + (p1[0] - p2[0]) * 2.0 / 3.0, p2[1] + (p1[1] - p2[1]) * 2.0 / 3.0)
    return _approximate_cubic(p0, c1, c2, p2, tol)


def _emit_rectangle(
    x: float,
    y: float,
    w: float,
    h: float,
    segments: List[Segment],
    max_dev: float,
    angle_spread_max: float,
):
    if w == 0 or h == 0:
        return
    pts = [
        [(x, y), (x + w, y)],
        [(x + w, y), (x + w, y + h)],
        [(x + w, y + h), (x, y + h)],
        [(x, y + h), (x, y)],
    ]
    for poly in pts:
        _emit_polyline(poly, segments, max_dev, angle_spread_max)


def _process_commands(
    commands: Sequence[Sequence],
    segments: List[Segment],
    curve_tol: float,
    max_dev: float,
    angle_spread_max: float,
):
    current: Tuple[float, float] | None = None
    start: Tuple[float, float] | None = None
    points: List[Tuple[float, float]] = []

    def flush() -> None:
        nonlocal points
        if points:
            _emit_polyline(points, segments, max_dev, angle_spread_max)
        points = []

    for raw_cmd in commands:
        if not raw_cmd:
            continue
        op = str(raw_cmd[0]).lower()
        if op == "m":
            flush()
            if len(raw_cmd) < 3:
                continue
            current = (float(raw_cmd[1]), float(raw_cmd[2]))
            start = current
            points = [current]
        elif op == "l" and current is not None:
            if len(raw_cmd) < 3:
                continue
            current = (float(raw_cmd[1]), float(raw_cmd[2]))
            points.append(current)
        elif op in {"c", "v", "y"} and current is not None:
            if op == "c" and len(raw_cmd) >= 7:
                ctrl1 = (float(raw_cmd[1]), float(raw_cmd[2]))
                ctrl2 = (float(raw_cmd[3]), float(raw_cmd[4]))
                end = (float(raw_cmd[5]), float(raw_cmd[6]))
            elif op == "v" and len(raw_cmd) >= 5:
                ctrl1 = current
                ctrl2 = (float(raw_cmd[1]), float(raw_cmd[2]))
                end = (float(raw_cmd[3]), float(raw_cmd[4]))
            elif op == "y" and len(raw_cmd) >= 5:
                ctrl1 = (float(raw_cmd[1]), float(raw_cmd[2]))
                end = (float(raw_cmd[3]), float(raw_cmd[4]))
                ctrl2 = end
            else:
                continue
            curve_points = _approximate_cubic(current, ctrl1, ctrl2, end, curve_tol)
            points.extend(curve_points[1:])
            current = end
        elif op == "q" and current is not None and len(raw_cmd) >= 5:
            ctrl = (float(raw_cmd[1]), float(raw_cmd[2]))
            end = (float(raw_cmd[3]), float(raw_cmd[4]))
            curve_points = _approximate_quadratic(current, ctrl, end, curve_tol)
            points.extend(curve_points[1:])
            current = end
        elif op == "h" and start is not None:
            points.append(start)
            flush()
            current = start
        elif op == "re" and len(raw_cmd) >= 5:
            flush()
            rx, ry, rw, rh = map(float, raw_cmd[1:5])
            _emit_rectangle(rx, ry, rw, rh, segments, max_dev, angle_spread_max)
            current = None
            start = None
        else:
            if op in {"n", "s", "f"}:
                flush()

    flush()


def load_pdf_segments(
    pdf_path: str, page_index: int, cfg: dict
) -> Tuple[List[Segment], Tuple[float, float]]:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        log.warning("pypdfium2 nicht gefunden, PyMuPDF-Fallback wird verwendet")
        return _load_pdf_segments_pymupdf(pdf_path, page_index, cfg)

    doc = pdfium.PdfDocument(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(
                f"Seite {page_index} existiert nicht (gültig: 0..{len(doc) - 1})."
            )
        page = doc[page_index]
        try:
            width, height = page.get_size()
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()
    finally:
        doc.close()

    width = float(width)
    height = float(height)
    diag = math.hypot(width, height) or 1.0

    curve_tol_rel = cfg.get("curve_tol_rel", 0.001)
    curve_tol = max(float(curve_tol_rel) * diag, 1e-6)

    parse_start = perf_counter()
    raw_segments = pdfium_extract_segments(pdf_path, page_index, curve_tol)
    duration = perf_counter() - parse_start

    segments = [
        Segment(float(seg["x1"]), float(seg["y1"]), float(seg["x2"]), float(seg["y2"]))
        for seg in raw_segments
    ]

    log.debug(
        "[pdf] Seite %d: Größe=(%.2f×%.2f), Diagonale=%.2f, Segmente=%d in %.3fs",
        page_index,
        width,
        height,
        diag,
        len(segments),
        duration,
    )

    return segments, (width, height)


def _load_pdf_segments_pymupdf(
    pdf_path: str, page_index: int, cfg: dict
) -> Tuple[List[Segment], Tuple[float, float]]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        width = float(page.rect.width)
        height = float(page.rect.height)
        diag = math.hypot(width, height) or 1.0
        parse_start = perf_counter()
        log.debug(
            "[pdf] Seite %d: Größe=(%.2f×%.2f), Diagonale=%.2f",
            page_index,
            width,
            height,
            diag,
        )

        curve_tol_rel = cfg.get("curve_tol_rel", 0.001)
        straight_max_dev_rel = cfg.get("straight_max_dev_rel", 0.002)
        angle_spread_max = cfg.get("straight_max_angle_spread_deg", 4.0)

        curve_tol = max(curve_tol_rel * diag, 1e-6)
        max_dev = straight_max_dev_rel * diag

        segments: List[Segment] = []
        drawings = page.get_drawings()
        log.debug("[pdf] Seite %d: %d drawings geladen", page_index, len(drawings))

        total_subpaths = 0
        total_items = 0
        for draw_idx, drawing in enumerate(drawings, start=1):
            segs_before = len(segments)
            path = drawing.get("path")
            if path:
                total_subpaths += len(path)
                for subpath in path:
                    _process_commands(
                        subpath, segments, curve_tol, max_dev, angle_spread_max
                    )
            items = drawing.get("items")
            if items:
                total_items += len(items)
                pseudo_commands: List[Tuple] = []
                for item in items:
                    if not item:
                        continue
                    op = str(item[0]).lower()
                    if op in {"m", "l"} and len(item) >= 2:
                        x, y = item[1]
                        pseudo_commands.append((op, float(x), float(y)))
                    elif op == "c" and len(item) >= 4:
                        p1, p2, p3 = item[1], item[2], item[3]
                        pseudo_commands.append(
                            (
                                "c",
                                float(p1[0]),
                                float(p1[1]),
                                float(p2[0]),
                                float(p2[1]),
                                float(p3[0]),
                                float(p3[1]),
                            )
                        )
                    elif op == "q" and len(item) >= 3:
                        ctrl, end = item[1], item[2]
                        pseudo_commands.append(
                            ("q", float(ctrl[0]), float(ctrl[1]), float(end[0]), float(end[1]))
                        )
                    elif op in {"v", "y"} and len(item) >= 3:
                        c1, c2 = item[1], item[2]
                        pseudo_commands.append(
                            (
                                op,
                                float(c1[0]),
                                float(c1[1]),
                                float(c2[0]),
                                float(c2[1]),
                            )
                        )
                    elif op == "re" and len(item) >= 2:
                        rx, ry, rw, rh = item[1]
                        pseudo_commands.append(
                            ("re", float(rx), float(ry), float(rw), float(rh))
                        )
                    elif op == "h":
                        pseudo_commands.append(("h",))
                if pseudo_commands:
                    _process_commands(
                        pseudo_commands, segments, curve_tol, max_dev, angle_spread_max
                    )

            produced = len(segments) - segs_before
            if produced:
                log.debug(
                    "[pdf] Seite %d: drawing #%d erzeugte %d Segmente (path=%d, items=%d)",
                    page_index,
                    draw_idx,
                    produced,
                    len(path) if path else 0,
                    len(items) if items else 0,
                )

        duration = perf_counter() - parse_start
        log.debug(
            "[pdf] Seite %d: %d Segmente aus %d drawings (%d Subpfade, %d Items) in %.3fs",
            page_index,
            len(segments),
            len(drawings),
            total_subpaths,
            total_items,
            duration,
        )

        return segments, (width, height)
    finally:
        doc.close()


def convert_pdf_to_svg_if_needed(
    pdf_path: str, page_index: int, outdir: str
) -> str:
    """Export a PDF page to SVG via PyMuPDF if no explicit SVG was provided."""

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    svg_out = os.path.join(outdir, f"{base}_p{page_index:03d}.svg")
    os.makedirs(outdir, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(
                f"Seite {page_index} existiert nicht (gültig: 0..{doc.page_count - 1})."
            )
        page = doc.load_page(page_index)
        height = page.rect.height
        matrix = fitz.Matrix(1, -1)
        matrix.pretranslate(0, -height)
        svg_xml = page.get_svg_image(matrix=matrix, text_as_path=False)
    finally:
        doc.close()

    with open(svg_out, "w", encoding="utf-8") as fh:
        fh.write(svg_xml)

    return svg_out


def _parse_length(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    if text.endswith("%"):
        return 0.0
    units = {
        "px": 1.0,
        "pt": 96.0 / 72.0,
        "pc": 16.0,
        "in": 96.0,
        "cm": 96.0 / 2.54,
        "mm": 96.0 / 25.4,
        "q": 96.0 / 101.6,
    }
    num = ""
    unit = ""
    for ch in text:
        if ch.isdigit() or ch in ".+-eE":
            num += ch
        else:
            unit += ch
    try:
        value = float(num)
    except ValueError:
        return 0.0
    unit = unit.strip().lower()
    factor = units.get(unit, 1.0)
    return value * factor


def _parse_viewbox(root) -> Tuple[float, float]:
    viewbox = root.get("viewBox")
    if not viewbox:
        return 0.0, 0.0
    parts = viewbox.replace(",", " ").split()
    if len(parts) != 4:
        return 0.0, 0.0
    try:
        width = float(parts[2])
        height = float(parts[3])
    except ValueError:
        return 0.0, 0.0
    return width, height


def load_svg_segments(svg_path: str, cfg: dict):
    segs = parse_svg_segments(svg_path, cfg)
    segs = [
        Segment(
            float(seg.x1) * PT_PER_PX,
            float(seg.y1) * PT_PER_PX,
            float(seg.x2) * PT_PER_PX,
            float(seg.y2) * PT_PER_PX,
        )
        for seg in segs
    ]
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return segs, (0.0, 0.0)
    root = tree.getroot()
    width_px = _parse_length(root.get("width"))
    height_px = _parse_length(root.get("height"))
    vb_w_px, vb_h_px = _parse_viewbox(root)
    if width_px == 0.0 and vb_w_px:
        width_px = vb_w_px
    if height_px == 0.0 and vb_h_px:
        height_px = vb_h_px
    width = float(width_px) * PT_PER_PX if width_px else 0.0
    height = float(height_px) * PT_PER_PX if height_px else 0.0
    return segs, (width, height)
