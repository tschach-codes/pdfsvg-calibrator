from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

import fitz

from .geom import fit_straight_segment
from .svg_path import parse_svg_segments
from .types import Segment


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
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        width = float(page.rect.width)
        height = float(page.rect.height)
        diag = math.hypot(width, height) or 1.0

        curve_tol_rel = cfg.get("curve_tol_rel", 0.001)
        straight_max_dev_rel = cfg.get("straight_max_dev_rel", 0.002)
        angle_spread_max = cfg.get("straight_max_angle_spread_deg", 4.0)

        curve_tol = max(curve_tol_rel * diag, 1e-6)
        max_dev = straight_max_dev_rel * diag

        segments: List[Segment] = []
        drawings = page.get_drawings()
        for drawing in drawings:
            path = drawing.get("path")
            if path:
                for subpath in path:
                    _process_commands(subpath, segments, curve_tol, max_dev, angle_spread_max)
            items = drawing.get("items")
            if items:
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

        return segments, (width, height)
    finally:
        doc.close()


def convert_pdf_to_svg_if_needed(
    pdf_path: str, page_index: int, outdir: str
) -> str:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    svg_out = os.path.join(outdir, f"{base}_p{page_index:03d}.svg")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(svg_out):
        raise FileNotFoundError(
            "No SVG found. Please export a vector SVG to:\n"
            f"  {svg_out}\n"
            "Hook an external converter if desired."
        )
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
    size = (0.0, 0.0)  # could parse viewBox later
    return segs, size
