from __future__ import annotations

import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import fitz
import numpy as np

from .geom import fit_straight_segment
from .debug.pdf_probe import probe_page
from .debug.pdf_segments_debug import analyze_segments_basic, debug_print_segments
from .pdfium_extract import debug_print_summary, extract_segments
from .types import Segment


log = logging.getLogger(__name__)

PT_PER_PX = 72.0 / 96.0


Matrix = Tuple[float, float, float, float, float, float]


def _matrix_identity() -> Matrix:
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return (
        a0 * b0 + a2 * b1,
        a1 * b0 + a3 * b1,
        a0 * b2 + a2 * b3,
        a1 * b2 + a3 * b3,
        a0 * b4 + a2 * b5 + a4,
        a1 * b4 + a3 * b5 + a5,
    )


def _matrix_translate(tx: float, ty: float) -> Matrix:
    return (1.0, 0.0, 0.0, 1.0, tx, ty)


def _matrix_scale(sx: float, sy: float) -> Matrix:
    return (sx, 0.0, 0.0, sy, 0.0, 0.0)


def _matrix_rotate(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> Matrix:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = (cos_t, sin_t, -sin_t, cos_t, 0.0, 0.0)
    if cx or cy:
        return _matrix_multiply(
            _matrix_multiply(_matrix_translate(cx, cy), rot),
            _matrix_translate(-cx, -cy),
        )
    return rot


def _matrix_skew_x(angle_deg: float) -> Matrix:
    theta = math.radians(angle_deg)
    return (1.0, 0.0, math.tan(theta), 1.0, 0.0, 0.0)


def _matrix_skew_y(angle_deg: float) -> Matrix:
    theta = math.radians(angle_deg)
    return (1.0, math.tan(theta), 0.0, 1.0, 0.0, 0.0)


_TRANSFORM_CMD_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _tokenize_path(d: str) -> List[str]:
    token_re = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    return [m.group(0) for m in token_re.finditer(d.replace(",", " "))]


def _local_name(node: ET.Element) -> str:
    if isinstance(node.tag, str):
        if node.tag.startswith("{"):
            return node.tag.split("}", 1)[1]
        return node.tag
    return ""


def _points_to_segments(points: Sequence[Tuple[float, float]]) -> List[dict]:
    segs: List[dict] = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        segs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return segs


def _parse_points_attribute(points: str) -> List[Tuple[float, float]]:
    cleaned = re.sub(r"[\s,]+", " ", points.strip())
    if not cleaned:
        return []
    coords = cleaned.split(" ")
    result: List[Tuple[float, float]] = []
    it = iter(coords)
    for x_str, y_str in zip(it, it):
        try:
            result.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return result


def _parse_transform(transform: Optional[str]) -> Matrix:
    if not transform:
        return _matrix_identity()
    transform = transform.strip()
    if not transform:
        return _matrix_identity()
    result = _matrix_identity()
    for match in _TRANSFORM_CMD_RE.finditer(transform):
        name = match.group(1).lower()
        params = [float(v) for v in _NUMBER_RE.findall(match.group(2))]
        if name == "matrix" and len(params) == 6:
            matrix = tuple(params)  # type: ignore[assignment]
        elif name == "translate":
            tx = params[0] if params else 0.0
            ty = params[1] if len(params) > 1 else 0.0
            matrix = _matrix_translate(tx, ty)
        elif name == "scale":
            sx = params[0] if params else 1.0
            sy = params[1] if len(params) > 1 else sx
            matrix = _matrix_scale(sx, sy)
        elif name == "rotate":
            if len(params) == 1:
                matrix = _matrix_rotate(params[0])
            elif len(params) == 3:
                matrix = _matrix_rotate(params[0], params[1], params[2])
            else:
                continue
        elif name == "skewx" and params:
            matrix = _matrix_skew_x(params[0])
        elif name == "skewy" and params:
            matrix = _matrix_skew_y(params[0])
        else:
            continue
        result = _matrix_multiply(result, matrix)  # type: ignore[arg-type]
    return result


def _apply_transform_point(point: Tuple[float, float], matrix: Matrix) -> Tuple[float, float]:
    x, y = point
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


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


def _arc_to_cubic(
    p0: Tuple[float, float],
    rx: float,
    ry: float,
    phi_deg: float,
    large_arc: int,
    sweep: int,
    p1: Tuple[float, float],
) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    if rx == 0 or ry == 0:
        return []

    phi = math.radians(phi_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    x1, y1 = p0
    x2, y2 = p1
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0

    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    rx_abs = abs(rx)
    ry_abs = abs(ry)
    lam = (x1p**2) / (rx_abs**2) + (y1p**2) / (ry_abs**2)
    if lam > 1:
        scale = math.sqrt(lam)
        rx_abs *= scale
        ry_abs *= scale

    sign = -1 if large_arc == sweep else 1
    numerator = rx_abs**2 * ry_abs**2 - rx_abs**2 * y1p**2 - ry_abs**2 * x1p**2
    denom = rx_abs**2 * y1p**2 + ry_abs**2 * x1p**2
    if denom == 0:
        denom = 1e-12
    coef = sign * math.sqrt(max(0.0, numerator / denom))
    cxp = coef * (rx_abs * y1p) / ry_abs
    cyp = coef * -(ry_abs * x1p) / rx_abs

    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    def angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
        ux, uy = u
        vx, vy = v
        dot = ux * vx + uy * vy
        det = ux * vy - uy * vx
        return math.atan2(det, dot)

    v1 = ((x1p - cxp) / rx_abs, (y1p - cyp) / ry_abs)
    v2 = ((-x1p - cxp) / rx_abs, (-y1p - cyp) / ry_abs)

    theta1 = angle((1.0, 0.0), v1)
    delta_theta = angle(v1, v2)
    if sweep == 0 and delta_theta > 0:
        delta_theta -= 2 * math.pi
    elif sweep == 1 and delta_theta < 0:
        delta_theta += 2 * math.pi

    segments = max(1, int(math.ceil(abs(delta_theta) / (math.pi / 2 + 1e-9))))
    delta = delta_theta / segments

    result = []
    for i in range(segments):
        t1 = theta1 + i * delta
        t2 = t1 + delta
        sin_t1 = math.sin(t1)
        cos_t1 = math.cos(t1)
        sin_t2 = math.sin(t2)
        cos_t2 = math.cos(t2)

        e = 4 * math.tan(delta / 4) / 3

        p_start = (
            cx + rx_abs * (cos_phi * cos_t1 - sin_phi * sin_t1),
            cy + ry_abs * (sin_phi * cos_t1 + cos_phi * sin_t1),
        )
        p_end = (
            cx + rx_abs * (cos_phi * cos_t2 - sin_phi * sin_t2),
            cy + ry_abs * (sin_phi * cos_t2 + cos_phi * sin_t2),
        )
        dx1 = -rx_abs * (cos_phi * sin_t1 + sin_phi * cos_t1)
        dy1 = -ry_abs * (sin_phi * sin_t1 - cos_phi * cos_t1)
        dx2 = -rx_abs * (cos_phi * sin_t2 + sin_phi * cos_t2)
        dy2 = -ry_abs * (sin_phi * sin_t2 - cos_phi * cos_t2)

        ctrl1 = (p_start[0] + dx1 * e, p_start[1] + dy1 * e)
        ctrl2 = (p_end[0] - dx2 * e, p_end[1] - dy2 * e)
        result.append((p_start, ctrl1, ctrl2, p_end))

    return result


def _flatten_arc(
    p0: Tuple[float, float],
    rx: float,
    ry: float,
    phi: float,
    large_arc: int,
    sweep: int,
    p1: Tuple[float, float],
    tol: float,
) -> List[Tuple[float, float]]:
    cubics = _arc_to_cubic(p0, rx, ry, phi, large_arc, sweep, p1)
    if not cubics:
        return [p0, p1]
    pts: List[Tuple[float, float]] = []
    for idx, (c0, c1, c2, c3) in enumerate(cubics):
        segment_pts = _approximate_cubic(c0, c1, c2, c3, tol)
        if idx:
            pts.extend(segment_pts[1:])
        else:
            pts.extend(segment_pts)
    return pts


def _path_to_polylines_absolute(d: str, tol: float) -> List[List[Tuple[float, float]]]:
    tokens = _tokenize_path(d)
    if not tokens:
        return []

    polylines: List[List[Tuple[float, float]]] = []
    idx = 0
    command = ""
    current = (0.0, 0.0)
    last_ctrl: Optional[Tuple[float, float]] = None
    last_cmd = ""
    current_poly: Optional[List[Tuple[float, float]]] = None

    def read_numbers(count: int) -> List[float]:
        nonlocal idx
        nums: List[float] = []
        for _ in range(count):
            if idx >= len(tokens):
                raise ValueError("Unexpected end of path data")
            nums.append(float(tokens[idx]))
            idx += 1
        return nums

    def ensure_poly(point: Tuple[float, float]) -> None:
        nonlocal current_poly
        if current_poly is None:
            current_poly = [point]
            polylines.append(current_poly)

    while idx < len(tokens):
        token = tokens[idx]
        if token.isalpha():
            command = token
            idx += 1
        elif not command:
            raise ValueError("Path data malformed")
        absolute = command.isupper()
        cmd = command.lower()

        if cmd == "m":
            coords = read_numbers(2)
            if absolute:
                current = (coords[0], coords[1])
            else:
                current = (current[0] + coords[0], current[1] + coords[1])
            current_poly = [current]
            polylines.append(current_poly)
            last_cmd = "m"
            last_ctrl = None
            while idx < len(tokens) and not tokens[idx].isalpha():
                coords = read_numbers(2)
                if absolute:
                    target = (coords[0], coords[1])
                else:
                    target = (current[0] + coords[0], current[1] + coords[1])
                current_poly.append(target)
                current = target
                last_cmd = "l"
            continue
        if cmd == "z":
            if current_poly and current_poly[0] != current_poly[-1]:
                current_poly.append(current_poly[0])
            current = current_poly[0] if current_poly else current
            last_cmd = "z"
            last_ctrl = None
            current_poly = None
            continue
        if cmd == "l":
            while idx < len(tokens) and not tokens[idx].isalpha():
                coords = read_numbers(2)
                target = (coords[0], coords[1]) if absolute else (current[0] + coords[0], current[1] + coords[1])
                ensure_poly(current)
                current_poly.append(target)
                current = target
            last_cmd = "l"
            last_ctrl = None
            continue
        if cmd == "h":
            while idx < len(tokens) and not tokens[idx].isalpha():
                (value,) = read_numbers(1)
                x = value if absolute else current[0] + value
                target = (x, current[1])
                ensure_poly(current)
                current_poly.append(target)
                current = target
            last_cmd = "h"
            last_ctrl = None
            continue
        if cmd == "v":
            while idx < len(tokens) and not tokens[idx].isalpha():
                (value,) = read_numbers(1)
                y = value if absolute else current[1] + value
                target = (current[0], y)
                ensure_poly(current)
                current_poly.append(target)
                current = target
            last_cmd = "v"
            last_ctrl = None
            continue
        if cmd == "c":
            while idx < len(tokens) and not tokens[idx].isalpha():
                values = read_numbers(6)
                c1 = (values[0], values[1])
                c2 = (values[2], values[3])
                end = (values[4], values[5])
                if not absolute:
                    c1 = (current[0] + c1[0], current[1] + c1[1])
                    c2 = (current[0] + c2[0], current[1] + c2[1])
                    end = (current[0] + end[0], current[1] + end[1])
                ensure_poly(current)
                curve_points = _approximate_cubic(current, c1, c2, end, tol)
                current_poly.extend(curve_points[1:])
                current = end
                last_ctrl = c2
            last_cmd = "c"
            continue
        if cmd == "s":
            while idx < len(tokens) and not tokens[idx].isalpha():
                values = read_numbers(4)
                c2 = (values[0], values[1])
                end = (values[2], values[3])
                if last_cmd in {"c", "s"} and last_ctrl is not None:
                    reflected = (2 * current[0] - last_ctrl[0], 2 * current[1] - last_ctrl[1])
                else:
                    reflected = current
                if not absolute:
                    c2 = (current[0] + c2[0], current[1] + c2[1])
                    end = (current[0] + end[0], current[1] + end[1])
                ensure_poly(current)
                curve_points = _approximate_cubic(current, reflected, c2, end, tol)
                current_poly.extend(curve_points[1:])
                current = end
                last_ctrl = c2
            last_cmd = "s"
            continue
        if cmd == "q":
            while idx < len(tokens) and not tokens[idx].isalpha():
                values = read_numbers(4)
                ctrl = (values[0], values[1])
                end = (values[2], values[3])
                if not absolute:
                    ctrl = (current[0] + ctrl[0], current[1] + ctrl[1])
                    end = (current[0] + end[0], current[1] + end[1])
                ensure_poly(current)
                curve_points = _approximate_quadratic(current, ctrl, end, tol)
                current_poly.extend(curve_points[1:])
                current = end
                last_ctrl = ctrl
            last_cmd = "q"
            continue
        if cmd == "t":
            while idx < len(tokens) and not tokens[idx].isalpha():
                values = read_numbers(2)
                if last_cmd in {"q", "t"} and last_ctrl is not None:
                    ctrl = (2 * current[0] - last_ctrl[0], 2 * current[1] - last_ctrl[1])
                else:
                    ctrl = current
                end = (values[0], values[1])
                if not absolute:
                    end = (current[0] + end[0], current[1] + end[1])
                ensure_poly(current)
                curve_points = _approximate_quadratic(current, ctrl, end, tol)
                current_poly.extend(curve_points[1:])
                current = end
                last_ctrl = ctrl
            last_cmd = "t"
            continue
        if cmd == "a":
            while idx < len(tokens) and not tokens[idx].isalpha():
                values = read_numbers(7)
                rx, ry, phi, large_arc, sweep, x, y = values
                target = (x, y)
                if not absolute:
                    target = (current[0] + target[0], current[1] + target[1])
                ensure_poly(current)
                arc_points = _flatten_arc(
                    current,
                    rx,
                    ry,
                    phi,
                    int(large_arc),
                    int(sweep),
                    target,
                    tol,
                )
                current_poly.extend(arc_points[1:])
                current = target
                last_ctrl = current
            last_cmd = "a"
            continue

        raise ValueError(f"Unsupported path command: {command}")

    return polylines


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
    pdf_path: str, page_index: int, cfg: dict, *, use_pdfium: bool = True
) -> Tuple[List[Segment], Tuple[float, float]]:
    if not use_pdfium:
        return _load_pdf_segments_pymupdf(pdf_path, page_index, cfg)

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

    probe_stats = None
    try:
        probe_stats = probe_page(pdf_path, page_index)
    except Exception as exc:  # pragma: no cover - diagnostic helper
        log.debug(
            "PDF-Probe vor Extraktion fehlgeschlagen: %s",
            exc,
            exc_info=isinstance(exc, RuntimeError),
        )
    else:
        counts = (probe_stats.get("counts") or {}).copy()
        notes = list(probe_stats.get("notes") or [])
        forms = int(probe_stats.get("forms", 0) or 0)
        log.debug(
            "[pdf] Seite %d: Probe counts=%s, forms=%d, notes=%s",
            page_index,
            counts,
            forms,
            notes,
        )

    parse_start = perf_counter()
    pdfium_segments = extract_segments(str(pdf_path), page_index=page_index, tol_pt=0.1)
    duration = perf_counter() - parse_start

    print(f"Seite {page_index}: pypdfium2-Segmente extrahiert={len(pdfium_segments)}")
    debug_print_summary("after CTM compose", pdfium_segments)
    stats = analyze_segments_basic(pdfium_segments, angle_tol_deg=2.0)
    debug_print_segments("PDFium raw", stats, pdfium_segments)

    if pdfium_segments:
        print("INFO: using PDFium segments (skip PyMuPDF fallback)")
        segments = [
            Segment(
                float(seg["x1"]),
                float(seg["y1"]),
                float(seg["x2"]),
                float(seg["y2"]),
            )
            for seg in pdfium_segments
        ]
    else:
        needs_raster_fallback = False

        if probe_stats is not None:
            notes = probe_stats.get("notes") or []
            for note in notes:
                if str(note) == "likely_scanned_raster_page":
                    log.error("Scan-Seite – kein Vektorinhalt")
                    raise ValueError("Seite enthält nur Rasterbild, keine Vektoren")

            counts = probe_stats.get("counts", {})
            text_count = int(counts.get("text", 0) or 0)
            path_count = int(counts.get("path", 0) or 0)
            form_count = int(probe_stats.get("forms", 0) or 0)

            if form_count > 0:
                log.error("FORM-Walker defekt – %d FORM-XObjects gefunden", form_count)

            if (
                text_count >= max(4, counts.get("total", 0) // 2)
                and path_count == 0
                and form_count == 0
            ):
                needs_raster_fallback = True

        if needs_raster_fallback:
            log.warning(
                "PDF enthält überwiegend Textobjekt-Outlines – Raster-Fallback wird verwendet",
            )
            return _load_pdf_segments_raster(pdf_path, page_index, cfg, width, height)

        print("WARN: PDFium returned 0 segments, trying PyMuPDF fallback …")
        segments, (width, height) = _load_pdf_segments_pymupdf(pdf_path, page_index, cfg)
        print(f"PyMuPDF fallback produced {len(segments)} segments")

    if len(segments) == 0:
        raise RuntimeError("PDF enthält keine vektoriellen Segmente -> Abbruch")

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

        result = segments, (width, height)
    finally:
        doc.close()

    return result


def _load_pdf_segments_raster(
    pdf_path: str,
    page_index: int,
    cfg: dict,
    page_width: float,
    page_height: float,
) -> Tuple[List[Segment], Tuple[float, float]]:
    raster_cfg = cfg.get("pdf_raster_fallback", {})
    zoom = float(raster_cfg.get("zoom", 4.0))
    edge_percentile = float(raster_cfg.get("edge_percentile", 97.0))
    edge_min_strength = float(raster_cfg.get("edge_min_strength", 10.0))
    max_segments = int(raster_cfg.get("max_segments", 200_000))
    if zoom <= 0:
        zoom = 4.0

    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
    finally:
        doc.close()

    width_px = pix.width
    height_px = pix.height
    if width_px <= 0 or height_px <= 0:
        return [], (page_width, page_height)

    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    img = buf.reshape(height_px, width_px).astype(np.float32)
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)

    if not np.any(mag):
        return [], (page_width, page_height)

    percentile = np.clip(edge_percentile, 0.0, 100.0)
    thresh = float(np.percentile(mag, percentile))
    thresh = max(thresh, edge_min_strength)
    edges = mag >= thresh
    if not np.any(edges):
        return [], (page_width, page_height)

    segments: List[Segment] = []

    def _pixel_center(col: int, row: int) -> Tuple[float, float]:
        x = (col + 0.5) / zoom
        y = page_height - (row + 0.5) / zoom
        return x, y

    edges = edges.astype(bool, copy=False)
    rows, cols = edges.shape
    for r in range(rows):
        row_mask = edges[r]
        for c in range(cols):
            if not row_mask[c]:
                continue
            if c + 1 < cols and row_mask[c + 1]:
                x1, y1 = _pixel_center(c, r)
                x2, y2 = _pixel_center(c + 1, r)
                if x1 != x2 or y1 != y2:
                    segments.append(Segment(x1, y1, x2, y2))
            if r + 1 < rows and edges[r + 1, c]:
                x1, y1 = _pixel_center(c, r)
                x2, y2 = _pixel_center(c, r + 1)
                if x1 != x2 or y1 != y2:
                    segments.append(Segment(x1, y1, x2, y2))

    if max_segments > 0 and len(segments) > max_segments:
        step = max(1, len(segments) // max_segments)
        segments = segments[::step]

    log.warning(
        "Raster-Fallback generierte %d Segmente (Zoom %.1f, Schwelle %.1f)",
        len(segments),
        zoom,
        thresh,
    )

    return segments, (page_width, page_height)


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


def _viewbox_matrix(node: ET.Element) -> Matrix:
    view_box = node.get("viewBox")
    width = _parse_length(node.get("width"))
    height = _parse_length(node.get("height"))
    x = _parse_length(node.get("x")) or 0.0
    y = _parse_length(node.get("y")) or 0.0
    matrix = _matrix_translate(x, y)
    if not view_box:
        return matrix

    parts = [float(v) for v in view_box.replace(",", " ").split() if v]
    if len(parts) != 4:
        return matrix
    min_x, min_y, vb_w, vb_h = parts
    if vb_w == 0 or vb_h == 0:
        return _matrix_multiply(matrix, _matrix_translate(-min_x, -min_y))

    preserve = (node.get("preserveAspectRatio") or "xMidYMid meet").strip()
    if preserve == "none":
        sx = (width / vb_w) if width else 1.0
        sy = (height / vb_h) if height else 1.0
        return _matrix_multiply(
            _matrix_multiply(matrix, _matrix_scale(sx if sx else 1.0, sy if sy else 1.0)),
            _matrix_translate(-min_x, -min_y),
        )

    sx = (width / vb_w) if width else 1.0
    sy = (height / vb_h) if height else 1.0
    scale = min(sx if sx else 1.0, sy if sy else 1.0)

    align = preserve.split()[0] if preserve else "xMidYMid"
    align_x = 0.0
    align_y = 0.0
    if width:
        extra_x = width - vb_w * scale
        if "xMid" in align:
            align_x = extra_x / 2.0
        elif "xMax" in align:
            align_x = extra_x
    if height:
        extra_y = height - vb_h * scale
        if "YMid" in align:
            align_y = extra_y / 2.0
        elif "YMax" in align:
            align_y = extra_y

    matrix = _matrix_multiply(matrix, _matrix_translate(align_x, align_y))
    matrix = _matrix_multiply(matrix, _matrix_scale(scale, scale))
    matrix = _matrix_multiply(matrix, _matrix_translate(-min_x, -min_y))
    return matrix


def _pdf_segments_pymupdf(pdf_path: str, page_index: int) -> List[dict]:
    cfg = {
        "curve_tol_rel": 0.001,
        "straight_max_dev_rel": 0.002,
        "straight_max_angle_spread_deg": 4.0,
    }
    segments, _ = _load_pdf_segments_pymupdf(pdf_path, page_index, cfg)
    return [
        {
            "x1": float(seg.x1),
            "y1": float(seg.y1),
            "x2": float(seg.x2),
            "y2": float(seg.y2),
        }
        for seg in segments
    ]


def pdf_to_segments(pdf_path: str, page: int, use_pdfium: bool = True) -> List[dict]:
    if use_pdfium:
        segs = extract_segments(pdf_path, page_index=page, tol_pt=0.1)
        return sanitize_segments(segs)

    segs = _pdf_segments_pymupdf(pdf_path, page)
    return sanitize_segments(segs)


def sanitize_segments(segments) -> List[dict]:
    out: List[dict] = []
    for s in segments:
        out.append(
            {
                "x1": float(s["x1"]),
                "y1": float(s["y1"]),
                "x2": float(s["x2"]),
                "y2": float(s["y2"]),
            }
        )
    return out


def svg_to_segments(svg_path: str, curve_tol_px: float = 0.25) -> List[dict]:
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as exc:
        log.warning("SVG parsing failed for %s: %s", svg_path, exc)
        return sanitize_segments([])

    root = tree.getroot()
    segments: List[dict] = []
    tol = max(float(curve_tol_px), 0.0)

    def traverse(node: ET.Element, matrix: Matrix) -> None:
        tag = _local_name(node)

        if tag in {"g", "svg", "symbol"}:
            child_matrix = matrix
            if tag in {"svg", "symbol"}:
                child_matrix = _matrix_multiply(child_matrix, _viewbox_matrix(node))
            child_matrix = _matrix_multiply(child_matrix, _parse_transform(node.get("transform")))
            for child in list(node):
                traverse(child, child_matrix)
            return

        shape_matrix = _matrix_multiply(matrix, _parse_transform(node.get("transform")))

        if tag == "path":
            d = node.get("d")
            if not d:
                return
            try:
                polylines = _path_to_polylines_absolute(d, tol)
            except ValueError as exc:
                log.debug("Pfad konnte nicht geparst werden (%s): %s", svg_path, exc)
                return
            for poly in polylines:
                if len(poly) < 2:
                    continue
                transformed = [_apply_transform_point(pt, shape_matrix) for pt in poly]
                segments.extend(_points_to_segments(transformed))
            return

        if tag == "line":
            try:
                x1 = float(node.get("x1") or 0.0)
                y1 = float(node.get("y1") or 0.0)
                x2 = float(node.get("x2") or 0.0)
                y2 = float(node.get("y2") or 0.0)
            except ValueError:
                return
            pts = [
                _apply_transform_point((x1, y1), shape_matrix),
                _apply_transform_point((x2, y2), shape_matrix),
            ]
            segments.extend(_points_to_segments(pts))
            return

        if tag in {"polyline", "polygon"}:
            points_attr = node.get("points")
            if not points_attr:
                return
            points = _parse_points_attribute(points_attr)
            if tag == "polygon" and points:
                points = points + [points[0]]
            if len(points) < 2:
                return
            transformed = [_apply_transform_point(pt, shape_matrix) for pt in points]
            segments.extend(_points_to_segments(transformed))
            return

        if tag == "rect":
            try:
                x = float(node.get("x") or 0.0)
                y = float(node.get("y") or 0.0)
                w = float(node.get("width") or 0.0)
                h = float(node.get("height") or 0.0)
            except ValueError:
                return
            if w == 0 or h == 0:
                return
            points = [
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
                (x, y),
            ]
            transformed = [_apply_transform_point(pt, shape_matrix) for pt in points]
            segments.extend(_points_to_segments(transformed))

    root_matrix = _viewbox_matrix(root)
    root_matrix = _matrix_multiply(root_matrix, _parse_transform(root.get("transform")))
    for child in list(root):
        traverse(child, root_matrix)

    return sanitize_segments(segments)


def load_svg_segments(svg_path: str, cfg: dict):
    curve_tol_rel = float(cfg.get("curve_tol_rel", 0.001))
    width_px = 0.0
    height_px = 0.0
    diag_px = 1.0
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        pass
    else:
        root = tree.getroot()
        width_px = _parse_length(root.get("width"))
        height_px = _parse_length(root.get("height"))
        vb_w_px, vb_h_px = _parse_viewbox(root)
        diag_candidates = []
        if vb_w_px and vb_h_px:
            diag_candidates.append(math.hypot(vb_w_px, vb_h_px))
        diag_candidates.append(math.hypot(width_px, height_px))
        diag_px = next((d for d in diag_candidates if d), 1.0)
        if width_px == 0.0 and vb_w_px:
            width_px = vb_w_px
        if height_px == 0.0 and vb_h_px:
            height_px = vb_h_px

    curve_tol_px = max(curve_tol_rel * diag_px, 1e-6)
    raw_segments = svg_to_segments(svg_path, curve_tol_px=curve_tol_px)
    segs = [
        Segment(
            float(seg["x1"]) * PT_PER_PX,
            float(seg["y1"]) * PT_PER_PX,
            float(seg["x2"]) * PT_PER_PX,
            float(seg["y2"]) * PT_PER_PX,
        )
        for seg in raw_segments
    ]

    width = float(width_px) * PT_PER_PX if width_px else 0.0
    height = float(height_px) * PT_PER_PX if height_px else 0.0
    return segs, (width, height)
