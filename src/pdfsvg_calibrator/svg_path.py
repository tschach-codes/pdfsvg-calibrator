from __future__ import annotations

import math
import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from lxml import etree as ET

from .types import Segment


log = logging.getLogger(__name__)


_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_WS_RE = re.compile(r"\s+")


def _parse_number(token: str) -> float:
    return float(token)


def _tokenise_path(d: str) -> Iterator[str]:
    token_re = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for tok in token_re.finditer(d.replace(",", " ")):
        yield tok.group(0)


def _parse_length(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    match = _FLOAT_RE.match(value)
    if not match:
        return None
    return float(match.group(0))


def _identity() -> np.ndarray:
    return np.eye(3)


def _translate(tx: float, ty: float) -> np.ndarray:
    m = _identity()
    m[0, 2] = tx
    m[1, 2] = ty
    return m


def _scale(sx: float, sy: float) -> np.ndarray:
    m = _identity()
    m[0, 0] = sx
    m[1, 1] = sy
    return m


def _rotate(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    if cx or cy:
        return _translate(cx, cy) @ rot @ _translate(-cx, -cy)
    return rot


def _skew_x(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    m = _identity()
    m[0, 1] = math.tan(theta)
    return m


def _skew_y(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    m = _identity()
    m[1, 0] = math.tan(theta)
    return m


def _parse_transform(transform: Optional[str]) -> np.ndarray:
    if not transform:
        return _identity()
    transform = transform.strip()
    if not transform:
        return _identity()
    pattern = re.compile(r"([a-zA-Z]+)\s*\(([^)]+)\)")
    result = _identity()
    for match in pattern.finditer(transform):
        name = match.group(1)
        params = [float(v) for v in _FLOAT_RE.findall(match.group(2))]
        name_lower = name.lower()
        if name_lower == "matrix" and len(params) == 6:
            a, b, c, d, e, f = params
            m = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name_lower == "translate":
            tx = params[0] if params else 0.0
            ty = params[1] if len(params) > 1 else 0.0
            m = _translate(tx, ty)
        elif name_lower == "scale":
            sx = params[0] if params else 1.0
            sy = params[1] if len(params) > 1 else sx
            m = _scale(sx, sy)
        elif name_lower == "rotate":
            if len(params) == 1:
                m = _rotate(params[0])
            elif len(params) == 3:
                m = _rotate(params[0], params[1], params[2])
            else:
                continue
        elif name_lower == "skewx" and params:
            m = _skew_x(params[0])
        elif name_lower == "skewy" and params:
            m = _skew_y(params[0])
        else:
            continue
        result = result @ m
    return result


def _local_name(node: ET.Element) -> str:
    if isinstance(node.tag, str):
        if node.tag.startswith("{"):
            return node.tag.split("}", 1)[1]
        return node.tag
    return ""


def _apply_transform(points: Iterable[Tuple[float, float]], matrix: np.ndarray) -> List[Tuple[float, float]]:
    pts = []
    for x, y in points:
        vec = matrix @ np.array([x, y, 1.0])
        pts.append((float(vec[0]), float(vec[1])))
    return pts


def _line_points(x1: float, y1: float, x2: float, y2: float) -> List[Tuple[float, float]]:
    return [(x1, y1), (x2, y2)]


def _rect_polylines(x: float, y: float, width: float, height: float) -> List[List[Tuple[float, float]]]:
    if width == 0 or height == 0:
        return []
    p1 = (x, y)
    p2 = (x + width, y)
    p3 = (x + width, y + height)
    p4 = (x, y + height)
    return [[p1, p2], [p2, p3], [p3, p4], [p4, p1]]


def _points_from_string(points_str: str) -> List[Tuple[float, float]]:
    cleaned = _WS_RE.sub(" ", points_str.strip().replace(",", " "))
    parts = [p for p in cleaned.split(" ") if p]
    coords: List[Tuple[float, float]] = []
    it = iter(parts)
    for x_str, y_str in zip(it, it):
        coords.append((float(x_str), float(y_str)))
    return coords


def _flatten_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tol: float) -> List[Tuple[float, float]]:
    def flat_enough(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray, pd: np.ndarray) -> bool:
        line = pd - pa
        if np.allclose(line[:2], 0.0):
            return True
        norm = math.hypot(line[0], line[1])
        if norm == 0:
            return True
        distances = []
        for ctrl in (pb, pc):
            vec = ctrl - pa
            dist = abs(line[0] * vec[1] - line[1] * vec[0]) / norm
            distances.append(dist)
        return max(distances) <= tol

    if flat_enough(p0, p1, p2, p3):
        return [(float(p0[0]), float(p0[1])), (float(p3[0]), float(p3[1]))]

    p01 = (p0 + p1) / 2.0
    p12 = (p1 + p2) / 2.0
    p23 = (p2 + p3) / 2.0
    p012 = (p01 + p12) / 2.0
    p123 = (p12 + p23) / 2.0
    p0123 = (p012 + p123) / 2.0

    left = _flatten_cubic(p0, p01, p012, p0123, tol)
    right = _flatten_cubic(p0123, p123, p23, p3, tol)
    return left[:-1] + right


def _flatten_quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, tol: float) -> List[Tuple[float, float]]:
    line = p2 - p0
    norm = math.hypot(line[0], line[1])
    if norm == 0:
        return [(float(p0[0]), float(p0[1])), (float(p2[0]), float(p2[1]))]
    vec = p1 - p0
    dist = abs(line[0] * vec[1] - line[1] * vec[0]) / norm
    if dist <= tol:
        return [(float(p0[0]), float(p0[1])), (float(p2[0]), float(p2[1]))]

    p01 = (p0 + p1) / 2.0
    p12 = (p1 + p2) / 2.0
    p012 = (p01 + p12) / 2.0
    left = _flatten_quadratic(p0, p01, p012, tol)
    right = _flatten_quadratic(p012, p12, p2, tol)
    return left[:-1] + right


def _arc_to_cubic(p0: Tuple[float, float], rx: float, ry: float, phi_deg: float, large_arc: int, sweep: int, p1: Tuple[float, float]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

        p_start = np.array([
            cx + rx_abs * (cos_phi * cos_t1 - sin_phi * sin_t1),
            cy + ry_abs * (sin_phi * cos_t1 + cos_phi * sin_t1),
        ])
        p_end = np.array([
            cx + rx_abs * (cos_phi * cos_t2 - sin_phi * sin_t2),
            cy + ry_abs * (sin_phi * cos_t2 + cos_phi * sin_t2),
        ])
        dx1 = -rx_abs * (cos_phi * sin_t1 + sin_phi * cos_t1)
        dy1 = -ry_abs * (sin_phi * sin_t1 - cos_phi * cos_t1)
        dx2 = -rx_abs * (cos_phi * sin_t2 + sin_phi * cos_t2)
        dy2 = -ry_abs * (sin_phi * sin_t2 - cos_phi * cos_t2)

        ctrl1 = p_start + np.array([dx1, dy1]) * e
        ctrl2 = p_end - np.array([dx2, dy2]) * e
        result.append((p_start, ctrl1, ctrl2, p_end))

    return result


def _flatten_arc(p0: Tuple[float, float], rx: float, ry: float, phi: float, large_arc: int, sweep: int, p1: Tuple[float, float], tol: float) -> List[Tuple[float, float]]:
    cubics = _arc_to_cubic(p0, rx, ry, phi, large_arc, sweep, p1)
    if not cubics:
        return [p0, p1]
    pts: List[Tuple[float, float]] = []
    for idx, (c0, c1, c2, c3) in enumerate(cubics):
        segment_pts = _flatten_cubic(c0, c1, c2, c3, tol)
        if idx:
            pts.extend(segment_pts[1:])
        else:
            pts.extend(segment_pts)
    return pts


def _path_to_polylines(d: str, tol: float) -> List[List[Tuple[float, float]]]:
    tokens = list(_tokenise_path(d))
    if not tokens:
        return []

    it = iter(tokens)
    polylines: List[List[Tuple[float, float]]] = []
    current_point = np.array([0.0, 0.0])
    start_point = np.array([0.0, 0.0])
    last_ctrl = np.array([0.0, 0.0])
    last_cmd = ""
    current_poly: Optional[List[Tuple[float, float]]] = None

    def ensure_poly(point: np.ndarray) -> None:
        nonlocal current_poly
        if current_poly is None:
            current_poly = [(float(point[0]), float(point[1]))]
            polylines.append(current_poly)

    command = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
        elif command is None:
            raise ValueError("Path data malformed")
        else:
            token = command

        cmd = token
        absolute = cmd.isupper()
        cmd_lower = cmd.lower()

        def read_numbers(count: int) -> List[float]:
            nonlocal index
            nums: List[float] = []
            for _ in range(count):
                if index >= len(tokens):
                    raise ValueError("Unexpected end of path data")
                nums.append(_parse_number(tokens[index]))
                index += 1
            return nums

        if cmd_lower == "m":
            coords = read_numbers(2)
            current_poly = None
            if absolute:
                current_point = np.array(coords)
            else:
                current_point = current_point + np.array(coords)
            start_point = current_point.copy()
            ensure_poly(current_point)
            last_cmd = "m"
            while index < len(tokens) and not tokens[index].isalpha():
                coords = read_numbers(2)
                delta = np.array(coords)
                target = delta if absolute else current_point + delta
                ensure_poly(current_point)
                current_poly.append((float(target[0]), float(target[1])))
                current_point = target
                last_cmd = "l"
        elif cmd_lower == "z":
            if current_poly is not None and current_poly[0] != current_poly[-1]:
                current_poly.append(current_poly[0])
            current_point = start_point.copy()
            current_poly = None
            last_cmd = "z"
        elif cmd_lower == "l":
            while index < len(tokens) and not tokens[index].isalpha():
                coords = read_numbers(2)
                delta = np.array(coords)
                target = delta if absolute else current_point + delta
                ensure_poly(current_point)
                current_poly.append((float(target[0]), float(target[1])))
                current_point = target
            last_cmd = "l"
        elif cmd_lower == "h":
            while index < len(tokens) and not tokens[index].isalpha():
                (value,) = read_numbers(1)
                x = value if absolute else current_point[0] + value
                target = np.array([x, current_point[1]])
                ensure_poly(current_point)
                current_poly.append((float(target[0]), float(target[1])))
                current_point = target
            last_cmd = "h"
        elif cmd_lower == "v":
            while index < len(tokens) and not tokens[index].isalpha():
                (value,) = read_numbers(1)
                y = value if absolute else current_point[1] + value
                target = np.array([current_point[0], y])
                ensure_poly(current_point)
                current_poly.append((float(target[0]), float(target[1])))
                current_point = target
            last_cmd = "v"
        elif cmd_lower == "c":
            while index < len(tokens) and not tokens[index].isalpha():
                values = read_numbers(6)
                pts = np.array(values).reshape(3, 2)
                if absolute:
                    c1 = pts[0]
                    c2 = pts[1]
                    end = pts[2]
                else:
                    c1 = current_point + pts[0]
                    c2 = current_point + pts[1]
                    end = current_point + pts[2]
                ensure_poly(current_point)
                flattened = _flatten_cubic(current_point, c1, c2, end, tol)
                current_poly.extend(flattened[1:])
                current_point = end
                last_ctrl = c2
            last_cmd = "c"
        elif cmd_lower == "s":
            while index < len(tokens) and not tokens[index].isalpha():
                values = read_numbers(4)
                pts = np.array(values).reshape(2, 2)
                if last_cmd in {"c", "s"}:
                    reflected = current_point * 2 - last_ctrl
                else:
                    reflected = current_point.copy()
                if absolute:
                    c2 = pts[0]
                    end = pts[1]
                else:
                    c2 = current_point + pts[0]
                    end = current_point + pts[1]
                c1 = reflected
                ensure_poly(current_point)
                flattened = _flatten_cubic(current_point, c1, c2, end, tol)
                current_poly.extend(flattened[1:])
                current_point = end
                last_ctrl = c2
            last_cmd = "s"
        elif cmd_lower == "q":
            while index < len(tokens) and not tokens[index].isalpha():
                values = read_numbers(4)
                pts = np.array(values).reshape(2, 2)
                if absolute:
                    ctrl = pts[0]
                    end = pts[1]
                else:
                    ctrl = current_point + pts[0]
                    end = current_point + pts[1]
                ensure_poly(current_point)
                flattened = _flatten_quadratic(current_point, ctrl, end, tol)
                current_poly.extend(flattened[1:])
                current_point = end
                last_ctrl = ctrl
            last_cmd = "q"
        elif cmd_lower == "t":
            while index < len(tokens) and not tokens[index].isalpha():
                values = read_numbers(2)
                if last_cmd in {"q", "t"}:
                    ctrl = current_point * 2 - last_ctrl
                else:
                    ctrl = current_point.copy()
                delta = np.array(values)
                end = delta if absolute else current_point + delta
                ensure_poly(current_point)
                flattened = _flatten_quadratic(current_point, ctrl, end, tol)
                current_poly.extend(flattened[1:])
                current_point = end
                last_ctrl = ctrl
            last_cmd = "t"
        elif cmd_lower == "a":
            while index < len(tokens) and not tokens[index].isalpha():
                values = read_numbers(7)
                rx, ry, phi, large_arc, sweep, x, y = values
                target = np.array([x, y])
                if not absolute:
                    target = current_point + target
                ensure_poly(current_point)
                arc_points = _flatten_arc(
                    (float(current_point[0]), float(current_point[1])),
                    rx,
                    ry,
                    phi,
                    int(large_arc),
                    int(sweep),
                    (float(target[0]), float(target[1])),
                    tol,
                )
                current_poly.extend(arc_points[1:])
                current_point = target
                last_ctrl = current_point.copy()
            last_cmd = "a"
        else:
            raise ValueError(f"Unsupported path command: {cmd}")

    return polylines


@dataclass
class LineFit:
    segment: Segment
    delta_max: float
    delta_rms: float
    angle_deg: float
    angle_spread_deg: float


def _fit_polyline(points: Sequence[Tuple[float, float]]) -> Optional[LineFit]:
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=float)
    diffs = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(diffs, axis=1)
    if np.allclose(lengths.sum(), 0.0):
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None
    direction = direction / norm

    projections = centered @ direction
    residuals = centered - np.outer(projections, direction)
    distances = np.linalg.norm(residuals, axis=1)
    delta_max = float(np.max(distances))
    delta_rms = float(math.sqrt(np.mean(distances**2)))

    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    p1 = centroid + direction * t_min
    p2 = centroid + direction * t_max

    angle = math.degrees(math.atan2(direction[1], direction[0]))

    unit_vectors = []
    for diff, length in zip(diffs, lengths):
        if length > 1e-9:
            unit_vectors.append(diff / length)
    angle_spread = 0.0
    for i in range(len(unit_vectors)):
        for j in range(i + 1, len(unit_vectors)):
            dot = float(np.clip(np.dot(unit_vectors[i], unit_vectors[j]), -1.0, 1.0))
            diff_angle = math.degrees(math.acos(dot))
            angle_spread = max(angle_spread, diff_angle)

    segment = Segment(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
    return LineFit(segment, delta_max, delta_rms, angle, angle_spread)


def _add_segments_from_polyline(points: List[Tuple[float, float]], diag: float, cfg: dict, segments: List[Segment]) -> None:
    tol_dev = float(cfg.get("straight_max_dev_rel", 0.001)) * diag
    tol_angle = float(cfg.get("straight_max_angle_spread_deg", 4.0))

    fit = _fit_polyline(points)
    if fit and fit.delta_max <= tol_dev and fit.angle_spread_deg <= tol_angle:
        if fit.segment.x1 != fit.segment.x2 or fit.segment.y1 != fit.segment.y2:
            segments.append(fit.segment)
        return

    for a, b in zip(points[:-1], points[1:]):
        if a == b:
            continue
        segments.append(Segment(a[0], a[1], b[0], b[1]))


def _segment_length(seg: Segment) -> float:
    return float(math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1))


def _segment_angle(seg: Segment) -> Optional[float]:
    length = _segment_length(seg)
    if length <= 1e-9:
        return None
    angle = math.degrees(math.atan2(seg.y2 - seg.y1, seg.x2 - seg.x1)) % 180.0
    if angle < 0:
        angle += 180.0
    return float(angle)


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return float(diff)


def _point_line_distance(pt: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    x0, y0 = pt
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom <= 1e-12:
        return math.hypot(x0 - x1, y0 - y1)
    return abs((x0 - x1) * dy - (y0 - y1) * dx) / denom


def _max_perpendicular_offset(a: Segment, b: Segment) -> float:
    start = (a.x1, a.y1)
    end = (a.x2, a.y2)
    offsets = [
        _point_line_distance((b.x1, b.y1), start, end),
        _point_line_distance((b.x2, b.y2), start, end),
    ]
    return float(max(offsets))


def _merge_collinear_segments(segments: List[Segment], diag: float, cfg: dict) -> List[Segment]:
    merge_cfg = cfg.get("merge") or {}
    if not merge_cfg.get("enable", False):
        return segments

    if len(segments) < 2:
        return segments

    angle_tol = float(merge_cfg.get("collinear_angle_tol_deg", 3.0))
    gap_tol = float(merge_cfg.get("gap_max_rel", 0.003)) * diag
    offset_tol = float(merge_cfg.get("offset_tol_rel", 0.002)) * diag

    angles: List[Optional[float]] = []
    for seg in segments:
        angles.append(_segment_angle(seg))

    bucket_size = max(gap_tol, 1e-6)

    parent = list(range(len(segments)))
    rank = [0] * len(segments)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1

    buckets: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = defaultdict(list)

    for idx, seg in enumerate(segments):
        endpoints = ((seg.x1, seg.y1), (seg.x2, seg.y2))
        for x, y in endpoints:
            cell = (int(round(x / bucket_size)), int(round(y / bucket_size)))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates = buckets.get((cell[0] + dx, cell[1] + dy))
                    if not candidates:
                        continue
                    for other_idx, ox, oy in candidates:
                        if other_idx == idx:
                            continue
                        ang_a = angles[idx]
                        ang_b = angles[other_idx]
                        if ang_a is None or ang_b is None:
                            continue
                        if _angle_diff_deg(ang_a, ang_b) > angle_tol:
                            continue
                        if math.hypot(x - ox, y - oy) > gap_tol:
                            continue
                        offset = max(
                            _max_perpendicular_offset(seg, segments[other_idx]),
                            _max_perpendicular_offset(segments[other_idx], seg),
                        )
                        if offset > offset_tol:
                            continue
                        union(idx, other_idx)
            buckets[cell].append((idx, x, y))

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(segments)):
        components[find(idx)].append(idx)

    merged: List[Segment] = []
    for _, comp in sorted(components.items(), key=lambda item: min(item[1])):
        if len(comp) == 1:
            merged.append(segments[comp[0]])
            continue

        points: List[Tuple[float, float]] = []
        for idx in comp:
            seg = segments[idx]
            points.append((seg.x1, seg.y1))
            points.append((seg.x2, seg.y2))

        fit = _fit_polyline(points)
        if fit and fit.segment and fit.delta_max <= offset_tol:
            seg_fit = fit.segment
            if _segment_length(seg_fit) > 1e-9:
                merged.append(seg_fit)
                continue

        for idx in comp:
            merged.append(segments[idx])

    return merged


def _extract_viewbox_matrix(node: ET.Element) -> np.ndarray:
    view_box = node.get("viewBox")
    width = _parse_length(node.get("width"))
    height = _parse_length(node.get("height"))
    x = _parse_length(node.get("x")) or 0.0
    y = _parse_length(node.get("y")) or 0.0
    matrix = _translate(x, y)
    if not view_box:
        return matrix

    parts = [float(v) for v in view_box.replace(",", " ").split() if v]
    if len(parts) != 4:
        return matrix
    min_x, min_y, vb_w, vb_h = parts
    if vb_w == 0 or vb_h == 0:
        return matrix @ _translate(-min_x, -min_y)

    preserve = node.get("preserveAspectRatio", "xMidYMid meet").strip()
    if preserve == "none":
        sx = (width / vb_w) if width else 1.0
        sy = (height / vb_h) if height else 1.0
        return matrix @ _scale(sx, sy) @ _translate(-min_x, -min_y)

    sx = (width / vb_w) if width else 1.0
    sy = (height / vb_h) if height else 1.0
    scale = min(sx if sx else 1.0, sy if sy else 1.0)

    align = preserve.split()[0] if preserve else "xMidYMid"
    align_x = 0.0
    align_y = 0.0
    if width:
        extra_x = (width - vb_w * scale)
        if "xMid" in align:
            align_x = extra_x / 2.0
        elif "xMax" in align:
            align_x = extra_x
    if height:
        extra_y = (height - vb_h * scale)
        if "YMid" in align:
            align_y = extra_y / 2.0
        elif "YMax" in align:
            align_y = extra_y

    return matrix @ _translate(align_x, align_y) @ _scale(scale, scale) @ _translate(-min_x, -min_y)


def _compute_diag(root: ET.Element) -> float:
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(v) for v in view_box.replace(",", " ").split() if v]
        if len(parts) == 4:
            _, _, w, h = parts
            diag = math.hypot(w, h)
            if diag > 0:
                return diag
    width = _parse_length(root.get("width")) or 0.0
    height = _parse_length(root.get("height")) or 0.0
    diag = math.hypot(width, height)
    if diag == 0:
        diag = 1.0
    return diag


def parse_svg_segments(svg_path: str, cfg: dict) -> List[Segment]:
    parser = ET.XMLParser(huge_tree=True, recover=True, remove_blank_text=False)
    tree = ET.parse(svg_path, parser=parser)
    root = tree.getroot()

    diag = _compute_diag(root)
    curve_tol = float(cfg.get("curve_tol_rel", 0.001)) * diag
    if curve_tol <= 0:
        curve_tol = 0.001 * diag

    segments: List[Segment] = []
    id_map: Dict[str, ET.Element] = {}
    for node in root.iter():
        node_id = node.get("id")
        if node_id:
            id_map[node_id] = node

    def process_node(node: ET.Element, matrix: np.ndarray) -> None:
        tag = _local_name(node)

        if tag in {"g", "svg", "symbol"}:
            child_matrix = matrix
            if tag in {"svg", "symbol"}:
                child_matrix = child_matrix @ _extract_viewbox_matrix(node)
            child_matrix = child_matrix @ _parse_transform(node.get("transform"))
            for child in node:
                process_node(child, child_matrix)
            return

        if tag == "use":
            href = node.get("href") or node.get("{http://www.w3.org/1999/xlink}href")
            if not href or not href.startswith("#"):
                return
            ref = id_map.get(href[1:])
            if ref is None:
                return
            tx = _parse_length(node.get("x")) or 0.0
            ty = _parse_length(node.get("y")) or 0.0
            child_matrix = matrix @ _translate(tx, ty) @ _parse_transform(node.get("transform"))
            process_node(ref, child_matrix)
            return

        shape_matrix = matrix @ _parse_transform(node.get("transform"))

        if tag == "line":
            x1 = float(node.get("x1") or 0.0)
            y1 = float(node.get("y1") or 0.0)
            x2 = float(node.get("x2") or 0.0)
            y2 = float(node.get("y2") or 0.0)
            pts = _apply_transform(_line_points(x1, y1, x2, y2), shape_matrix)
            _add_segments_from_polyline(pts, diag, cfg, segments)
        elif tag == "rect":
            x = float(node.get("x") or 0.0)
            y = float(node.get("y") or 0.0)
            width = float(node.get("width") or 0.0)
            height = float(node.get("height") or 0.0)
            polylines = _rect_polylines(x, y, width, height)
            for poly in polylines:
                pts = _apply_transform(poly, shape_matrix)
                _add_segments_from_polyline(pts, diag, cfg, segments)
        elif tag in {"polyline", "polygon"}:
            points_attr = node.get("points")
            if not points_attr:
                return
            points = _points_from_string(points_attr)
            if tag == "polygon" and points:
                points = points + [points[0]]
            pts = _apply_transform(points, shape_matrix)
            _add_segments_from_polyline(pts, diag, cfg, segments)
        elif tag == "path":
            d = node.get("d")
            if not d:
                return
            try:
                polylines = _path_to_polylines(d, curve_tol)
            except ValueError:
                return
            for poly in polylines:
                if len(poly) < 2:
                    continue
                pts = _apply_transform(poly, shape_matrix)
                _add_segments_from_polyline(pts, diag, cfg, segments)

    root_matrix = _identity()
    root_matrix = root_matrix @ _extract_viewbox_matrix(root)
    root_matrix = root_matrix @ _parse_transform(root.get("transform"))

    for child in root:
        process_node(child, root_matrix)

    merged = _merge_collinear_segments(segments, diag, cfg)
    merge_cfg = cfg.get("merge") or {}
    if merge_cfg.get("enable", False):
        before = len(segments)
        after = len(merged)
        if before == 0:
            percent_str = "0%"
        else:
            percent = ((after - before) / before) * 100.0
            if abs(percent) < 0.05:
                percent_str = "0%"
            else:
                sign = "-" if percent < 0 else "+"
                percent_str = f"{sign}{abs(percent):.0f}%"
        log.info(
            "SVG segments: %s → %s after merge (%s)",
            f"{before:,}",
            f"{after:,}",
            percent_str,
        )

    return merged


__all__ = ["parse_svg_segments", "_fit_polyline", "LineFit"]

