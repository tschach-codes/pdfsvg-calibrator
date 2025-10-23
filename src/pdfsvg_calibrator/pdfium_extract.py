from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


Matrix = Tuple[float, float, float, float, float, float]
Point = Tuple[float, float]


def _mmul(a: Matrix, b: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = a
    A, B, C, D, E, F = b
    return (
        a1 * A + b1 * D,
        a1 * B + b1 * E,
        a1 * C + b1 * F + c1,
        d1 * A + e1 * D,
        d1 * B + e1 * E,
        d1 * C + e1 * F + f1,
    )


def _mapply(m: Matrix, x: float, y: float) -> Point:
    a, b, c, d, e, f = m
    return (a * x + b * y + c, d * x + e * y + f)


def _flatten_cubic(
    p0: Point, p1: Point, p2: Point, p3: Point, tol: float = 0.1
) -> List[Point]:
    def flat_enough(a0: Point, a1: Point, a2: Point, a3: Point) -> bool:
        x0, y0 = a0
        x3, y3 = a3
        dx = x3 - x0
        dy = y3 - y0
        denom = math.hypot(dx, dy)
        if denom <= 1e-12:
            return True
        distances = []
        for ctrl in (a1, a2):
            cx, cy = ctrl
            dist = abs((cx - x0) * dy - (cy - y0) * dx) / denom
            distances.append(dist)
        return max(distances, default=0.0) <= tol

    if flat_enough(p0, p1, p2, p3):
        return [p0, p3]

    mid01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
    mid12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    mid23 = ((p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5)
    mid012 = ((mid01[0] + mid12[0]) * 0.5, (mid01[1] + mid12[1]) * 0.5)
    mid123 = ((mid12[0] + mid23[0]) * 0.5, (mid12[1] + mid23[1]) * 0.5)
    mid = ((mid012[0] + mid123[0]) * 0.5, (mid012[1] + mid123[1]) * 0.5)

    left = _flatten_cubic(p0, mid01, mid012, mid, tol)
    right = _flatten_cubic(mid, mid123, mid23, p3, tol)
    return left[:-1] + right


def _as_segments(poly: Sequence[Point]) -> List[Dict[str, float]]:
    points = list(poly)
    out: List[Dict[str, float]] = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 == x2 and y1 == y2:
            continue
        out.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        })
    return out


def _extract_point(value) -> Point | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], (int, float)):
            return float(value[0]), float(value[1])
        if len(value) >= 1:
            first = value[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                return float(first[0]), float(first[1])
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return None


def _iter_children(container) -> Iterable:
    get_count = getattr(container, "get_objects_count", None)
    get_object = getattr(container, "get_object", None)
    if callable(get_count) and callable(get_object):
        count = get_count()
        for idx in range(count):
            child = get_object(idx)
            if child is not None:
                yield child
        return
    objects = getattr(container, "objects", None)
    if objects is not None:
        for child in objects:
            if child is not None:
                yield child


def extract_segments(
    pdf_path: str, page_index: int = 0, curve_tol_pt: float = 0.1
) -> List[Dict[str, float]]:
    """Extract flattened line segments from a PDF page using pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"Page {page_index} out of range (0..{len(doc) - 1})")
        page = doc[page_index]
        try:
            out: List[Dict[str, float]] = []
            identity: Matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

            def walk(obj, parent_matrix: Matrix) -> None:
                matrix_attr = getattr(obj, "get_matrix", None)
                if callable(matrix_attr):
                    local = matrix_attr()
                else:
                    local = getattr(obj, "matrix", identity)
                if isinstance(local, Sequence):
                    local_tuple = tuple(float(v) for v in local[:6])
                    if len(local_tuple) == 6:
                        composed = _mmul(parent_matrix, local_tuple)
                    else:
                        composed = parent_matrix
                else:
                    composed = parent_matrix

                obj_type = getattr(obj, "type", None)
                if obj_type is None and isinstance(obj, dict):
                    obj_type = obj.get("type")

                if obj_type == "path":
                    path = getattr(obj, "get_path", None)
                    if callable(path):
                        segments = path()
                    else:
                        segments = getattr(obj, "path", None)
                    if not segments:
                        return
                    current: Point | None = None
                    poly: List[Point] = []
                    for seg in segments:
                        seg_type = getattr(seg, "type", None)
                        if seg_type is None and isinstance(seg, dict):
                            seg_type = seg.get("type")
                        if seg_type is None:
                            continue
                        seg_type = str(seg_type).lower()
                        if seg_type == "move":
                            point = _extract_point(getattr(seg, "pos", None))
                            if point is None and isinstance(seg, dict):
                                point = _extract_point(seg.get("pos"))
                            if point is None:
                                continue
                            mapped = _mapply(composed, point[0], point[1])
                            if poly:
                                out.extend(_as_segments(poly))
                            poly = [mapped]
                            current = mapped
                        elif seg_type in {"line", "line_to"}:
                            point = _extract_point(getattr(seg, "pos", None))
                            if point is None and isinstance(seg, dict):
                                point = _extract_point(seg.get("pos"))
                            if point is None:
                                continue
                            mapped = _mapply(composed, point[0], point[1])
                            if not poly:
                                poly = [mapped]
                            else:
                                poly.append(mapped)
                            current = mapped
                        elif seg_type in {"bezier", "curve", "cubic", "cubic_bezier"}:
                            if current is None:
                                continue
                            pos = _extract_point(getattr(seg, "pos", None))
                            ctrl1 = _extract_point(getattr(seg, "ctrl1", None))
                            ctrl2 = _extract_point(getattr(seg, "ctrl2", None))
                            if isinstance(seg, dict):
                                pos = pos or _extract_point(seg.get("pos"))
                                ctrl1 = ctrl1 or _extract_point(seg.get("ctrl1"))
                                ctrl2 = ctrl2 or _extract_point(seg.get("ctrl2"))
                            if pos is None or ctrl1 is None or ctrl2 is None:
                                continue
                            p0 = current
                            p1 = _mapply(composed, ctrl1[0], ctrl1[1])
                            p2 = _mapply(composed, ctrl2[0], ctrl2[1])
                            p3 = _mapply(composed, pos[0], pos[1])
                            flat = _flatten_cubic(p0, p1, p2, p3, tol=curve_tol_pt)
                            if not poly:
                                poly = [flat[0]]
                            poly.extend(flat[1:])
                            current = p3
                        elif seg_type in {"close", "closepath"}:
                            if poly and poly[0] != poly[-1]:
                                poly.append(poly[0])
                            current = poly[0] if poly else None
                        else:
                            continue
                    if poly:
                        out.extend(_as_segments(poly))
                elif obj_type == "form":
                    form = getattr(obj, "get_form", None)
                    if callable(form):
                        container = form()
                    else:
                        container = getattr(obj, "form", None)
                    if container is not None:
                        for child in _iter_children(container):
                            walk(child, composed)
                else:
                    for child in _iter_children(obj):
                        walk(child, composed)

            for child in _iter_children(page):
                walk(child, identity)

            return out
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()
    finally:
        doc.close()

    return out


__all__ = ["extract_segments"]
