from __future__ import annotations

import importlib
import math
import sys
from typing import Any, Dict, List, Tuple, cast

try:  # pragma: no cover - import is validated via tests with stubs
    import pypdfium2 as pdfium  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency missing
    pdfium = None  # type: ignore[assignment]


# Matrix is always 2x3 affine in PDFium: (a,b,c,d,e,f)
# apply: (x',y') = (a*x + b*y + c, d*x + e*y + f)


Matrix = Tuple[float, float, float, float, float, float]
Point = Tuple[float, float]


def _get_pdfium() -> Any:
    """Return the pypdfium2 module, honoring monkeypatched stubs."""

    module = sys.modules.get("pypdfium2")
    if module is not None:
        return module
    if pdfium is not None:
        return pdfium
    return importlib.import_module("pypdfium2")


def _mmul(
    A: Matrix,
    B: Matrix,
) -> Matrix:
    """Compose matrices so that B is applied first, then A."""

    a, b, c, d, e, f = A
    A2, B2, C2, D2, E2, F2 = B
    return (
        a * A2 + b * D2,
        a * B2 + b * E2,
        a * C2 + b * F2 + c,
        d * A2 + e * D2,
        d * B2 + e * E2,
        d * C2 + e * F2 + f,
    )


def _mapply(M: Matrix, x: float, y: float) -> Point:
    a, b, c, d, e, f = M
    return (a * x + b * y + c, d * x + e * y + f)


def _flatten_cubic(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    tol: float = 0.1,
) -> List[Point]:
    """Adaptive subdivision of a cubic Bezier curve using De Casteljau."""

    def _max_distance(a0: Point, a1: Point, a2: Point, a3: Point) -> float:
        x0, y0 = a0
        x3, y3 = a3
        dx = x3 - x0
        dy = y3 - y0
        denom = math.hypot(dx, dy)
        if denom <= 1e-12:
            # Degenerate case – fall back to point distances from start.
            d1 = math.hypot(a1[0] - x0, a1[1] - y0)
            d2 = math.hypot(a2[0] - x0, a2[1] - y0)
            return max(d1, d2)
        dist1 = abs((a1[0] - x0) * dy - (a1[1] - y0) * dx) / denom
        dist2 = abs((a2[0] - x0) * dy - (a2[1] - y0) * dx) / denom
        return max(dist1, dist2)

    if _max_distance(p0, p1, p2, p3) <= tol:
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


def _segments_from_poly(poly_pts: List[Point]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for i in range(len(poly_pts) - 1):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[i + 1]
        if x1 == x2 and y1 == y2:
            continue
        out.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
        })
    return out


def extract_segments(
    pdf_path: str,
    page_index: int = 0,
    tol_pt: float = 0.1,
    **legacy_kwargs: float,
) -> List[Dict[str, float]]:
    """
    Return all path segments on the given page as a flat list of dicts:
    [{"x1":..,"y1":..,"x2":..,"y2":..}, ...]
    Recurses into form XObjects. Applies full CTM chain.
    """

    if "curve_tol_pt" in legacy_kwargs:
        tol_pt = float(legacy_kwargs.pop("curve_tol_pt"))
    if legacy_kwargs:
        extra = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword arguments: {extra}")

    pdfium_mod = _get_pdfium()

    doc = pdfium_mod.PdfDocument(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"Page {page_index} out of range (0..{len(doc) - 1})")

        page = doc[page_index]
        try:
            PAGE_M: Matrix = (
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            )

            page_matrix_attr = getattr(page, "get_matrix", None)
            if callable(page_matrix_attr):
                try:
                    raw_page_matrix = page_matrix_attr()
                    if hasattr(raw_page_matrix, "get"):
                        raw_page_matrix = raw_page_matrix.get()
                    if hasattr(raw_page_matrix, "tolist"):
                        raw_page_matrix = raw_page_matrix.tolist()
                    if isinstance(raw_page_matrix, (list, tuple)) and len(raw_page_matrix) >= 6:
                        PAGE_M = cast(Matrix, tuple(float(raw_page_matrix[i]) for i in range(6)))
                except Exception:
                    pass

            out: List[Dict[str, float]] = []

            def walk(obj, M_parent: Matrix) -> None:
                get_m = getattr(obj, "get_matrix", None)
                if callable(get_m):
                    try:
                        local = get_m()
                        if hasattr(local, "get"):
                            local = local.get()
                        if hasattr(local, "tolist"):
                            local = local.tolist()
                        if isinstance(local, (list, tuple)) and len(local) >= 6:
                            M_here = _mmul(M_parent, cast(Matrix, tuple(float(local[i]) for i in range(6))))
                        else:
                            M_here = M_parent
                    except Exception:
                        M_here = M_parent
                else:
                    M_here = M_parent

                t = getattr(obj, "type", None)

                if t == "path":
                    if not hasattr(obj, "get_path"):
                        return
                    path = obj.get_path()
                    current_pt: Point | None = None
                    poly: List[Point] = []
                    polys_out: List[List[Point]] = []

                    for seg in path:
                        st = getattr(seg, "type", None)
                        st_str = str(st).lower() if st is not None else ""

                        if st_str in {"move", "moveto", "move_to"}:
                            p = _mapply(M_here, seg.pos[0], seg.pos[1])
                            if poly:
                                polys_out.append(poly)
                            poly = [p]
                            current_pt = p
                        elif st_str in {"line", "lineto", "line_to"}:
                            p = _mapply(M_here, seg.pos[0], seg.pos[1])
                            if not poly:
                                if current_pt is not None:
                                    poly = [current_pt, p]
                                else:
                                    poly = [p]
                            else:
                                poly.append(p)
                            current_pt = p
                        elif st_str in {"rect", "rectangle"}:
                            pos = getattr(seg, "pos", None)
                            width = getattr(seg, "width", None)
                            height = getattr(seg, "height", None)
                            if (
                                pos is None
                                or len(pos) < 2
                                or width is None
                                or height is None
                            ):
                                continue
                            try:
                                w = float(width)
                                h = float(height)
                            except (TypeError, ValueError):
                                continue
                            x0, y0 = float(pos[0]), float(pos[1])
                            rect_points = [
                                _mapply(M_here, x0, y0),
                                _mapply(M_here, x0 + w, y0),
                                _mapply(M_here, x0 + w, y0 + h),
                                _mapply(M_here, x0, y0 + h),
                                _mapply(M_here, x0, y0),
                            ]
                            polys_out.append(rect_points)
                            poly = []
                            current_pt = rect_points[-1]
                        elif st_str in {"bezier", "bezierto", "bezier_to", "cubic", "cubic_bezier"}:
                            if current_pt is None:
                                continue
                            ctrl1 = getattr(seg, "ctrl1", None)
                            ctrl2 = getattr(seg, "ctrl2", None)
                            pos = getattr(seg, "pos", None)
                            if (
                                ctrl1 is None
                                or ctrl2 is None
                                or pos is None
                                or len(ctrl1) < 2
                                or len(ctrl2) < 2
                                or len(pos) < 2
                            ):
                                continue
                            p1 = _mapply(M_here, ctrl1[0], ctrl1[1])
                            p2 = _mapply(M_here, ctrl2[0], ctrl2[1])
                            p3 = _mapply(M_here, pos[0], pos[1])
                            curve_pts = _flatten_cubic(current_pt, p1, p2, p3, tol=tol_pt)
                            if not poly:
                                poly = [curve_pts[0]]
                            poly.extend(curve_pts[1:])
                            current_pt = p3
                        elif st_str in {"close", "closepath", "close_path"}:
                            if poly and poly[0] != poly[-1]:
                                poly.append(poly[0])
                            if poly:
                                polys_out.append(poly)
                            poly = []
                            current_pt = None

                    if poly:
                        polys_out.append(poly)

                    for pl in polys_out:
                        out.extend(_segments_from_poly(pl))

                elif t == "form":
                    if hasattr(obj, "get_objects_count"):
                        n = obj.get_objects_count()
                        for i in range(n):
                            child = obj.get_object(i)
                            if child is not None:
                                walk(child, M_here)
                    elif hasattr(obj, "get_objects"):
                        for child in obj.get_objects():
                            if child is not None:
                                walk(child, M_here)

                else:
                    if hasattr(obj, "get_objects_count"):
                        n = obj.get_objects_count()
                        for i in range(n):
                            child = obj.get_object(i)
                            if child is not None:
                                walk(child, M_here)
                    elif hasattr(obj, "get_objects"):
                        for child in obj.get_objects():
                            if child is not None:
                                walk(child, M_here)

            if hasattr(page, "get_objects_count"):
                count = page.get_objects_count()
                for i in range(count):
                    obj = page.get_object(i)
                    if obj is not None:
                        walk(obj, PAGE_M)
            elif hasattr(page, "get_objects"):
                for obj in page.get_objects():
                    if obj is not None:
                        walk(obj, PAGE_M)

            return out
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()
    finally:
        doc.close()


__all__ = ["extract_segments"]
