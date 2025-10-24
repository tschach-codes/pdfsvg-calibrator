from __future__ import annotations

import importlib
import math
import sys
from ctypes import c_float
from collections.abc import Iterable as IterableABC
from typing import Any, Dict, Iterable, List, Tuple, cast

try:  # pragma: no cover - import is validated via tests with stubs
    import pypdfium2 as pdfium  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover - fallback when dependency missing
    pdfium = None  # type: ignore[assignment]


# Matrix is always 2x3 affine in PDFium: (a,b,c,d,e,f)
# apply: (x',y') = (a*x + b*y + c, d*x + e*y + f)

Matrix = Tuple[float, float, float, float, float, float]
Point = Tuple[float, float]
_SEGMENT_KEYS = ("x1", "y1", "x2", "y2")

# Introspection snapshot (via debug.pdfium_introspect.introspect_page)
# === PAGE DUMP ===
# page type: <class 'pypdfium2._helpers.page.PdfPage'>
# page dir : ['close', 'formenv', 'gen_content', 'get_artbox', 'get_bbox', 'get_bleedbox', 'get_cropbox',
#             'get_height', 'get_mediabox', 'get_objects', 'get_rotation', 'get_size', 'get_textpage',
#             'get_trimbox', 'get_width', 'insert_obj', 'parent', 'pdf', 'raw', 'remove_obj', 'render',
#             'set_artbox', 'set_bleedbox', 'set_cropbox', 'set_mediabox', 'set_rotation', 'set_trimbox']
# -- OBJ 0 --
# type(obj): <class 'pypdfium2._helpers.pageobjects.PdfObject'>
# dir(obj): ['close', 'get_matrix', 'get_pos', 'level', 'page', 'parent', 'pdf', 'raw', 'set_matrix', 'transform', 'type']
# obj.type : 2 FPDF_PAGEOBJ_PATH
# has get_path  : False (path data must be accessed via pdfium.raw.FPDFPath_* APIs)


def _get_pdfium() -> Any:
    """Return the pypdfium2 module, honoring monkeypatched stubs."""

    module = sys.modules.get("pypdfium2")
    if module is not None:
        return module
    if pdfium is not None:
        return pdfium
    return importlib.import_module("pypdfium2")


def _compose(parent: Matrix, local: Matrix) -> Matrix:
    """Return the composition parent ∘ local (apply *local* first, then *parent*)."""

    a, b, c, d, e, f = parent
    A2, B2, C2, D2, E2, F2 = local
    return (
        a * A2 + b * D2,
        a * B2 + b * E2,
        a * C2 + b * F2 + c,
        d * A2 + e * D2,
        d * B2 + e * E2,
        d * C2 + e * F2 + f,
    )


def _apply(matrix: Matrix, x: float, y: float) -> Point:
    a, b, c, d, e, f = matrix
    return (a * x + b * y + c, d * x + e * y + f)


def _matrix_from(pdfium_matrix: Any) -> Matrix:
    values: Iterable[float]
    if hasattr(pdfium_matrix, "get"):
        values = pdfium_matrix.get()  # PdfMatrix helper exposes ``get``
    else:
        values = pdfium_matrix
    raw_vals = list(values)  # type: ignore[arg-type]
    if len(raw_vals) < 6:  # pragma: no cover - defensive, not expected
        raise ValueError("Matrix must contain 6 values")
    seq = tuple(float(raw_vals[i]) for i in range(6))
    return cast(Matrix, seq)


def _safe_matrix(obj: Any) -> Matrix:
    get_matrix = getattr(obj, "get_matrix", None)
    if not callable(get_matrix):
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    try:
        raw_matrix = get_matrix()
        return _matrix_from(raw_matrix)
    except Exception:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _flatten_cubic(p0: Point, p1: Point, p2: Point, p3: Point, tol: float) -> List[Point]:
    """Adaptive subdivision of a cubic Bezier curve using De Casteljau."""

    def _max_distance(a0: Point, a1: Point, a2: Point, a3: Point) -> float:
        x0, y0 = a0
        x3, y3 = a3
        dx = x3 - x0
        dy = y3 - y0
        denom = math.hypot(dx, dy)
        if denom <= 1e-12:
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


def _segments_from_poly(points: List[Point]) -> List[Dict[str, float]]:
    result: List[Dict[str, float]] = []
    if len(points) < 2:
        return result
    prev = points[0]
    for current in points[1:]:
        if prev == current:
            prev = current
            continue
        segment = {
            "x1": float(prev[0]),
            "y1": float(prev[1]),
            "x2": float(current[0]),
            "y2": float(current[1]),
        }
        result.append(segment)
        prev = current
    return result


def _path_segments_raw(obj: Any, matrix: Matrix, raw_mod: Any, tol_pt: float) -> List[Dict[str, float]]:
    path_handle = getattr(obj, "raw", None)
    if path_handle is None:
        return []

    count = raw_mod.FPDFPath_CountSegments(path_handle)
    polylines: List[List[Point]] = []
    current_poly: List[Point] = []
    current_point: Point | None = None

    def _flush(close: bool = False) -> None:
        nonlocal current_poly, current_point
        if close and current_poly and current_poly[0] != current_poly[-1]:
            current_poly.append(current_poly[0])
        if len(current_poly) > 1:
            polylines.append(current_poly)
        current_poly = []
        current_point = None

    def _segment_point(segment: Any) -> Point:
        x = c_float()
        y = c_float()
        raw_mod.FPDFPathSegment_GetPoint(segment, x, y)
        return _apply(matrix, float(x.value), float(y.value))

    idx = 0
    while idx < count:
        segment = raw_mod.FPDFPath_GetPathSegment(path_handle, idx)
        seg_type = raw_mod.FPDFPathSegment_GetType(segment)
        close_flag = bool(raw_mod.FPDFPathSegment_GetClose(segment))

        if seg_type == getattr(raw_mod, "FPDF_SEGMENT_MOVETO", object()):
            point = _segment_point(segment)
            if len(current_poly) > 1:
                _flush()
            current_poly = [point]
            current_point = point
            if close_flag:
                _flush(close=True)
            idx += 1
            continue

        if seg_type == getattr(raw_mod, "FPDF_SEGMENT_LINETO", object()):
            point = _segment_point(segment)
            if not current_poly:
                if current_point is not None:
                    current_poly = [current_point]
                else:
                    current_poly = [point]
            current_poly.append(point)
            current_point = point
            if close_flag:
                _flush(close=True)
            idx += 1
            continue

        if seg_type == getattr(raw_mod, "FPDF_SEGMENT_BEZIERTO", object()):
            if idx + 2 >= count:
                break
            ctrl1 = _segment_point(segment)
            ctrl2_segment = raw_mod.FPDFPath_GetPathSegment(path_handle, idx + 1)
            ctrl2 = _segment_point(ctrl2_segment)
            end_segment = raw_mod.FPDFPath_GetPathSegment(path_handle, idx + 2)
            end_point = _segment_point(end_segment)
            close_flag = bool(raw_mod.FPDFPathSegment_GetClose(end_segment))

            if current_point is None:
                if current_poly:
                    current_point = current_poly[-1]
                else:
                    current_point = ctrl1
                    current_poly = [current_point]

            curve_points = _flatten_cubic(current_point, ctrl1, ctrl2, end_point, tol=tol_pt)
            if not current_poly:
                current_poly = [curve_points[0]]
            current_poly.extend(curve_points[1:])
            current_point = end_point

            if close_flag:
                _flush(close=True)
            idx += 3
            continue

        # Unknown segment types are skipped gracefully
        idx += 1

    if len(current_poly) > 1:
        polylines.append(current_poly)

    segments: List[Dict[str, float]] = []
    for poly in polylines:
        segments.extend(_segments_from_poly(poly))
    return segments


def _path_segments_stub(obj: Any, matrix: Matrix, tol_pt: float) -> List[Dict[str, float]]:
    if not hasattr(obj, "get_path"):
        return []
    path = obj.get_path()
    if path is None:
        return []

    current_point: Point | None = None
    current_poly: List[Point] = []
    polylines: List[List[Point]] = []

    def _start_new(point: Point) -> None:
        nonlocal current_point, current_poly
        if len(current_poly) > 1:
            polylines.append(current_poly)
        current_poly = [point]
        current_point = point

    def _close_poly() -> None:
        nonlocal current_poly, current_point
        if current_poly and current_poly[0] != current_poly[-1]:
            current_poly.append(current_poly[0])
        if len(current_poly) > 1:
            polylines.append(current_poly)
        current_poly = []
        current_point = None

    for segment in path:
        seg_type = getattr(segment, "type", None)
        seg_name = str(seg_type).lower()

        if seg_name.startswith("move"):
            pos = getattr(segment, "pos", None)
            if pos is None or len(pos) < 2:
                continue
            point = _apply(matrix, float(pos[0]), float(pos[1]))
            _start_new(point)
        elif seg_name.startswith("line"):
            pos = getattr(segment, "pos", None)
            if pos is None or len(pos) < 2:
                continue
            point = _apply(matrix, float(pos[0]), float(pos[1]))
            if not current_poly:
                if current_point is None:
                    current_poly = [point]
                else:
                    current_poly = [current_point]
            current_poly.append(point)
            current_point = point
        elif seg_name in {"rect", "rectangle"}:
            pos = getattr(segment, "pos", None)
            width = getattr(segment, "width", None)
            height = getattr(segment, "height", None)
            if pos is None or width is None or height is None:
                continue
            x0, y0 = float(pos[0]), float(pos[1])
            w = float(width)
            h = float(height)
            rect = [
                _apply(matrix, x0, y0),
                _apply(matrix, x0 + w, y0),
                _apply(matrix, x0 + w, y0 + h),
                _apply(matrix, x0, y0 + h),
                _apply(matrix, x0, y0),
            ]
            polylines.append(rect)
            current_poly = []
            current_point = rect[-1]
        elif seg_name.startswith("bezier") or seg_name.endswith("bezier"):
            if current_point is None:
                continue
            ctrl1 = getattr(segment, "ctrl1", None)
            ctrl2 = getattr(segment, "ctrl2", None)
            pos = getattr(segment, "pos", None)
            if (
                ctrl1 is None
                or ctrl2 is None
                or pos is None
                or len(ctrl1) < 2
                or len(ctrl2) < 2
                or len(pos) < 2
            ):
                continue
            p1 = _apply(matrix, float(ctrl1[0]), float(ctrl1[1]))
            p2 = _apply(matrix, float(ctrl2[0]), float(ctrl2[1]))
            p3 = _apply(matrix, float(pos[0]), float(pos[1]))
            curve_points = _flatten_cubic(current_point, p1, p2, p3, tol=tol_pt)
            if not current_poly:
                current_poly = [curve_points[0]]
            current_poly.extend(curve_points[1:])
            current_point = p3
        elif seg_name.startswith("close"):
            if current_poly:
                _close_poly()

    if len(current_poly) > 1:
        polylines.append(current_poly)

    segments: List[Dict[str, float]] = []
    for poly in polylines:
        segments.extend(_segments_from_poly(poly))
    return segments


def _path_segments(obj: Any, matrix: Matrix, raw_mod: Any | None, tol_pt: float) -> List[Dict[str, float]]:
    if raw_mod is not None and getattr(obj, "raw", None) is not None:
        try:
            return _path_segments_raw(obj, matrix, raw_mod, tol_pt)
        except Exception:  # pragma: no cover - raw iteration fallback
            pass
    return _path_segments_stub(obj, matrix, tol_pt)


def _iter_objects(container: Any) -> Iterable[Any]:
    getter = getattr(container, "get_objects", None)
    if callable(getter):
        objs = getter()
        if objs is not None:
            for item in objs:
                yield item
            return

    count_getter = getattr(container, "get_objects_count", None)
    index_getter = getattr(container, "get_object", None)
    if callable(count_getter) and callable(index_getter):
        try:
            count = int(count_getter())
        except Exception:
            count = 0
        for idx in range(max(0, count)):
            yield index_getter(idx)
        return

    objects_attr = getattr(container, "objects", None)
    if isinstance(objects_attr, IterableABC):
        for item in objects_attr:
            yield item


def _is_path_type(obj_type: Any, raw_mod: Any | None) -> bool:
    if obj_type is None:
        return False
    if isinstance(obj_type, str):
        return obj_type.lower() == "path"
    if raw_mod is not None and hasattr(raw_mod, "FPDF_PAGEOBJ_PATH"):
        return obj_type == raw_mod.FPDF_PAGEOBJ_PATH
    return False


def _is_form_type(obj_type: Any, raw_mod: Any | None) -> bool:
    if obj_type is None:
        return False
    if isinstance(obj_type, str):
        return obj_type.lower() == "form"
    if raw_mod is not None and hasattr(raw_mod, "FPDF_PAGEOBJ_FORM"):
        return obj_type == raw_mod.FPDF_PAGEOBJ_FORM
    return False


def _iter_form_children(obj: Any, raw_mod: Any | None, pdfium_mod: Any) -> Iterable[Any]:
    if raw_mod is not None and getattr(obj, "raw", None) is not None:
        try:
            count = raw_mod.FPDFFormObj_CountObjects(obj.raw)
        except Exception:
            count = 0
        for idx in range(count):
            child_raw = raw_mod.FPDFFormObj_GetObject(obj.raw, idx)
            if not child_raw:
                continue
            yield pdfium_mod.PdfObject(
                child_raw,
                page=getattr(obj, "page", None),
                pdf=getattr(obj, "pdf", None),
                level=getattr(obj, "level", 0) + 1,
            )
        return

    yield from _iter_objects(obj)


def extract_segments(
    pdf_path: str,
    page_index: int = 0,
    tol_pt: float = 0.1,
    **legacy_kwargs: float,
) -> List[Dict[str, float]]:
    """Return all path segments on the given page (recursing into Form XObjects)."""

    if "curve_tol_pt" in legacy_kwargs:
        tol_pt = float(legacy_kwargs.pop("curve_tol_pt"))
    if legacy_kwargs:
        extras = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword arguments: {extras}")

    pdfium_mod = _get_pdfium()
    raw_mod: Any | None = getattr(pdfium_mod, "raw", None)
    if raw_mod is None:
        try:
            raw_mod = importlib.import_module("pypdfium2.raw")
        except Exception:  # pragma: no cover - fallback when raw bindings unavailable
            raw_mod = None

    doc = pdfium_mod.PdfDocument(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"Page {page_index} out of range (0..{len(doc) - 1})")
        page = doc[page_index]
        try:
            base_matrix: Matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            if hasattr(page, "get_matrix") and callable(page.get_matrix):  # type: ignore[attr-defined]
                try:
                    base_matrix = _matrix_from(page.get_matrix())  # type: ignore[attr-defined]
                except Exception:
                    base_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

            segments: List[Dict[str, float]] = []

            def walk(obj: Any, parent_matrix: Matrix) -> None:
                local_matrix = _safe_matrix(obj)
                composed = _compose(parent_matrix, local_matrix)
                obj_type = getattr(obj, "type", None)

                if _is_path_type(obj_type, raw_mod):
                    segments.extend(_path_segments(obj, composed, raw_mod, tol_pt))
                    return

                if _is_form_type(obj_type, raw_mod):
                    for child in _iter_form_children(obj, raw_mod, pdfium_mod):
                        if child is not None:
                            walk(child, composed)
                    return

                for child in _iter_objects(obj):
                    if child is not None:
                        walk(child, composed)

            for top_obj in _iter_objects(page):
                if top_obj is not None:
                    walk(top_obj, base_matrix)

            return segments
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()
    finally:
        doc.close()


def debug_print_summary(segments: List[Dict[str, float]]) -> None:
    if not segments:
        print("segments: 0 entries")
        return

    xs = [seg["x1"] for seg in segments] + [seg["x2"] for seg in segments]
    ys = [seg["y1"] for seg in segments] + [seg["y2"] for seg in segments]
    print(f"segments: {len(segments)} entries")
    print(f"  x range: {min(xs):.3f} .. {max(xs):.3f}")
    print(f"  y range: {min(ys):.3f} .. {max(ys):.3f}")
    sample = segments[: min(5, len(segments))]
    for idx, seg in enumerate(sample):
        coords = ", ".join(f"{key}={seg[key]:.3f}" for key in _SEGMENT_KEYS)
        print(f"  [{idx}] {coords}")


__all__ = ["extract_segments", "debug_print_summary"]
