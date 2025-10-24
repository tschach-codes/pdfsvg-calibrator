from __future__ import annotations
from typing import Any, List, Dict, Tuple, cast
import sys
import math
import numpy as np
import pypdfium2 as pdfium

Affine = Tuple[float, float, float, float, float, float]  # (a,b,c,d,e,f)

IDENTITY: Affine = (
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
)


def _get_pdfium() -> Any:
    module = sys.modules.get("pypdfium2")
    if module is not None:
        return module
    return pdfium


def compose(parent: Affine, local: Affine) -> Affine:
    """
    Compose two 2x3 affine transforms.
    We interpret `local` as mapping local->parent.
    Then a point P_local is transformed to page space as:
        P_page = parent( local(P_local) )
    So global = parent ∘ local.
    """
    a, b, c, d, e, f = parent
    A, B, C, D, E, F = local
    return (
        a * A + b * D,
        a * B + b * E,
        a * C + b * F + c,
        d * A + e * D,
        d * B + e * E,
        d * C + e * F + f,
    )


def apply(M: Affine, x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = M
    return (a * x + b * y + c, d * x + e * y + f)


def _flatness_sq(p0, p1, p2, p3):
    # squared distance of control points from the baseline p0->p3.
    # used for adaptive subdivision
    (x0, y0) = p0
    (x3, y3) = p3
    dx = x3 - x0
    dy = y3 - y0
    # line length squared
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return 0.0

    def dist_sq(pt):
        (x, y) = pt
        # perpendicular distance from pt to line p0->p3
        # area*2 / |v|  => using cross product magnitude
        cross = abs((x - x0) * dy - (y - y0) * dx)
        # squared perpendicular distance:
        return (cross * cross) / denom

    return max(dist_sq(p1), dist_sq(p2))


def flatten_cubic(p0, p1, p2, p3, tol=0.1, out=None):
    """
    Adaptive De Casteljau subdivision.
    tol is max allowed perpendicular deviation in page units (pt).
    Returns list of points [p0, ..., p3].
    """
    if out is None:
        out = []
    if _flatness_sq(p0, p1, p2, p3) <= tol * tol:
        if not out:
            out.append(p0)
        out.append(p3)
        return out

    # subdivide
    # De Casteljau midpoints:
    def lerp(a, b):
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    p01 = lerp(p0, p1)
    p12 = lerp(p1, p2)
    p23 = lerp(p2, p3)
    p012 = lerp(p01, p12)
    p123 = lerp(p12, p23)
    p0123 = lerp(p012, p123)  # split point

    flatten_cubic(p0, p01, p012, p0123, tol, out)
    flatten_cubic(p0123, p123, p23, p3, tol, out)
    return out


def _poly_to_segments(points: List[Tuple[float, float]]) -> List[Dict[str, float]]:
    segs: List[Dict[str, float]] = []
    if len(points) < 2:
        return segs
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if x1 == x2 and y1 == y2:
            continue
        segs.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return segs


def _collect_path_segments(obj, M_here: Affine, tol_pt: float) -> List[Dict[str, float]]:
    """
    Turn a PDFium path object into straight line segments in page coords.
    We assume obj.get_path() yields an iterable of segments,
    each with .type in {"move","line","bezier","close"}, and .pos / .ctrl1 / .ctrl2.
    """
    out: List[Dict[str, float]] = []

    try:
        path = obj.get_path()
    except Exception:
        return out
    current_pt = None
    active_poly: List[Tuple[float, float]] = []
    polys: List[List[Tuple[float, float]]] = []

    for seg in path:
        st = getattr(seg, "type", None)

        if st == "move":
            if seg.pos is None:
                continue
            p = apply(M_here, seg.pos[0], seg.pos[1])
            # flush old poly
            if active_poly:
                polys.append(active_poly)
            active_poly = [p]
            current_pt = p

        elif st == "line":
            if seg.pos is None:
                continue
            p = apply(M_here, seg.pos[0], seg.pos[1])
            if not active_poly:
                active_poly = [p]
            else:
                active_poly.append(p)
            current_pt = p

        elif st == "bezier":
            if current_pt is None:
                if seg.pos is None:
                    continue
                current_pt = apply(M_here, seg.pos[0], seg.pos[1])
                if not active_poly:
                    active_poly = [current_pt]
                else:
                    active_poly.append(current_pt)
                continue
            # cubic bezier: current_pt -> ctrl1 -> ctrl2 -> pos
            if seg.ctrl1 is None or seg.ctrl2 is None or seg.pos is None:
                continue
            p0 = current_pt
            p1 = apply(M_here, seg.ctrl1[0], seg.ctrl1[1])
            p2 = apply(M_here, seg.ctrl2[0], seg.ctrl2[1])
            p3 = apply(M_here, seg.pos[0], seg.pos[1])
            flat_pts = flatten_cubic(p0, p1, p2, p3, tol=tol_pt)
            if not active_poly:
                active_poly = [flat_pts[0]]
            active_poly.extend(flat_pts[1:])
            current_pt = p3

        elif st == "rect":
            if seg.pos is None:
                continue
            x0, y0 = seg.pos
            width = float(getattr(seg, "width", 0.0) or 0.0)
            height = float(getattr(seg, "height", 0.0) or 0.0)
            rect_pts_local = [
                (x0, y0),
                (x0 + width, y0),
                (x0 + width, y0 + height),
                (x0, y0 + height),
                (x0, y0),
            ]
            rect_pts = [apply(M_here, px, py) for (px, py) in rect_pts_local]
            if active_poly:
                polys.append(active_poly)
            polys.append(rect_pts)
            active_poly = []
            current_pt = None

        elif st == "close":
            if active_poly and active_poly[0] != active_poly[-1]:
                active_poly.append(active_poly[0])
            if active_poly:
                polys.append(active_poly)
            active_poly = []
            current_pt = None

    if active_poly:
        polys.append(active_poly)

    for poly in polys:
        out.extend(_poly_to_segments(poly))

    return out


def _walk_object(obj, M_parent: Affine, tol_pt: float, sink: List[Dict[str, float]]):
    """
    Recursively descend into path/form objects, applying CTM.
    """
    # local matrix if available
    if hasattr(obj, "get_matrix"):
        try:
            raw = obj.get_matrix()
            if isinstance(raw, tuple):
                raw_vals = raw
            else:
                raw_vals = tuple(raw)
            if len(raw_vals) != 6:
                raise ValueError
            M_local = cast(Affine, tuple(float(v) for v in raw_vals))
        except Exception:
            M_local = IDENTITY
    else:
        M_local = IDENTITY

    # accumulate global M = parent ∘ local
    M_here = compose(M_parent, M_local)

    otype = getattr(obj, "type", None)

    if otype == "path" and hasattr(obj, "get_path"):
        segs = _collect_path_segments(obj, M_here, tol_pt)
        if segs:
            sink.extend(segs)

    elif otype == "form":
        # recurse into form children
        children = None
        if hasattr(obj, "get_objects"):
            try:
                children = obj.get_objects()
            except Exception:
                children = None
        if children is None and hasattr(obj, "get_objects_count"):
            try:
                n = obj.get_objects_count()
                tmp = []
                for i in range(n):
                    tmp.append(obj.get_object(i))
                children = tmp
            except Exception:
                children = None

        if children:
            for ch in children:
                _walk_object(ch, M_here, tol_pt, sink)

    else:
        # ignore text, image, shading, etc. for calibration
        pass


def extract_segments(
    pdf_path: str,
    page_index: int = 0,
    tol_pt: float = 0.1,
    curve_tol_pt: float | None = None,
) -> List[Dict[str, float]]:
    """
    Extract all path geometry on a page, recursively (including form XObjects),
    as straight line segments in page coordinates.
    """
    if curve_tol_pt is not None:
        tol_pt = float(curve_tol_pt)

    pdf_mod = _get_pdfium()
    doc = pdf_mod.PdfDocument(str(pdf_path))
    try:
        page = doc[page_index]
        try:
            # collect all top-level objects
            objects = None
            if hasattr(page, "get_objects"):
                objects = page.get_objects()
            elif hasattr(page, "get_objects_count"):
                n = page.get_objects_count()
                tmp = []
                for i in range(n):
                    tmp.append(page.get_object(i))
                objects = tmp
            else:
                objects = []

            segs: List[Dict[str, float]] = []
            for obj in objects:
                _walk_object(obj, IDENTITY, tol_pt, segs)

            # quick bbox + aspect sanity
            if len(segs) == 0:
                return segs

            xs = []
            ys = []
            for s in segs:
                xs.append(s["x1"])
                xs.append(s["x2"])
                ys.append(s["y1"])
                ys.append(s["y2"])
            xs = np.array(xs)
            ys = np.array(ys)
            span_x = float(xs.max() - xs.min())
            span_y = float(ys.max() - ys.min())

            pw, ph = page.get_size()  # (width, height) in pt
            page_ratio = ph / pw if pw != 0 else 1.0
            bbox_ratio = span_y / span_x if span_x != 0 else 1.0

            # If ratios are wildly off, warn (we keep going anyway for robustness).
            # This protects us from broken matrix chaining across arbitrary PDFs.
            if page_ratio > 0 and bbox_ratio > 0:
                rel = max(page_ratio / bbox_ratio, bbox_ratio / page_ratio)
                if rel > 1.5:
                    print(
                        "WARN: bbox/page aspect ratio mismatch "
                        f"page_ratio={page_ratio:.3f} bbox_ratio={bbox_ratio:.3f}"
                    )

            return segs
        finally:
            close_page = getattr(page, "close", None)
            if callable(close_page):
                close_page()
    finally:
        doc.close()


def debug_print_summary(label: str, segs: List[Dict[str, float]], angle_tol_deg: float = 2.0):
    """
    Optional debug helper you can call right after extract_segments().
    """
    print(f"[PDFIUM SUMMARY] {label}")
    print(f"  count={len(segs)}")
    if not segs:
        return
    xs = []
    ys = []
    angs = []
    horiz_cnt = 0
    vert_cnt = 0
    for s in segs[:20000]:  # sample
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        if ang > 90.0:
            ang = 180.0 - ang
        angs.append(ang)
        if abs(ang - 0.0) <= angle_tol_deg:
            horiz_cnt += 1
        if abs(ang - 90.0) <= angle_tol_deg:
            vert_cnt += 1
    xs = np.array(xs)
    ys = np.array(ys)
    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())
    span_x = float(max_x - min_x)
    span_y = float(max_y - min_y)
    print(
        f"  bbox=({min_x},{min_y})..({max_x},{max_y}) span=({span_x},{span_y})"
    )
    if angs:
        angs = np.array(angs)
        print(
            "  angle_folded_deg p50="
            f"{float(np.percentile(angs, 50)):.3f} p95={float(np.percentile(angs, 95)):.3f}"
        )
    print(f"  horiz_cnt≈0deg={horiz_cnt} vert_cnt≈90deg={vert_cnt}")
