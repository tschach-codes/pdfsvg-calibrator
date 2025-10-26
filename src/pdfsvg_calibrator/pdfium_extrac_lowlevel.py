from __future__ import annotations
from ctypes import c_double, byref
import pypdfium2 as pdfium
from pypdfium2 import raw as pdfraw

# PDFium object type enums (stable in PDFium)
PDFIUM_PAGEOBJ_PATH    = 1
PDFIUM_PAGEOBJ_TEXT    = 2
PDFIUM_PAGEOBJ_IMAGE   = 3
PDFIUM_PAGEOBJ_SHADING = 4
PDFIUM_PAGEOBJ_FORM    = 5

def _compose_matrix(parent, child):
    """
    Compose two PdfMatrix transforms: parent ∘ child.
    PdfMatrix is (a,b,c,d,e,f) meaning:
      [ a c e ]
      [ b d f ]
      [ 0 0 1 ]
    """
    pa, pb, pc, pd, pe, pf = parent
    ca, cb, cc, cd, ce, cf = child

    return pdfium.PdfMatrix(
        pa*ca + pc*cb,
        pb*ca + pd*cb,
        pa*cc + pc*cd,
        pb*cc + pd*cd,
        pa*ce + pc*cf + pe,
        pb*ce + pd*cf + pf,
    )

def _apply_matrix(mat, x, y):
    """
    Apply PdfMatrix to point (x,y).
    """
    a,b,c,d,e,f = mat
    X = a*x + c*y + e
    Y = b*x + d*y + f
    return (X, Y)

def _iter_page_objects(page):
    """
    Yield all top-level objects from a PdfPage, regardless if get_objects()
    returns generator or if we have to index manually.
    """
    if hasattr(page, "get_objects"):
        try:
            for obj in page.get_objects():
                yield obj
            return
        except Exception:
            pass

    if hasattr(page, "get_objects_count"):
        try:
            n = page.get_objects_count()
            for i in range(n):
                yield page.get_object(i)
            return
        except Exception:
            pass

    # nothing else we can do
    return

def _iter_form_children(obj):
    """
    If obj is a FORM object (XObject), try to iterate its kids.
    On your dump, get_objects / get_objects_count was False,
    aber wir bauen es trotzdem ein für Robustheit.
    """
    if hasattr(obj, "get_objects"):
        try:
            for kid in obj.get_objects():
                yield kid
            return
        except Exception:
            pass

    if hasattr(obj, "get_objects_count"):
        try:
            n = obj.get_objects_count()
            for i in range(n):
                yield obj.get_object(i)
            return
        except Exception:
            pass

    # form without exposed children API in this build -> nothing
    return

def _extract_path_segments_from_obj(obj, ctm, out_segments):
    """
    obj: PdfPageObject with type == PATH
    ctm: current transform matrix (PdfMatrix)
    out_segments: append tuples (x0,y0,x1,y1)
    """
    handle = getattr(obj, "_obj", None)
    if handle is None:
        return

    # Count PDFium path segments
    count = pdfraw.FPDFPath_CountSegments(handle)

    # We'll reconstruct line strips:
    # keep track of current pen position from MoveTo / LineTo / BezierTo
    pen = None
    subpath_start = None  # track polygon starts for Close

    for i in range(count):
        seg = pdfraw.FPDFPath_GetPathSegment(handle, i)
        seg_type = pdfraw.FPDFPathSegment_GetType(seg)
        # PDFium values:
        # 1 = MoveTo
        # 2 = LineTo
        # 3 = BezierTo
        # 4 = Close

        x = c_double()
        y = c_double()
        pdfraw.FPDFPathSegment_GetPoint(seg, byref(x), byref(y))
        px, py = float(x.value), float(y.value)

        X, Y = _apply_matrix(ctm, px, py)

        if seg_type == 1:  # MoveTo
            pen = (X, Y)
            subpath_start = (X, Y)

        elif seg_type == 2:  # LineTo
            if pen is not None:
                x0, y0 = pen
                x1, y1 = X, Y
                if x0 != x1 or y0 != y1:
                    out_segments.append((x0, y0, x1, y1))
            pen = (X, Y)

        elif seg_type == 3:  # BezierTo
            # We don't approximate curve; we just advance pen to this new point.
            pen = (X, Y)

        elif seg_type == 4:  # ClosePath
            # Close current subpath: draw line to start if distinct
            if pen is not None and subpath_start is not None:
                x0, y0 = pen
                x1, y1 = subpath_start
                if x0 != x1 or y0 != y1:
                    out_segments.append((x0, y0, x1, y1))
            # after close, pen becomes start
            pen = subpath_start

def _walk_object(obj, parent_ctm, out_segments):
    """
    Walk one object (PATH, FORM, ...) and append all line segments.
    This handles transform composition.
    """
    otype = getattr(obj, "type", None)

    # get this object's own matrix (or identity if missing)
    if hasattr(obj, "get_matrix"):
        try:
            mat_local = obj.get_matrix()
        except Exception:
            mat_local = pdfium.PdfMatrix(1,0,0,1,0,0)
    else:
        mat_local = pdfium.PdfMatrix(1,0,0,1,0,0)

    # compose
    ctm = _compose_matrix(parent_ctm, mat_local)

    if otype == PDFIUM_PAGEOBJ_PATH:
        _extract_path_segments_from_obj(obj, ctm, out_segments)

    elif otype == PDFIUM_PAGEOBJ_FORM:
        # recurse for each kid in the form, using ctm as new parent_ctm
        for kid in _iter_form_children(obj):
            _walk_object(kid, ctm, out_segments)

    # TEXT / IMAGE etc. are ignored

def extract_segments_pdfium_lowlevel(pdf_path: str, page_index: int = 0):
    """
    Public: read a page, return list of (x0,y0,x1,y1) line segments in page coords.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]

    segments: list[tuple[float,float,float,float]] = []
    identity = pdfium.PdfMatrix(1,0,0,1,0,0)

    for obj in _iter_page_objects(page):
        _walk_object(obj, identity, segments)

    return segments
