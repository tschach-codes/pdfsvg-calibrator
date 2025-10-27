from __future__ import annotations
from ctypes import (
    c_double,
    byref,
    POINTER,
)
import pypdfium2 as pdfium
from pypdfium2 import raw as pdfraw

# PDFium object type enums
PDFIUM_PAGEOBJ_PATH    = 1
PDFIUM_PAGEOBJ_TEXT    = 2
PDFIUM_PAGEOBJ_IMAGE   = 3
PDFIUM_PAGEOBJ_SHADING = 4
PDFIUM_PAGEOBJ_FORM    = 5


# ---------------------------------------------------------------------------------
# Helpers to introspect / unify FS_MATRIX from your pdfraw build
# ---------------------------------------------------------------------------------

def _find_fs_matrix_cls():
    """
    Try to locate the FS_MATRIX struct class that pypdfium2.raw exposes.
    We look for something with fields a,b,c,d,e,f (case-insensitive),
    or exactly 6 float/double fields.
    Return (cls, field_names_lowercase_ordered)
    """
    candidates = []
    for name in dir(pdfraw):
        if "MATRIX" in name.upper():
            cls = getattr(pdfraw, name)
            # only consider ctypes-like classes
            if hasattr(cls, "_fields_"):
                fields = [f[0] for f in getattr(cls, "_fields_", [])]
                if len(fields) == 6:
                    candidates.append((name, cls, fields))

    # heuristic: prefer something literally named FS_MATRIX / struct__FS_MATRIX_
    # but fall back to first 6-field struct
    if not candidates:
        raise RuntimeError("Could not locate FS_MATRIX-like struct in pypdfium2.raw")

    # rank
    def score(entry):
        name, cls, fields = entry
        s = 0
        if "FS_MATRIX" in name.upper():
            s += 10
        if "MATRIX" == name.upper():
            s += 5
        return -s  # smaller is worse, but we want highest score -> lowest sort
    candidates.sort(key=score)

    best_name, best_cls, best_fields = candidates[0]

    # build a lowercase map so we can read values later regardless of exact field caps
    lower_fields = [f.lower() for f in best_fields]

    return best_cls, best_fields, lower_fields


_FS_MATRIX_CLS, _FS_FIELDS, _FS_FIELDS_LC = _find_fs_matrix_cls()


def _matrix_struct_to_tuple(mat_obj):
    """
    Convert whatever struct instance pdfraw.FS_MATRIX is into (a,b,c,d,e,f).
    We'll try to read attributes matching 'a','b','c','d','e','f' (case-insensitive),
    or fallback to positional via the declared _fields_ order.
    """
    # try by attribute names first (case-insensitive)
    vals = [None]*6
    for idx, want in enumerate(["a","b","c","d","e","f"]):
        # find field index whose lowercase matches this
        try:
            field_index = _FS_FIELDS_LC.index(want)
        except ValueError:
            field_index = None
        if field_index is not None:
            real_name = _FS_FIELDS[field_index]
            vals[idx] = float(getattr(mat_obj, real_name))
        else:
            # fallback: if attribute with that exact name exists
            if hasattr(mat_obj, want):
                vals[idx] = float(getattr(mat_obj, want))

    # fill anything missing by raw _fields_ order
    for i in range(6):
        if vals[i] is None:
            fname = _FS_FIELDS[i]
            vals[i] = float(getattr(mat_obj, fname))

    return tuple(vals)  # (a,b,c,d,e,f)


def _get_obj_matrix6(obj_handle):
    """
    Get the CTM of this PDF object by calling FPDFPageObj_GetMatrix with
    the EXACT struct type pdfraw expects.

    Your build's signature wants a pointer to *its* FS_MATRIX struct type,
    NOT a byref() of our own struct. So:
    - create an instance of that struct class
    - create POINTER(thatclass)(instance)
    - call
    """
    mat = _FS_MATRIX_CLS()
    mat_ptr = POINTER(_FS_MATRIX_CLS)(mat)
    ok = pdfraw.FPDFPageObj_GetMatrix(obj_handle, mat_ptr)
    if ok:
        return _matrix_struct_to_tuple(mat)
    # identity if PDFium says no matrix
    return (1.0,0.0,0.0,1.0,0.0,0.0)


# ---------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------

def _compose_tuple(parent6, child6):
    # both 6-tuples (a,b,c,d,e,f)
    pa,pb,pc,pd,pe,pf = parent6
    ca,cb,cc,cd,ce,cf = child6
    return (
        pa*ca + pc*cb,
        pb*ca + pd*cb,
        pa*cc + pc*cd,
        pb*cc + pd*cd,
        pa*ce + pc*cf + pe,
        pb*ce + pd*cf + pf,
    )

def _apply_tuple(mat6, x, y):
    a,b,c,d,e,f = mat6
    X = a*x + c*y + e
    Y = b*x + d*y + f
    return (X, Y)


def _extract_path_segments(obj_handle, ctm6, out_segments):
    """
    For PAGEOBJ_PATH:
    Walk FPDFPath_* and append straight line segments as (x0,y0,x1,y1).
    """
    count = pdfraw.FPDFPath_CountSegments(obj_handle)

    pen = None
    subpath_start = None

    for i in range(count):
        seg = pdfraw.FPDFPath_GetPathSegment(obj_handle, i)
        seg_type = pdfraw.FPDFPathSegment_GetType(seg)
        # 1=MoveTo, 2=LineTo, 3=BezierTo, 4=Close

        x = c_double()
        y = c_double()
        pdfraw.FPDFPathSegment_GetPoint(seg, byref(x), byref(y))
        px, py = float(x.value), float(y.value)

        X, Y = _apply_tuple(ctm6, px, py)

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
            # no curve approximation, just advance pen
            pen = (X, Y)

        elif seg_type == 4:  # Close
            if pen is not None and subpath_start is not None:
                x0, y0 = pen
                x1, y1 = subpath_start
                if x0 != x1 or y0 != y1:
                    out_segments.append((x0, y0, x1, y1))
            pen = subpath_start


def _walk_obj_recursive(obj_handle, parent_ctm6, out_segments):
    """
    Recurse PDF objects via PDFium RAW API:
    - PATH -> extract lines
    - FORM -> recurse into child objects
    """
    otype = pdfraw.FPDFPageObj_GetType(obj_handle)

    local6 = _get_obj_matrix6(obj_handle)
    ctm6   = _compose_tuple(parent_ctm6, local6)

    if otype == PDFIUM_PAGEOBJ_PATH:
        _extract_path_segments(obj_handle, ctm6, out_segments)
        return

    if otype == PDFIUM_PAGEOBJ_FORM:
        child_count = pdfraw.FPDFFormObj_CountObjects(obj_handle)
        for i in range(child_count):
            child_handle = pdfraw.FPDFFormObj_GetObject(obj_handle, i)
            _walk_obj_recursive(child_handle, ctm6, out_segments)
        return

    # TEXT / IMAGE / SHADING ignored


def extract_segments_pdfium_lowlevel(pdf_path: str, page_index: int = 0):
    """
    Fully raw PDFium walker.
    Returns list[(x0,y0,x1,y1), ...] in page coords.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]

    page_handle = getattr(page, "_obj", None)
    if page_handle is None:
        return []

    segments = []
    identity6 = (1.0,0.0,0.0,1.0,0.0,0.0)

    obj_count = pdfraw.FPDFPage_CountObjects(page_handle)
    for i in range(obj_count):
        obj_handle = pdfraw.FPDFPage_GetObject(page_handle, i)
        _walk_obj_recursive(obj_handle, identity6, segments)

    if len(segments) == 0:
        # Deep debug mode: inspect PATH objects and FORM objects explicitly
        print(f"[PDFDBG-RAW] obj_count={obj_count}")

        path_debug_lines = []
        form_debug_lines = []

        for i in range(min(obj_count, 50)):
            obj_handle = pdfraw.FPDFPage_GetObject(page_handle, i)
            otype = pdfraw.FPDFPageObj_GetType(obj_handle)

            if otype == PDFIUM_PAGEOBJ_PATH:
                # check segment count
                try:
                    segcnt = pdfraw.FPDFPath_CountSegments(obj_handle)
                except Exception as e:
                    segcnt = f"<err {e!r}>"

                # check matrix
                try:
                    m6 = _get_obj_matrix6(obj_handle)
                except Exception as e:
                    m6 = f"<m_err {e!r}>"

                path_debug_lines.append(f"[PATH #{i}] segcnt={segcnt} matrix={m6}")

            elif otype == PDFIUM_PAGEOBJ_FORM:
                # check child count
                try:
                    child_cnt = pdfraw.FPDFFormObj_CountObjects(obj_handle)
                except Exception as e:
                    child_cnt = f"<err {e!r}>"

                form_debug_lines.append(f"[FORM #{i}] children={child_cnt}")

        if path_debug_lines:
            print("[PDFDBG-RAW] PATH objects:")
            for ln in path_debug_lines:
                print(" ", ln)

        if form_debug_lines:
            print("[PDFDBG-RAW] FORM objects:")
            for ln in form_debug_lines:
                print(" ", ln)

        # Also dump first ~10 object types just like before
        sample = []
        for j in range(min(obj_count, 10)):
            h = pdfraw.FPDFPage_GetObject(page_handle, j)
            sample.append(str(pdfraw.FPDFPageObj_GetType(h)))
        print("[PDFDBG-RAW] first object types:", ", ".join(sample))

    return segments

