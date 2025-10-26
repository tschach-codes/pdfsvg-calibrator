import pypdfium2 as pdfium
from typing import Iterable, Any

def _to_list_safe(maybe_iter: Any, limit: int | None = None):
    if maybe_iter is None:
        return []
    out = []
    try:
        for idx, x in enumerate(maybe_iter):
            if limit is not None and idx >= limit:
                break
            out.append(x)
    except Exception as e:
        print("[probe] iter -> list failed:", e)
    return out

def _safe_has(obj, name: str) -> bool:
    try:
        return hasattr(obj, name)
    except Exception:
        return False

def _safe_call(obj, name: str, *args, **kwargs):
    if not _safe_has(obj, name):
        return False, f"<missing {name}>"
    fn = getattr(obj, name)
    if not callable(fn):
        return False, f"<attr {name} not callable>"
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        return False, f"<{name} raised {e!r}>"

def _collect_page_objects(page):
    if _safe_has(page, "get_objects"):
        ok, objs = _safe_call(page, "get_objects")
        if ok and objs is not None:
            return _to_list_safe(objs)  # normalize possible generator
        else:
            print("[probe] page.get_objects() failed or None:", objs)

    if _safe_has(page, "get_objects_count"):
        ok_cnt, cnt = _safe_call(page, "get_objects_count")
        if ok_cnt and isinstance(cnt, int) and cnt >= 0:
            out = []
            for i in range(cnt):
                ok_obj, child = _safe_call(page, "get_object", i)
                if ok_obj:
                    out.append(child)
                else:
                    print(f"[probe] page.get_object({i}) failed:", child)
            return out
        else:
            print("[probe] get_objects_count failed:", cnt)

    return []

def _dump_basic_obj(obj, idx: int, indent=""):
    otype = getattr(obj, "type", "<no .type>")
    print(f"{indent}[probe] OBJ {idx}: type={otype!r}")

    if _safe_has(obj, "get_matrix"):
        ok, mat = _safe_call(obj, "get_matrix")
        print(f"{indent}  get_matrix(): {mat}")
    else:
        print(f"{indent}  get_matrix(): <missing>")

    print(f"{indent}  children API: "
          f"get_objects={_safe_has(obj,'get_objects')}, "
          f"get_objects_count={_safe_has(obj,'get_objects_count')}")

def _introspect_path_obj(obj):
    """
    Deep dive on a PATH object (type=1):
    - show dir(obj)
    - try known candidate methods to access geometry
    """
    print("---- PATH OBJECT INTROSPECTION ----")
    # Show attribute names
    try:
        attrs = dir(obj)
        print(f"dir(obj) -> {len(attrs)} attrs")
        # print first ~40 attrs for sanity
        for name in attrs[:40]:
            print("   ", name)
    except Exception as e:
        print("dir(obj) failed:", e)

    # Try likely geometry getters
    candidates = [
        "get_path",
        "get_pathdata",
        "get_path_data",
        "get_points",
        "get_vertices",
        "get_segment",
        "get_segments",
        "get_points_count",
        "get_segment_count",
    ]
    for cand in candidates:
        ok, res = _safe_call(obj, cand)
        print(f"try {cand}():", "OK" if ok else "NO", "->", type(res), repr(res)[:200])

        # If it looked iterable, try to peek inside
        if ok:
            preview = _to_list_safe(res, limit=3)
            print(f"  {cand}() preview of first 3:", preview)
    print("---- END PATH OBJECT INTROSPECTION ----")

def probe_types(pdf_path: str, page_index: int = 0, max_objects: int = 50):
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]

    objs = _collect_page_objects(page)
    print(f"[probe] found {len(objs)} top-level objects\n")

    first_path_obj = None

    for idx, obj in enumerate(objs[:max_objects]):
        _dump_basic_obj(obj, idx)
        otype = getattr(obj, "type", None)
        if first_path_obj is None and otype == 1:
            first_path_obj = obj
        print("")

    if first_path_obj is None:
        print("[probe] no PATH object (type==1) found in first", max_objects)
    else:
        _introspect_path_obj(first_path_obj)
