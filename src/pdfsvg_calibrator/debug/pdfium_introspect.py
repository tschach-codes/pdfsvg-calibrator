import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c


def _safecall(obj, name):
    if hasattr(obj, name):
        attr = getattr(obj, name)
        if callable(attr):
            try:
                return attr()
            except Exception as e:
                return f"<callable {name}() raised {e!r}>"
        return attr
    return "<missing>"


def _dir_public(obj):
    return [n for n in dir(obj) if not n.startswith("_")]


def introspect_page(pdf_path: str, page_index: int = 0):
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]

    print("=== PAGE DUMP ===")
    print("page type:", type(page))
    print("page dir :", _dir_public(page))

    # Try to get page objects using all plausible APIs
    candidates = []
    for guess in ["get_objects", "get_objects_count", "get_object",
                  "get_pageobjects", "iter_objects"]:
        if hasattr(page, guess):
            candidates.append(guess)
    print("page candidate accessors:", candidates)

    # Try common patterns:
    objs = None
    if hasattr(page, "get_objects"):
        try:
            objs = page.get_objects()
        except Exception as e:
            print("page.get_objects() raised", e)
    if objs is None and hasattr(page, "get_objects_count"):
        try:
            cnt = page.get_objects_count()
            tmp = []
            for i in range(cnt):
                tmp.append(page.get_object(i))
            objs = tmp
        except Exception as e:
            print("page.get_objects_count/page.get_object failed:", e)
    if objs is None:
        print("Could not list page objects with known guesses.")
        return

    if not isinstance(objs, list):
        try:
            objs = list(objs)
        except TypeError:
            objs = list(iter(objs))

    print(f"found {len(objs)} top-level objects")

    type_names = {getattr(pdfium_c, name): name for name in dir(pdfium_c) if name.startswith("FPDF_PAGEOBJ_")}

    for idx, obj in enumerate(objs[:20]):  # cap to first 20 for sanity
        print(f"\n-- OBJ {idx} --")
        print("type(obj):", type(obj))
        print("dir(obj):", _dir_public(obj))
        obj_type = getattr(obj, "type", "<no .type>")
        print("obj.type :", obj_type, type_names.get(obj_type, "<unknown>"))
        print("has get_matrix:", hasattr(obj, "get_matrix"))
        print("has get_path  :", hasattr(obj, "get_path"))
        print("has get_objects/get_objects_count:", hasattr(obj, "get_objects"), hasattr(obj, "get_objects_count"))

        if hasattr(obj, "get_matrix"):
            print("matrix:", _safecall(obj, "get_matrix"))
        if hasattr(obj, "get_path"):
            p = _safecall(obj, "get_path")
            print("path type:", type(p))
            if hasattr(p, "__iter__"):
                # dump first 5 path segments
                try:
                    it = list(p)[:5]
                    for si, seg in enumerate(it):
                        print(f"  seg[{si}].type:", getattr(seg, "type", None))
                        print("   dir(seg):", _dir_public(seg))
                        # typical names: pos, ctrl1, ctrl2
                        print("   pos:", getattr(seg, "pos", None))
                        print("   ctrl1:", getattr(seg, "ctrl1", None))
                        print("   ctrl2:", getattr(seg, "ctrl2", None))
                except Exception as e:
                    print("   iter path failed:", e)

        # if it's a form, introspect children
        if getattr(obj, "type", None) == "form" or hasattr(obj, "get_objects"):
            try:
                subobjs = None
                if hasattr(obj, "get_objects"):
                    subobjs = obj.get_objects()
                elif hasattr(obj, "get_objects_count"):
                    cnt = obj.get_objects_count()
                    tmp = []
                    for j in range(cnt):
                        tmp.append(obj.get_object(j))
                    subobjs = tmp
                if subobjs is not None:
                    print(f"  form has {len(subobjs)} children")
                    if len(subobjs):
                        sub0 = subobjs[0]
                        print("  first child dir:", _dir_public(sub0))
                        print("  first child .type:", getattr(sub0, "type", None))
            except Exception as e:
                print("  child introspection failed:", e)
