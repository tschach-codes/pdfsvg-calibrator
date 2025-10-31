from __future__ import annotations
import inspect
from types import SimpleNamespace

def install_lxml_tag_guard(verbose: bool = False):
    """
    Monkey-patch lxml.etree.Element and SubElement to validate tag parameters.
    Only meant for diagnostics; call early (e.g., at CLI start when --verbose).
    """
    try:
        from lxml import etree
    except Exception:
        # lxml not in use; nothing to guard
        return SimpleNamespace(enabled=False)

    orig_Element = etree.Element
    orig_SubElement = etree.SubElement

    def _fmt_caller():
        frm = inspect.stack()[2]  # caller of Element/SubElement
        return f"{frm.filename}:{frm.lineno} in {frm.function}"

    def _check_tag(tag):
        # Allow strings and 'ns{tag}'-style tags
        if isinstance(tag, (str, bytes)):
            return
        # Disallow callables (e.g., etree.Comment function object)
        if callable(tag):
            raise TypeError(
                f"Invalid XML tag (callable) passed: {tag!r}\n"
                f"Did you mean to append Comment(...) instead of using it as a tag?\n"
                f"Caller: {_fmt_caller()}"
            )
        # Anything else (e.g., Element objects) is also invalid as tag
        raise TypeError(
            f"Invalid XML tag (type={type(tag)}): {tag!r}\n"
            f"Caller: {_fmt_caller()}"
        )

    def Element(tag, *args, **kwargs):
        _check_tag(tag)
        return orig_Element(tag, *args, **kwargs)

    def SubElement(parent, tag, *args, **kwargs):
        _check_tag(tag)
        return orig_SubElement(parent, tag, *args, **kwargs)

    # Only patch once
    if getattr(etree, "_pdfsvg_guard_installed", False):
        return SimpleNamespace(enabled=True)

    etree.Element = Element  # type: ignore
    etree.SubElement = SubElement  # type: ignore
    etree._pdfsvg_guard_installed = True  # type: ignore

    if verbose:
        print("[pdfsvg] XML tag guard installed for lxml.etree.Element/SubElement")

    return SimpleNamespace(enabled=True)
