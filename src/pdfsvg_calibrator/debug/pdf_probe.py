from __future__ import annotations

from typing import Any, Dict

import pypdfium2 as pdfium


__all__ = ["probe_page"]


def probe_page(pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
    doc = pdfium.PdfDocument(pdf_path)
    try:
        page = doc[page_index]

        stats: Dict[str, Any] = {
            "page_index": page_index,
            "boxes": {
                "mediabox": tuple(page.get_size()),
            },
            "counts": {
                "total": 0,
                "path": 0,
                "form": 0,
                "image": 0,
                "text": 0,
                "shading": 0,
                "other": 0,
            },
            "depth_max": 0,
            "forms": 0,
            "images": [],
            "notes": [],
        }

        def _mmul(
            a: tuple[float, float, float, float, float, float],
            b: tuple[float, float, float, float, float, float],
        ) -> tuple[float, float, float, float, float, float]:
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

        I = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        def walk(obj: Any, depth: int, m_parent: tuple[float, float, float, float, float, float]) -> None:
            stats["depth_max"] = max(stats["depth_max"], depth)
            stats["counts"]["total"] += 1

            m_obj_attr = getattr(obj, "get_matrix", None)
            m_obj = m_obj_attr() if callable(m_obj_attr) else I
            m_combined = _mmul(m_parent, m_obj)

            obj_type = obj.type
            if obj_type in stats["counts"]:
                stats["counts"][obj_type] += 1
            else:
                stats["counts"]["other"] += 1

            if obj_type == "image":
                try:
                    width, height = obj.get_size()
                except Exception:  # pragma: no cover - best effort diagnostic
                    width, height = None, None
                stats["images"].append({"w_px": width, "h_px": height, "depth": depth, "matrix": m_combined})

            if obj_type == "form":
                stats["forms"] += 1
                child_count = obj.get_objects_count()
                for idx in range(child_count):
                    child = obj.get_object(idx)
                    walk(child, depth + 1, m_combined)

        object_count = page.get_objects_count()
        for index in range(object_count):
            walk(page.get_object(index), 0, I)

        if stats["counts"]["image"] >= 1 and stats["counts"]["path"] == 0 and stats["forms"] == 0:
            stats["notes"].append("likely_scanned_raster_page")

        if stats["forms"] > 0 and stats["counts"]["path"] == 0:
            stats["notes"].append("paths_may_be_inside_forms_but_not_counted")

        if stats["counts"]["text"] > 0 and stats["counts"]["path"] == 0:
            stats["notes"].append("geometry_might_be_in_text_outlines/type3")

        return stats
    finally:
        doc.close()
