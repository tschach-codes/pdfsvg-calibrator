from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import math
import numpy as np

SegmentTuple = Tuple[float, float, float, float]


def _tuples_to_dicts(segments: Sequence[SegmentTuple]) -> List[Dict[str, float]]:
    return [
        {
            "x1": float(x0),
            "y1": float(y0),
            "x2": float(x1),
            "y2": float(y1),
        }
        for x0, y0, x1, y1 in segments
    ]


def extract_segments(
    pdf_path: str,
    page_index: int = 0,
    *,
    tol_pt: float | None = None,
    curve_tol_pt: float | None = None,
) -> List[Dict[str, float]]:
    """Legacy compatibility wrapper for the PDFium segment extractor.

    The old high-level walker accepted a ``tol_pt`` argument.  The new low-level
    implementation performs its own flattening, so ``tol_pt`` is ignored but
    kept for API compatibility.
    """

    del tol_pt, curve_tol_pt  # unused but kept for signature compatibility
    from .pdfium_extrac_lowlevel import (
        extract_segments_pdfium_lowlevel as _extract_segments_pdfium_lowlevel,
    )

    segments = _extract_segments_pdfium_lowlevel(pdf_path, page_index)
    return _tuples_to_dicts(segments)


def debug_print_summary(
    label: str, segs: Sequence[Dict[str, float]], angle_tol_deg: float = 2.0
) -> None:
    """Print a quick statistical summary of the extracted PDF segments."""

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
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    min_x = float(xs_arr.min())
    max_x = float(xs_arr.max())
    min_y = float(ys_arr.min())
    max_y = float(ys_arr.max())
    span_x = float(max_x - min_x)
    span_y = float(max_y - min_y)
    print(
        f"  bbox=({min_x},{min_y})..({max_x},{max_y}) span=({span_x},{span_y})"
    )
    if angs:
        angs_arr = np.array(angs)
        print(
            "  angle_folded_deg p50="
            f"{float(np.percentile(angs_arr, 50)):.3f} p95={float(np.percentile(angs_arr, 95)):.3f}"
        )
    print(f"  horiz_cnt≈0deg={horiz_cnt} vert_cnt≈90deg={vert_cnt}")


__all__ = ["extract_segments", "debug_print_summary"]
