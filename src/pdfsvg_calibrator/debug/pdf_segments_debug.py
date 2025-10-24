from __future__ import annotations

from typing import List, Dict
import math
import numpy as np


def analyze_segments_basic(segments: List[Dict[str, float]], angle_tol_deg: float = 2.0) -> dict:
    if not segments:
        return {"count": 0}

    xs = []
    ys = []
    lens = []
    folded_angles = []
    horiz_cnt = 0
    vert_cnt = 0

    for s in segments:
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])

        dx = x2 - x1
        dy = y2 - y1
        L = math.hypot(dx, dy)
        if L == 0:
            continue
        lens.append(L)

        # Winkel in Grad, modulo 180
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        # Falte auf [0,90]
        if ang > 90.0:
            ang = 180.0 - ang
        folded_angles.append(ang)

        # horizontnah?
        if abs(ang - 0.0) <= angle_tol_deg:
            horiz_cnt += 1
        # vertikalnah?
        if abs(ang - 90.0) <= angle_tol_deg:
            vert_cnt += 1

    xs = np.array(xs)
    ys = np.array(ys)
    lens = np.array(lens) if lens else np.array([0.0])
    folded_angles = np.array(folded_angles) if folded_angles else np.array([0.0])

    return {
        "count": len(segments),
        "bbox": {
            "min_x": float(xs.min()),
            "max_x": float(xs.max()),
            "min_y": float(ys.min()),
            "max_y": float(ys.max()),
            "span_x": float(xs.max() - xs.min()),
            "span_y": float(ys.max() - ys.min()),
        },
        "length": {
            "min": float(lens.min()),
            "p50": float(np.percentile(lens, 50)),
            "p90": float(np.percentile(lens, 90)),
            "max": float(lens.max()),
        },
        "angle_folded_deg_summary": {
            "p5": float(np.percentile(folded_angles, 5)),
            "p50": float(np.percentile(folded_angles, 50)),
            "p95": float(np.percentile(folded_angles, 95)),
        },
        "horiz_cnt": int(horiz_cnt),
        "vert_cnt": int(vert_cnt),
    }


def debug_print_segments(label: str, stats: dict, examples: List[Dict[str, float]]):
    print(f"[SEGDBG] {label}:")
    print(f"  count={stats.get('count')}")
    if stats.get("count", 0) == 0:
        return
    print(f"  bbox={stats['bbox']}")
    print(f"  length={stats['length']}")
    print(f"  angle_folded_deg_summary={stats['angle_folded_deg_summary']}")
    print(f"  horiz_cnt@±2deg={stats['horiz_cnt']}")
    print(f"  vert_cnt@±2deg={stats['vert_cnt']}")
    print("  examples:")
    for s in examples[:5]:
        print(f"    {s}")
