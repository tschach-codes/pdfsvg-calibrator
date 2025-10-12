from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .geom import classify_hv
from .types import Model, Segment


@dataclass(frozen=True)
class Candidate:
    idx: int
    seg: Segment
    axis: str
    score: float
    length: float
    center: Tuple[float, float]
    tie: float


class SegmentGrid:
    """Simple uniform grid spatial index for segment proximity queries."""

    def __init__(self, segments: Sequence[Segment], cell_size: float, bbox: Tuple[float, float, float, float]):
        self._segments: Sequence[Segment] = segments
        self.cell_size = max(cell_size, 1e-6)
        self.min_x, self.min_y, self.max_x, self.max_y = bbox
        self._cells: Dict[Tuple[int, int], List[int]] = {}
        for idx, seg in enumerate(segments):
            x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            ix0 = self._cell_index(min_x, self.min_x)
            ix1 = self._cell_index(max_x, self.min_x)
            iy0 = self._cell_index(min_y, self.min_y)
            iy1 = self._cell_index(max_y, self.min_y)
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    self._cells.setdefault((ix, iy), []).append(idx)

    @property
    def segments(self) -> Sequence[Segment]:
        return self._segments

    def _cell_index(self, value: float, offset: float) -> int:
        return int(math.floor((value - offset) / self.cell_size))

    def query(self, x: float, y: float, radius: float) -> Sequence[Segment]:
        if not self._segments:
            return []
        if radius <= 0:
            radius = self.cell_size
        reach = max(1, int(math.ceil(radius / self.cell_size)))
        ix = self._cell_index(x, self.min_x)
        iy = self._cell_index(y, self.min_y)
        seen: set[int] = set()
        results: List[Segment] = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for idx in self._cells.get((ix + dx, iy + dy), []):
                    if idx not in seen:
                        seen.add(idx)
                        results.append(self._segments[idx])
        if not results:
            return self._segments
        return results


def _segment_length(seg: Segment) -> float:
    return math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)


def _segment_center(seg: Segment) -> Tuple[float, float]:
    return (0.5 * (seg.x1 + seg.x2), 0.5 * (seg.y1 + seg.y2))


def _bbox_and_diag(segments: Iterable[Segment]) -> Tuple[float, float, float, float, float]:
    segs = list(segments)
    if not segs:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    min_x = min(min(s.x1, s.x2) for s in segs)
    max_x = max(max(s.x1, s.x2) for s in segs)
    min_y = min(min(s.y1, s.y2) for s in segs)
    max_y = max(max(s.y1, s.y2) for s in segs)
    diag = math.hypot(max_x - min_x, max_y - min_y)
    return min_x, min_y, max_x, max_y, diag


def _point_to_segment_distance(px: float, py: float, seg: Segment) -> float:
    x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _apply_model(model: Model, x: float, y: float) -> Tuple[float, float]:
    rot = model.rot_deg % 360
    if rot == 0:
        rx, ry = x, y
    elif rot == 180:
        rx, ry = -x, -y
    elif rot == 90:
        rx, ry = -y, x
    elif rot == 270:
        rx, ry = y, -x
    else:
        raise ValueError(f"Unsupported rotation: {model.rot_deg}")
    return model.sx * rx + model.tx, model.sy * ry + model.ty


def build_seg_grid(svg_segments: Sequence[Segment], cfg: dict, diag_hint: float) -> SegmentGrid:
    _, _, _, _, diag_svg = _bbox_and_diag(svg_segments)
    diag_ref = diag_svg if diag_svg > 0 else diag_hint
    cell_rel = cfg.get("grid_cell_rel", 0.02)
    cell_size = max(diag_ref * cell_rel, 1.0 if diag_ref == 0 else 1e-6)
    min_x, min_y, max_x, max_y, _ = _bbox_and_diag(svg_segments)
    return SegmentGrid(svg_segments, cell_size, (min_x, min_y, max_x, max_y))


def score_line_support(pdf_seg: Segment, model: Model, svg_grid: SegmentGrid, cfg: dict) -> float:
    length = _segment_length(pdf_seg)
    if length <= 0:
        return 0.0
    diag_pdf = cfg.get("_diag_pdf", length if length > 0 else 1.0)
    sampling_cfg = cfg.get("sampling", {})
    step_rel = sampling_cfg.get("step_rel", 0.02)
    max_points = int(sampling_cfg.get("max_points", 5000))
    step = max(step_rel * diag_pdf, 1e-9)
    if max_points <= 1:
        num_points = 2
    else:
        num_points = min(max_points, max(2, int(math.ceil(length / step)) + 1))
    chamfer_cfg = cfg.get("chamfer", {})
    sigma_rel = chamfer_cfg.get("sigma_rel", 0.004)
    hard_mul = chamfer_cfg.get("hard_mul", 3.0)
    sigma = max(sigma_rel * diag_pdf, 1e-9)
    hard_limit = sigma * hard_mul
    accum = 0.0
    for i in range(num_points):
        t = i / (num_points - 1)
        px = pdf_seg.x1 + (pdf_seg.x2 - pdf_seg.x1) * t
        py = pdf_seg.y1 + (pdf_seg.y2 - pdf_seg.y1) * t
        sx, sy = _apply_model(model, px, py)
        candidates = svg_grid.query(sx, sy, hard_limit)
        if not candidates:
            continue
        dist = min(_point_to_segment_distance(sx, sy, seg) for seg in candidates)
        if dist > hard_limit:
            score = 0.0
        else:
            score = math.exp(-0.5 * (dist / sigma) ** 2)
        accum += score
    if num_points == 0:
        return 0.0
    mean_score = accum / num_points
    return mean_score * length


def _sort_key(candidate: Candidate) -> Tuple[float, float, float, float, float]:
    axis_order = 0.0 if candidate.axis == "H" else 1.0
    cx, cy = candidate.center
    return (-candidate.score, -candidate.length, candidate.tie, axis_order, cx + cy)


def _respect_diversity(candidate: Candidate, selected: Sequence[Candidate], min_dist: float) -> bool:
    if min_dist <= 0:
        return True
    cx, cy = candidate.center
    for other in selected:
        ox, oy = other.center
        if math.hypot(cx - ox, cy - oy) < min_dist:
            return False
    return True


def select_lines(
    pdf_segs: List[Segment],
    model: Model,
    svg_segs: List[Segment],
    cfg: dict,
) -> Tuple[List[Segment], Dict[str, object]]:
    verify_cfg = cfg.get("verify", {})
    pick_k = int(verify_cfg.get("pick_k", 5))
    if pick_k <= 0:
        return [], {"had_enough_H": False, "had_enough_V": False, "diversity_enforced": True, "notes": ["pick_k <= 0"]}

    _, _, _, _, diag_pdf = _bbox_and_diag(pdf_segs)
    if diag_pdf <= 0:
        diag_pdf = max((_segment_length(seg) for seg in pdf_segs), default=1.0)
        if diag_pdf <= 0:
            diag_pdf = 1.0
    local_cfg = dict(cfg)
    local_cfg["_diag_pdf"] = diag_pdf

    svg_grid = build_seg_grid(svg_segs, cfg, diag_pdf)

    diversity_rel = float(verify_cfg.get("diversity_rel", 0.1))
    diversity_threshold = diag_pdf * diversity_rel

    rng_seed = cfg.get("rng_seed")
    if rng_seed is not None:
        import random

        rng = random.Random(rng_seed)
    else:
        rng = None

    h_segs, v_segs = classify_hv(pdf_segs, angle_tol_deg=verify_cfg.get("dir_tol_deg", cfg.get("angle_tol_deg", 6.0)))

    candidates: List[Candidate] = []
    axis_lists: Dict[str, List[Candidate]] = {"H": [], "V": []}
    for axis, seg_list in (("H", h_segs), ("V", v_segs)):
        for seg in seg_list:
            length = _segment_length(seg)
            if length <= 0:
                continue
            score = score_line_support(seg, model, svg_grid, local_cfg)
            tie = rng.random() if rng is not None else 0.0
            candidate = Candidate(
                idx=len(candidates),
                seg=seg,
                axis=axis,
                score=score,
                length=length,
                center=_segment_center(seg),
                tie=tie,
            )
            candidates.append(candidate)
            axis_lists[axis].append(candidate)

    for cand_list in axis_lists.values():
        cand_list.sort(key=_sort_key)
    all_sorted = sorted(candidates, key=_sort_key)

    had_enough_h = len(axis_lists["H"]) >= 2
    had_enough_v = len(axis_lists["V"]) >= 2

    target_h = min(2, pick_k, len(axis_lists["H"]))
    target_v = min(2, pick_k, len(axis_lists["V"]))

    req_h = target_h
    req_v = target_v
    while req_h + req_v > pick_k:
        if req_h >= req_v and req_h > 0:
            req_h -= 1
        elif req_v > 0:
            req_v -= 1
        else:
            break

    selected: List[Candidate] = []
    selected_ids: set[int] = set()
    counts = {"H": 0, "V": 0}

    for axis, req, cand_list in (("H", req_h, axis_lists["H"]), ("V", req_v, axis_lists["V"])):
        for cand in cand_list:
            if counts[axis] >= req:
                break
            if cand.idx in selected_ids:
                continue
            if not _respect_diversity(cand, selected, diversity_threshold):
                continue
            selected.append(cand)
            selected_ids.add(cand.idx)
            counts[axis] += 1

    # Fill remaining slots with best-scoring candidates respecting diversity.
    for cand in all_sorted:
        if len(selected) >= pick_k:
            break
        if cand.idx in selected_ids:
            continue
        if not _respect_diversity(cand, selected, diversity_threshold):
            continue
        selected.append(cand)
        selected_ids.add(cand.idx)
        counts[cand.axis] += 1

    diversity_enforced = True
    if len(selected) < min(pick_k, len(all_sorted)):
        diversity_enforced = False
        for cand in all_sorted:
            if len(selected) >= min(pick_k, len(all_sorted)):
                break
            if cand.idx in selected_ids:
                continue
            selected.append(cand)
            selected_ids.add(cand.idx)
            counts[cand.axis] += 1

    if len(selected) > pick_k:
        selected = selected[:pick_k]

    counts = {"H": 0, "V": 0}
    for cand in selected:
        counts[cand.axis] += 1

    notes: List[str] = []
    if counts["H"] < target_h:
        notes.append(f"Selected {counts['H']} horizontal segments (target {target_h}).")
    if counts["V"] < target_v:
        notes.append(f"Selected {counts['V']} vertical segments (target {target_v}).")
    if not had_enough_h:
        notes.append("Fewer than two horizontal candidates available.")
    if not had_enough_v:
        notes.append("Fewer than two vertical candidates available.")
    if len(selected) < pick_k:
        notes.append(f"Only {len(selected)} segments selected out of requested {pick_k}.")
    if not candidates:
        notes.append("No eligible candidates after classification.")
    if not diversity_enforced and len(selected) >= min(pick_k, len(all_sorted)):
        notes.append("Diversity threshold relaxed to reach requested count.")

    info = {
        "had_enough_H": had_enough_h,
        "had_enough_V": had_enough_v,
        "diversity_enforced": diversity_enforced,
        "notes": notes,
    }

    return [cand.seg for cand in selected], info
