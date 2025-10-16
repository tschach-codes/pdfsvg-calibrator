from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .geom import classify_hv
from .thirdparty.hungarian import solve as hungarian_solve
from .types import Match, Model, Segment


@dataclass
class _SegmentInfo:
    seg: Segment
    axis: Optional[str]
    length: float
    center: Tuple[float, float]
    signature: List[Tuple[float, float, float]]


@dataclass(frozen=True)
class Candidate:
    idx: int
    seg: Segment
    axis: str
    score: float
    length: float
    center: Tuple[float, float]
    tie: float


def _segment_length(seg: Segment) -> float:
    return math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)


def _segment_center(seg: Segment) -> Tuple[float, float]:
    return (0.5 * (seg.x1 + seg.x2), 0.5 * (seg.y1 + seg.y2))


def _segment_direction(seg: Segment) -> Tuple[float, float]:
    length = _segment_length(seg)
    if length == 0:
        return (1.0, 0.0)
    return ((seg.x2 - seg.x1) / length, (seg.y2 - seg.y1) / length)


def _segment_angle_deg(seg: Segment) -> float:
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    if dx == 0 and dy == 0:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _is_axis_aligned(seg: Segment, angle_tol_deg: float) -> Optional[str]:
    angle = _segment_angle_deg(seg)
    delta_h = min(angle, 180.0 - angle)
    delta_v = abs(angle - 90.0)
    tol = max(angle_tol_deg, 0.0)
    if delta_h <= tol and delta_h <= delta_v:
        return "H"
    if delta_v <= tol:
        return "V"
    return None


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


def _segment_distance(seg_a: Segment, seg_b: Segment) -> float:
    distances = [
        _point_to_segment_distance(seg_a.x1, seg_a.y1, seg_b),
        _point_to_segment_distance(seg_a.x2, seg_a.y2, seg_b),
        _point_to_segment_distance(seg_b.x1, seg_b.y1, seg_a),
        _point_to_segment_distance(seg_b.x2, seg_b.y2, seg_a),
    ]
    return min(distances)


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


def neighbor_signature(
    ref_seg: Segment,
    all_segs: Sequence[Segment],
    radius: float,
    angle_tol_deg: float,
) -> List[Tuple[float, float, float]]:
    length_ref = _segment_length(ref_seg)
    if length_ref == 0:
        return []
    ux = (ref_seg.x2 - ref_seg.x1) / length_ref
    uy = (ref_seg.y2 - ref_seg.y1) / length_ref
    features: List[Tuple[float, float, float]] = []
    radius = max(radius, 0.0)
    center_ref = _segment_center(ref_seg)
    for seg in all_segs:
        if seg is ref_seg:
            continue
        if radius > 0:
            cx, cy = _segment_center(seg)
            if math.hypot(cx - center_ref[0], cy - center_ref[1]) > radius:
                continue
        axis = _is_axis_aligned(seg, angle_tol_deg)
        if axis is None:
            continue
        length_nei = _segment_length(seg)
        if length_nei == 0:
            continue
        cx, cy = _segment_center(seg)
        vx = cx - ref_seg.x1
        vy = cy - ref_seg.y1
        proj = vx * ux + vy * uy
        t = 0.0 if length_ref == 0 else proj / length_ref
        t = max(0.0, min(1.0, t))
        angle_ref = _segment_angle_deg(ref_seg)
        angle_nei = _segment_angle_deg(seg)
        dtheta = abs(angle_ref - angle_nei) % 180.0
        if dtheta > 90.0:
            dtheta = 180.0 - dtheta
        rho = length_nei / length_ref
        rho = max(0.0, min(rho, 5.0))
        features.append((t, dtheta, rho))
    features.sort(key=lambda item: (item[0], item[1], item[2]))
    return features


def _direction_term(pdf_seg: Segment, svg_seg: Segment, dir_tol_deg: float) -> Tuple[float, bool]:
    upx, upy = _segment_direction(pdf_seg)
    usx, usy = _segment_direction(svg_seg)
    dot = abs(upx * usx + upy * usy)
    dot = max(-1.0, min(1.0, dot))
    dot = min(dot, 1.0)
    angle = math.degrees(math.acos(dot))
    if angle > dir_tol_deg:
        return 1.0 + angle, False
    return 1.0 - dot, True


def _endpoint_term(pdf_seg: Segment, svg_seg: Segment, radius: float) -> float:
    radius = max(radius, 1e-6)
    p_endpoints = [(pdf_seg.x1, pdf_seg.y1), (pdf_seg.x2, pdf_seg.y2)]
    s_endpoints = [(svg_seg.x1, svg_seg.y1), (svg_seg.x2, svg_seg.y2)]
    d1 = math.hypot(p_endpoints[0][0] - s_endpoints[0][0], p_endpoints[0][1] - s_endpoints[0][1]) + math.hypot(
        p_endpoints[1][0] - s_endpoints[1][0], p_endpoints[1][1] - s_endpoints[1][1]
    )
    d2 = math.hypot(p_endpoints[0][0] - s_endpoints[1][0], p_endpoints[0][1] - s_endpoints[1][1]) + math.hypot(
        p_endpoints[1][0] - s_endpoints[0][0], p_endpoints[1][1] - s_endpoints[0][1]
    )
    return min(d1, d2) / (2.0 * radius)


def _midpoint_term(pdf_seg: Segment, svg_seg: Segment, radius: float) -> float:
    radius = max(radius, 1e-6)
    pcx, pcy = _segment_center(pdf_seg)
    scx, scy = _segment_center(svg_seg)
    return math.hypot(pcx - scx, pcy - scy) / radius


def _neighbor_cost(
    sig_pdf: Sequence[Tuple[float, float, float]],
    sig_svg: Sequence[Tuple[float, float, float]],
    neighbors_cfg: Dict[str, float],
) -> float:
    if not sig_pdf or not sig_svg:
        return neighbors_cfg.get("penalty_empty", 5.0)
    dt_max = max(neighbors_cfg.get("dt", 0.05), 1e-6)
    dtheta_max = max(neighbors_cfg.get("dtheta_deg", 8.0), 1e-6)
    rho_soft = max(neighbors_cfg.get("rho_soft", 0.25), 1e-6)
    penalty_miss = neighbors_cfg.get("penalty_miss", 1.5)
    used: set[int] = set()
    cost = 0.0
    for t_pdf, dtheta_pdf, rho_pdf in sig_pdf:
        best_idx = None
        best_val = float("inf")
        for idx, (t_svg, dtheta_svg, rho_svg) in enumerate(sig_svg):
            if idx in used:
                continue
            dt = abs(t_pdf - t_svg)
            dtheta = abs(dtheta_pdf - dtheta_svg)
            if dt > dt_max or dtheta > dtheta_max:
                continue
            rho_diff = abs(rho_pdf - rho_svg)
            rho_rel = rho_diff / max(max(rho_pdf, rho_svg), 1e-6)
            val = (dt / dt_max) + (dtheta / dtheta_max) + (rho_rel / rho_soft)
            if val < best_val:
                best_val = val
                best_idx = idx
        if best_idx is None:
            cost += penalty_miss
        else:
            used.add(best_idx)
            cost += best_val / 3.0
    unmatched_svg = len(sig_svg) - len(used)
    if unmatched_svg > 0:
        cost += unmatched_svg * 0.5
    return cost


def candidate_cost(
    pdf_seg_T: Segment,
    svg_seg: Segment,
    weights: Dict[str, float],
    sig_pdf: Sequence[Tuple[float, float, float]],
    sig_svg: Sequence[Tuple[float, float, float]],
    cfg: Dict[str, object],
) -> float:
    verify_cfg = cfg.get("verify", {})
    dir_tol_deg = float(verify_cfg.get("dir_tol_deg", 6.0))
    radius = float(verify_cfg.get("radius_px", 1.0))
    dir_term, within_tol = _direction_term(pdf_seg_T, svg_seg, dir_tol_deg)
    if not within_tol:
        return float("inf")
    endpoint_term = _endpoint_term(pdf_seg_T, svg_seg, radius)
    midpoint_term = _midpoint_term(pdf_seg_T, svg_seg, radius)
    neighbors_cfg = cfg.get("neighbors", {})
    neighbor_term = _neighbor_cost(sig_pdf, sig_svg, neighbors_cfg)
    w_endpoint = float(weights.get("endpoint", 1.0))
    w_midpoint = float(weights.get("midpoint", 1.0))
    w_direction = float(weights.get("direction", 1.0))
    w_neighbors = float(weights.get("neighbors", 1.0))
    cost = (
        w_endpoint * endpoint_term
        + w_midpoint * midpoint_term
        + w_direction * dir_term
        + w_neighbors * neighbor_term
    )
    return cost


def _prepare_segment_infos(
    segments: Sequence[Segment],
    radius: float,
    angle_tol_deg: float,
) -> List[_SegmentInfo]:
    infos: List[_SegmentInfo] = []
    for seg in segments:
        length = _segment_length(seg)
        axis = _is_axis_aligned(seg, angle_tol_deg)
        center = _segment_center(seg)
        signature = neighbor_signature(seg, segments, radius, angle_tol_deg)
        infos.append(_SegmentInfo(seg, axis, length, center, signature))
    return infos


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
        return [], {
            "had_enough_H": False,
            "had_enough_V": False,
            "diversity_enforced": True,
            "notes": ["pick_k <= 0"],
        }

    _, _, _, _, diag_pdf = _bbox_and_diag(pdf_segs)
    if diag_pdf <= 0:
        diag_pdf = max((_segment_length(seg) for seg in pdf_segs), default=1.0)
        if diag_pdf <= 0:
            diag_pdf = 1.0
    local_cfg = dict(cfg)
    local_cfg["_diag_pdf"] = diag_pdf
    sampling_cfg = dict(cfg.get("sampling", {}))
    sampling_max_override = verify_cfg.get("sampling_max_points")
    if sampling_max_override is not None:
        sampling_cfg["max_points"] = min(
            int(sampling_cfg.get("max_points", sampling_max_override)),
            int(sampling_max_override),
        )
    local_cfg["sampling"] = sampling_cfg

    svg_grid = build_seg_grid(svg_segs, cfg, diag_pdf)

    diversity_rel = float(verify_cfg.get("diversity_rel", 0.1))
    diversity_threshold = diag_pdf * diversity_rel

    rng_seed = cfg.get("rng_seed")
    if rng_seed is not None:
        import random

        rng = random.Random(rng_seed)
    else:
        rng = None

    h_segs, v_segs = classify_hv(
        pdf_segs,
        angle_tol_deg=verify_cfg.get("dir_tol_deg", cfg.get("angle_tol_deg", 6.0)),
    )

    max_candidates_per_axis = verify_cfg.get("max_candidates_per_axis")
    limit_adjusted = False
    prefilter_dropped: Dict[str, int] = {"H": 0, "V": 0}
    if max_candidates_per_axis is not None:
        limit = max(int(max_candidates_per_axis), pick_k)
        if limit > int(max_candidates_per_axis):
            limit_adjusted = True
        filtered_axis: Dict[str, List[Segment]] = {}
        for axis, seg_list in (("H", h_segs), ("V", v_segs)):
            if limit <= 0 or len(seg_list) <= limit:
                filtered_axis[axis] = list(seg_list)
                continue
            sorted_by_length = sorted(seg_list, key=_segment_length, reverse=True)
            filtered_axis[axis] = sorted_by_length[:limit]
            prefilter_dropped[axis] = len(seg_list) - len(filtered_axis[axis])
        h_segs = filtered_axis["H"]
        v_segs = filtered_axis["V"]

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
    scored_per_axis: Dict[str, int] = {axis: len(cands) for axis, cands in axis_lists.items()}
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
    if prefilter_dropped["H"]:
        notes.append(f"Prefilter dropped {prefilter_dropped['H']} horizontal candidates.")
    if prefilter_dropped["V"]:
        notes.append(f"Prefilter dropped {prefilter_dropped['V']} vertical candidates.")
    if limit_adjusted:
        notes.append("max_candidates_per_axis raised to satisfy pick_k diversity requirement.")

    info = {
        "had_enough_H": had_enough_h,
        "had_enough_V": had_enough_v,
        "diversity_enforced": diversity_enforced,
        "notes": notes,
        "scored_candidates": dict(scored_per_axis),
        "prefilter_dropped": dict(prefilter_dropped),
    }

    return [cand.seg for cand in selected], info


def match_lines(
    pdf_lines: Sequence[Segment],
    svg_segs: Sequence[Segment],
    model: Model,
    cfg: Dict[str, object],
) -> List[Match]:
    verify_cfg = cfg.get("verify", {})
    neighbors_cfg = cfg.get("neighbors", {})
    cost_weights = cfg.get("cost_weights", {})
    dir_tol_deg = float(verify_cfg.get("dir_tol_deg", cfg.get("angle_tol_deg", 6.0)))
    tol_rel = float(verify_cfg.get("tol_rel", 0.01))
    radius_px = float(verify_cfg.get("radius_px", 1.0))
    _, _, _, _, diag_svg = _bbox_and_diag(svg_segs)
    neighbor_radius = float(neighbors_cfg.get("radius_rel", 0.06)) * diag_svg
    if neighbor_radius <= 0:
        neighbor_radius = radius_px

    transformed_pdf: List[Segment] = []
    for seg in pdf_lines:
        x1, y1 = _apply_model(model, seg.x1, seg.y1)
        x2, y2 = _apply_model(model, seg.x2, seg.y2)
        transformed_pdf.append(Segment(x1, y1, x2, y2))

    pdf_infos = _prepare_segment_infos(transformed_pdf, neighbor_radius, dir_tol_deg)
    svg_infos = _prepare_segment_infos(svg_segs, neighbor_radius, dir_tol_deg)

    svg_axis_map = {idx: info.axis for idx, info in enumerate(svg_infos)}

    candidates: List[List[Tuple[int, float]]] = []
    candidate_sets: List[set[int]] = []
    for pdf_info in pdf_infos:
        row_candidates: List[Tuple[int, float]] = []
        row_indices: set[int] = set()
        if pdf_info.axis is not None:
            for idx, svg_info in enumerate(svg_infos):
                if svg_axis_map[idx] != pdf_info.axis or svg_info.axis is None:
                    continue
                dist = _segment_distance(pdf_info.seg, svg_info.seg)
                if dist > radius_px:
                    continue
                cost = candidate_cost(
                    pdf_info.seg,
                    svg_info.seg,
                    cost_weights,
                    pdf_info.signature,
                    svg_info.signature,
                    cfg,
                )
                if math.isinf(cost):
                    continue
                row_candidates.append((idx, cost))
                row_indices.add(idx)
        candidates.append(row_candidates)
        candidate_sets.append(row_indices)

    all_candidate_indices: List[int] = sorted(set().union(*candidate_sets))
    num_pdf = len(pdf_infos)
    unavailable_cost = 1e6
    dummy_cost = 5e5
    columns: List[Optional[int]] = list(all_candidate_indices)
    while len(columns) < num_pdf:
        columns.append(None)

    cost_matrix: List[List[float]] = []
    row_best_costs: List[float] = []
    for row, row_candidates in enumerate(candidates):
        mapping = {idx: cost for idx, cost in row_candidates}
        best = min((cost for cost in mapping.values()), default=float("inf"))
        row_best_costs.append(best)
        costs_row: List[float] = []
        for col_idx in columns:
            if col_idx is None:
                costs_row.append(dummy_cost)
            else:
                cost_val = mapping.get(col_idx, unavailable_cost)
                costs_row.append(cost_val)
        cost_matrix.append(costs_row)

    assignments: List[Tuple[int, int]] = []
    if cost_matrix and cost_matrix[0]:
        assignments = hungarian_solve(cost_matrix)

    row_to_col: Dict[int, Optional[int]] = {row: None for row in range(num_pdf)}
    for row, col in assignments:
        if row < num_pdf and col < len(columns):
            row_to_col[row] = col

    finite_row_costs = [c for c in row_best_costs if math.isfinite(c)]
    if verify_cfg.get("max_cost") is not None:
        max_cost = float(verify_cfg["max_cost"])
    elif finite_row_costs:
        max_cost = 3.0 * median(finite_row_costs)
    else:
        max_cost = dummy_cost

    matches: List[Match] = []
    for idx, (original_seg, pdf_info) in enumerate(zip(pdf_lines, pdf_infos)):
        col_idx = row_to_col.get(idx)
        chosen_cost = dummy_cost
        svg_seg: Optional[Segment] = None
        svg_info: Optional[_SegmentInfo] = None
        if col_idx is not None and col_idx < len(columns):
            svg_index = columns[col_idx]
            if svg_index is not None:
                info = svg_infos[svg_index]
                candidate_dict = {cid: cost for cid, cost in candidates[idx]}
                chosen_cost = candidate_dict.get(svg_index, unavailable_cost)
                if chosen_cost <= max_cost:
                    svg_seg = info.seg
                    svg_info = info
        if svg_seg is None:
            matches.append(
                Match(
                    id=idx,
                    axis=pdf_info.axis or "H",
                    pdf_seg=original_seg,
                    svg_seg=None,
                    cost=float(chosen_cost),
                    confidence=0.0,
                    pdf_len=_segment_length(original_seg),
                    svg_len=None,
                    rel_error=None,
                    pass01=None,
                )
            )
            continue
        pdf_len = _segment_length(original_seg)
        svg_len = _segment_length(svg_seg)
        ratio = svg_len / pdf_len if pdf_len > 0 else float("inf")
        axis_scale_expected = abs(model.sx) if (pdf_info.axis or "H") == "H" else abs(model.sy)
        rel_error = abs(ratio / axis_scale_expected - 1.0) if axis_scale_expected > 0 else float("inf")
        pass01 = 1 if rel_error <= tol_rel else 0
        threshold = max(max_cost, 1e-6)
        base_conf = max(0.0, min(1.0, 1.0 - chosen_cost / (threshold * 1.5)))
        neigh_ratio = 0.0
        if pdf_info.signature and svg_info.signature:
            neigh_ratio = min(len(pdf_info.signature), len(svg_info.signature)) / max(
                len(pdf_info.signature), len(svg_info.signature)
            )
        confidence = max(0.0, min(1.0, base_conf * (0.5 + 0.5 * neigh_ratio)))
        matches.append(
            Match(
                id=idx,
                axis=pdf_info.axis or "H",
                pdf_seg=original_seg,
                svg_seg=svg_seg,
                cost=float(chosen_cost),
                confidence=confidence,
                pdf_len=pdf_len,
                svg_len=svg_len,
                rel_error=rel_error,
                pass01=pass01,
            )
        )
    return matches
