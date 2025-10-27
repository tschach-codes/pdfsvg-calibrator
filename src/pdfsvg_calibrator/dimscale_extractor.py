from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import math
import io
import numpy as np
from PIL import Image, ImageDraw
from lxml import etree

# OCR optional
try:
    import pytesseract  # requires tesseract installed on system
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False


###############################################################################
# Data structures
###############################################################################

@dataclass
class DimCandidate:
    # one candidate measurement association between a long line and a nearby text-ish thing
    p1: Tuple[float, float]   # line start (x1,y1)
    p2: Tuple[float, float]   # line end   (x2,y2)
    length: float             # line length in SVG units
    orientation: str          # "h" or "v" or "diag"
    numeric_value: Optional[float]  # parsed number from text
    raw_text: str             # raw string (from <text> or OCR)
    text_bbox: Tuple[float,float,float,float]  # (minx,miny,maxx,maxy) of label region
    dist_normal: float        # signed distance along normal from line midpoint
    # ratio_x/ratio_y not filled yet, will be computed later


@dataclass
class DimScaleResult:
    ok: bool
    reason: str

    scale_x_svg_per_unit: Optional[float]
    scale_y_svg_per_unit: Optional[float]

    scale_x_unit_per_svg: Optional[float]
    scale_y_unit_per_svg: Optional[float]

    candidates_used: int
    inlier_cluster_size: int

    # debug crops (PIL Images) for visual QA
    preview_horizontal: Optional[Image.Image]
    preview_vertical: Optional[Image.Image]


###############################################################################
# SVG parsing helpers
###############################################################################

def _parse_svg(svg_path: str):
    """
    Parse SVG with lxml and return root and namespace map.
    We'll assume a fairly standard SVG.
    """
    parser = etree.XMLParser(remove_comments=False, recover=True)
    tree = etree.parse(svg_path, parser)
    root = tree.getroot()
    return root, tree


def _svg_get_segments(root) -> List[Tuple[float,float,float,float]]:
    """
    Very naive segment extractor from <line> elements for now.
    TODO: extend to path 'd' parsing if needed.
    Returns list of segments as (x1,y1,x2,y2).
    Note: We'll treat <line> first because dimension lines often survive as <line>.
    """
    segs: List[Tuple[float,float,float,float]] = []

    # typical SVG ns handling
    # elements may look like {http://www.w3.org/2000/svg}line
    for el in root.iter():
        tag = etree.QName(el.tag).localname.lower()
        if tag == "line":
            try:
                x1 = float(el.get("x1", "0"))
                y1 = float(el.get("y1", "0"))
                x2 = float(el.get("x2", "0"))
                y2 = float(el.get("y2", "0"))
            except ValueError:
                continue
            segs.append((x1,y1,x2,y2))

    return segs


def _svg_get_text_elems(root) -> List[Dict[str,Any]]:
    """
    Collect <text> elements with rough bounding boxes.
    For <text>, we estimate bbox from x,y and font-size.
    This is crude but good enough to get us started.
    We will also grab the string content.
    Returns list of dicts: {
        "text": "...",
        "bbox": (minx,miny,maxx,maxy)
    }
    """
    texts: List[Dict[str,Any]] = []
    for el in root.iter():
        tag = etree.QName(el.tag).localname.lower()
        if tag == "text":
            raw_txt = "".join(el.itertext()).strip()
            if not raw_txt:
                continue
            try:
                x = float(el.get("x", "0"))
                y = float(el.get("y", "0"))
            except ValueError:
                continue

            # font-size heuristic
            fs_attr = el.get("font-size", None)
            if fs_attr is not None:
                try:
                    fs_val = float(fs_attr)
                except ValueError:
                    fs_val = 10.0
            else:
                fs_val = 10.0

            # naive bbox: assume left= x, top= y - fs_val, right ~ x+len*fs_val*0.6
            # This is crude but works for rough spatial matching.
            w_est = max(len(raw_txt),1) * fs_val * 0.6
            minx = x
            maxx = x + w_est
            # SVG text y = baseline, so approximate ascender above baseline
            miny = y - fs_val
            maxy = y + fs_val*0.2

            texts.append({
                "text": raw_txt,
                "bbox": (minx,miny,maxx,maxy),
            })
    return texts


# NOTE:
# Outline glyph clusters (<path> groups forming digits) is the hard part.
# We'll scaffold a placeholder that finds small <path> groups near a query point
# and (optionally) OCR them.
# For v1 we'll keep this minimal: we won't try to proactively index all clusters.
# We'll just sample around candidate lines on demand.


def _collect_paths_in_bbox(root, bbox: Tuple[float,float,float,float]) -> List[etree._Element]:
    """
    Return list of <path> elements whose 'd' we found and whose bbox
    (roughly estimated via parsing numbers in 'd') intersects bbox.
    WARNING: rough and expensive, but this runs only on local crop queries.
    """
    minx,miny,maxx,maxy = bbox
    hits = []

    for el in root.iter():
        if etree.QName(el.tag).localname.lower() == "path":
            d_attr = el.get("d")
            if not d_attr:
                continue
            # extract all coordinate pairs from the path 'd'
            # We'll do a super naive parse: grab all floats, pair consecutive.
            # This is approximate bbox, good enough for presence testing.
            nums = _extract_floats_from_path_d(d_attr)
            if len(nums) < 2:
                continue
            xs = nums[0::2]
            ys = nums[1::2]
            if not xs or not ys:
                continue
            bb_minx = min(xs)
            bb_maxx = max(xs)
            bb_miny = min(ys)
            bb_maxy = max(ys)

            # check intersection with bbox
            if not (bb_maxx < minx or bb_minx > maxx or bb_maxy < miny or bb_miny > maxy):
                hits.append(el)

    return hits


def _extract_floats_from_path_d(d_str: str) -> List[float]:
    """
    crude float extractor for SVG path 'd' content.
    """
    out = []
    # split by space/commas/letters
    # We treat command letters as separators. We'll replace them with spaces.
    cleaned = []
    cur = []
    for ch in d_str:
        if ch.isalpha():
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    cleaned = "".join(cleaned)
    for tok in cleaned.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


###############################################################################
# Geometry helpers
###############################################################################

def _length_and_orientation(p1, p2) -> Tuple[float,str]:
    x1,y1 = p1
    x2,y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx))
    # classify orientation
    # horizontal-ish vs vertical-ish vs diag
    # We'll snap if within 10 degrees.
    if abs(ang) < 10 or abs(ang-180) < 10 or abs(ang+180)<10:
        ori = "h"
    elif abs(abs(ang)-90) < 10:
        ori = "v"
    else:
        ori = "diag"
    return L, ori


def _midpoint(p1, p2):
    return ((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5)


def _normal_vector(p1, p2):
    # line dir = (dx,dy)
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    # normal ~ (-dy, dx)
    nx, ny = -dy, dx
    nlen = math.hypot(nx, ny)
    if nlen < 1e-9:
        return (0.0, 0.0)
    return (nx/nlen, ny/nlen)


def _bbox_center(b):
    x0,y0,x1,y1 = b
    return ((x0+x1)*0.5, (y0+y1)*0.5)


def _dist_along_normal(p_mid, nvec, pt):
    # project (pt - p_mid) onto nvec
    vx = pt[0]-p_mid[0]
    vy = pt[1]-p_mid[1]
    return vx*nvec[0] + vy*nvec[1]


def _closest_point_dist_to_line(p1,p2,pt):
    # perpendicular distance from pt to infinite line p1->p2
    x1,y1 = p1
    x2,y2 = p2
    x0,y0 = pt
    dx = x2-x1
    dy = y2-y1
    num = abs(dy*x0 - dx*y0 + x2*y1 - y2*x1)
    den = math.hypot(dx, dy)
    if den < 1e-9:
        return 1e9
    return num/den


def _bbox_of_line_and_label(line_p1, line_p2, label_bbox, margin=50.0):
    """
    We produce a crop bbox for debugging/preview:
    include line segment and the label bbox, plus some margin.
    """
    x1,y1 = line_p1
    x2,y2 = line_p2
    lx0,ly0,lx1,ly1 = label_bbox
    minx = min(x1,x2,lx0) - margin
    miny = min(y1,y2,ly0) - margin
    maxx = max(x1,x2,lx1) + margin
    maxy = max(y1,y2,ly1) + margin
    return (minx,miny,maxx,maxy)


###############################################################################
# OCR / text parsing helpers
###############################################################################

def _try_parse_number(s: str) -> Optional[float]:
    """
    Extract something numeric from the string (e.g. "2450", "1.63", "10,00").
    We'll:
      - replace comma with dot
      - take first contiguous [0-9 .] chunk
    If that fails, return None.
    """
    # normalize decimal comma -> dot
    s2 = s.replace(",", ".")
    # keep digits and dot/space
    cleaned_chars = []
    for ch in s2:
        if ch.isdigit() or ch == "." or ch == " ":
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    cleaned = "".join(cleaned_chars)
    parts = cleaned.split()
    if not parts:
        return None
    # take the longest numeric-ish token
    parts_sorted = sorted(parts, key=len, reverse=True)
    for token in parts_sorted:
        # must contain at least one digit
        if any(c.isdigit() for c in token):
            try:
                val = float(token)
                return val
            except ValueError:
                continue
    return None


def _ocr_crop_from_svg(root, crop_bbox: Tuple[float,float,float,float], dpi=200) -> Optional[str]:
    """
    Render a local crop around crop_bbox from SVG via Pillow vector draw? We don't have a full SVG rasterizer
    here without bringing in cairosvg again. We WILL bring in cairosvg - we already imported it elsewhere
    in raster_align. We'll replicate minimal logic here.

    Strategy:
      - We'll render the FULL svg to bitmap using cairosvg at 'dpi'
      - Then we'll crop the bbox region (properly scaled).
    This is heavier, but simpler to keep consistent.

    WARNING: this means for each OCR attempt we rasterize whole SVG. We'll accept that for v1.
    Optimization later: cache rendered SVG once.
    """
    try:
        import cairosvg
    except Exception:
        return None

    # get full svg bounding box from viewBox if present, else fallback
    vb = _get_viewbox_from_root(root)
    if vb is None:
        # can't determine a stable coordinate system, skip
        return None
    minx, miny, vbw, vbh = vb
    full_w = vbw
    full_h = vbh

    # target pixel size at given dpi: we pretend 1 SVG unit == 1 px@dpi
    # Then crop_bbox maps 1:1 into that raster.
    # (This is approximate but good enough for OCR on digits.)
    px_w = int(math.ceil(full_w))
    px_h = int(math.ceil(full_h))
    if px_w <= 0 or px_h <= 0:
        return None

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=etree.tostring(root),
            output_width=px_w,
            output_height=px_h,
            background_color="white",
        )
    except Exception:
        return None

    img_full = Image.open(io.BytesIO(png_bytes)).convert("L")

    # now crop the bbox
    cx0,cy0,cx1,cy1 = crop_bbox
    # clamp
    cx0 = max(minx, cx0)
    cy0 = max(miny, cy0)
    cx1 = min(minx+full_w, cx1)
    cy1 = min(miny+full_h, cy1)

    # translate SVG coords -> pixel coords:
    # SVG coords assumed origin at (minx,miny)
    crop_px0 = int(max(0, math.floor(cx0 - minx)))
    crop_py0 = int(max(0, math.floor(cy0 - miny)))
    crop_px1 = int(min(px_w, math.ceil(cx1 - minx)))
    crop_py1 = int(min(px_h, math.ceil(cy1 - miny)))

    if crop_px1 <= crop_px0 or crop_py1 <= crop_py0:
        return None

    crop_img = img_full.crop((crop_px0, crop_py0, crop_px1, crop_py1))

    # upscale for OCR so digits are crisp at ~200 dpi-ish.
    # If the crop is tiny, we can scale it up 2x
    up_w = max(crop_img.width, 1)*2
    up_h = max(crop_img.height,1)*2
    crop_img_up = crop_img.resize((up_w, up_h), Image.BICUBIC)

    if not _HAS_TESSERACT:
        return None

    try:
        txt = pytesseract.image_to_string(
            crop_img_up,
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,",
        )
        txt = txt.strip()
    except Exception:
        return None

    if not txt:
        return None
    return txt


def _get_viewbox_from_root(root) -> Optional[Tuple[float,float,float,float]]:
    vb = root.get("viewBox")
    if not vb:
        return None
    parts = vb.strip().split()
    if len(parts) != 4:
        return None
    try:
        minx = float(parts[0])
        miny = float(parts[1])
        w    = float(parts[2])
        h    = float(parts[3])
    except ValueError:
        return None
    return (minx,miny,w,h)


###############################################################################
# Core logic
###############################################################################

def _gather_dim_candidates(
    root,
    segments: List[Tuple[float,float,float,float]],
    min_len_rel_diag: float = 0.07,
    normal_dist_max: float = 50.0,
    require_hv_only: bool = True,
    enable_ocr_paths: bool = False,
) -> List[DimCandidate]:
    """
    1. compute doc diagonal in SVG coords from viewBox
    2. keep only long lines >= min_len_rel_diag * diag
    3. for each such line:
        - midpoint M
        - normal vector n
        - search text candidates (true <text> elems)
          where bbox center projects within ±normal_dist_max along n
          AND distance to the line is small (< normal_dist_max)
        - if no <text>, optionally try OCR on path clusters in a 100x100 bbox
          around M + n * ?? (but for v1 let's reuse same bbox as text-check)
    """
    vb = _get_viewbox_from_root(root)
    if vb is None:
        return []
    vb_minx, vb_miny, vb_w, vb_h = vb
    diag = math.hypot(vb_w, vb_h)
    min_len_abs = min_len_rel_diag * diag

    text_elems = _svg_get_text_elems(root)

    cands: List[DimCandidate] = []

    for (x1,y1,x2,y2) in segments:
        p1 = (x1,y1)
        p2 = (x2,y2)
        L, ori = _length_and_orientation(p1,p2)
        if L < min_len_abs:
            continue
        if require_hv_only and (ori not in ("h","v")):
            continue

        mid = _midpoint(p1,p2)
        nvec = _normal_vector(p1,p2)

        # search <text> candidates
        best_local: Optional[DimCandidate] = None

        for t in text_elems:
            raw_txt = t["text"]
            bbox = t["bbox"]
            center_txt = _bbox_center(bbox)

            # Check perpendicular distance to line is not crazy
            perp_dist = _closest_point_dist_to_line(p1,p2, center_txt)
            if perp_dist > normal_dist_max:
                continue

            # Check projection along normal is within ±normal_dist_max
            dist_n = _dist_along_normal(mid, nvec, center_txt)
            if abs(dist_n) > normal_dist_max:
                continue

            # We got a plausible measurement label.
            val = _try_parse_number(raw_txt)

            cand = DimCandidate(
                p1=p1,
                p2=p2,
                length=L,
                orientation=ori,
                numeric_value=val,
                raw_text=raw_txt,
                text_bbox=bbox,
                dist_normal=dist_n,
            )
            # prefer ones with a parsed numeric value
            if best_local is None:
                best_local = cand
            else:
                # heuristic: choose the closer one in |dist_n|
                if abs(cand.dist_normal) < abs(best_local.dist_normal):
                    best_local = cand

        # no <text> match -> optionally OCR path cluster in vicinity
        if best_local is None and enable_ocr_paths:
            # define a crop bbox that we'd OCR
            # We'll look normal_dist_max above/below the line midpoint
            cx0 = mid[0] - normal_dist_max
            cy0 = mid[1] - normal_dist_max
            cx1 = mid[0] + normal_dist_max
            cy1 = mid[1] + normal_dist_max
            cropbox = (cx0,cy0,cx1,cy1)

            # (1) collect <path> elements in that crop (not used in v1 except to decide "is there anything")
            paths_here = _collect_paths_in_bbox(root, cropbox)
            if paths_here:
                # (2) OCR that region
                raw_ocr = _ocr_crop_from_svg(root, cropbox, dpi=200)
                if raw_ocr:
                    val = _try_parse_number(raw_ocr)
                    if val is not None:
                        best_local = DimCandidate(
                            p1=p1,
                            p2=p2,
                            length=L,
                            orientation=ori,
                            numeric_value=val,
                            raw_text=raw_ocr,
                            text_bbox=cropbox,
                            dist_normal=0.0,
                        )

        if best_local is not None:
            cands.append(best_local)

    return cands


def _cluster_scale_ratios(
    cands: List[DimCandidate],
    min_total: int = 3,
    rel_tol: float = 0.01,
) -> Tuple[bool,str,Optional[float],Optional[float],int,int]:
    """
    Compute ratio_x = length_x / value, ratio_y = length_y / value
    (length_x == length for horizontal, length_y == length for vertical.
     For 'diag' we can project onto x/y, aber wir ignorieren diag für jetzt.)
    Then cluster by closeness (±1 %).
    We need:
      - >= min_total candidates overall
      - and >=2 inliers in at least one cluster

    Returns:
      ok(bool), reason(str),
      best_ratio_x, best_ratio_y,
      total_candidates, inlier_cluster_size
    """
    usable = []
    for c in cands:
        if c.numeric_value is None:
            continue
        if c.numeric_value == 0:
            continue

        dx = abs(c.p2[0] - c.p1[0])
        dy = abs(c.p2[1] - c.p1[1])

        # ratio_x meaningful if horizontal-ish
        ratio_x = None
        ratio_y = None
        if c.orientation == "h":
            ratio_x = dx / c.numeric_value
        elif c.orientation == "v":
            ratio_y = dy / c.numeric_value
        # (diag we ignore for now)

        usable.append((ratio_x, ratio_y))

    if len(usable) < min_total:
        return (False, f"not enough dimension candidates ({len(usable)})", None, None, len(usable), 0)

    # cluster ratio_x and ratio_y separately, pick biggest consistent group
    best_ratio_x, inliers_x = _cluster_1d(
        [r[0] for r in usable if r[0] is not None],
        rel_tol=rel_tol
    )
    best_ratio_y, inliers_y = _cluster_1d(
        [r[1] for r in usable if r[1] is not None],
        rel_tol=rel_tol
    )

    inlier_cluster_size = max(inliers_x, inliers_y)

    if inlier_cluster_size < 2:
        return (False, "no stable 1% cluster", best_ratio_x, best_ratio_y, len(usable), inlier_cluster_size)

    return (True, "ok", best_ratio_x, best_ratio_y, len(usable), inlier_cluster_size)


def _cluster_1d(values: List[float], rel_tol: float) -> Tuple[Optional[float], int]:
    """
    Very simple clustering: sort, slide window, find largest group where
    max/min <= (1+rel_tol)*min.
    Return median and group size.
    """
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return (None, 0)
    vals.sort()
    best_group = []
    start = 0
    for start in range(len(vals)):
        vmin = vals[start]
        vmax_allowed = vmin * (1.0 + rel_tol)
        group = [vmin]
        for j in range(start+1, len(vals)):
            if vals[j] <= vmax_allowed:
                group.append(vals[j])
            else:
                break
        if len(group) > len(best_group):
            best_group = group[:]

    if not best_group:
        return (None, 0)
    # median of best_group
    m = np.median(best_group)
    return (float(m), len(best_group))


def _raster_debug_crop(
    root,
    bbox: Tuple[float,float,float,float],
    dpi: int = 200
) -> Optional[Image.Image]:
    """
    Render full SVG (like in OCR), crop bbox+margin and return PIL.Image for QA.
    This is similar to _ocr_crop_from_svg but returns the image.
    """
    try:
        import cairosvg
    except Exception:
        return None

    vb = _get_viewbox_from_root(root)
    if vb is None:
        return None
    minx, miny, vbw, vbh = vb
    px_w = int(math.ceil(vbw))
    px_h = int(math.ceil(vbh))
    if px_w <= 0 or px_h <= 0:
        return None

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=etree.tostring(root),
            output_width=px_w,
            output_height=px_h,
            background_color="white",
        )
    except Exception:
        return None

    img_full = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    cx0,cy0,cx1,cy1 = bbox
    cx0 = max(minx, cx0)
    cy0 = max(miny, cy0)
    cx1 = min(minx+vbw, cx1)
    cy1 = min(miny+vbh, cy1)

    crop_px0 = int(max(0, math.floor(cx0 - minx)))
    crop_py0 = int(max(0, math.floor(cy0 - miny)))
    crop_px1 = int(min(px_w, math.ceil(cx1 - minx)))
    crop_py1 = int(min(px_h, math.ceil(cy1 - miny)))

    if crop_px1 <= crop_px0 or crop_py1 <= crop_py0:
        return None

    crop_img = img_full.crop((crop_px0, crop_py0, crop_px1, crop_py1))
    return crop_img


def estimate_dimline_scale(
    svg_path: str,
    enable_ocr_paths: bool = False,
) -> DimScaleResult:
    """
    High-level entry:
    - parse SVG
    - extract <line> segments
    - gather DimCandidates via long-line + nearby text logic
    - cluster ratios (x and y separately)
    - pick best cluster if >=3 total, >=2 inliers in 1% band
    - pick one horiz and one vert candidate to render debug crops

    Returns DimScaleResult with scales + debug crop images.
    """

    try:
        root, _tree = _parse_svg(svg_path)
    except Exception as e:
        return DimScaleResult(
            ok=False,
            reason=f"SVG parse failed: {e}",
            scale_x_svg_per_unit=None,
            scale_y_svg_per_unit=None,
            scale_x_unit_per_svg=None,
            scale_y_unit_per_svg=None,
            candidates_used=0,
            inlier_cluster_size=0,
            preview_horizontal=None,
            preview_vertical=None,
        )

    segments = _svg_get_segments(root)

    # Gather candidates
    cands = _gather_dim_candidates(
        root,
        segments,
        min_len_rel_diag=0.07,        # 7% der globalen Diagonale
        normal_dist_max=50.0,         # ±50 SVG units orthogonal
        require_hv_only=True,         # h/v only for now
        enable_ocr_paths=enable_ocr_paths,
    )

    ok, reason, best_ratio_x, best_ratio_y, tot_cands, inliers = _cluster_scale_ratios(
        cands,
        min_total=3,
        rel_tol=0.01,   # ±1%
    )

    # prepare debug crops:
    prev_h = None
    prev_v = None
    # pick ONE best horizontal cand and ONE best vertical cand (if exist)
    # We'll choose the first from cands matching orientation 'h'/'v'
    for c in cands:
        dbg_bbox = _bbox_of_line_and_label(c.p1, c.p2, c.text_bbox, margin=50.0)
        if c.orientation == "h" and prev_h is None:
            prev_h = _raster_debug_crop(root, dbg_bbox, dpi=200)
        if c.orientation == "v" and prev_v is None:
            prev_v = _raster_debug_crop(root, dbg_bbox, dpi=200)
        if prev_h is not None and prev_v is not None:
            break

    # convert ratio to both svg_per_unit and unit_per_svg
    # best_ratio_x means [svg units] / [number units]
    # => svg_per_unit = best_ratio_x
    # => unit_per_svg = 1/best_ratio_x (if nonzero)
    def invert_or_none(v):
        if v is None: return None
        if abs(v) < 1e-12: return None
        return 1.0/v

    scale_x_svg_per_unit = best_ratio_x
    scale_y_svg_per_unit = best_ratio_y
    scale_x_unit_per_svg = invert_or_none(best_ratio_x)
    scale_y_unit_per_svg = invert_or_none(best_ratio_y)

    return DimScaleResult(
        ok=ok,
        reason=reason,
        scale_x_svg_per_unit=scale_x_svg_per_unit,
        scale_y_svg_per_unit=scale_y_svg_per_unit,
        scale_x_unit_per_svg=scale_x_unit_per_svg,
        scale_y_unit_per_svg=scale_y_unit_per_svg,
        candidates_used=tot_cands,
        inlier_cluster_size=inliers,
        preview_horizontal=prev_h,
        preview_vertical=prev_v,
    )
