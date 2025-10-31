from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import math
import io
import numpy as np
from PIL import Image, ImageDraw
from lxml import etree
import re


def _safe_local_tag(el) -> Optional[str]:
    """
    Return lowercase localname for real element nodes.
    Skip comments / processing instructions / special nodes.
    """
    tag_obj = getattr(el, "tag", None)
    if isinstance(tag_obj, bytes):
        try:
            tag_obj = tag_obj.decode("utf-8", errors="ignore")
        except Exception:
            pass
    if not isinstance(tag_obj, str):
        return None
    return etree.QName(tag_obj).localname.lower()

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
    pair_index: Optional[int] = None


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

    debug: Dict[str, Any] = field(default_factory=dict)


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


def _is_large_enough_to_consider(seg_list: List[Tuple[float,float,float,float]], min_len: float) -> Optional[Tuple[float,float,float,float]]:
    """
    Given a list of raw line segments (x1,y1,x2,y2), pick the single
    longest segment whose length >= min_len. If none pass, return None.
    """
    best = None
    best_L = 0.0
    for (x1,y1,x2,y2) in seg_list:
        L = math.hypot(x2-x1, y2-y1)
        if L >= min_len and L > best_L:
            best_L = L
            best = (x1,y1,x2,y2)
    return best


def _extract_line_segments_from_path_d(d_attr: str) -> List[Tuple[float,float,float,float]]:
    """
    Parse a subset of SVG path syntax and extract straight line segments.

    We handle commands:
      M x y   (move absolute)
      m dx dy (move relative)
      L x y   (line absolute)
      l dx dy (line relative)
      H x     (horizontal line abs)
      h dx    (horizontal line rel)
      V y     (vertical line abs)
      v dy    (vertical line rel)
    We IGNORE curves (C,Q,A,S,T, etc.) for now.
    We also ignore Z/z close-path for now because that usually indicates
    closed polygons, not dimension leaders.

    We return a list of (x1,y1,x2,y2) for every straight subsegment we see.
    """
    # Tokenize path data into [ (cmd, [floats...]), ... ]
    # We'll do a very basic parser: iterate through chars, detect letters as commands.
    # Then consume the following numbers until the next command letter.
    tokens: List[Tuple[str,List[float]]] = []
    cur_cmd = None
    cur_nums: List[float] = []

    def flush():
        nonlocal tokens, cur_cmd, cur_nums
        if cur_cmd is not None:
            tokens.append((cur_cmd, cur_nums))
        cur_cmd = None
        cur_nums = []

    # pass 1: break into cmd + numeric chunks
    buf = ""
    for ch in d_attr:
        if ch.isalpha():
            # flush any pending
            if cur_cmd is not None:
                # finish previous cmd before switching
                flush()
            cur_cmd = ch
            cur_nums = []
            buf = ""
        else:
            # keep building numeric buffer with commas/space
            buf += ch
            # split on space/comma
            # we only finalize nums on actual split, we'll post-process after loop
            # so do nothing here
            pass
        # whenever we hit a command char, we continue reading numbers for that command
        # until next alpha or end.

        # We don't finalize nums here; we'll do after the loop
        # because we collected them in buf but not yet splitted.

        # Actually we do need to push tokens per command if multiple commands appear
        # in a row without numeric separation. But typical path data won't do that
        # in a way that breaks us.

        # We'll parse numbers for each command after the loop ends.

    # end loop
    # We consumed chars, but we didn't yet extract floats from buf for last command.
    # Let's do a second pass to rebuild properly in an easier way.

    # The above quick attempt is a bit too naive: we lost track of some numeric sets.
    # Let's do a second, more robust approach instead:
    # We'll do a real mini state machine in one go:

    tokens = []
    cur_cmd = None
    cur_nums = []

    def push_token():
        nonlocal tokens, cur_cmd, cur_nums
        if cur_cmd is not None:
            tokens.append((cur_cmd, cur_nums))
        cur_cmd = None
        cur_nums = []

    # helper to finalize numeric parse from a buffer string of mixed commas/spaces
    def parse_nums(s: str) -> List[float]:
        out = []
        # break on commas
        for part in s.replace(",", " ").split():
            try:
                out.append(float(part))
            except ValueError:
                pass
        return out

    # second, more robust pass:
    num_buf = ""
    for ch in d_attr:
        if ch.isalpha():
            # new command
            # flush previous command+nums if any
            if cur_cmd is not None:
                # we still might have trailing nums in num_buf
                cur_nums.extend(parse_nums(num_buf))
                num_buf = ""
                push_token()
            # start new
            cur_cmd = ch
            cur_nums = []
        else:
            # numeric or punctuation
            num_buf += ch
            # if we see another letter next iteration, we'll parse_nums then

        # note: we'll add the last num_buf after loop as well
    # end for
    if cur_cmd is not None:
        cur_nums.extend(parse_nums(num_buf))
        push_token()

    # now tokens = [(cmd, [numbers...]), ...]

    segs: List[Tuple[float,float,float,float]] = []

    # interpret tokens as successive draw ops
    # We'll maintain a current point (cx,cy).
    cx, cy = 0.0, 0.0
    have_pos = False

    def emit_line(nx, ny, ox, oy):
        # add segment ox,oy -> nx,ny
        if ox is None or oy is None:
            return
        segs.append((ox, oy, nx, ny))

    i = 0
    while i < len(tokens):
        cmd, nums = tokens[i]
        i += 1

        c = cmd
        # absolute vs relative logic:
        is_rel = c.islower()
        c_up = c.upper()

        # M/m can have multiple coordinate pairs => move + implicit line-tos
        if c_up == "M":
            # nums are [x,y, x,y, ...]
            it = iter(nums)
            first_pair = True
            for x in it:
                y = next(it, None)
                if y is None:
                    break
                if is_rel and have_pos:
                    nx = cx + x
                    ny = cy + y
                else:
                    nx = x
                    ny = y
                cx, cy = nx, ny
                if not have_pos:
                    have_pos = True
                    first_pair = False
                else:
                    if first_pair:
                        first_pair = False
                    else:
                        # implicit line-to after the first pair
                        # previous point -> cx,cy
                        # Actually we updated cx,cy to the new point, we lost old.
                        # let's fix by storing old then updating.
                        pass
            # NOTE: proper M parsing with multiple pairs is messy.
            # We'll keep it simple: just set cx,cy to the last pair.
            # This is good enough for dimension-ish geometry where M is usually single.
            continue

        # L/l = line to (x,y) pairs
        if c_up == "L":
            it = iter(nums)
            for x in it:
                y = next(it, None)
                if y is None:
                    break
                ox, oy = cx, cy
                if is_rel:
                    cx += x
                    cy += y
                else:
                    cx, cy = x, y
                if have_pos:
                    emit_line(cx, cy, ox, oy)
                else:
                    have_pos = True
            continue

        # H/h = horizontal line
        if c_up == "H":
            for x in nums:
                ox, oy = cx, cy
                if is_rel:
                    cx += x
                else:
                    cx = x
                if have_pos:
                    emit_line(cx, cy, ox, oy)
                else:
                    have_pos = True
            continue

        # V/v = vertical line
        if c_up == "V":
            for y in nums:
                ox, oy = cx, cy
                if is_rel:
                    cy += y
                else:
                    cy = y
                if have_pos:
                    emit_line(cx, cy, ox, oy)
                else:
                    have_pos = True
            continue

        # Z/z -> closepath: treat as a segment back to start-of-subpath.
        # For dimension lines, typically not relevant, we skip.

        # All curve commands: C,Q,S,T,A...
        # We ignore, because these are not straight dimension lines.
        # We'll just leave cx,cy as last known (SVG spec moves current point).
        # (In actual SVG, those also update cx,cy to the new end point,
        # but we don't generate linear segments for them.)
        # For dimension lines that use arrowheads etc. with curves,
        # we'll miss them, but that's acceptable for now.

    return segs


def _svg_get_segments(root) -> List[Tuple[float,float,float,float]]:
    """
    Unified segment extractor:
    - Collect <line> elements directly
    - Collect the LONGEST straight subsegment from <path> elements,
      if that subsegment is not trivially tiny.
    Return list[(x1,y1,x2,y2)] across both.

    NOTE:
    We don't try to merge collinear chunks etc. We just need good long candidates.
    """
    segs: List[Tuple[float,float,float,float]] = []

    # Pass 1: direct <line> elems
    for el in root.iter("*"):
        tag = _safe_local_tag(el)
        if tag is None:
            continue
        if tag == "line":
            try:
                x1 = float(el.get("x1", "0"))
                y1 = float(el.get("y1", "0"))
                x2 = float(el.get("x2", "0"))
                y2 = float(el.get("y2", "0"))
            except ValueError:
                continue
            segs.append((x1,y1,x2,y2))

    # Pass 2: <path> elems, longest straight chunk
    for el in root.iter("*"):
        tag = _safe_local_tag(el)
        if tag is None:
            continue
        if tag == "path":
            d_attr = el.get("d")
            if not d_attr:
                continue

            raw_line_segs = _extract_line_segments_from_path_d(d_attr)
            # Heuristik: wir wollen NICHT winzige Pfeilspitzenfragmente.
            # Setzen wir erstmal eine untere Grenze rein, damit wir nicht alles ansammeln:
            # -> wir entscheiden "groß genug" erst später, weil wir die globale Seite erst kennen,
            #    aber wir können hier trotzdem schonmal einfach 'max segment wins' nehmen.
            best_seg = _is_large_enough_to_consider(raw_line_segs, min_len=0.0)
            if best_seg is not None:
                segs.append(best_seg)

    return segs


def _make_text_entry(
    raw_text: str,
    bbox_xywh: Tuple[float, float, float, float],
    *,
    value: Optional[float] = None,
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    x, y, w, h = bbox_xywh
    if value is None or unit is None:
        parsed_val, parsed_unit = _extract_value_and_unit(raw_text)
        if value is None:
            value = parsed_val
        if unit is None:
            unit = parsed_unit
    center = (x + w * 0.5, y + h * 0.5)
    entry: Dict[str, Any] = {
        "text": raw_text,
        "unit": unit,
        "value": value,
        "bbox": bbox_xywh,
        "center": center,
    }
    return entry


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
    for el in root.iter("*"):
        tag = _safe_local_tag(el)
        if tag is None:
            continue
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

            bbox_xywh = _bbox_minmax_to_xywh((minx, miny, maxx, maxy))
            entry = _make_text_entry(raw_txt, bbox_xywh)
            entry["_debug_index"] = len(texts)
            texts.append(entry)
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

    for el in root.iter("*"):
        tag = _safe_local_tag(el)
        if tag is None:
            continue
        if tag == "path":
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


def _bbox_center(b: Tuple[float, float, float, float]):
    x0, y0, w, h = b
    return (x0 + w * 0.5, y0 + h * 0.5)


def _bbox_xywh_to_minmax(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def _bbox_minmax_to_xywh(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = b
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


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


_UNIT_RE = re.compile(r"(\d[\d\s\.,]*)\s*([a-zA-Zµμ]+)")


def _extract_value_and_unit(s: str) -> Tuple[Optional[float], Optional[str]]:
    value = _try_parse_number(s)
    unit = None
    if s:
        match = _UNIT_RE.search(s)
        if match:
            unit = match.group(2)
    return value, unit


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
    text_elems: List[Dict[str, Any]],
    pairs_debug: List[Dict[str, Any]],
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

    cands: List[DimCandidate] = []

    for seg_idx, (x1,y1,x2,y2) in enumerate(segments):
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
            raw_txt = t.get("text", "")
            bbox_xywh = t.get("bbox")
            if not bbox_xywh:
                continue
            center_txt = t.get("center") or _bbox_center(bbox_xywh)

            # Check perpendicular distance to line is not crazy
            perp_dist = _closest_point_dist_to_line(p1,p2, center_txt)
            if perp_dist > normal_dist_max:
                continue

            # Check projection along normal is within ±normal_dist_max
            dist_n = _dist_along_normal(mid, nvec, center_txt)
            if abs(dist_n) > normal_dist_max:
                continue

            # We got a plausible measurement label.
            val = t.get("value")
            bbox_minmax = _bbox_xywh_to_minmax(bbox_xywh)
            txt_idx = t.get("_debug_index")
            if txt_idx is None:
                txt_idx = text_elems.index(t)

            pairs_debug.append({
                "seg": seg_idx,
                "txt": txt_idx,
                "dist": dist_n,
                "value": val,
                "unit": t.get("unit"),
            })
            pair_idx = len(pairs_debug) - 1

            cand = DimCandidate(
                p1=p1,
                p2=p2,
                length=L,
                orientation=ori,
                numeric_value=val,
                raw_text=raw_txt,
                text_bbox=bbox_minmax,
                dist_normal=dist_n,
                pair_index=pair_idx,
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
                        bbox_xywh = _bbox_minmax_to_xywh(cropbox)
                        ocr_entry = _make_text_entry(raw_ocr, bbox_xywh, value=val)
                        ocr_entry["_debug_index"] = len(text_elems)
                        text_elems.append(ocr_entry)
                        txt_idx = ocr_entry["_debug_index"]
                        pairs_debug.append({
                            "seg": seg_idx,
                            "txt": txt_idx,
                            "dist": 0.0,
                            "value": val,
                            "unit": ocr_entry.get("unit"),
                        })
                        pair_idx = len(pairs_debug) - 1
                        best_local = DimCandidate(
                            p1=p1,
                            p2=p2,
                            length=L,
                            orientation=ori,
                            numeric_value=val,
                            raw_text=raw_ocr,
                            text_bbox=cropbox,
                            dist_normal=0.0,
                            pair_index=pair_idx,
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

    debug_dict: Dict[str, Any] = {}
    segments_svg: List[Tuple[float, float, float, float]] = []
    text_elems: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []

    try:
        root, _tree = _parse_svg(svg_path)
    except Exception as e:
        dbg = debug_dict.setdefault("dimscale", {})
        dbg["segments_svg"] = segments_svg
        dbg["texts_svg"] = text_elems
        dbg["pairs"] = pairs
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
            debug=debug_dict,
        )

    segments = _svg_get_segments(root)
    segments_svg = list(segments)
    text_elems = _svg_get_text_elems(root)

    # Gather candidates
    cands = _gather_dim_candidates(
        root,
        segments,
        text_elems,
        pairs,
        min_len_rel_diag=0.07,        # 7% der globalen Diagonale
        normal_dist_max=50.0,         # ±50 SVG units orthogonal
        require_hv_only=True,         # h/v only for now
        enable_ocr_paths=enable_ocr_paths,
    )

    cluster_rel_tol = 0.01
    ok, reason, best_ratio_x, best_ratio_y, tot_cands, inliers = _cluster_scale_ratios(
        cands,
        min_total=3,
        rel_tol=cluster_rel_tol,   # ±1%
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

    # mark inlier pairs for debug overlays
    def _within_rel_tol(value: Optional[float], reference: Optional[float]) -> bool:
        if value is None or reference is None:
            return False
        if abs(reference) < 1e-12:
            return False
        return abs(value - reference) <= abs(reference) * cluster_rel_tol

    inlier_pair_indices: set[int] = set()
    for cand in cands:
        if cand.numeric_value is None or cand.numeric_value == 0 or cand.pair_index is None:
            continue
        dx = abs(cand.p2[0] - cand.p1[0])
        dy = abs(cand.p2[1] - cand.p1[1])
        ratio = None
        if cand.orientation == "h":
            ratio = dx / cand.numeric_value
            if _within_rel_tol(ratio, best_ratio_x):
                inlier_pair_indices.add(cand.pair_index)
        elif cand.orientation == "v":
            ratio = dy / cand.numeric_value
            if _within_rel_tol(ratio, best_ratio_y):
                inlier_pair_indices.add(cand.pair_index)

    for idx in inlier_pair_indices:
        if 0 <= idx < len(pairs):
            pairs[idx]["inlier"] = True

    texts_svg_debug = [
        {
            "text": t.get("text"),
            "unit": t.get("unit"),
            "value": t.get("value"),
            "bbox": t.get("bbox"),
            "center": t.get("center"),
        }
        for t in text_elems
    ]

    dbg = debug_dict.setdefault("dimscale", {})
    dbg["segments_svg"] = segments_svg
    dbg["texts_svg"] = texts_svg_debug
    dbg["pairs"] = pairs

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
        debug=debug_dict,
    )
