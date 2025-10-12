from typing import List
from .types import Segment, Match
import math

def select_lines(pdf_segs: List[Segment], model, cfg: dict)->List[Segment]:
    ss = sorted(pdf_segs, key=lambda s: math.hypot(s.x2-s.x1, s.y2-s.y1), reverse=True)
    return ss[:5]

def match_lines(pdf_lines: List[Segment], svg_segs: List[Segment], model, cfg: dict)->List[Match]:
    out=[]
    for i,s in enumerate(pdf_lines, start=1):
        pdf_len = math.hypot(s.x2-s.x1, s.y2-s.y1)
        out.append(Match(id=i, axis="H", pdf_seg=s, svg_seg=None, cost=1e9,
                         confidence=0.0, pdf_len=pdf_len, svg_len=None,
                         rel_error=None, pass01=None))
    return out
