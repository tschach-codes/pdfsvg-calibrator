from typing import List, Tuple
from .types import Segment
import math

def classify_hv(segments: List[Segment], angle_tol_deg: float=6.0)->Tuple[List[Segment], List[Segment]]:
    H,V=[],[]
    for s in segments:
        ang = abs((math.degrees(math.atan2(s.y2-s.y1, s.x2-s.x1)) + 360.0) % 180.0)
        if min(ang, 180-ang) < angle_tol_deg:
            H.append(s)
        elif abs(90.0-ang) < angle_tol_deg:
            V.append(s)
    return H,V

def merge_collinear(segments: List[Segment])->List[Segment]:
    # TODO: implement merging (placeholder returns as-is)
    return segments
