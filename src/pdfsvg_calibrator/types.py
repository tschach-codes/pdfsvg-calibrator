from dataclasses import dataclass
from typing import Literal, Optional, Tuple

@dataclass
class Segment:
    x1: float; y1: float; x2: float; y2: float

Axis = Literal["H","V"]

@dataclass
class Model:
    rot_deg: int
    sx: float; sy: float; tx: float; ty: float
    score: float; rmse: float; p95: float; median: float
    quality_notes: Tuple[str, ...] = ()

@dataclass
class Match:
    id: int
    axis: Axis
    pdf_seg: Segment
    svg_seg: Optional[Segment]
    cost: float
    confidence: float
    pdf_len: float
    svg_len: Optional[float]
    rel_error: Optional[float]
    pass01: Optional[int]
