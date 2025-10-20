from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .transform import Transform2D

@dataclass
class Segment:
    x1: float; y1: float; x2: float; y2: float

Axis = Literal["H","V"]

@dataclass
class Model:
    rot_deg: int
    sx: float; sy: float; tx: float; ty: float
    score: float; rmse: float; p95: float; median: float
    flip_x: float = 1.0
    flip_y: float = 1.0
    transform: "Transform2D | None" = None
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
