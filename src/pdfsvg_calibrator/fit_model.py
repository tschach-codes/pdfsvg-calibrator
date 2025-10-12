from .types import Model, Segment
from typing import List, Tuple

def calibrate(pdf_segs: List[Segment], svg_segs: List[Segment], pdf_size, svg_size, cfg: dict)->Model:
    # TODO: implement full RANSAC with flips/rotations; placeholder identity model
    return Model(rot_deg=0, sx=1.0, sy=1.0, tx=0.0, ty=0.0, score=1.0, rmse=0.0, p95=0.0, median=0.0)
