from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0


@dataclass
class Transform2D:
    flip: tuple[float, float]
    rot_deg: int
    sx: float
    sy: float
    tx: float
    ty: float

    def F(self) -> np.ndarray:
        fx, fy = self.flip
        return np.array([[fx, 0.0], [0.0, fy]], dtype=float)

    def R(self) -> np.ndarray:
        a = (self.rot_deg % 360) * DEG
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca, -sa], [sa, ca]], dtype=float)

    def S(self) -> np.ndarray:
        assert self.sx > 0 and self.sy > 0
        return np.array([[self.sx, 0.0], [0.0, self.sy]], dtype=float)

    def A(self) -> np.ndarray:
        return self.F() @ self.R() @ self.S()

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return (pts @ self.A().T) + np.array([self.tx, self.ty])

    def summary(self) -> str:
        fx, fy = self.flip
        return (
            f"rot={self.rot_deg}° | flips=({int(fx)},{int(fy)}) | "
            f"sx={self.sx:.6f} sy={self.sy:.6f} tx={self.tx:.3f} ty={self.ty:.3f}"
        )
