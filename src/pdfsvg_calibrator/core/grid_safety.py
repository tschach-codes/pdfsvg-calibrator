from __future__ import annotations

from typing import Any

import numpy as np


def zeros2d(h: int, w: int, dtype=float) -> np.ndarray:
    """Create a true 2D numpy array with guaranteed independence of rows."""
    return np.zeros((int(h), int(w)), dtype=dtype, order="C")


def ensure_ndarray2d(name: str, arr: Any) -> np.ndarray:
    """Convert to 2D ndarray and assert shape correctness."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got ndim={a.ndim}")
    return a


def safe_index(i: Any, j: Any) -> tuple[int, int]:
    """Force integer indices (fix float/tuple leaks)."""
    return int(i), int(j)


def dbg_array(name: str, a: Any):
    """Quick one-line debug print for type & shape."""
    print(f"DBG {name}: type={type(a)}, shape={getattr(a, 'shape', None)}")
