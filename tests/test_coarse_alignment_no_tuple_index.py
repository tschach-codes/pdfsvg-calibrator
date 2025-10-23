import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.core.grid_safety import zeros2d, ensure_ndarray2d, safe_index


def fake_coarse(acc):
    acc = ensure_ndarray2d("acc", acc)
    for i in range(3):
        for j in range(4):
            ii, jj = safe_index(i, j)
            acc[ii, jj] += 1
    return acc


def test_coarse_grid_is_ndarray_and_indexable():
    acc = zeros2d(3, 4, dtype=np.int32)
    out = fake_coarse(acc)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 4)
    assert out.sum() == 12
