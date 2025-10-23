import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.core.grid_safety import zeros2d, ensure_ndarray2d, safe_index


def test_zeros2d_shape_and_type():
    A = zeros2d(10, 7, dtype=np.int32)
    assert isinstance(A, np.ndarray)
    assert A.shape == (10, 7)
    assert A.dtype == np.int32


def test_ensure_ndarray2d_converts_list():
    L = [[0] * 5 for _ in range(3)]
    A = ensure_ndarray2d("L", L)
    assert isinstance(A, np.ndarray)
    assert A.shape == (3, 5)


def test_safe_index_ints():
    assert safe_index(3.9, 4.1) == (3, 4)


def test_tuple_index_does_not_exist_anymore():
    A = zeros2d(3, 3, dtype=np.int32)
    i, j = safe_index(1.0, 2.0)
    A[i, j] += 1
    assert A[1, 2] == 1
