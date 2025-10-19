import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdfsvg_calibrator.orientation import phase_correlation


def test_phase_correlation_recovers_integer_shift() -> None:
    base = np.zeros((96, 128), dtype=np.float32)
    base[20:60, 30:90] = 1.0
    base[65:80, 95:110] = 0.5

    shift_x = 7
    shift_y = -5

    shifted = np.roll(base, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)

    du, dv, response = phase_correlation(shifted, base)

    assert abs(du + shift_x) <= 1.0
    assert abs(dv + shift_y) <= 1.0
    assert response > 0.1
