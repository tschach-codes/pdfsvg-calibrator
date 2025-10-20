from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def xcorr_1d(a: np.ndarray, b: np.ndarray):
    # normalisierte Kreuzkorrelation via FFT; gibt (shift, peak, corr_curve) zurück
    n = int(1 << int(np.ceil(np.log2(len(a) + len(b) - 1))))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    c = np.fft.irfft(A * np.conj(B), n)
    # zirkular -> auf lineare Verschiebung umlegen (max bei idx)
    idx = int(np.argmax(c))
    peak = float(c[idx] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    shift = idx if idx < n // 2 else idx - n
    return shift, peak, c


@dataclass
class Orientation:
    rot_deg: int  # 0,90,180,270
    flip: str  # 'none'|'x'|'y'


def adjust_profile_for_orientation(Hx, Hy, orientation: Orientation):
    # rotiere/flippe Profile logisch:
    # 90/270 => Achsentausch; flip => Reverse der jeweiligen Achse
    def rev(x):
        return x[::-1].copy()

    Hx2, Hy2 = Hx.copy(), Hy.copy()
    if orientation.rot_deg in (90, 270):
        Hx2, Hy2 = Hy2, Hx2
    if orientation.flip == 'x':
        Hx2 = rev(Hx2)
    elif orientation.flip == 'y':
        Hy2 = rev(Hy2)
    return Hx2, Hy2
