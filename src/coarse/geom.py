import numpy as np

from pdfsvg_calibrator.core.grid_safety import ensure_ndarray2d

from .corr import Orientation


def apply_orientation_pts(P: np.ndarray, orientation: Orientation, bbox):
    # P: (N,2) Punkte, bbox: (xmin,ymin,xmax,ymax)
    P = ensure_ndarray2d("points", P).astype(float, copy=False)
    if P.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {P.shape}")
    xmin, ymin, xmax, ymax = bbox
    W = xmax - xmin
    H = ymax - ymin
    Q = P.copy()
    # auf [0,W]x[0,H] normalisieren
    Q[:, 0] -= xmin
    Q[:, 1] -= ymin
    if orientation.rot_deg == 90:
        Q = np.stack([Q[:, 1], H - Q[:, 0]], axis=1)
        W, H = H, W
    elif orientation.rot_deg == 180:
        Q = np.stack([W - Q[:, 0], H - Q[:, 1]], axis=1)
    elif orientation.rot_deg == 270:
        Q = np.stack([W - Q[:, 1], Q[:, 0]], axis=1)
        W, H = H, W
    if orientation.flip == 'x':
        Q[:, 0] = W - Q[:, 0]
    elif orientation.flip == 'y':
        Q[:, 1] = H - Q[:, 1]
    Q[:, 0] += xmin
    Q[:, 1] += ymin
    return Q


def apply_scale_shift_pts(P, sx, sy, tx, ty):
    P = ensure_ndarray2d("points", P).astype(float, copy=False)
    if P.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {P.shape}")
    Q = P.copy()
    Q[:, 0] = sx * Q[:, 0] + tx
    Q[:, 1] = sy * Q[:, 1] + ty
    return Q


def transform_segments(segments, fn_pts):
    segs = ensure_ndarray2d("segments", segments).astype(float, copy=False)
    if segs.shape[1] != 4:
        raise ValueError(f"segments must have shape (N,4), got {segs.shape}")
    A = ensure_ndarray2d("fn_pts(segs[:,0:2])", fn_pts(segs[:, 0:2])).astype(float, copy=False)
    B = ensure_ndarray2d("fn_pts(segs[:,2:4])", fn_pts(segs[:, 2:4])).astype(float, copy=False)
    if A.shape[1] != 2 or B.shape[1] != 2:
        raise ValueError("transformed segments must have 2 columns")
    return np.concatenate([A, B], axis=1)
