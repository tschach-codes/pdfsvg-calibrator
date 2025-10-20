import numpy as np
from .corr import Orientation


def apply_orientation_pts(P: np.ndarray, orientation: Orientation, bbox):
    # P: (N,2) Punkte, bbox: (xmin,ymin,xmax,ymax)
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
    Q = P.copy()
    Q[:, 0] = sx * Q[:, 0] + tx
    Q[:, 1] = sy * Q[:, 1] + ty
    return Q


def transform_segments(segments, fn_pts):
    segs = np.asarray(segments, float)
    A = fn_pts(segs[:, 0:2])
    B = fn_pts(segs[:, 2:4])
    return np.concatenate([A, B], axis=1)
