import numpy as np
from scipy.spatial import cKDTree

def gate_matches(pdf_segs, svg_segs, angle_tol_deg, len_tol_rel, dist_tol_px, max_pairs=3000):
    def _feat(segs):
        v = segs[:, 2:4] - segs[:, 0:2]
        center = 0.5 * (segs[:, 0:2] + segs[:, 2:4])
        length = np.linalg.norm(v, axis=1) + 1e-9
        angle = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi)
        return center, length, angle

    Cp, Lp, Ap = _feat(pdf_segs)
    Cs, Ls, As = _feat(svg_segs)

    # Subsample auf längste
    idx_pdf = np.argsort(Lp)[-max_pairs:]
    idx_svg = np.argsort(Ls)[-max_pairs:]

    Cp, Lp, Ap = Cp[idx_pdf], Lp[idx_pdf], Ap[idx_pdf]
    Cs, Ls, As = Cs[idx_svg], Ls[idx_svg], As[idx_svg]

    tree = cKDTree(Cp)
    # Vorselektion via Radius
    cand = tree.query_ball_point(Cs, r=dist_tol_px)
    pairs = []
    ang_tol = np.deg2rad(angle_tol_deg)
    for j, neigh in enumerate(cand):
        if not neigh:
            continue
        for i in neigh:
            if min(abs(Ap[i] - As[j]), np.pi - abs(Ap[i] - As[j])) > ang_tol:
                continue
            r = Ls[j] / Lp[i]
            if not (1.0 - len_tol_rel <= r <= 1.0 + len_tol_rel):
                continue
            pairs.append((i, j))
    if not pairs:
        return 0.0, []
    # Inlier-Score: länge-gewichteter Anteil
    w = np.array([min(Lp[i], Ls[j]) for i, j in pairs])
    score = float(w.sum() / (Ls.sum() + 1e-9))
    return score, pairs
