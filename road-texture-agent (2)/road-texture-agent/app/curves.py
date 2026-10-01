"""
Curved placements: copies of an object along a smooth line.

The line passes through every point placed on the map. Between points it is a
centripetal Catmull-Rom curve: it bends smoothly through each point, and,
unlike the plain kind, never loops or overshoots where points are unevenly
spaced. Two points give a straight line.

Copies fill the line at the chosen gap, the run centred on its length, so a
longer line holds more copies. Each copy is turned with the curve: its own X
(its width) runs along the line and its front (its +Y) faces the left of the
line's direction, or the right when flipped. Rows are parallel lines beside it.

ui/app.js repeats these steps (curveSamples, curveCopies) for the preview on
the map, so what the map shows is what the 3D model gets: keep the two alike.
"""

import math

import numpy as np

STEPS = 24      # samples per span between two points


def samples(points):
    """The smooth line through the points, as a dense polyline (N, 2)."""
    P = [np.asarray(p, float) for p in points]
    if len(P) < 2:
        return np.array(P, float).reshape(-1, 2)
    # a mirrored point beyond each end, so the line leaves its ends straight
    ext = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        # centripetal: knots spaced by the square root of the distance
        t0 = 0.0
        t1 = t0 + max(math.sqrt(math.hypot(*(p1 - p0))), 1e-6)
        t2 = t1 + max(math.sqrt(math.hypot(*(p2 - p1))), 1e-6)
        t3 = t2 + max(math.sqrt(math.hypot(*(p3 - p2))), 1e-6)
        for k in range(STEPS):
            t = t1 + (t2 - t1) * k / STEPS
            a1 = ((t1 - t) * p0 + (t - t0) * p1) / (t1 - t0)
            a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
            a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
            b1 = ((t2 - t) * a1 + (t - t0) * a2) / (t2 - t0)
            b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
            out.append(((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1))
    out.append(P[-1])
    return np.array(out, float)


def copies(points, w, d, gap_x=0.0, gap_y=0.0, ny=1, flip=False):
    """
    Where the copies go along the line through the points.

    Points and sizes in one unit (metres here), image axes: x right, y down.
    w is a copy's size along the line, d across it. Returns the copies as
    (x, y, angle), the angle in radians being the direction of the copy's own
    X; the number of copies along the line; and the line's length.
    """
    S = samples(points)
    seg = np.hypot(*np.diff(S, axis=0).T) if len(S) > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    step = max(w + gap_x, 1e-9)
    n = max(1, int(math.floor((L + gap_x) / step + 1e-6)))
    first = (L - (n * w + (n - 1) * gap_x)) / 2 + w / 2
    Ly = ny * d + (ny - 1) * gap_y
    out = []
    for i in range(n):
        at = min(max(first + i * step, 0.0), L)
        if not len(seg) or L <= 0:
            pos, ang = (S[0] if len(S) else np.zeros(2)), 0.0
        else:
            k = min(max(int(np.searchsorted(s, at, side="right")) - 1, 0), len(seg) - 1)
            pos = S[k] + (S[k + 1] - S[k]) * ((at - s[k]) / seg[k] if seg[k] > 0 else 0.0)
            j = k                                       # the direction of the nearest span with length
            while j < len(seg) - 1 and seg[j] <= 1e-12:
                j += 1
            while j > 0 and seg[j] <= 1e-12:
                j -= 1
            t = S[j + 1] - S[j]
            ang = math.atan2(t[1], t[0])
        if flip:
            ang += math.pi
        v = np.array([-math.sin(ang), math.cos(ang)])  # across the line, towards the copy's back
        for r in range(ny):
            c = pos + v * (-Ly / 2 + d / 2 + r * (d + gap_y))
            out.append((float(c[0]), float(c[1]), ang))
    return out, n, L
