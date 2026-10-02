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

ui/app.js repeats these steps (curveSamples, curveCopies, packageCopies) for
the preview on the map, so what the map shows is what the 3D model gets: keep
the two alike.
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


def _line(points):
    """The line through the points: its samples, their spacing, distance along it, length."""
    S = samples(points)
    seg = np.hypot(*np.diff(S, axis=0).T) if len(S) > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return S, seg, s, float(s[-1])


def _at(line, at):
    """The point at distance `at` along the line, and the line's direction there."""
    S, seg, s, L = line
    if not len(seg) or L <= 0:
        return (S[0] if len(S) else np.zeros(2)), 0.0
    k = min(max(int(np.searchsorted(s, at, side="right")) - 1, 0), len(seg) - 1)
    pos = S[k] + (S[k + 1] - S[k]) * ((at - s[k]) / seg[k] if seg[k] > 0 else 0.0)
    j = k                                               # the direction of the nearest span with length
    while j < len(seg) - 1 and seg[j] <= 1e-12:
        j += 1
    while j > 0 and seg[j] <= 1e-12:
        j -= 1
    t = S[j + 1] - S[j]
    return pos, math.atan2(t[1], t[0])


def copies(points, w, d, gap_x=0.0, gap_y=0.0, ny=1, flip=False, spaces=None):
    """
    Where the copies go along the line through the points.

    Points and sizes in one unit (metres here), image axes: x right, y down.
    w is a copy's size along the line, d across it. With random spaces (see
    spacer) each row has its own gaps, so its own number of copies. Returns
    the copies as (x, y, angle), the angle in radians being the direction of
    the copy's own X; the copies in each row; and the line's length.
    """
    line = _line(points)
    L = line[3]
    fronts, Ly = row_fronts(d, ny, row_gaps(spaces, ny - 1, gap_y))
    out, counts = [], []
    if not spaces:
        step = max(w + gap_x, 1e-9)
        n = max(1, int(math.floor((L + gap_x) / step + 1e-6)))
        first = (L - (n * w + (n - 1) * gap_x)) / 2 + w / 2
        for i in range(n):
            pos, ang = _at(line, min(max(first + i * step, 0.0), L))
            if flip:
                ang += math.pi
            v = np.array([-math.sin(ang), math.cos(ang)])  # across the line, towards the copy's back
            for r in range(ny):
                c = pos + v * (fronts[r] + d / 2)
                out.append((float(c[0]), float(c[1]), ang))
        return out, [n] * ny, L
    for r in range(ny):
        gap = spacer(spaces, r, gap_x)
        gaps, used = [0.0], max(w, 1e-6)
        while len(gaps) < 5000:
            g = gap()
            if used + g + max(w, 1e-6) > L + 1e-9:
                break                                   # the row is full: always at least one
            gaps.append(g)
            used += g + max(w, 1e-6)
        counts.append(len(gaps))
        at = (L - used) / 2
        for i, g in enumerate(gaps):
            at += g
            pos, ang = _at(line, min(max(at + w / 2, 0.0), L))
            at += max(w, 1e-6)
            if flip:
                ang += math.pi
            v = np.array([-math.sin(ang), math.cos(ang)])
            c = pos + v * (fronts[r] + d / 2)
            out.append((float(c[0]), float(c[1]), ang))
    return out, counts, L


def grid_copies(centre, angle, w, d, nx, ny, gap_x=0.0, gap_y=0.0, spaces=None):
    """
    A rectangle of copies: nx along its width in each of ny rows, centred on
    centre, the rows running along angle (radians, image axes). With random
    spaces each row has its own gaps, so its own length (each row centred).
    Returns the copies as (x, y, angle), each row's length, and the depth.
    """
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    c0 = np.asarray(centre, float)
    fronts, Ly = row_fronts(d, ny, row_gaps(spaces, ny - 1, gap_y))
    out, lengths = [], []
    for r in range(ny):
        gap = spacer(spaces, r, gap_x)
        gaps = [0.0] + [gap() for _ in range(nx - 1)]
        Lr = nx * w
        for g in gaps:
            Lr += g                                     # summed in the same order as ui/app.js
        lengths.append(Lr)
        at = -Lr / 2
        for g in gaps:
            at += g
            c = c0 + u * (at + w / 2) + v * (fronts[r] + d / 2)
            at += w
            out.append((float(c[0]), float(c[1]), angle))
    return out, lengths, Ly


# ----------------------------------------------------------------- packages
# A package mixes several objects in one placement. Each row is filled with
# random picks (weighted, never the same object twice in a row), each taking
# its own width with the same gap between every pair, the run centred on the
# line. Fronts are in line: every object's front side sits on the row's front
# edge, deeper objects reach further back. The random numbers come from a
# seeded generator (mulberry32) that ui/app.js repeats bit for bit, so the map
# and the 3D model get the same mix, and the mix never changes on its own.

_M32 = 0xFFFFFFFF


def _rng(seed):
    a = seed & _M32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & _M32
        t = ((a ^ (a >> 15)) * (a | 1)) & _M32
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & _M32)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0
    return nxt


def row_seed(seed, row):
    """Each row its own stream, so a longer row never changes the others."""
    return (seed ^ (((row + 1) * 0x9E3779B9) & _M32)) & _M32


def _pick(rnd, weights, prev):
    """A weighted random slot, not the same as the previous one when there is a choice."""
    cand = [k for k, w in enumerate(weights) if w > 0]
    if len(cand) > 1 and prev in cand:
        cand.remove(prev)
    total = 0.0
    for k in cand:
        total += weights[k]
    r = rnd() * total
    acc = 0.0
    for k in cand:
        acc += weights[k]
        if r < acc:
            return k
    return cand[-1]


# ----------------------------------------------------------------- random spaces
# Spaces can be random: each gap along a row a random distance between the X
# min and max, each gap between two rows one between the Y min and max (in
# metres). spaces = {"on": True, "x": [min, max], "y": [min, max], "seed": n};
# None (or "on" false) is even spacing, the placement's own gaps. The random
# numbers have their own seed, apart from a package's mix, so neither changes
# the other, and each row its own stream, so a longer row only adds to its end.

def spacer(spaces, row, even):
    """The gaps along one row, one call per gap: random, or the even gap."""
    if not spaces:
        return lambda: even
    lo, hi = sorted(max(0.0, float(x)) for x in spaces["x"])
    rnd = _rng(row_seed((int(spaces["seed"]) ^ 0x5BD1E995) & _M32, row))
    return lambda: lo + rnd() * (hi - lo)


def row_gaps(spaces, n, even):
    """The n gaps between rows: random, or all the even gap."""
    if not spaces:
        return [even] * n
    lo, hi = sorted(max(0.0, float(x)) for x in spaces["y"])
    rnd = _rng((int(spaces["seed"]) ^ 0xA5A5A5A5) & _M32)
    return [lo + rnd() * (hi - lo) for _ in range(n)]


def row_fronts(depth, ny, gaps):
    """Each row's front edge across the line, the rows centred on it, and their total depth."""
    Ly = ny * depth
    for g in gaps:
        Ly += g
    fronts, f = [], -Ly / 2
    for r in range(ny):
        if r:
            f += depth + gaps[r - 1]
        fronts.append(f)
    return fronts, Ly


def package_copies(points, slots, gap_x=0.0, gap_y=0.0, ny=1, flip=False, seed=1, spaces=None):
    """
    A package along the line through the points. slots: (w, d, weight) per
    object, w along the line and d across, in the points' unit. Returns the
    copies as (x, y, angle, slot), the copies in each row, and the line's length.
    """
    line = _line(points)
    L = line[3]
    weights = [max(0.0, float(wt)) for _, _, wt in slots]
    if not any(weights):
        weights = [1.0] * len(slots)
    D = max(d for _, d, _ in slots)                     # rows are as deep as the deepest object
    fronts, Ly = row_fronts(D, ny, row_gaps(spaces, ny - 1, gap_y))
    out, counts = [], []
    for r in range(ny):
        rnd, gap = _rng(row_seed(seed, r)), spacer(spaces, r, gap_x)
        picks, gaps, used, prev = [], [], 0.0, None
        while len(picks) < 5000:
            k = _pick(rnd, weights, prev)
            g = gap() if picks else 0.0
            need = max(slots[k][0], 1e-6) + g
            if picks and used + need > L + 1e-9:
                break                                   # the row is full: always at least one
            picks.append(k)
            gaps.append(g)
            used += need
            prev = k
        counts.append(len(picks))
        at = (L - used) / 2
        for k, g in zip(picks, gaps):
            w, d, _ = slots[k]
            at += g
            pos, ang = _at(line, min(max(at + w / 2, 0.0), L))
            at += max(w, 1e-6)
            if flip:
                ang += math.pi
            v = np.array([-math.sin(ang), math.cos(ang)])
            c = pos + v * (fronts[r] + d / 2)           # fronts in line on the row's front edge
            out.append((float(c[0]), float(c[1]), ang, k))
    return out, counts, L
