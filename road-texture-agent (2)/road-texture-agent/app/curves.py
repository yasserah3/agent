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

A curved placement can have several curve lines, each with as many points (a
point added goes on every line). The first shapes the front row, the last the
back row, the others are spread evenly between, and a row between two lines
follows a blend of the two, point by point: on 5 rows and 2 lines the second
row is 3/4 of the first line and 1/4 of the second. Each row keeps its place
across the lines (its gaps and depth), so the lines bend the rows, and every
row has its own length along its own line.

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


def row_lines(points, ny):
    """
    Each row's line (see _line) and its points. points: one line, or a list of
    curve lines with as many points each (see above). Rows on the same line
    share it; with one line the points come back as None.
    """
    P = np.asarray(points, float)
    if P.ndim == 2 or len(P) == 1:
        line = _line(P if P.ndim == 2 else P[0])
        return [line] * ny, None
    m, lines, pts, made = len(P), [], [], {}
    for r in range(ny):
        x = r * (m - 1) / (ny - 1) if ny > 1 else 0.0  # where the row is among the lines
        a = min(int(math.floor(x)), m - 2)
        f = x - a
        if (a, f) not in made:
            q = P[a] if f == 0 else (1 - f) * P[a] + f * P[a + 1]
            made[(a, f)] = (q, _line(q))
        pts.append(made[(a, f)][0])
        lines.append(made[(a, f)][1])
    return lines, pts


def carry(la, lb, s):
    """
    A distance along line la to the same place along line lb. Lines with as
    many points have as many samples, and the same sample on each is the same
    place, point by point.
    """
    if la is lb:
        return s
    S, seg, sa, L = la
    if not len(seg):
        return s
    k = min(max(int(np.searchsorted(sa, s, side="right")) - 1, 0), len(seg) - 1)
    t = (s - sa[k]) / seg[k] if seg[k] > 0 else 0.0
    return float(lb[2][k] + t * lb[1][k])


def turned(w, d, deg):
    """
    A footprint's size along its row and across it once turned by deg degrees:
    a turned object takes its whole turned outline, so the spaces around it
    stay as set (neighbours and the rows behind move instead).
    """
    if not deg:
        return w, d
    t = math.radians(deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    return w * c + d * s, w * s + d * c


def _turn(turns, r, i):
    """A copy's own turn, by its id "row-column" (both from 1)."""
    return float((turns or {}).get("%d-%d" % (r + 1, i + 1), 0.0))


# ----------------------------------------------------------------- objects alignment
# align = {"rows": .., "last": .., "columns": .., "exclude": ..}, each true or false:
#   rows     every second row turned round: row 1 as placed, row 2 facing the
#            other way, row 3 as row 1, and so on
#   last     the last row turned round (back again, if rows had turned it)
#   columns  the objects at either end of a row face out of that end, in every
#            row, turned round or not: the first column 270 degrees from row 1's
#            direction (clockwise), the last column 90. With the line flipped
#            (or a rectangle mirrored) the two swap, so the ends still face out.
#            One object alone in its row is not an end
#   exclude  the first and last rows keep their ends facing as their row
# A row turned round lines its fronts up on its back edge, the side it faces.
# An object's own turn (turns) adds to these, and every turned object takes
# its turned outline, so the spaces stay as set.

def _aligned(align):
    return bool(align) and any(align.get(k) for k in ("rows", "last", "columns"))


def _round(align, r, ny):
    """Whether row r is turned round."""
    return bool(align.get("rows") and r % 2) != bool(align.get("last") and r == ny - 1)


def _ends(align, r, ny):
    """Whether row r's end objects face out of its ends."""
    return bool(align.get("columns")) and not (align.get("exclude") and (r == 0 or r == ny - 1))


def _auto(align, r, i, n, ny, flip):
    """
    An object's turn from the alignment, in degrees clockwise. n: the objects
    in its row, or None while the row is being filled (it is not the last yet).
    """
    if not align:
        return 0.0
    if _ends(align, r, ny) and (n is None or n > 1):
        if i == 0:
            return 90.0 if flip else 270.0
        if n is not None and i == n - 1:
            return 270.0 if flip else 90.0
    return 180.0 if _round(align, r, ny) else 0.0


def copies(points, w, d, gap_x=0.0, gap_y=0.0, ny=1, flip=False, spaces=None, layout=None, turns=None, align=None):
    """
    Where the copies go along the line through the points.

    Points and sizes in one unit (metres here), image axes: x right, y down.
    points: one line, or several curve lines (see row_lines). w is a copy's
    size along the line, d across it. With random spaces (see spacer) or
    turned copies (turns: {id: degrees}) each row has its own gaps and sizes,
    so its own number of copies, as has each row on a line of its own length.
    The alignment (align, see above) turns rows round and the ends of rows.
    Returns the copies as (x, y, angle), the angle in radians being the
    direction of the copy's own X with the alignment's turn (without its own
    turn); the copies in each row; and the (first) line's length. A layout
    dict, if given, is filled in (see new_layout).
    """
    lines, rpts = row_lines(points, ny)
    if not spaces and not turns and not _aligned(align):
        fronts, Ly = row_fronts(d, ny, row_gaps(spaces, ny - 1, gap_y))
        lay = new_layout(layout, points, flip, fronts, [d] * ny, rpts)
        step = max(w + gap_x, 1e-9)
        out, counts = [], []
        for r in range(ny):
            line = lines[r]
            L = line[3]
            n = max(1, int(math.floor((L + gap_x) / step + 1e-6)))
            first = (L - (n * w + (n - 1) * gap_x)) / 2 + w / 2
            lay["rows"][r]["items"] = [[first + i * step - w / 2, first + i * step + w / 2] for i in range(n)]
            counts.append(n)
            for i in range(n):
                pos, ang = _at(line, min(max(first + i * step, 0.0), L))
                if flip:
                    ang += math.pi
                v = np.array([-math.sin(ang), math.cos(ang)])  # across the line, towards the copy's back
                c = pos + v * (fronts[r] + d / 2)
                out.append((float(c[0]), float(c[1]), ang))
                lay["ids"].append("%d-%d" % (r + 1, i + 1))
        return out, counts, lines[0][3]
    # each row first: its copies' (turned) sizes, as many as fit with the gaps
    rows = []
    for r in range(ny):
        L = lines[r][3]
        gap = spacer(spaces, r, gap_x)
        tt = lambda i, n=None: _auto(align, r, i, n, ny, flip) + _turn(turns, r, i)
        sizes = [turned(w, d, tt(0))]
        gaps, used = [0.0], max(sizes[0][0], 1e-6)
        while len(gaps) < 5000:
            nxt = turned(w, d, tt(len(gaps)))
            g = gap()
            if used + g + max(nxt[0], 1e-6) > L + 1e-9:
                break                                   # the row is full: always at least one
            gaps.append(g)
            sizes.append(nxt)
            used += g + max(nxt[0], 1e-6)
        if align and _ends(align, r, ny):
            # the last one faces out of its end too: turned, and as many as still fit
            while True:
                n = len(sizes)
                sizes[-1] = turned(w, d, tt(n - 1, n))
                used = max(sizes[0][0], 1e-6)
                for g, z in zip(gaps[1:], sizes[1:]):
                    used += g + max(z[0], 1e-6)
                if n == 1 or used <= L + 1e-9:
                    break
                gaps.pop()
                sizes.pop()
        rows.append((gaps, sizes, used))
    # then the rows, each as deep as its deepest (turned) copy
    depths = [max([d] + [z[1] for z in sz]) for _, sz, _ in rows]
    fronts, Ly = row_fronts(depths, ny, row_gaps(spaces, ny - 1, gap_y), base=d)
    lay = new_layout(layout, points, flip, fronts, depths, rpts)
    out, counts = [], []
    for r, (gaps, sizes, used) in enumerate(rows):
        line = lines[r]
        L = line[3]
        n = len(gaps)
        counts.append(n)
        back = bool(align) and _round(align, r, ny)
        at = (L - used) / 2
        for i, (g, (ww, dd)) in enumerate(zip(gaps, sizes)):
            at += g
            pos, ang = _at(line, min(max(at + ww / 2, 0.0), L))
            lay["rows"][r]["items"].append([at, at + max(ww, 1e-6)])
            at += max(ww, 1e-6)
            if flip:
                ang += math.pi
            v = np.array([-math.sin(ang), math.cos(ang)])
            # fronts in line on the row's front edge, or its back edge when turned round
            c = pos + v * (fronts[r] + depths[r] - dd / 2 if back else fronts[r] + dd / 2)
            out.append((float(c[0]), float(c[1]), ang + math.radians(_auto(align, r, i, n, ny, flip))))
            lay["ids"].append("%d-%d" % (r + 1, i + 1))
    return out, counts, lines[0][3]


def grid_copies(centre, angle, w, d, nx, ny, gap_x=0.0, gap_y=0.0, spaces=None, layout=None, turns=None,
                mirror=False, align=None):
    """
    A rectangle of copies: nx along its width in each of ny rows, centred on
    centre, the rows running along angle (radians, image axes). With random
    spaces or turned copies each row has its own gaps and sizes, so its own
    length (each row centred), and each row is as deep as its deepest copy.
    Mirrored, the rows are laid out from the other end: the mirror image, the
    copies still facing the same way. The alignment (align, see above) turns
    rows round and the ends of rows. Returns the copies as (x, y, angle),
    each row's length, and the depth.
    """
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    du = -u if mirror else u                            # the way the rows are laid out
    c0 = np.asarray(centre, float)
    rows = []
    for r in range(ny):
        gap = spacer(spaces, r, gap_x)
        gaps = [0.0] + [gap() for _ in range(nx - 1)]
        sizes = [turned(w, d, _auto(align, r, i, nx, ny, mirror) + _turn(turns, r, i)) for i in range(nx)]
        Lr = 0.0
        for g, (ww, _) in zip(gaps, sizes):
            Lr += g                                     # summed in the same order as ui/app.js
            Lr += ww
        rows.append((gaps, sizes, Lr))
    depths = [max([d] + [z[1] for z in sz]) for _, sz, _ in rows]
    fronts, Ly = row_fronts(depths, ny, row_gaps(spaces, ny - 1, gap_y), base=d)
    out, lengths, starts = [], [], []
    for r, (gaps, sizes, Lr) in enumerate(rows):
        lengths.append(Lr)
        at = -Lr / 2
        row = []
        back = bool(align) and _round(align, r, ny)
        for i, (g, (ww, dd)) in enumerate(zip(gaps, sizes)):
            at += g
            c = c0 + du * (at + ww / 2) + v * (fronts[r] + depths[r] - dd / 2 if back else fronts[r] + dd / 2)
            row.append([at, at + ww])
            at += ww
            out.append((float(c[0]), float(c[1]), angle + math.radians(_auto(align, r, i, nx, ny, mirror))))
        starts.append(row)
    if layout is not None:
        # the rectangle's middle line, as long as its longest row: distances along
        # it start at its left end (mirrored, its right end, the rows on the same side)
        half = max(lengths) / 2
        lay = new_layout(layout, [c0 - du * half, c0 + du * half], mirror, fronts, depths)
        for r in range(ny):
            lay["rows"][r]["items"] = [[a + half, b + half] for a, b in starts[r]]
            lay["ids"] += ["%d-%d" % (r + 1, i + 1) for i in range(nx)]
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


def row_fronts(depth, ny, gaps, base=None):
    """
    Each row's front edge across the line, the rows centred on it, and their
    total depth. depth: one for every row, or a list, one per row. With base
    (the rows' depth before any copy was turned), the rows are placed as for
    base and keep their front: rows made deeper by a turned copy push only the
    rows behind them back.
    """
    depths = list(depth) if isinstance(depth, (list, tuple)) else [depth] * ny
    Ly = 0.0
    for dd in depths:
        Ly += dd
    for g in gaps:
        Ly += g
    Ly0 = Ly
    if base is not None:
        Ly0 = 0.0
        for _ in range(ny):
            Ly0 += base
        for g in gaps:
            Ly0 += g
    fronts, f = [], -Ly0 / 2
    for r in range(ny):
        if r:
            f += depths[r - 1] + gaps[r - 1]
        fronts.append(f)
    return fronts, Ly


def package_copies(points, slots, gap_x=0.0, gap_y=0.0, ny=1, flip=False, seed=1, spaces=None, layout=None,
                   turns=None, align=None):
    """
    A package along the line through the points (one line, or several curve
    lines: see row_lines). slots: (w, d, weight) per object, w along the line
    and d across, in the points' unit. Turned copies (turns: {id: degrees})
    take their turned size, and the alignment (align, see above) turns rows
    round and the ends of rows. Returns the copies as (x, y, angle, slot), the
    copies in each row, and the (first) line's length.
    """
    lines, rpts = row_lines(points, ny)
    weights = [max(0.0, float(wt)) for _, _, wt in slots]
    if not any(weights):
        weights = [1.0] * len(slots)
    D = max(d for _, d, _ in slots)                     # rows are as deep as the deepest object
    rows = []
    for r in range(ny):
        L = lines[r][3]
        rnd, gap = _rng(row_seed(seed, r)), spacer(spaces, r, gap_x)
        tt = lambda i, n=None: _auto(align, r, i, n, ny, flip) + _turn(turns, r, i)
        picks, gaps, sizes, used, prev = [], [], [], 0.0, None
        while len(picks) < 5000:
            k = _pick(rnd, weights, prev)
            ww, dd = turned(slots[k][0], slots[k][1], tt(len(picks)))
            g = gap() if picks else 0.0
            need = max(ww, 1e-6) + g
            if picks and used + need > L + 1e-9:
                break                                   # the row is full: always at least one
            picks.append(k)
            gaps.append(g)
            sizes.append((ww, dd))
            used += need
            prev = k
        if align and _ends(align, r, ny):
            # the last one faces out of its end too: turned, and as many as still fit
            while True:
                n = len(picks)
                sizes[-1] = turned(slots[picks[-1]][0], slots[picks[-1]][1], tt(n - 1, n))
                used = 0.0
                for g, z in zip(gaps, sizes):
                    used += max(z[0], 1e-6) + g
                if n == 1 or used <= L + 1e-9:
                    break
                picks.pop()
                gaps.pop()
                sizes.pop()
        rows.append((picks, gaps, sizes, used))
    # each row as deep as the package's deepest object, or a deeper turned one in it
    depths = [max([D] + [z[1] for z in sz]) for _, _, sz, _ in rows]
    fronts, Ly = row_fronts(depths, ny, row_gaps(spaces, ny - 1, gap_y), base=D)
    lay = new_layout(layout, points, flip, fronts, depths, rpts)
    out, counts = [], []
    for r, (picks, gaps, sizes, used) in enumerate(rows):
        line = lines[r]
        L = line[3]
        n = len(picks)
        counts.append(n)
        back = bool(align) and _round(align, r, ny)
        at = (L - used) / 2
        for i, (k, g, (ww, dd)) in enumerate(zip(picks, gaps, sizes)):
            at += g
            pos, ang = _at(line, min(max(at + ww / 2, 0.0), L))
            lay["rows"][r]["items"].append([at, at + max(ww, 1e-6)])
            lay["ids"].append("%d-%d" % (r + 1, i + 1))
            at += max(ww, 1e-6)
            if flip:
                ang += math.pi
            v = np.array([-math.sin(ang), math.cos(ang)])
            # fronts in line on the row's front edge, or its back edge when turned round
            c = pos + v * (fronts[r] + depths[r] - dd / 2 if back else fronts[r] + dd / 2)
            out.append((float(c[0]), float(c[1]), ang + math.radians(_auto(align, r, i, n, ny, flip)), k))
    return out, counts, lines[0][3]


# ----------------------------------------------------------------- layout and cells
# Every placement is rows of objects along a line (a rectangle's is its middle
# line). Its layout says where: the line (points), whether flipped, and per row
# its front edge and depth across the line and each object's stretch along it
# (distances from the line's start). With several curve lines each row has its
# own line too (its points), and its distances are along it. Objects have ids
# "row-column", counted from the top left: row 1 is the front row, column 1
# the first along the line.
#
# The spaces between objects split into cells, for drawing inner streets:
#   x  the space between two neighbours in a row      ("x1-2": row 1, after column 2)
#   y  the space between two rows, beside the objects  ("y1-1": rows 1 and 2, first)
#   j  a junction: where an x space of either row meets the space between rows
# A cell's distances along are along its row's line (for y and j, the front one's).

def new_layout(layout, points, flip, fronts, depths, row_points=None):
    """Start a layout in the given dict (or a throwaway one); depths: one per row."""
    lay = layout if layout is not None else {}
    P = np.asarray(points, float)
    first = P if P.ndim == 2 else P[0]
    lay.update({"line": [[float(p[0]), float(p[1])] for p in first], "flip": bool(flip),
                "rows": [{"front": float(f), "depth": float(dd), "items": []} for f, dd in zip(fronts, depths)],
                "ids": []})
    if row_points is not None:
        for row, q in zip(lay["rows"], row_points):
            row["line"] = [[float(p[0]), float(p[1])] for p in q]
    return lay


def row_line(layout, r, cache=None):
    """Row r's line (see _line): its own with several curve lines, else the layout's. cache: a dict, reused."""
    cache = {} if cache is None else cache
    key = r if "line" in layout["rows"][r] else -1
    if key not in cache:
        cache[key] = _line(layout["rows"][r]["line"] if key >= 0 else layout["line"])
    return cache[key]


def cells(layout, min_m=0.05, cache=None):
    """The cells of a layout's spaces: id, kind, row, along [s0, s1] and across [o0, o1]."""
    rows, out = layout["rows"], []
    for r, row in enumerate(rows):
        it = row["items"]
        for i in range(len(it) - 1):
            if it[i + 1][0] - it[i][1] > min_m:
                out.append({"id": "x%d-%d" % (r + 1, i + 1), "kind": "x", "row": r, "s": [it[i][1], it[i + 1][0]],
                            "o": [row["front"], row["front"] + row["depth"]]})
    cache = {} if cache is None else cache
    for r in range(len(rows) - 1):
        a, b = rows[r], rows[r + 1]
        o0, o1 = a["front"] + a["depth"], b["front"]
        if o1 - o0 <= min_m or not a["items"] or not b["items"]:
            continue
        bi = b["items"]
        if "line" in a or "line" in b:
            # the row behind on a line of its own: its objects where they are along the front one's
            la, lb = row_line(layout, r, cache), row_line(layout, r + 1, cache)
            bi = [[carry(lb, la, s0), carry(lb, la, s1)] for s0, s1 in bi]
        lo = min(a["items"][0][0], bi[0][0])
        hi = max(a["items"][-1][1], bi[-1][1])
        # where either row's x spaces meet the space between the rows: junctions
        gaps = sorted([it[i][1], it[i + 1][0]] for it in (a["items"], bi)
                      for i in range(len(it) - 1) if it[i + 1][0] - it[i][1] > min_m)
        merged = []
        for g in gaps:
            if merged and g[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], g[1])
            else:
                merged.append(list(g))
        cur, ky, kj = lo, 0, 0
        for g0, g1 in merged:
            g0, g1 = max(g0, lo), min(g1, hi)
            if g0 - cur > min_m:
                ky += 1
                out.append({"id": "y%d-%d" % (r + 1, ky), "kind": "y", "row": r, "s": [cur, g0], "o": [o0, o1]})
            kj += 1
            out.append({"id": "j%d-%d" % (r + 1, kj), "kind": "j", "row": r, "s": [g0, g1], "o": [o0, o1]})
            cur = g1
        if hi - cur > min_m:
            ky += 1
            out.append({"id": "y%d-%d" % (r + 1, ky), "kind": "y", "row": r, "s": [cur, hi], "o": [o0, o1]})
    return out


def strip_polygon(layout, s0, s1, o0, o1, step=1.0, line=None, far=None):
    """
    The area between distances s0..s1 along the layout's line (or the given
    one) and o0..o1 across it. With far, another row's line, the far edge
    runs along that one, at the same places.
    """
    line = line or _line(layout["line"])
    L = line[3]
    n = max(1, int(math.ceil((s1 - s0) / step - 1e-9)))
    sign = -1.0 if layout["flip"] else 1.0
    near, back = [], []
    for k in range(n + 1):
        s = min(max(s0 + (s1 - s0) * k / n, 0.0), L)
        pos, ang = _at(line, s)
        v = np.array([-math.sin(ang), math.cos(ang)]) * sign
        near.append((float(pos[0] + v[0] * o0), float(pos[1] + v[1] * o0)))
        if far is not None and far is not line:
            pos, ang = _at(far, min(max(carry(line, far, s), 0.0), far[3]))
            v = np.array([-math.sin(ang), math.cos(ang)]) * sign
        back.append((float(pos[0] + v[0] * o1), float(pos[1] + v[1] * o1)))
    return near + back[::-1]


def cell_polygon(layout, cell, step=1.0, cache=None):
    """A cell's area: along its row's line, a y or j cell reaching across to the next row's."""
    cache = {} if cache is None else cache
    r = cell.get("row", 0)
    line = row_line(layout, r, cache)
    far = row_line(layout, r + 1, cache) if cell["kind"] != "x" and r + 1 < len(layout["rows"]) else line
    return strip_polygon(layout, cell["s"][0], cell["s"][1], cell["o"][0], cell["o"][1], step, line, far)
