"""
Highways' centrelines, whole (Road markings, Highways).

A highway's street segments (app/junctions.py) are cut out around every
junction it passes, and on a wide road with side streets every hundred metres
the cut circles (twice the road's half width each way) overlap: most of a
highway has no segment left to lay lines along. So a highway's centreline is
found here on its own, from the mask's whole skeleton:

1. the skeleton where the road is at least the highway's width (with a little
   give, so a road measuring a few centimetres under it in places stays one);
2. as a graph: the short stubs a side street leaves on it taken off, two
   branches meeting at a point joined into one;
3. a highway keeps its identity through bends and ordinary width variation;
   when genuine highways meet, every approach stops short of that junction;
   narrower side-road stubs do not stop the main road's continuous lines;
4. each line put midway between its kerbs, measured across it every pixel; where
   a side street opens one kerb (or a driveway, a bus bay), the middle and the
   width carry on straight from either side.

Each chain: {"pts": [[x, y], ...] (mask pixels, along it), "w": [width in
metres at each point], "length_m"}. The lines of a highway are laid along its chain
(app/generation.py, app/quadmesh.py), and its own street segments lay none.
"""
import math

import numpy as np
from scipy import ndimage as ndi

from app import lanes as LN

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MIN_M = 30.0                    # a highway chain at least this long
IN_LINE = 35.0                  # two branches run on through a meeting point within this many degrees
STEP_PX = 2.0                   # a chain's points this far apart, as kept


def _graph(mask):
    """A skeleton's pixels as a graph: (coords (N, 2) y x, neighbours [list of indices], degree)."""
    coords = np.argwhere(mask)
    idx = np.full(mask.shape, -1, np.int64)
    idx[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
    H, W = mask.shape
    nbr = [[] for _ in range(len(coords))]
    for dy, dx in N8:
        ys, xs = coords[:, 0] + dy, coords[:, 1] + dx
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        if dy and dx:
            # A diagonal beside an orthogonal step is a shortcut, not another arm.
            # Keeping both creates tiny triangles and false junctions on bends.
            diagonal = np.nonzero(ok)[0]
            shortcut = mask[coords[diagonal, 0], xs[diagonal]] | mask[ys[diagonal], coords[diagonal, 1]]
            ok[diagonal[shortcut]] = False
        j = np.full(len(coords), -1, np.int64)
        j[ok] = idx[ys[ok], xs[ok]]
        for i in np.nonzero(j >= 0)[0]:
            nbr[i].append(int(j[i]))
    deg = np.array([len(n) for n in nbr])
    return coords, nbr, deg


def _branches(coords, nbr, deg):
    """
    The graph cut at its meeting points (pixels of 3 or more neighbours, taken
    together where they touch): (nodes: [set of pixel indices], branches: [{"px":
    [indices in order], "a", "b": the node at each end or None (a loose end)}]).
    """
    n = len(coords)
    is_node = deg >= 3
    node_of = np.full(n, -1, np.int64)
    nodes = []
    for i in np.nonzero(is_node)[0]:
        if node_of[i] >= 0:
            continue
        k = len(nodes)
        stack, members = [i], set()
        node_of[i] = k
        while stack:
            p = stack.pop()
            members.add(p)
            for q in nbr[p]:
                if is_node[q] and node_of[q] < 0:
                    node_of[q] = k
                    stack.append(q)
        nodes.append(members)
    seen = np.zeros(n, bool)
    branches = []

    def walk(start, prev):
        path = [start]
        seen[start] = True
        cur, last = start, prev
        while True:
            nxt = [q for q in nbr[cur] if q != last and not is_node[q] and not seen[q]]
            ends = [q for q in nbr[cur] if q != last and is_node[q]]
            if ends and (not nxt or len(path) > 1):
                return path, int(node_of[ends[0]])
            if not nxt:
                return path, None
            last, cur = cur, nxt[0]
            seen[cur] = True
            path.append(cur)

    # from each node out along each of its branches
    for k, members in enumerate(nodes):
        for p in members:
            for q in nbr[p]:
                if is_node[q] or seen[q]:
                    continue
                path, b = walk(q, p)
                branches.append({"px": path, "a": k, "b": b})
    # pieces with no meeting point: from a loose end, then what is left (loops) from anywhere
    for pass_ in (0, 1):
        for i in range(n):
            if seen[i] or is_node[i] or (pass_ == 0 and deg[i] != 1):
                continue
            path, b = walk(i, -1)
            branches.append({"px": path, "a": None, "b": b})
    return nodes, branches


def _direction(coords, path, from_end, reach):
    """A branch's way out of a node: from its first pixel (or its last) to the one reach pixels along it."""
    P = coords[path][:, ::-1].astype(float)                 # x y
    if from_end:
        P = P[::-1]
    k = min(len(P) - 1, max(1, int(reach)))
    v = P[k] - P[0]
    n = float(np.hypot(*v))
    return v / n if n > 0 else np.array([1.0, 0.0])


def _smooth(P, window):
    """A line (N, 2) smoothed by a moving average (ends kept)."""
    w = int(window) | 1
    if len(P) < 3 or w < 3:
        return P.copy()
    w = min(w, (len(P) // 2) * 2 - 1)
    if w < 3:
        return P.copy()
    k = np.ones(w) / w
    pad = w // 2
    Q = np.pad(P, ((pad, pad), (0, 0)), mode="edge")
    out = np.column_stack([np.convolve(Q[:, c], k, mode="valid") for c in range(P.shape[1])])
    out[0], out[-1] = P[0], P[-1]
    return out


def _resample(P, step):
    seg = np.hypot(*np.diff(P, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] <= 0:
        return P[:1]
    t = np.linspace(0.0, arc[-1], max(2, int(math.ceil(arc[-1] / step)) + 1))
    return np.column_stack([np.interp(t, arc, P[:, 0]), np.interp(t, arc, P[:, 1])])


def _rolling(v, window, fn):
    w = max(1, int(window))
    pad = w // 2
    Q = np.pad(v, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(Q, 2 * pad + 1)
    return fn(view, axis=1)


def _refine(P, road, half_px, px):
    """
    A chain put midway between its kerbs: across it at every pixel, how far
    the road reaches each way; where the road is as wide as it usually is there
    (not where a side street, a bay or a corner's rounding opens it), the kerbs'
    middle and the width; elsewhere carried on from either side. Twice,
    the second time square to the first's line (a skeleton bends towards every
    opening, so rays square to it are a little askew there). Returns (points,
    widths in metres).
    """
    H, W = road.shape
    P = _resample(P, 1.0)
    if len(P) < 5:
        return P, np.full(len(P), 2 * half_px * px)
    win = max(5, int(2.0 * half_px))
    # measured from a smooth line, not the skeleton itself (it steps towards every side street's
    # branch point, and a kerb measured from there looks moved)
    P = _smooth(P, 4 * win + 1)
    arc = np.arange(len(P), dtype=float)
    t = np.arange(0.0, 1.8 * half_px + 3.0, 0.5)
    wid = np.full(len(P), 2 * half_px)
    for _ in range(2):
        T = np.gradient(_smooth(P, 2 * win + 1), axis=0)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        N = np.column_stack([-T[:, 1], T[:, 0]])

        def extent(sign):
            X = P[:, 0:1] + sign * N[:, 0:1] * t[None, :]
            Y = P[:, 1:2] + sign * N[:, 1:2] * t[None, :]
            xi, yi = np.round(X).astype(int), np.round(Y).astype(int)
            ins = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            on = np.zeros(X.shape, bool)
            on[ins] = road[yi[ins], xi[ins]]
            off = ~on
            hit = off.any(axis=1)
            return t[np.where(hit, off.argmax(axis=1), len(t) - 1)], hit
        a, ha = extent(1.0)
        b, hb = extent(-1.0)
        ok = ha & hb
        if ok.sum() < 3:
            break
        # the width against the road's own width here, its narrower readings nearby (a side
        # street, a bay or a corner's rounding only ever widens it, where the road itself
        # narrows or widens along its length the readings follow): the line being off the
        # middle changes neither
        w = a + b
        ref = _rolling(np.where(ok, w, np.nan), 3 * win, lambda v, axis: np.nanpercentile(v, 25, axis=axis))
        ref = np.where(np.isfinite(ref), ref, np.nanmedian(w[ok]))
        ok &= w <= ref + np.maximum(1.5, 0.04 * ref)
        if ok.sum() < 3:
            break
        # each kept reading steadied by its neighbours (a median keeps a change of width sharp),
        # carried across the openings, then smoothed a little
        mid = np.interp(arc, arc[ok], ndi.median_filter(((a - b) / 2)[ok], size=9, mode="nearest"))
        wid = np.interp(arc, arc[ok], ndi.median_filter((a + b)[ok], size=9, mode="nearest"))
        mid = _smooth(mid[:, None], max(5, win // 2))[:, 0]
        wid = _smooth(wid[:, None], max(5, win // 2))[:, 0]
        P = P + N * mid[:, None]
    return _smooth(P, 5), np.maximum(0.0, wid - 0.5) * px


def chains(det, mpp, hw, road=None):
    """The highways' centrelines (see above): [{"pts", "w"}], mask pixels and metres."""
    if not hw:
        return []
    px = max(mpp, 1e-6)
    sk = det["skeleton"].astype(bool)
    dt = det["dt"]
    road = det["road"] if road is None else road
    # the skeleton of roads at least the highway's width (a road's width reads 2 (dt - 0.5)):
    # strong where it is, kept a pixel under it where it joins such a stretch
    T = (float(hw["min_m"]) - LN.TOL_M) / (2 * px) + 0.5
    strong = sk & (dt >= T)
    # A highway remains one road through ordinary width variation. Classify
    # whole branches below; a widened junction alone is not a highway seed.
    weak = sk & (dt >= max(1.5, T * 0.72))
    lab, n = ndi.label(weak, structure=np.ones((3, 3)))
    if not n:
        return []
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    core = keep[lab]
    coords, nbr, deg = _graph(core)
    if not len(coords):
        return []
    nodes, branches = _branches(coords, nbr, deg)
    active, eligible = set(), set()
    incident = {}
    for bi, br in enumerate(branches):
        values = dt[coords[br['px'], 0], coords[br['px'], 1]]
        if not len(values):
            continue
        trim = max(1, int(len(values) * 0.15))
        middle = values[trim:-trim] if len(values) > 2 * trim + 2 else values
        typical = float(np.median(middle))
        if typical >= T * 0.72:
            eligible.add(bi)
        # A real stretch of road, not a junction's enlarged middle.
        if typical >= T - 1.0 and len(values) * px >= max(MIN_M, 4 * typical * px):
            active.add(bi)
        for side in ('a', 'b'):
            if br[side] is not None:
                incident.setdefault(br[side], []).append((bi, side))
    # Carry a classification through a narrower part of the same corridor,
    # choosing its continuation rather than promoting intersecting side roads.
    while True:
        added = set()
        for node, inc in incident.items():
            candidates = [(bi, side) for bi, side in inc if bi in eligible]
            for bi, side in candidates:
                if bi not in active:
                    continue
                direction = _direction(coords, branches[bi]['px'], side == 'b', max(6, T * 2))
                options = []
                for other, end in candidates:
                    if other == bi or other in active:
                        continue
                    vector = _direction(coords, branches[other]['px'], end == 'b', max(6, T * 2))
                    angle = float(direction @ vector)
                    if angle <= -math.cos(math.radians(IN_LINE)) or len(candidates) == 2:
                        options.append((angle, other))
                if options:
                    added.add(min(options)[1])
        added -= active
        if not added:
            break
        active |= added
    if not active:
        return []
    retained = set()
    for bi in active:
        retained.update(branches[bi]['px'])
        for side in ('a', 'b'):
            if branches[bi][side] is not None:
                retained.update(nodes[branches[bi][side]])
    clean = np.zeros_like(core)
    pixels = coords[list(retained)]
    clean[pixels[:, 0], pixels[:, 1]] = True
    core = clean
    coords, nbr, deg = _graph(core)
    # stubs: a loose branch shorter than the road is wide (a side street's start, a bump)
    for _ in range(4):
        nodes, branches = _branches(coords, nbr, deg)
        drop = set()
        for br in branches:
            loose = (br["a"] is None) + (br["b"] is None)
            L = len(br["px"])
            hw_px = float(np.median(dt[coords[br["px"], 0], coords[br["px"], 1]]))
            if (loose == 1 and L < max(10.0 / px, 2.0 * hw_px)) or (loose == 2 and L < MIN_M / px):
                drop.update(br["px"])
        if not drop:
            break
        m = core.copy()
        c = coords[list(drop)]
        m[c[:, 0], c[:, 1]] = False
        core = m
        coords, nbr, deg = _graph(core)
    nodes, branches = _branches(coords, nbr, deg)
    # a meeting point's own pixels can loop back to it: no branch
    branches = [br for br in branches if not (br["a"] is not None and br["a"] == br["b"] and len(br["px"]) < 10)]
    # at each meeting point: which branches run on through it
    ends_at = {}
    for bi, br in enumerate(branches):
        for side, k in (("a", br["a"]), ("b", br["b"])):
            if k is not None:
                ends_at.setdefault(k, []).append((bi, side))
    links = {}                                               # (branch, side) -> (branch, side) it runs on into
    stops = set()                                            # (branch, side) ending at a crossing: cut back
    cos_in = math.cos(math.radians(IN_LINE))
    for k, inc in ends_at.items():
        if len(inc) == 2:
            links[inc[0]], links[inc[1]] = inc[1], inc[0]
            continue
        dirs = []
        for bi, side in inc:
            br = branches[bi]
            hw_px = float(np.median(dt[coords[br["px"], 0], coords[br["px"], 1]]))
            dirs.append(_direction(coords, br["px"], side == "b", max(6.0, hw_px)))
        if len(inc) == 3:
            best = None
            for i in range(3):
                for j in range(i + 1, 3):
                    c = float(dirs[i] @ dirs[j])
                    if c <= -cos_in and (best is None or c < best[0]):
                        best = (c, i, j)
            if best:
                _, i, j = best
                other, _ = inc[3 - i - j]
                branch = branches[other]['px']
                widths = dt[coords[branch, 0], coords[branch, 1]]
                trim = max(1, int(len(widths) * .15))
                middle = widths[trim:-trim] if len(widths) > 2 * trim + 2 else widths
                # A genuine third highway makes a highway-to-highway junction.
                # Only a narrower side-road stub may leave the main lines unbroken.
                if float(np.median(middle)) < T - 1.0 or len(branch) * px < MIN_M:
                    links[inc[i]], links[inc[j]] = inc[j], inc[i]
                    stops.add(inc[3 - i - j])
                    continue
        stops.update(inc)
    # chains: branches strung together through their links
    used = set()
    out = []
    for bi in range(len(branches)):
        if bi in used:
            continue
        # walk back to this chain's first branch
        cur, side = bi, "a"
        guard = 0
        while (cur, side) in links and guard < len(branches):
            nb, ns = links[(cur, side)]
            if nb == bi:
                break
            cur, side = nb, ("b" if ns == "a" else "a")
            guard += 1
        # then forward, from cur entering at side
        seq, c, s_in = [], cur, side
        while c not in used:
            used.add(c)
            px_list = branches[c]["px"] if s_in == "a" else branches[c]["px"][::-1]
            s_out = "b" if s_in == "a" else "a"
            seq.append((c, px_list, s_in, s_out))
            if (c, s_out) not in links:
                break
            c, s_in = links[(c, s_out)]
        pix = np.concatenate([coords[p] for _, p, _, _ in seq]) if seq else np.zeros((0, 2), int)
        if len(pix) < 2:
            continue
        # cut back from a crossing to where the road is its own width again
        med = float(np.median(dt[pix[:, 0], pix[:, 1]]))
        lo, hi = 0, len(pix)
        if (seq[0][0], seq[0][2]) in stops:
            while lo < hi - 1 and dt[pix[lo, 0], pix[lo, 1]] > 1.15 * med:
                lo += 1
            lo = min(hi - 1, lo + int(med))
        if (seq[-1][0], seq[-1][3]) in stops:
            while hi - 1 > lo and dt[pix[hi - 1, 0], pix[hi - 1, 1]] > 1.15 * med:
                hi -= 1
            hi = max(lo + 1, hi - int(med))
        pix = pix[lo:hi]
        P = pix[:, ::-1].astype(float)
        if len(P) < 2 or float(np.hypot(*np.diff(P, axis=0).T).sum()) * px < MIN_M:
            continue
        Q, wid = _refine(_smooth(P, 5), road, med, px)
        Qs = _resample(Q, STEP_PX)
        arc_q = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
        arc_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Qs, axis=0).T))])
        ws = np.interp(arc_s, arc_q, wid)
        out.append({"pts": [[round(float(x), 2), round(float(y), 2)] for x, y in Qs],
                    "w": [round(float(v), 2) for v in ws], "length_m": round(float(arc_s[-1]) * px, 1),
                    "stop_start": (seq[0][0], seq[0][2]) in stops,
                    "stop_end": (seq[-1][0], seq[-1][3]) in stops})
    return out
