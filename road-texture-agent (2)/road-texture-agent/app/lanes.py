"""
Lane lines by road width: how many lanes a street carries on each of its two
sides, where its lines go, and on the widest roads a raised island between the
two directions. The road texture (app/generation.py) and the 3D model
(app/quadmesh.py, app/model3d.py) both lay a street's lines from layout(), and
the Generate tab's Lanes tool (ui/app.js) draws the same layout from a copy of
these few lines, so all three agree.

The limit: the most lanes a road can hold, lanes being at least 3 m wide,

    limit = floor((width - island) / 3 m)

with island = 1 m from 22 m up (a raised island between the two directions)
and 0 below. A width read from a mask is a little short or long, so each width
counts from TOL_M below it (a road measured at 5.7 m holds 2 lanes).

Even (the default): both sides the same, floor(limit / 2) lanes each; an odd
lane left over is dropped and the lanes widen to fill the road. So 6 m is 1 + 1
(one line), 9 m 1 + 1 of 4.5 m, 12 m 2 + 2, 15 m 2 + 2 of 3.75 m, 18 m 3 + 3,
22 m 3 + 3 of 3.5 m around the island, 25 m 4 + 4, 31 m 5 + 5. Under 6 m: one
lane, no lines.

Custom (a street picked in the Lanes tool): the lanes asked for on each side,
cut back to the limit if they add up to more (one off the larger side at a
time). A side may have none: a one-way road.

Highways (Road markings, Highways; hw= {"min_m", "lanes"}): a street at least
min_m wide (18 m: 3 + 3 lanes of 3 m) is a highway. Unless the Lanes tool gives
it its own, a highway has lanes each way (3), however wide it is: the lanes
widen to fill the road (a 30 m road: 3 + 3 lanes of 4.8 m around the island).
Its lines run on through the junctions where it meets side streets and through
T-junctions, laid along its whole centreline (app/highways.py), and stop only
where it meets another highway at a crossroads. hw None: no highways (as
before).

All lanes are equal across the road; the lines between them are dashed. From
22 m, with lanes on both sides, the island lies between the two sides, and a
solid line runs along both kerbs and along both sides of the island.

Two sets of rules for the lines (rules=):
- "simple" (as before): every line between two lanes is dashed alike.
- "guide" (the road marking guide, UTTIPEC Delhi 2009, after IRC 35): the line
  between the two directions is the centre line: broken with one lane each way
  ("cdash"), a single solid line with two ("csolid"), a double solid line from
  three ("cdouble"); the lines between lanes going the same way stay dashed
  lane lines ("dash"). One-way streets and roads with an island have no centre
  line. app/roadmarks.py has the sizes and colours of each kind.

Positions are across the road, in metres from its centre: from -width/2 (side
1's kerb) to +width/2 (side 2's). Side 1 is the left of the street's canonical
direction (canonical()), so every picture picks the same side.
"""
import math

TOL_M = 0.4                 # widths count from this much below them
LANE_M = 3.0                # a lane's least width
HIGHWAY_M = 22.0            # from here a two-way road has a raised island between its sides
ISLAND_M = 1.0              # the island's width
ISLAND_H_M = 0.15           # and height above the road (a kerb's)
EDGE_INSET_M = 0.3          # a solid edge line's centre from the kerb or the island
LINE_REF_M = 9.0            # a learned line width (a share of the road's width) is taken of at most this
HW_MIN_M = 18.0             # a street this wide or wider is a highway (Road markings, Highways)
HW_LANES = 3                # and has this many lanes each way, unless the Lanes tool gives it its own


def is_highway_width(width_m):
    """Whether a road this wide has a raised island between its two directions (from 22 m)."""
    return float(width_m) >= HIGHWAY_M - TOL_M


def is_highway(width_m, hw):
    """Whether a street this wide is a highway under these settings (hw: {"min_m", "lanes"}, None: no highways)."""
    return bool(hw) and float(width_m) >= float(hw["min_m"]) - TOL_M


def limit(width_m):
    """The most lanes a road this wide can hold (3 m each at least, less the island from 22 m)."""
    W = float(width_m)
    isl = ISLAND_M if is_highway_width(W) else 0.0
    return max(0, int(math.floor((W + TOL_M - isl) / LANE_M)))


def sides_for(width_m, want=None, hw=None):
    """
    The lanes on each side, (side 1, side 2, cut): even when want is None
    (on a highway, hw's lanes each way, as many as fit), else want cut back to
    the limit (one lane off the larger side at a time); cut says whether it had
    to be.
    """
    lim = limit(width_m)
    if want is None:
        if is_highway(width_m, hw):
            n = min(max(1, int(hw["lanes"])), lim // 2)
            return n, n, False
        return lim // 2, lim // 2, False
    a, b = max(0, int(want[0])), max(0, int(want[1]))
    cut = False
    while a + b > lim:
        if a >= b:
            a -= 1
        else:
            b -= 1
        cut = True
    return a, b, cut


CENTRE_KINDS = ("cdash", "csolid", "cdouble")
SOLID_KINDS = ("solid", "csolid", "cdouble")      # laid whole, not dashed


def centre_kind(a, b):
    """The guide's centre line for a lanes each way and b the other: broken, solid, or double solid."""
    m = max(a, b)
    return "cdash" if m <= 1 else "csolid" if m == 2 else "cdouble"


def layout(width_m, sides=None, rules="simple", edge_inset=None, hw=None, island=None):
    """
    The lines (and island) of a road width_m wide, with sides = (side 1, side 2)
    lanes or None for even: {"width_m", "limit", "sides", "even", "cut",
    "lanes", "lane_m", "lines": [(x, kind)], "island": (x0, x1) or None,
    "kind": "none", "road" or "highway" (with a raised island), "hw": whether
    it is a highway (hw's settings: its lanes each way when even)}; x across,
    metres from the centre. kind: "dash" (a lane line) or "solid" (an edge
    line), and with the guide's rules the centre line, "cdash", "csolid" or
    "cdouble". edge_inset: an edge line's middle from the kerb or the island
    (EDGE_INSET_M when None). island: True or False to have the island or not
    whatever the width (a highway keeps its layout the length of it as its
    width changes), None: from 22 m.
    """
    W = float(width_m)
    half = W / 2
    a, b, cut = sides_for(W, sides, hw)
    lanes = a + b
    ei = EDGE_INSET_M if edge_inset is None else float(edge_inset)
    out = {"width_m": W, "limit": limit(W), "sides": (a, b), "even": sides is None, "cut": cut, "lanes": lanes,
           "hw": is_highway(W, hw)}
    if lanes < 2:
        return dict(out, lane_m=W, kind="none", lines=[], island=None)
    if (is_highway_width(W) if island is None else island) and a >= 1 and b >= 1:
        s = (W - ISLAND_M) / lanes
        i0 = a * s                                         # the island, from side 1's kerb
        lines = [(s * k - half, "dash") for k in range(1, a)]
        lines += [(i0 + ISLAND_M + s * k - half, "dash") for k in range(1, b)]
        lines += [(x - half, "solid") for x in (ei, i0 - ei, i0 + ISLAND_M + ei, W - ei)]
        return dict(out, lane_m=s, kind="highway", lines=sorted(lines), island=(i0 - half, i0 + ISLAND_M - half))
    s = W / lanes
    if rules == "guide" and a >= 1 and b >= 1:
        lines = [(s * k - half, "dash" if k != a else centre_kind(a, b)) for k in range(1, lanes)]
        return dict(out, lane_m=s, kind="road", lines=lines, island=None)
    return dict(out, lane_m=s, kind="road", lines=[(s * k - half, "dash") for k in range(1, lanes)], island=None)


def lane_spans(lay):
    """
    Each side's lanes across the road, (x0, x1) in metres from the centre, side
    1's from its kerb to the middle, side 2's from the middle to its kerb, and
    where the two directions meet (the centre line's x, or the island's (x0,
    x1)): {"side1", "side2", "centre"}. A road with no lanes (too narrow for
    lines) counts as one lane each way, half the road each.
    """
    W = lay["width_m"]
    half = W / 2
    a, b = lay["sides"]
    if lay["kind"] == "none":
        return {"side1": [(-half, 0.0)], "side2": [(0.0, half)], "centre": 0.0}
    s = lay["lane_m"]
    s1 = [(-half + k * s, -half + (k + 1) * s) for k in range(a)]
    if lay["island"]:
        x0 = lay["island"][1]
        s2 = [(x0 + k * s, x0 + (k + 1) * s) for k in range(b)]
        return {"side1": s1, "side2": s2, "centre": tuple(lay["island"])}
    s2 = [(-half + (a + k) * s, -half + (a + k + 1) * s) for k in range(b)]
    return {"side1": s1, "side2": s2, "centre": (-half + a * s) if a and b else None}


def end_setbacks(path_xy, ends, default_px, px, inside=False):
    """
    How far short of each end a street's lines stop (pixels along it, at its
    first point and its last): where a junction's markings ask (ends: [{"x",
    "y" (the junction edge, this picture's pixels), "ox", "oy" (out of the
    junction, along the street), "m" (metres out from the edge the lines stop
    at), "reach_m" (how far out from the edge the street's end may lie),
    "half_m" (the road's half width there)}]), else
    default_px. An end belongs to the street end it lies within reach of and
    faces along. Below 0: the lines run on past the street's end (into the
    junction's circle) to the stop line. inside: the street may end inside the
    junction, past the edge (a highway's centreline runs on to where the
    crossing highway is): its lines still stop at the markings.
    """
    import numpy as np
    P = np.asarray(path_xy, float)
    out = []
    for i, j in ((0, min(len(P) - 1, 4)), (len(P) - 1, max(0, len(P) - 5))):
        p0 = P[i]
        t = P[j] - p0                                         # into the street from this end
        n = float(np.hypot(*t)) or 1.0
        t = t / n
        best = None
        for e in ends or ():
            if e.get("chain") or e["ox"] * t[0] + e["oy"] * t[1] < 0.8:
                continue                                      # a highway's centreline, or not along this street
            # this end must lie on the approach: out from the edge along it, and across it
            # no further than the road's half width (a stub of centreline beside it is not it)
            dx, dy = p0[0] - e["x"], p0[1] - e["y"]
            fwd = dx * e["ox"] + dy * e["oy"]
            side_ = abs(-dx * e["oy"] + dy * e["ox"])
            lo = -e.get("reach_m", 30.0) / px if inside else -2.0 / px
            if not (lo <= fwd <= e.get("reach_m", 30.0) / px) or side_ > (e.get("half_m", 6.0) + 1.0) / px:
                continue
            # the stop point, measured along the street from this end
            d = (e["x"] + e["ox"] * e["m"] / px - p0[0]) * t[0] + (e["y"] + e["oy"] * e["m"] / px - p0[1]) * t[1]
            if best is None or side_ < best[0]:
                best = (side_, d)
        out.append(best[1] if best else default_px)
    return out[0], out[1]


def extend_ends(P, a_px, b_px):
    """
    A line (N x 2 points) run on straight past its first point by a_px and past
    its last by b_px (along its direction there), a point every pixel (so
    smoothing it treats the new part as it treats the rest): (points, before,
    after), before and after the distances of the points added at each end
    from the old end (before: from the furthest to the nearest).
    """
    import numpy as np
    P = np.asarray(P, float)
    none = np.zeros(0)
    if len(P) < 2:
        return P, none, none
    out, before, after = [P], none, none
    if a_px > 0:
        t = P[0] - P[min(4, len(P) - 1)]
        t = t / (np.hypot(*t) or 1.0)
        before = np.linspace(a_px, 0.0, int(math.ceil(a_px)) + 1)[:-1]
        out.insert(0, P[0] + t * before[:, None])
    if b_px > 0:
        t = P[-1] - P[max(0, len(P) - 5)]
        t = t / (np.hypot(*t) or 1.0)
        after = np.linspace(0.0, b_px, int(math.ceil(b_px)) + 1)[1:]
        out.append(P[-1] + t * after[:, None])
    return np.vstack(out), before, after


# ---------------------------------------------------------------- highways' centrelines
# A highway's lines are laid along its whole centreline (app/highways.py: a
# chain, kept with the lines' ends as {"chain": True, "pts": [[x, y], ...],
# "w": [width in metres at each point]}), past every side street and
# T-junction, and its own street segments (cut up at every junction, and on a
# busy highway mostly cut away) lay none. In the street lists a chain is the
# street numbered CHAIN_ID plus its place among the chains.

CHAIN_ID = 1_000_000


def chains_of(ends):
    """The highways' centrelines among the lines' ends."""
    return [e for e in ends or () if e.get("chain")]


class ChainIndex:
    """Where the highways' centrelines run: which street points lie on a highway's carriageway."""

    def __init__(self, chains, px):
        import numpy as np
        from scipy.spatial import cKDTree
        P = [np.asarray(c["pts"], float) for c in chains if len(c["pts"])]
        W = [np.asarray(c["w"], float) for c in chains if len(c["pts"])]
        self.ok = bool(P)
        if self.ok:
            self.P, self.W = np.vstack(P), np.concatenate(W)
            self.tree = cKDTree(self.P)
        self.px = px

    def along(self, pts_xy):
        """Per point: whether it lies on a highway's carriageway (within half its width of its centreline)."""
        import numpy as np
        pts = np.asarray(pts_xy, float)
        if not self.ok or not len(pts):
            return np.zeros(len(pts), bool)
        d, i = self.tree.query(pts)
        return d <= np.maximum(1.5, 0.5 * self.W[i]) / self.px

    def share(self, pts_xy):
        """How much of a street's centreline runs along a highway's (0 to 1)."""
        a = self.along(pts_xy)
        return float(a.mean()) if len(a) else 0.0


def chain_offsets(lay, widths, rules, edge_inset=None):
    """
    Each of a highway chain's lines' x (and its island's edges') at every
    point, the road there as wide as widths says: its layout's lanes and island
    kept the length of it, the lines moving with the width as layout() moves
    them. Returns ([array per line], [array, array] or None).
    """
    import numpy as np
    w = np.asarray(widths, float)
    W0 = float(lay["width_m"])
    isl = bool(lay["island"])
    L1 = layout(W0, lay["sides"], rules=rules, edge_inset=edge_inset, island=isl)
    L2 = layout(W0 + 1.0, lay["sides"], rules=rules, edge_inset=edge_inset, island=isl)
    if [k for _, k in L1["lines"]] != [k for _, k in L2["lines"]] or bool(L1["island"]) != bool(L2["island"]):
        return ([np.full(len(w), x) for x, _ in lay["lines"]],
                [np.full(len(w), x) for x in lay["island"]] if lay["island"] else None)
    xs = [x1 + (x2 - x1) * (w - W0) for (x1, _), (x2, _) in zip(L1["lines"], L2["lines"])]
    il = ([a1 + (a2 - a1) * (w - W0) for a1, a2 in zip(L1["island"], L2["island"])] if L1["island"] else None)
    return xs, il


def line_width(width_m, ratio, fixed_m):
    """The paint's width: fixed, or a learned share of the road's width (of at most LINE_REF_M)."""
    return float(ratio) * min(float(width_m), LINE_REF_M) if ratio else float(fixed_m)


def canonical(p0, p1):
    """
    Whether a street from p0 to p1 (x, y in a picture, y down) runs the
    canonical way: along its main axis, towards larger x (or larger y). A street
    drawn the other way is read reversed, so its side 1 and its dashes come out
    the same whichever end it was traced from.
    """
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    return dx >= 0 if abs(dx) >= abs(dy) else dy >= 0


def picks_list(raw):
    """Lane choices as sent by the page, checked: [{"pt": [x, y] (mask pixels), "sides": [a, b]}]."""
    out = []
    for p in (raw or [])[:5000]:
        try:
            x, y = float(p["pt"][0]), float(p["pt"][1])
            a, b = int(p["sides"][0]), int(p["sides"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        out.append({"pt": [round(x, 1), round(y, 1)], "sides": [max(0, min(40, a)), max(0, min(40, b))]})
    return out


def match_picks(picks, points, labels, reach):
    """
    Which street each lane choice is for: {street label: (a, b)}. points: every
    street's centreline points (N, 2) in the picks' pixels, labels: their
    street, reach(label, distance): whether a pick that far from the street is
    on it. A later choice wins a street picked twice.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    out = {}
    if not picks or len(points) == 0:
        return out
    tree = cKDTree(np.asarray(points, float))
    d, i = tree.query(np.array([p["pt"] for p in picks], float))
    for p, dd, ii in zip(picks, d, i):
        lab = int(labels[int(ii)])
        if reach(lab, float(dd)):
            out[lab] = tuple(p["sides"])
    return out


def summary(layouts):
    """For the log: how many streets got each split (layout() results)."""
    out = {}
    for L in layouts:
        a, b = L["sides"]
        key = "none" if L["kind"] == "none" else f"{a} + {b}" + (" highway" if L["kind"] == "highway" else "")
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------- markings area
# One rectangle, set on the map (the Generate tab's Markings area), outside which
# no lane lines and no decals are laid: {"cx", "cy" (its middle), "w", "h" (its
# width along its turn and its height across it), "angle" (degrees, from the
# picture's x axis towards its y axis, as the map turns it)}, in mask pixels. None:
# markings everywhere. A dash is kept whole when its middle lies inside, a solid
# line stops at the edge, a decal is kept when its middle lies inside.

def area_check(raw):
    """A markings area as the page sends it, checked; None when there is none."""
    if not isinstance(raw, dict):
        return None
    try:
        a = {k: float(raw[k]) for k in ("cx", "cy", "w", "h")}
        a["angle"] = float(raw.get("angle", 0.0)) % 360.0
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in a.values()) or a["w"] <= 0 or a["h"] <= 0:
        return None
    return {k: round(v, 3) for k, v in a.items()}


def area_scaled(area, scale):
    """
    The area in the pixels of a picture scale times the mask's size, counted as
    the code counts them (a pixel's middle at its whole number; the map counts
    from its corner).
    """
    if not area:
        return None
    return dict(area, cx=area["cx"] * scale - 0.5, cy=area["cy"] * scale - 0.5, w=area["w"] * scale, h=area["h"] * scale)


def area_outside(area, x, y):
    """
    How far each point (x, y, arrays or numbers, the area's pixels) lies outside
    the area: below 0 inside (minus the distance to the nearest edge), above 0
    outside (the distance past the edge, at least). No area: everywhere inside.
    """
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not area:
        return np.full(np.broadcast(x, y).shape, -np.inf)
    a = math.radians(area["angle"])
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - area["cx"], y - area["cy"]
    u, v = dx * c + dy * s, -dx * s + dy * c
    return np.maximum(np.abs(u) - area["w"] / 2, np.abs(v) - area["h"] / 2)


def in_area(area, x, y):
    """Whether each point (x, y) lies inside the area (always, with none)."""
    return area_outside(area, x, y) <= 0


def area_status(area, x, y, reach):
    """
    Whether a street with these centreline points, whose lines reach this far
    from them, lies all "in" the area, all "out" of it, or across its edge
    ("edge").
    """
    if not area:
        return "in"
    d = area_outside(area, x, y)
    if (d <= -reach).all():
        return "in"
    if (d > reach).all():
        return "out"
    return "edge"

