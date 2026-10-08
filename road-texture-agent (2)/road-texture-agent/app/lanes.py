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

All lanes are equal across the road; the lines between them are dashed. From
22 m, with lanes on both sides, the island lies between the two sides, and a
solid line runs along both kerbs and along both sides of the island.

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


def is_highway_width(width_m):
    return float(width_m) >= HIGHWAY_M - TOL_M


def limit(width_m):
    """The most lanes a road this wide can hold (3 m each at least, less the island from 22 m)."""
    W = float(width_m)
    isl = ISLAND_M if is_highway_width(W) else 0.0
    return max(0, int(math.floor((W + TOL_M - isl) / LANE_M)))


def sides_for(width_m, want=None):
    """
    The lanes on each side, (side 1, side 2, cut): even when want is None,
    else want cut back to the limit (one lane off the larger side at a time);
    cut says whether it had to be.
    """
    lim = limit(width_m)
    if want is None:
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


def layout(width_m, sides=None):
    """
    The lines (and island) of a road width_m wide, with sides = (side 1, side 2)
    lanes or None for even: {"width_m", "limit", "sides", "even", "cut",
    "lanes", "lane_m", "lines": [(x, "dash" or "solid")], "island": (x0, x1) or
    None, "kind": "none", "road" or "highway"}; x across, metres from the centre.
    """
    W = float(width_m)
    half = W / 2
    a, b, cut = sides_for(W, sides)
    lanes = a + b
    out = {"width_m": W, "limit": limit(W), "sides": (a, b), "even": sides is None, "cut": cut, "lanes": lanes}
    if lanes < 2:
        return dict(out, lane_m=W, kind="none", lines=[], island=None)
    if is_highway_width(W) and a >= 1 and b >= 1:
        s = (W - ISLAND_M) / lanes
        i0 = a * s                                         # the island, from side 1's kerb
        lines = [(s * k - half, "dash") for k in range(1, a)]
        lines += [(i0 + ISLAND_M + s * k - half, "dash") for k in range(1, b)]
        lines += [(x - half, "solid") for x in (EDGE_INSET_M, i0 - EDGE_INSET_M,
                                                 i0 + ISLAND_M + EDGE_INSET_M, W - EDGE_INSET_M)]
        return dict(out, lane_m=s, kind="highway", lines=sorted(lines), island=(i0 - half, i0 + ISLAND_M - half))
    s = W / lanes
    return dict(out, lane_m=s, kind="road", lines=[(s * k - half, "dash") for k in range(1, lanes)], island=None)


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
