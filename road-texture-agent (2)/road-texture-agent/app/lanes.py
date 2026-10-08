"""
Lane lines by road width: how many lines a street carries and where, and on
the widest roads a raised island down the middle. The road texture
(app/generation.py) and the 3D model (app/quadmesh.py, app/model3d.py) both lay
a street's lines from layout(), so the two agree.

- Under 6 m: no lines.
- 6 to 9 m: one line, in the centre. 9 to 12 m: two; 12 to 15 m: three; 15 to
  22 m: four. The lines split the road into equal lanes, all dashed.
- From 22 m: a two-way highway. Lanes of 3 m (stretched a little to fill the
  width exactly) and a raised island of 1 m: five lanes on the first side
  (lines 1 to 4 between them), the island, then the other direction's lanes
  (lines 5, 6 ...): 22 m = 7 lanes x 3 m + 1 m (5 lines), 25 m 8 lanes, 28 m 9,
  31 m 10 (5 each way, 8 lines). Wider still, each further lane goes to the
  direction with fewer, so both stay equal. The lane lines are dashed; a solid
  line runs along both kerbs and along both sides of the island.

A road's width is measured from a mask, so it reads a little short or long:
each width counts from TOL_M below it (a 5.7 m road has the 6 m line).

Positions are across the road, in metres from its centre: from -width/2 (the
first side's kerb) to +width/2. A street's first side is the left of its
canonical direction (canonical()), so both pictures pick the same side.
"""
import math

TOL_M = 0.4                 # widths count from this much below them
STEPS = ((6.0, 1), (9.0, 2), (12.0, 3), (15.0, 4))
HIGHWAY_M = 22.0            # from here a two-way highway with an island
LANE_M = 3.0
ISLAND_M = 1.0              # the island's width
ISLAND_H_M = 0.15           # and height above the road (a kerb's)
FIRST_SIDE = 5              # lanes on the first side before the island
EDGE_INSET_M = 0.3          # a solid edge line's centre from the kerb or the island
LINE_REF_M = 9.0            # a learned line width (a share of the road's width) is taken of at most this


def layout(width_m):
    """
    The lines (and island) of a road width_m wide: {"width_m", "lanes",
    "lane_m", "lines": [(x, "dash" or "solid")], "island": (x0, x1) or None,
    "kind": "none", "road" or "highway"}; x across, metres from the centre.
    """
    W = float(width_m)
    half = W / 2
    if W >= HIGHWAY_M - TOL_M:
        lanes = max(FIRST_SIDE + 2, int(math.floor((W - ISLAND_M + TOL_M) / LANE_M)))
        a = max(FIRST_SIDE, (lanes + 1) // 2)
        b = lanes - a
        s = (W - ISLAND_M) / lanes
        i0 = a * s                                         # the island, from the first kerb
        lines = [(s * k - half, "dash") for k in range(1, a)]
        lines += [(i0 + ISLAND_M + s * k - half, "dash") for k in range(1, b)]
        lines += [(x - half, "solid") for x in (EDGE_INSET_M, i0 - EDGE_INSET_M,
                                                 i0 + ISLAND_M + EDGE_INSET_M, W - EDGE_INSET_M)]
        return {"width_m": W, "lanes": lanes, "lane_m": s, "sides": (a, b), "kind": "highway",
                "lines": sorted(lines), "island": (i0 - half, i0 + ISLAND_M - half)}
    n = 0
    for t, k in STEPS:
        if W >= t - TOL_M:
            n = k
    return {"width_m": W, "lanes": n + 1 if n else 1, "lane_m": W / (n + 1), "sides": None,
            "kind": "road" if n else "none",
            "lines": [(W * k / (n + 1) - half, "dash") for k in range(1, n + 1)], "island": None}


def count(width_m):
    """How many lines a road of this width has, solid edge lines included."""
    return len(layout(width_m)["lines"])


def line_width(width_m, ratio, fixed_m):
    """The paint's width: fixed, or a learned share of the road's width (of at most LINE_REF_M)."""
    return float(ratio) * min(float(width_m), LINE_REF_M) if ratio else float(fixed_m)


def canonical(p0, p1):
    """
    Whether a street from p0 to p1 (x, y in a picture, y down) runs the
    canonical way: along its main axis, towards larger x (or larger y). A street
    drawn the other way is read reversed, so its first side and its dashes come
    out the same whichever end it was traced from.
    """
    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    return dx >= 0 if abs(dx) >= abs(dy) else dy >= 0


def summary(widths):
    """For the log: how many streets got each kind of layout."""
    out = {}
    for w in widths:
        L = layout(w)
        key = "highway" if L["kind"] == "highway" else f"{sum(1 for _, k in L['lines'] if k == 'dash')} line"
        out[key] = out.get(key, 0) + 1
    return out
