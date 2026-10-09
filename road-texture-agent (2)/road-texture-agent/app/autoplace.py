"""
Automatic placement: objects laid on islands by rule, not by hand.

Each island's outline is cut into its sides (straight runs of its kerb). An
object stands behind a side with its front (+Y) to the street, its front
SETBACK metres in from the kerb, its back towards the middle of the island.

The rule, for an object W wide (its X) and D deep (its Y), a weight K
(objects per side), a setback S and a gap G between objects:

1. The sides are taken longest first.
2. A side's stretch is where an object's middle can be along it with its
   whole footprint on the island, at least S from every kerb, less D + G at
   each end whose neighbouring side can hold an object (the corners stay free
   for the neighbours' objects).
3. The side holds n = min(K, capacity) objects, capacity = how many fit side
   by side with G between them: floor((stretch + G) / (W + G)), where stretch
   is the length their footprints can cover (the middles' range plus W). A
   side with no room after its corners but room for one gets one in its
   middle.
4. One object stands in the middle of the stretch; more are spread evenly,
   the same space before, between and after them.
5. An object that would come within G of one laid before it (a narrow island:
   the opposite side, back to back) is left out, never moved.
6. An island that got none gets one object in its middle (the point furthest
   from every kerb), front to its nearest street, when it fits there.

So the weight only ever adds objects where they fit: a big island takes K on
every side; a narrow one keeps its long sides (back to back when it is deep
enough for two objects and a gap, one long side when not); a tiny one, one
object in the middle or none. For a rectangular island L x B (L >= B):
    long sides:   capacity = floor((L - 2S + G) / (W + G)); both long sides
                  when B >= 2(S + D) + G, one when B >= S + D, none otherwise;
                  n objects spread with e = (L - 2S - nW) / (n + 1) metres
                  before, between and after them;
    short sides:  the same with B, when their objects stay G clear of the
                  long sides' (the corners: e >= D + G, or the island wide
                  enough that (B - W) / 2 >= S + D + G).
All lengths in metres; positions in the map's metres (x right, y down).
"""

import math

import numpy as np

SETBACK_M = 3.0          # from the kerb to an object's front (a 2.5 m sidewalk, and half a metre)
GAP_M = 2.0              # between two objects
EPS = 0.05               # metres of slack in the fit checks (a front exactly on its line fits)


def _shapely():
    import shapely
    from shapely.geometry import Polygon
    return shapely, Polygon


def island_polygon(island, mpp):
    """An island (app/islands.py: rings in mask pixels) as a polygon in metres."""
    shapely, Polygon = _shapely()
    rings = [np.asarray(r, float) * mpp for r in island["rings"] if len(r) >= 4]
    if not rings:
        return None
    # the outline is the ring with the largest area; any other is a hole
    areas = [abs(Polygon(r).area) for r in rings]
    k = int(np.argmax(areas))
    poly = Polygon(rings[k], [r for i, r in enumerate(rings) if i != k])
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    return shapely.geometry.polygon.orient(poly, 1.0)


def oriented_size(poly):
    """The island's own X and Y: the sides of its smallest turned rectangle (long, short) in metres."""
    r = poly.minimum_rotated_rectangle
    c = np.asarray(r.exterior.coords)[:4]
    a, b = np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1]))
    return max(a, b), min(a, b)


def sides(poly, min_len=1.0, kerb_tol=0.0):
    """
    The island's sides: its outline simplified (bumps under about 1.5% of its
    size, or under kerb_tol, smoothed away), runs that turn less than 20 degrees merged, as
    [(start, end, length, along, out, near)]: along the unit vector from start
    to end, out the unit normal pointing out of the island, to the street,
    near how far the kerb may stray from the straight side.
    """
    # bumps smaller than this are not sides: 1.5% of the island's size, and never under the
    # outline's own precision (kerb_tol: a mask pixel, whose steps are not bends of the kerb)
    tol = max(float(kerb_tol), float(np.clip(0.015 * math.sqrt(poly.area), 0.5, 5.0)))
    ring = np.asarray(poly.exterior.simplify(tol).coords)[:-1]
    if len(ring) < 3:
        ring = np.asarray(poly.exterior.coords)[:-1]
    # merge runs that hardly turn, as long as the merged side stays close to the kerb: a jog
    # of up to 4% of the island's length (never a fifth of its width, nor more than a couple
    # of mask pixels: the drawing's own unevenness, never a street poking in) is still one
    # side, but bend after bend of a curved kerb does not become a chord across the island
    outline = poly.exterior
    long_m, short_m = oriented_size(poly)
    near = max(tol, min(0.04 * long_m, 0.2 * short_m, max(8.0, 2.0 * kerb_tol)))
    pts = list(ring)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            u, v = b - a, c - b
            nu, nv = np.hypot(*u), np.hypot(*v)
            if nu < 1e-9 or nv < 1e-9:
                pts.pop(i); changed = True
                break
            if abs(math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))) < 20 and \
                    _deviation(outline, a, c) <= near:
                pts.pop(i)
                changed = True
                break
    out = []
    for i in range(len(pts)):
        a, b = np.asarray(pts[i], float), np.asarray(pts[(i + 1) % len(pts)], float)
        L = float(np.hypot(*(b - a)))
        if L < min_len:
            continue
        t = (b - a) / L
        # the outline runs anticlockwise (as numbers), so outside is to the right of it
        out.append((a, b, L, t, np.array([t[1], -t[0]]), near))
    return out


def _deviation(outline, a, c, n=9):
    """How far a straight side from a to c strays from the kerb: the furthest of points along it."""
    from shapely.geometry import Point
    return max(outline.distance(Point(*(a + (c - a) * f))) for f in np.linspace(0, 1, n))


def _boxes(centres, along, out, w, d):
    """Footprints (shapely polygons), w along the side and d deep, their fronts at the centres' line."""
    shapely, _ = _shapely()
    c = np.asarray(centres, float).reshape(-1, 2)
    hw = along * (w / 2)
    back = -out * d
    corners = np.stack([c - hw, c + hw, c + hw + back, c - hw + back], axis=1)
    return shapely.polygons(corners)


def _facing(out):
    """The turn (degrees, clockwise from the top of the map) that points an object's +Y along out."""
    return (math.degrees(math.atan2(out[0], -out[1])) + 360.0) % 360.0


def _on_border(a, b, border, tol):
    """Is a side (a to b, metres) on the map's edge (border: x0, y0, x1, y1)? Then it is not a street."""
    if border is None:
        return False
    x0, y0, x1, y1 = border
    for k, v in ((0, x0), (0, x1), (1, y0), (1, y1)):
        if abs(a[k] - v) <= tol and abs(b[k] - v) <= tol:
            return True
    return False


def place_on_island(poly, w, d, weight, setback=SETBACK_M, gap=GAP_M, step=None, kerb_tol=0.0, border=None):
    """
    The objects for one island: [{"x", "y" (its footprint's middle, metres), "angle" (its +Y,
    degrees clockwise from the top of the map), "side" (index, longest first; -1: the middle
    of the island), "front": [x, y] (its front's middle)}]. kerb_tol: how precise the outline
    is (a mask pixel, metres); border: the map's edge (x0, y0, x1, y1 metres), not a street.
    """
    shapely, Polygon = _shapely()
    if poly is None or poly.is_empty or weight < 1:
        return []
    inside = poly.buffer(-(setback - EPS))
    if inside.is_empty:
        return []
    shapely.prepare(inside)
    step0 = step
    ring = sides(poly, min_len=0.0, kerb_tol=kerb_tol)    # in order round the island: the neighbours
    # a side along the edge of the map is not a street: nothing faces it, it is nobody's neighbour
    edge_tol = max(0.5, 1.5 * kerb_tol)
    street = [not _on_border(sd[0], sd[1], border, edge_tol) for sd in ring]
    laid, taken, result = [], None, []
    order = sorted(range(len(ring)), key=lambda i: -ring[i][2])
    for rank, i in enumerate(order):
        a, b, L, t, n, near = ring[i]
        if L < w or not street[i]:
            continue
        step = step0 or float(np.clip(max(w / 6.0, L / 300.0), 0.25, 5.0))
        # where the object's middle may be along the side: every footprint wholly on the island
        # and S from every kerb (sampled); the stretch is the run of those round the side's middle
        u = np.arange(w / 2, L - w / 2 + 1e-9, step) if L > w else np.array([L / 2])
        if u[-1] < L - w / 2 - 1e-6:
            u = np.append(u, L - w / 2)
        # the front S in from the straight side, or further in where the kerb dips in from it
        # (a side straightened over a notch): the least of a few steps back that fits
        backs = setback + near * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        fits = np.stack([shapely.within(_boxes(a[None, :] + t[None, :] * u[:, None] - n[None, :] * s_, t, n, w, d),
                                        inside) for s_ in backs])
        ok = fits.any(axis=0)
        if not ok.any():
            continue

        def front_at(uc):
            for s_ in backs:
                f = a + t * uc - n * s_
                if shapely.within(_boxes(f, t, n, w, d)[0], inside):
                    return f
            return a + t * uc - n * setback
        idx = np.flatnonzero(ok)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        mid = L / 2
        run = min(runs, key=lambda r: (0 if u[r[0]] - 1e-9 <= mid <= u[r[-1]] + 1e-9 else 1,
                                       min(abs(u[r[0]] - mid), abs(u[r[-1]] - mid)), -len(r)))
        lo, hi = float(u[run[0]]), float(u[run[-1]])

        def fits_at(uc):
            return any(shapely.within(_boxes(a + t * uc - n * s_, t, n, w, d)[0], inside) for s_ in backs)

        # the stretch's ends exactly (to a centimetre), not at the samples: the middle is the middle
        if run[0] > 0:
            bad_u = float(u[run[0] - 1])
            while lo - bad_u > 0.01:
                m = (lo + bad_u) / 2
                lo, bad_u = (m, bad_u) if fits_at(m) else (lo, m)
        if run[-1] < len(u) - 1:
            bad_u = float(u[run[-1] + 1])
            while bad_u - hi > 0.01:
                m = (hi + bad_u) / 2
                hi, bad_u = (m, bad_u) if fits_at(m) else (hi, m)
        # the corners stay free for the neighbouring sides' objects (when a neighbour is long
        # enough for one): their depth and a gap kept clear at that end
        j = (i + 1) % len(ring)
        prev_long, next_long = ring[i - 1][2] >= w and street[i - 1], ring[j][2] >= w and street[j]
        lo2, hi2 = lo + (d + gap if prev_long else 0.0), hi - (d + gap if next_long else 0.0)
        if hi2 >= lo2:
            stretch = (hi2 - lo2) + w
            cap = int(math.floor((stretch + gap) / (w + gap) + 1e-9))
            k = min(int(weight), max(1, cap))
            if k == 1:
                centres = [(lo2 + hi2) / 2]
            else:
                e = (stretch - k * w) / (k + 1)                  # the same space before, between and after
                centres = ([lo2 - w / 2 + e * (j + 1) + w * (j + 0.5) for j in range(k)] if e >= gap
                           else list(np.linspace(lo2, hi2, k)))
        else:
            centres = [(lo + hi) / 2]          # room for one only, in the middle (it may take the corners)
        for uc in centres:
            front = front_at(uc)
            box = _boxes(front, t, n, w, d)[0]
            if not shapely.within(box, inside):
                continue
            if taken is not None and box.buffer(gap / 2 - EPS, join_style="mitre").intersects(taken):
                continue                       # it would touch an object laid before: left out, never moved
            laid.append(box.buffer(gap / 2, join_style="mitre"))
            taken = shapely.union_all(laid)
            mid_pt = front - n * (d / 2)
            result.append({"x": round(float(mid_pt[0]), 3), "y": round(float(mid_pt[1]), 3),
                           "angle": round(_facing(n), 2), "side": rank,
                           "front": [round(float(front[0]), 3), round(float(front[1]), 3)]})
    if not result:
        # too small for any side: one in its middle, front to the nearest kerb, if it fits there
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import polylabel, nearest_points
        try:
            c = polylabel(poly, tolerance=0.25)
        except Exception:
            c = poly.representative_point()
        kerbs = [LineString([tuple(sd[0]), tuple(sd[1])]) for sd, st in zip(ring, street) if st]
        if not kerbs:
            return []                                        # no street at all (the map's edge all round)
        near = nearest_points(MultiLineString(kerbs), c)[0]
        v = np.array([near.x - c.x, near.y - c.y])
        nv = float(np.hypot(*v))
        if nv > 1e-6:
            n = v / nv
            t = np.array([-n[1], n[0]])
            centre = np.array([c.x, c.y])
            front = centre + n * (d / 2)
            box = _boxes(front, t, n, w, d)[0]
            if shapely.within(box, inside):
                result.append({"x": round(float(centre[0]), 3), "y": round(float(centre[1]), 3),
                               "angle": round(_facing(n), 2), "side": -1,
                               "front": [round(float(front[0]), 3), round(float(front[1]), 3)]})
    return result


def footprint(p, w, d):
    """A placed object's footprint corners (metres), for drawing and checks."""
    a = math.radians(p["angle"])
    fy = np.array([math.sin(a), -math.cos(a)])          # its +Y on the map
    fx = np.array([-fy[1], fy[0]])                       # its +X: to the right of facing
    c = np.array([p["x"], p["y"]])
    return np.array([c + fx * w / 2 + fy * d / 2, c - fx * w / 2 + fy * d / 2,
                     c - fx * w / 2 - fy * d / 2, c + fx * w / 2 - fy * d / 2])

