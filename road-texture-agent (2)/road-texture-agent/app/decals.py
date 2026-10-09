"""
Decals: pictures laid on the streets (arrows, crossings, manhole covers,
stains, patches). Each imported picture is a layer with where it goes; a
layer can use several of these at once:

- Place intersections: across the street at the junctions' edges, right
  where the street meets the junction, its top towards the junction (a
  crossing, a stop line); at some of the edges ("chance", %), at all of
  them, or at two opposite edges per junction (the two arms most in line).
  "fit" scales it to the road's width there, "back_m" moves it further back.
- Place on streets: anywhere on the road, streets and junctions alike,
  at random; "streets_w" (weight) sets how many (50: about one per 400 m2 of
  road; double the weight, double the decals).
- Place on junctions: inside the junctions only (T, Y, crossroads and
  others), at random; "jn_w": 100 is one per junction on average, 50 one in
  two junctions, 200 two each.
- Place junction edge: next to a junction, against the right or the left
  kerb of the street ("edge_side", as seen driving towards the junction), its
  top towards the junction (an arrow before a junction); "edge_w": 100 is
  every street end, 50 half of them.

A picture's top is its front; its width goes across the road, its length
along it, in proportion. On streets and junctions it lies along the road
(facing either way), or any way with "spin".

The junctions' edges are found where each street really meets its junction:
walking along the street towards the junction, the place where the road
starts to widen (the kerb corners begin). Everything comes from the
generation's mask (app/junctions.py: its junctions, their arms, the streets'
centre lines), in mask pixels, so the texture and the 3D model have the
decals in the same places. Decals over a bridge's road are left out (the
deck lifts it), and two decals of the same layer never lie on each other.
"""
import io
import json
import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GAP_M = 0.2            # between a junction's edge and a decal laid there
KERB_M = 0.3           # between a kerb and a decal laid against it
LIFT_SEPARATE = 0.015  # metres above the road: a decal as its own object (the dashes are at 0.01)
LIFT_PAINTED = 0.006   # a decal that is part of the road object
STREET_M2 = 400.0      # Place on streets at weight 50: one decal per this much road


def layer_list(raw, known):
    """Decal layers as the page sends them, checked; known: the decal ids there are pictures for."""
    out = []
    for l in (raw or [])[:64]:
        if not isinstance(l, dict) or str(l.get("decal")) not in known:
            continue
        num = lambda k, lo, hi, d: min(hi, max(lo, float(l.get(k, d) if l.get(k) not in (None, "") else d)))
        L = {"decal": str(l["decal"]), "width_m": num("width_m", 0.1, 60.0, 3.0), "spin": bool(l.get("spin")),
             # Place intersections, at some, all or two opposite edges
             "inter": bool(l.get("inter")), "junctions": bool(l.get("junctions")), "all_edges": bool(l.get("all_edges")),
             "even_edges": bool(l.get("even_edges")), "chance": num("chance", 0.0, 100.0, 50.0),
             "fit": bool(l.get("fit")), "back_m": num("back_m", 0.0, 500.0, 0.0),
             # Place on streets, on junctions, junction edge (right or left), each with its weight
             "streets": bool(l.get("streets")), "streets_w": num("streets_w", 0.0, 1000.0, 50.0),
             "jn": bool(l.get("jn")), "jn_w": num("jn_w", 0.0, 1000.0, 50.0),
             "edge": bool(l.get("edge")), "edge_side": "left" if l.get("edge_side") == "left" else "right",
             "edge_w": num("edge_w", 0.0, 1000.0, 100.0)}
        if "inter" not in l and "streets" not in l and l.get("random", False):
            # a layer from before: "Place randomly" with a junction choice, or along the streets
            if l.get("junctions") or l.get("all_edges") or l.get("even_edges"):
                L["inter"] = True
            else:
                L["streets"] = True
                L["streets_w"] = min(1000.0, max(1.0, 50.0 * 40.0 / max(2.0, float(l.get("every_m") or 40.0))))
        if L["inter"] and not (L["junctions"] or L["all_edges"] or L["even_edges"]):
            L["all_edges"] = True
        out.append(L)
    return out


def _in_bridge(x, y, bridges):
    if not bridges:
        return False
    from app import quadmesh as QM
    return any(QM.inside_rect(np.array([[x, y]]), b)[0] for b in bridges)


def _centre_line_edge(j, arm, det):
    """Where the arm's centre line crosses the junction's circle, and the street's direction there, outwards."""
    cx, cy, r = j["cx"], j["cy"], j["r"]
    s = arm.get("segment")
    pts = det["segment_pixels"][s] if s else None
    if pts is None or len(pts) < 3:
        if arm.get("off_image"):
            return None
        t = math.radians(arm["angle"])
        ox, oy = math.cos(t), -math.sin(t)
        return cx + ox * r, cy + oy * r, ox, oy
    d = np.hypot(pts[:, 1] - cx, pts[:, 0] - cy)
    k = int(np.argmin(np.abs(d - r)))
    ey, ex = float(pts[k, 0]), float(pts[k, 1])
    near = pts[(d >= d[k] - 1) & (d <= d[k] + max(8.0, 3 * float(arm["width_px"])))]
    if len(near) >= 3:
        c = near - near.mean(axis=0)
        ev = np.linalg.eigh(c.T @ c)[1][:, -1]
        ox, oy = float(ev[1]), float(ev[0])
    else:
        ox, oy = ex - cx, ey - cy
    n = math.hypot(ox, oy) or 1.0
    ox, oy = ox / n, oy / n
    if ox * (ex - cx) + oy * (ey - cy) < 0:                    # pointing out of the junction
        ox, oy = -ox, -oy
    return ex, ey, ox, oy


def _edge(j, arm, det):
    """
    A junction edge: where the street really meets the junction, (x, y, out x,
    out y, half width), in mask pixels. Walking along the street's centre line
    towards the junction, the road's width across the street is measured at
    every half pixel (both ways from the line to the kerbs); the edge is the
    last place before it widens (the kerb corners begin, or a street opens on
    one side), centred between the kerbs there.
    """
    e = _centre_line_edge(j, arm, det)
    if e is None:
        return None
    ex, ey, ox, oy = e
    road = det["road"]
    H, W = road.shape
    px, py = -oy, ox                                          # across the street
    w_ref = max(2.0, float(arm["width_px"]))
    reach = 3.0 * w_ref + 4
    s = np.arange(-2.0 * w_ref, j["r"] + 0.25, 0.5)           # along: outwards (-) and into the junction (+)
    t = np.arange(0.0, reach, 0.5)
    def extent(sign):
        X = ex - ox * s[:, None] + sign * px * t[None, :]
        Y = ey - oy * s[:, None] + sign * py * t[None, :]
        xi, yi = np.round(X).astype(int), np.round(Y).astype(int)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        on = np.zeros(X.shape, bool)
        on[inside] = road[yi[inside], xi[inside]]
        off = ~on
        first = np.where(off.any(axis=1), off.argmax(axis=1), len(t))
        return t[np.minimum(first, len(t) - 1)]
    a, b = extent(1.0), extent(-1.0)                          # to the kerb on each side
    width = a + b
    outside = s < 0
    ref = float(np.median(width[outside])) if outside.any() else w_ref
    wide = width > ref * 1.2 + 1.0
    wide[:np.argmax(s >= 0)] = False                          # only from the circle inwards
    # the last place still as wide as the street (no widening found: where the line meets the circle)
    k = max(0, int(np.argmax(wide)) - 1) if wide.any() else int(np.argmax(s >= 0))
    # centred between the street's kerbs, and its half width, as the street is just outside
    mid = float(np.median(((a - b) / 2)[outside])) if outside.any() else (a[k] - b[k]) / 2
    x = ex - ox * s[k] + px * mid
    y = ey - oy * s[k] + py * mid
    return float(x), float(y), ox, oy, float(ref) / 2


def _fits(cx, cy, ux, uy, w_px, h_px, road):
    """Does a decal (centre, top direction, size in pixels) lie on the road, corners and middle?"""
    H, W = road.shape
    rx, ry = -uy, ux
    for sr, su in ((0, 0), (-1, -1), (1, -1), (1, 1), (-1, 1)):
        x = cx + rx * sr * w_px / 2 * 0.9 + ux * su * h_px / 2 * 0.9
        y = cy + ry * sr * w_px / 2 * 0.9 + uy * su * h_px / 2 * 0.9
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H) or not road[yi, xi]:
            return False
    return True


def _corners(p, mpp, grow=1.0, across=1.0):
    """A placed decal's four corners (mask pixels); across stretches its width only."""
    ux, uy = p["ux"], p["uy"]
    rx, ry = -uy, ux
    hw, hh = p["w_m"] / mpp / 2 * grow * across, p["h_m"] / mpp / 2 * grow
    return np.array([[p["x"] + rx * sr * hw + ux * su * hh, p["y"] + ry * sr * hw + uy * su * hh]
                     for sr, su in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _overlap(a, b, share=0.0):
    """
    Do two rectangles (4 corners each) overlap, by more than share of the
    smaller one? (0: at all, by separating axes)
    """
    for poly in (a, b):
        for i in range(2):
            e = poly[i + 1] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    if share <= 0:
        return True
    from shapely.geometry import Polygon
    A, B = Polygon(a), Polygon(b)
    return A.intersection(B).area > share * min(A.area, B.area)


class _Near:
    """
    Decals laid so far, by where they are: a grid of cells, so a new decal is
    checked only against those that could reach it, not every one on the map
    (a city asks for hundreds of thousands: checking each against all of them
    took a day). far: the furthest any of them reaches from its middle (its
    half diagonal, a crossing's width stretched as free() stretches it).
    """
    CELL = 32.0                                               # mask pixels

    def __init__(self, mpp):
        self.mpp, self.cells, self.far = mpp, {}, 0.0

    def add(self, q):
        c = self.CELL
        self.cells.setdefault((int(q["x"] // c), int(q["y"] // c)), []).append(q)
        self.far = max(self.far, math.hypot(q["w_m"] * 1.5, q["h_m"]) / self.mpp)

    def near(self, x, y, reach):
        """Every decal whose middle is within reach + far of (x, y), and maybe a few more."""
        c, r = self.CELL, reach + self.far
        for i in range(int((x - r) // c), int((x + r) // c) + 1):
            for j in range(int((y - r) // c), int((y + r) // c) + 1):
                yield from self.cells.get((i, j), ())


def _count(weight, rng):
    """How many for a weight of 100 per one: 250 is two, and a third half the time."""
    w = max(0.0, weight) / 100.0
    return int(w) + (1 if rng.random() < w - int(w) else 0)


_det = {}            # the last mask's junctions (finding them is the slow part of placing)


def _detect(gray):
    import hashlib
    from app import junctions as J
    g = np.ascontiguousarray(np.asarray(gray))
    key = (g.shape, hashlib.sha1(g.tobytes()).hexdigest())
    if key not in _det:
        _det.clear()
        _det[key] = J.detect(g)
    return _det[key]


def place(gray, mpp, layers, sizes, seed=7, bridges=None):
    """
    Where every decal goes: [{"layer", "decal", "x", "y" (mask pixels, its
    centre), "ux", "uy" (the way its top faces, in image axes), "w_m", "h_m",
    "at" ("intersection", "street", "junction" or "edge")}].
    sizes: {decal id: (width px, height px)}.
    """
    det = _detect(gray)
    road = det["road"]
    H, W = road.shape
    junctions = [j for j in det["junctions"] if j["type"] != "interchange (flagged)"]
    edges = {}                                                # (junction, arm) -> edge, worked out once

    def edge_of(ji, ai):
        if (ji, ai) not in edges:
            edges[(ji, ai)] = _edge(junctions[ji], junctions[ji]["arms"][ai], det)
        return edges[(ji, ai)]

    # the street's direction at any road pixel: that of the nearest centre-line pixel
    sk = det["skeleton"].astype(bool)
    near_sk = ndi.distance_transform_edt(~sk, return_distances=False, return_indices=True) if sk.any() else None

    def along(x, y):
        if near_sk is None:
            return 1.0, 0.0
        yi, xi = int(min(H - 1, max(0, round(y)))), int(min(W - 1, max(0, round(x))))
        sy, sx = near_sk[0][yi, xi], near_sk[1][yi, xi]
        win = sk[max(0, sy - 6):sy + 7, max(0, sx - 6):sx + 7]
        pts = np.argwhere(win)
        if len(pts) < 3:
            return 1.0, 0.0
        c = pts - pts.mean(axis=0)
        ev = np.linalg.eigh(c.T @ c)[1][:, -1]
        return float(ev[1]), float(ev[0])

    # each kind of place in turn, every layer's: the crossings first (they belong right at the
    # edges), then the decals beside the kerbs at the edges (behind any crossing there), then
    # inside the junctions, then anywhere on the road; each keeps clear of those laid before it
    placed = {li: [] for li in range(len(layers))}
    # by where they are: each layer's own, and the crossings and edge decals of every layer
    near_mine = {li: _Near(mpp) for li in range(len(layers))}
    near_marks = _Near(mpp)

    def free(p, li, gap, share, others):
        c = _corners(p, mpp, gap)
        reach = math.hypot(p["w_m"], p["h_m"]) * gap / mpp
        for group in (near_mine[li], near_marks) if others else (near_mine[li],):
            for q in group.near(p["x"], p["y"], reach):
                mine = q["layer"] == li                       # one of its own layer's
                # a crossing counts as reaching from kerb to kerb: nothing squeezes in beside it
                across = 1.5 if q["at"] == "intersection" and not mine else 1.0
                if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) < reach + math.hypot(q["w_m"] * across, q["h_m"]) / mpp \
                        and _overlap(c, _corners(q, mpp, gap, across), share if mine else 0.0):
                    return False
        return True

    def make(li, cx, cy, ux, uy, w, h, at, gap=1.0, share=0.0, others=True):
        """Lay one decal there if it can go (on the road, off any bridge, clear of the others), else None."""
        xi, yi = int(round(cx)), int(round(cy))
        if not (0 <= xi < W and 0 <= yi < H) or not road[yi, xi] or _in_bridge(cx, cy, bridges):
            return None
        n = math.hypot(ux, uy) or 1.0
        p = {"layer": li, "decal": layers[li]["decal"], "x": round(float(cx), 2), "y": round(float(cy), 2),
             "ux": round(ux / n, 5), "uy": round(uy / n, 5), "w_m": round(w, 3), "h_m": round(h, 3), "at": at}
        # two of a layer never on each other (a short street between two junctions: one crossing, not
        # two); share lets two crossings' corners meet at a junction's corner; gap keeps room between
        if not free(p, li, gap, share, others):
            return None
        placed[li].append(p)
        near_mine[li].add(p)
        if at in ("intersection", "edge"):
            near_marks.add(p)
        return p

    ctx = []
    for li, L in enumerate(layers):
        pw, ph = sizes[L["decal"]]
        ctx.append((L, ph / max(1, pw), np.random.default_rng([int(seed), 3101, li])))

    def lying(L, rng, x, y):
        """A decal's top direction: along the road (either way), or any way with spin."""
        if L["spin"]:
            a = rng.uniform(0, 2 * math.pi)
            return math.cos(a), math.sin(a)
        dx, dy = along(x, y)
        s = 1.0 if rng.random() < 0.5 else -1.0
        return dx * s, dy * s

    # ---- Place intersections: across the street, right at the junction's edge
    for li, (L, aspect, rng) in enumerate(ctx):
        if not L["inter"]:
            continue
        mode = "all" if L["all_edges"] else "even" if L["even_edges"] else "chance"
        for ji, j in enumerate(junctions):
            idx = list(range(len(j["arms"])))
            if mode == "even":
                arms = j["arms"]
                pairs = [(abs(((arms[a]["angle"] - arms[b]["angle"]) % 360) - 180), a, b)
                         for a in idx for b in idx if a < b]
                pairs = [p for p in pairs if p[0] <= 35]
                if not pairs:
                    continue
                best = min(p[0] for p in pairs)
                near = [p for p in pairs if p[0] <= best + 10]      # a crossroads: either road, at random
                _, a, b = near[int(rng.integers(len(near)))]
                idx = [a, b]
            for ai in idx:
                if mode == "chance" and rng.uniform(0, 100) >= L["chance"]:
                    continue
                e = edge_of(ji, ai)
                if e is None:
                    continue
                ex, ey, ox, oy, half = e
                w = 2 * half * mpp * 0.96 if L["fit"] else L["width_m"]
                h = w * aspect
                far = (L["back_m"] + h / 2 + GAP_M) / mpp
                make(li, ex + ox * far, ey + oy * far, -ox, -oy, w, h, "intersection", share=0.25, others=False)

    # ---- Place junction edge: beside the right (or left) kerb, next to the junction (behind a crossing)
    for li, (L, aspect, rng) in enumerate(ctx):
        if not L["edge"]:
            continue
        w, h = L["width_m"], L["width_m"] * aspect
        for ji, j in enumerate(junctions):
            for ai in range(len(j["arms"])):
                e = edge_of(ji, ai)
                if e is None:
                    continue
                ex, ey, ox, oy, half = e
                # driving towards the junction (-out), the right is (out y, -out x) on the map
                rx, ry = (oy, -ox) if L["edge_side"] == "right" else (-oy, ox)
                # clear of the kerb by KERB_M, plus half a mask pixel (the kerb line is smoothed across it)
                side = (half * mpp - w / 2 - KERB_M - 0.5 * mpp) / mpp
                if side < 0:
                    continue                                          # the street is too narrow for it there
                back = 0.0
                for _ in range(_count(L["edge_w"], rng)):            # more than one: one behind the other
                    while back < 30.0:
                        far = (back + h / 2 + GAP_M) / mpp
                        if make(li, ex + ox * far + rx * side, ey + oy * far + ry * side, -ox, -oy, w, h, "edge",
                                share=0.0):
                            back += h + 1.0
                            break
                        back += 0.5

    # ---- Place on junctions: inside the junctions only
    for li, (L, aspect, rng) in enumerate(ctx):
        if not L["jn"]:
            continue
        w, h = L["width_m"], L["width_m"] * aspect
        for ji, j in enumerate(junctions):
            # the junction itself: inside its edges (the nearest edge's distance from its middle)
            dists = [math.hypot(e[0] - j["cx"], e[1] - j["cy"])
                     for e in (edge_of(ji, ai) for ai in range(len(j["arms"]))) if e is not None]
            core = (min(dists) if dists else j["hw"]) * 0.9
            for _ in range(_count(L["jn_w"], rng)):
                for _try in range(12):
                    a, r = rng.uniform(0, 2 * math.pi), core * math.sqrt(rng.random())
                    cx, cy = j["cx"] + r * math.cos(a), j["cy"] + r * math.sin(a)
                    if L["spin"]:
                        t2 = rng.uniform(0, 2 * math.pi)
                        ux, uy = math.cos(t2), math.sin(t2)
                    else:                                         # along one of its streets
                        arm = j["arms"][int(rng.integers(len(j["arms"])))] if j["arms"] else {"angle": 0}
                        t2 = math.radians(arm["angle"]) + (math.pi if rng.random() < 0.5 else 0)
                        ux, uy = math.cos(t2), -math.sin(t2)
                    if _fits(cx, cy, ux, uy, w / mpp, h / mpp, road) and make(li, cx, cy, ux, uy, w, h, "junction", gap=1.2):
                        break

    # ---- Place on streets: anywhere on the road, streets and junctions, at random
    ys, xs = np.nonzero(road)
    for li, (L, aspect, rng) in enumerate(ctx):
        if not L["streets"] or not len(ys):
            continue
        w, h = L["width_m"], L["width_m"] * aspect
        n = int(rng.poisson(len(ys) * mpp * mpp / STREET_M2 * L["streets_w"] / 50.0))
        laid = 0
        for k in rng.integers(0, len(ys), size=min(4 * n, 200000)) if n else []:
            if laid >= n:
                break
            cx, cy = xs[k] + rng.random() - 0.5, ys[k] + rng.random() - 0.5
            ux, uy = lying(L, rng, cx, cy)
            if _fits(cx, cy, ux, uy, w / mpp, h / mpp, road) and make(li, cx, cy, ux, uy, w, h, "street", gap=1.2):
                laid += 1
    return [p for li in range(len(layers)) for p in placed[li]]


# ------------------------------------------------------------------ the texture
def footprint(p, mpp_out, scale, pad=3):
    """The texture pixels a placed decal can cover, turned any way: (x0, y0, x1, y1)."""
    r = math.hypot(p["w_m"], p["h_m"]) / 2 / mpp_out + pad
    cx, cy = p["x"] * scale, p["y"] * scale
    return int(math.floor(cx - r)), int(math.floor(cy - r)), int(math.ceil(cx + r)) + 1, int(math.ceil(cy + r)) + 1


def _key(p):
    return (p["decal"], p["x"], p["y"], p["ux"], p["uy"], p["w_m"], p["h_m"])


def changed(old, new):
    """The decals laid before and not now, and now and not before (each as it lies): [placement]."""
    ko, kn = {_key(p) for p in old}, {_key(p) for p in new}
    return [p for p in old if _key(p) not in kn] + [p for p in new if _key(p) not in ko]


def _turned(p, img, mpp_out, sized=None):
    """
    A placed decal's picture, sized and turned as it lies (float RGBA 0-1).
    sized: a dict that keeps each picture at each size, for the next decal of
    that size (a layer's decals are all one size: sized once, not each time).
    """
    w = max(1, int(round(p["w_m"] / mpp_out)))
    h = max(1, int(round(p["h_m"] / mpp_out)))
    k = (p["decal"], w, h)
    im = sized.get(k) if sized is not None else None
    if im is None:
        im = img.resize((w, h), Image.LANCZOS if w < img.width else Image.BICUBIC)
        if sized is not None:
            sized[k] = im
    # turned so its top (0, -1) faces (ux, uy): PIL turns anticlockwise as seen
    deg = math.degrees(math.atan2(-p["ux"], -p["uy"]))
    im = im.rotate(deg, resample=Image.BICUBIC, expand=True)
    return np.asarray(im, np.float32) / 255.0


def paint(shape, placements, images, mpp_out, scale, coverage=None, window=None, turned=None):
    """
    The decals as a picture of the texture's size (RGBA, float 0-1, H x W x 4),
    each turned and sized as placed; only on the road (coverage).
    images: {decal id: PIL RGBA image}. window: (x0, y0, x1, y1), only that
    part of it (the same pixels the whole picture has there; coverage is still
    the whole texture's). turned: a dict that keeps each decal's sized and
    turned picture for the next window.
    """
    H, W = shape
    wx0, wy0, wx1, wy1 = window if window is not None else (0, 0, W, H)
    acc = np.zeros((wy1 - wy0, wx1 - wx0, 4), np.float32)
    sized = {}
    for p in placements:
        img = images.get(p["decal"])
        if img is None:
            continue
        if window is not None:
            fx0, fy0, fx1, fy1 = footprint(p, mpp_out, scale)
            if fx1 <= wx0 or fx0 >= wx1 or fy1 <= wy0 or fy0 >= wy1:
                continue                                     # nowhere near this part
        if turned is None:
            a = _turned(p, img, mpp_out, sized)
        else:
            k = _key(p)
            if k not in turned:
                turned[k] = _turned(p, img, mpp_out, sized)
            a = turned[k]
        ih, iw = a.shape[:2]
        cx, cy = p["x"] * scale, p["y"] * scale
        x0, y0 = int(round(cx - iw / 2)), int(round(cy - ih / 2))
        xa, ya = max(0, x0, wx0), max(0, y0, wy0)
        xb, yb = min(W, x0 + iw, wx1), min(H, y0 + ih, wy1)
        if xa >= xb or ya >= yb:
            continue
        src = a[ya - y0:yb - y0, xa - x0:xb - x0]
        al = src[..., 3:4]
        if coverage is not None:
            al = al * coverage[ya:yb, xa:xb, None]
        dst = acc[ya - wy0:yb - wy0, xa - wx0:xb - wx0]
        # premultiplied "over"
        dst[..., :3] = src[..., :3] * al + dst[..., :3] * (1 - al)
        dst[..., 3:4] = al + dst[..., 3:4] * (1 - al)
    return acc


def layer_picture(rgba):
    """The Decals layer as saved (RGBA uint8, colour not premultiplied) from paint's picture."""
    return np.clip(np.concatenate([rgba[..., :3] / np.maximum(rgba[..., 3:], 1e-6), rgba[..., 3:]], -1) * 255 + 0.5,
                   0, 255).astype(np.uint8)


def repaint(changes, placements, images, mpp_out, scale, coverage, base_rgb, layer, result, painted):
    """
    Only where decals changed (changes: placements laid before or now, as
    changed() finds them): the Decals layer (RGBA uint8, as saved) and the
    road texture (painted: the texture as generated, base_rgb, with the decals
    in; else the texture as generated) are worked out again from every decal
    now there, in each changed decal's footprint, and written in place. The
    pixels come out as painting the whole picture again gives them. Returns
    the number of pixels gone over.
    """
    H, W = result.shape[:2]
    # the decals now laid, by the 64-pixel cells their footprints touch, to find those near a footprint
    cells = {}
    for i, p in enumerate(placements):
        x0, y0, x1, y1 = footprint(p, mpp_out, scale)
        for cy in range(max(0, y0) // 64, max(0, min(H, y1) - 1) // 64 + 1):
            for cx in range(max(0, x0) // 64, max(0, min(W, x1) - 1) // 64 + 1):
                cells.setdefault((cx, cy), []).append(i)
    done, turned = 0, {}
    for q in changes:
        x0, y0, x1, y1 = footprint(q, mpp_out, scale)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x0 >= x1 or y0 >= y1:
            continue
        near = sorted({i for cy in range(y0 // 64, (y1 - 1) // 64 + 1) for cx in range(x0 // 64, (x1 - 1) // 64 + 1)
                       for i in cells.get((cx, cy), [])})
        rgba = paint((H, W), [placements[i] for i in near], images, mpp_out, scale, coverage, (x0, y0, x1, y1), turned)
        layer[y0:y1, x0:x1] = layer_picture(rgba)
        result[y0:y1, x0:x1] = composite(base_rgb[y0:y1, x0:x1], rgba) if painted else base_rgb[y0:y1, x0:x1]
        done += (x1 - x0) * (y1 - y0)
    return done


def composite(base_rgb, decal_rgba):
    """The road texture with the decals painted in (uint8 RGB)."""
    a = decal_rgba[..., 3:4]
    rgb = decal_rgba[..., :3] / np.maximum(a, 1e-6)
    out = base_rgb.astype(np.float32) / 255.0 * (1 - a) + rgb * a
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ the 3D model
def mesh_prims(placements, images, materials, image_list, W, H, mpp_out, scale, lift, image_index, names=None):
    """
    Each decal as a flat quad lift metres above the road, one primitive per
    picture with a see-through material of its own. W, H: the texture's size
    (the model's ground is W x H texture pixels of mpp_out metres).
    image_index(bytes, mime): the GLB's picture index. Returns primitives.
    """
    by = {}
    for p in placements:
        by.setdefault(p["decal"], []).append(p)
    prims = []
    for did, ps in by.items():
        img = images.get(did)
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, "PNG")
        ti = image_index(buf.getvalue(), "image/png")
        label = "".join(c if c.isalnum() else "_" for c in (names or {}).get(did, did)).strip("_") or did
        materials.append({"name": f"Decal_{label}", "doubleSided": True, "alphaMode": "BLEND",
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": ti}, "metallicFactor": 0.0,
                                                   "roughnessFactor": 0.75}})
        mat = len(materials) - 1
        pos, uv, idx = [], [], []
        for p in ps:
            cx = p["x"] * scale * mpp_out - W * mpp_out / 2
            cz = p["y"] * scale * mpp_out - H * mpp_out / 2
            u = np.array([p["ux"], p["uy"]])                   # its top, in image axes = world X, Z
            r = np.array([-u[1], u[0]])                        # its right
            hw, hh = p["w_m"] / 2, p["h_m"] / 2
            base = len(pos)
            for (sr, su), t in (((-1, 1), (0, 0)), ((1, 1), (1, 0)), ((1, -1), (1, 1)), ((-1, -1), (0, 1))):
                q = np.array([cx, cz]) + r * sr * hw + u * su * hh
                pos.append((q[0], lift, q[1])); uv.append(t)
            # facing up: (0, 1, 0)
            a, b, c, d = base, base + 1, base + 2, base + 3
            P = np.array(pos[-4:])
            ny = np.cross(P[1] - P[0], P[2] - P[0])[1]
            idx += [[a, b, c], [a, c, d]] if ny > 0 else [[a, c, b], [a, d, c]]
        P = np.array(pos)
        prims.append({"positions": P, "normals": np.tile([0.0, 1.0, 0.0], (len(P), 1)), "uv0": np.array(uv, float),
                      "indices": np.array(idx), "material": mat})
    return prims


def load_pictures(index, folder):
    """{decal id: PIL RGBA image} for the decals in index."""
    out = {}
    for d in index:
        p = folder / f"{d['id']}.png"
        if p.exists():
            out[d["id"]] = Image.open(p).convert("RGBA")
    return out


def summary(placements, layers, names):
    """Per layer: how many, and how many of each kind of place."""
    out = []
    for li, L in enumerate(layers):
        mine = [p for p in placements if p["layer"] == li]
        out.append({"layer": li, "name": names.get(L["decal"], L["decal"]), "count": len(mine),
                    **{k: sum(p["at"] == k for p in mine) for k in ("intersection", "street", "junction", "edge")}})
    return out


def save(placements, path):
    path.write_text(json.dumps({"placements": placements}, separators=(",", ":")))
