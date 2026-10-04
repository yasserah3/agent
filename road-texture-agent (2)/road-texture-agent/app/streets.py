"""
Inner streets: spaces between a placement's objects drawn as streets.

The cells of the spaces (app/curves.py cells) drawn on the map become road in a
copy of the street mask when Generate runs, so everything after it, the
texture, junctions, kerbs, sidewalks, markings and the 3D model, treats them
as streets, and the areas between them become islands like any city block,
paved like the sidewalks.

Per placement, on a fine grid (about 0.25 m) around it:
  1. the drawn cells; where one reaches the edge of the placement it continues
     straight on to the nearest street, up to 50 m, unless it would cross an
     object of any placement (else it stops there);
  2. the road of each drawn space is a straight strip down its middle, a
     sidewalk in from either side, along the space as drawn: a street across
     the rows runs straight on through each drawn junction, a street along
     the rows too, and the ones going on to a street keep their width. Where
     full sidewalks would leave less road than a lane (3 m), the sidewalks are
     narrowed: the road keeps a lane, or half of a space narrower than two
     lanes, so every drawn street shows;
  3. the corners where streets meet are rounded to the corner radius, never
     closer to an object than half a sidewalk; specks of island left between
     streets, too small to stand on, become road;
  4. back to the mask's own size, a pixel being road when at least half of it is.

For the 3D model the same road also comes as exact shapes, the strips as
drawn and the rounded corners, with the centreline of each street between
junctions for its markings: the model is built from these, so the inner
streets keep their straight edges (app/quadmesh.py, exact).
"""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from app import curves as CV
from app import placements as PL

REACH_M = 50.0      # how far an inner street goes on to meet a street
FINE_M = 0.25       # the grid the streets are shaped on, at most this coarse
SPECK_M2 = 40.0     # islands smaller than this, left between streets, become road
LANE_M = 3.0        # the road a narrow space keeps before its sidewalks are narrowed


def width_of(cell):
    """A space's width across the street it makes: a junction, its narrower side."""
    along, across = cell["s"][1] - cell["s"][0], cell["o"][1] - cell["o"][0]
    return along if cell["kind"] == "x" else across if cell["kind"] == "y" else min(along, across)


def fitted_sidewalk(sw, width):
    """
    The sidewalk a space gets: as set while the road keeps at least a lane,
    else narrower, so the road keeps a lane, or half the space when that is
    narrower than two lanes.
    """
    width = max(width, 0.0)
    road = max(width - 2 * sw, min(LANE_M, width / 2))
    return min(sw, (width - road) / 2)


def _road_strips(layout, cache, sel, fit):
    """
    The road of the drawn spaces (polygons in metres): a straight strip down
    each, its sidewalk (fit) in from either side. A drawn junction carries on
    the streets that meet it: straight through when one goes on beyond it, to
    the far side of the street it meets at a T or a corner, and to a
    sidewalk short of its edge where it ends. A junction drawn on its own is
    a square of road.
    """
    xs = [c for c in sel if c["kind"] == "x"]
    ys = [c for c in sel if c["kind"] == "y"]
    out = []

    def strip(r, s0, s1, o0, o1, between):
        # between two rows: from row r's line to row r + 1's
        line = CV.row_line(layout, r, cache)
        far = CV.row_line(layout, r + 1, cache) if between else line
        out.append(CV.strip_polygon(layout, s0, s1, o0, o1, 0.5, line, far))

    for c in xs:
        f, (s0, s1), (o0, o1) = fit[c["id"]], c["s"], c["o"]
        if s1 - s0 - 2 * f > 1e-6:
            strip(c["row"], s0 + f, s1 - f, o0, o1, False)
    for c in ys:
        f, (s0, s1), (o0, o1) = fit[c["id"]], c["s"], c["o"]
        if o1 - o0 - 2 * f > 1e-6:
            strip(c["row"], s0, s1, o0 + f, o1 - f, True)
    for c in sel:
        if c["kind"] != "j":
            continue
        r, (s0, s1), (o0, o1) = c["row"], c["s"], c["o"]
        la, lb = CV.row_line(layout, r, cache), CV.row_line(layout, r + 1, cache)
        # the streets across the rows meeting it, from the row in front (above) and behind (below),
        # as road bands along the junction's s
        bands = {"above": [], "below": []}
        for x in xs:
            if x["row"] not in (r, r + 1):
                continue
            a, b = x["s"] if x["row"] == r else (CV.carry(lb, la, x["s"][0]), CV.carry(lb, la, x["s"][1]))
            f = fit[x["id"]]
            a, b = max(a + f, s0), min(b - f, s1)
            if b - a > 1e-6:
                bands["above" if x["row"] == r else "below"].append((a, b, f))
        # the streets along the rows meeting it, from the left and the right, as their road band across
        left = [y for y in ys if y["row"] == r and abs(y["s"][1] - s0) < 1e-6]
        right = [y for y in ys if y["row"] == r and abs(y["s"][0] - s1) < 1e-6]
        fh = min([fit[y["id"]] for y in left + right] or [0.0])
        h = (o0 + fh, o1 - fh) if (left or right) and o1 - o0 - 2 * fh > 1e-6 else None
        vb = bands["above"] + bands["below"]
        for side, other in (("above", "below"), ("below", "above")):
            for a, b, f in bands[side]:
                through = any(min(b, b2) - max(a, a2) > 1e-6 for a2, b2, _ in bands[other])
                if side == "above":
                    strip(r, a, b, o0, o1 if through else (h[1] if h else o1 - f), True)
                else:
                    strip(r, a, b, o0 if through else (h[0] if h else o0 + f), o1, True)
        if h:
            if left:
                strip(r, s0, s1 if right else (max(b for _, b, _ in vb) if vb else s1 - fh), h[0], h[1], True)
            if right:
                strip(r, s0 if left else (min(a for a, _, _ in vb) if vb else s0 + fh), s1, h[0], h[1], True)
        f = fit[c["id"]]
        if not vb and not h and s1 - s0 - 2 * f > 1e-6 and o1 - o0 - 2 * f > 1e-6:
            strip(r, s0 + f, s1 - f, o0 + f, o1 - f, True)
    return out


def _junction_bands(layout, cache, sel):
    """Per drawn junction: whether a street across the rows meets it, and one along the rows."""
    xs = [c for c in sel if c["kind"] == "x"]
    ys = [c for c in sel if c["kind"] == "y"]
    out = {}
    for c in sel:
        if c["kind"] != "j":
            continue
        r, (s0, s1) = c["row"], c["s"]
        la, lb = CV.row_line(layout, r, cache), CV.row_line(layout, r + 1, cache)
        across = []
        for x in xs:
            if x["row"] in (r, r + 1):
                a, b = x["s"] if x["row"] == r else (CV.carry(lb, la, x["s"][0]), CV.carry(lb, la, x["s"][1]))
                if min(b, s1) - max(a, s0) > 1e-6:
                    across.append(x["id"])
        along = [y["id"] for y in ys if y["row"] == r and (abs(y["s"][1] - s0) < 1e-6 or abs(y["s"][0] - s1) < 1e-6)]
        out[c["id"]] = (across, along)
    return out


def _centrelines(layout, cache, sel, fit, ext):
    """
    The middle of each drawn street between junctions, for its markings:
    (points in metres, road half-width in metres). A street runs on through a
    drawn junction that no other street meets; one that ends at the edge and
    goes on to a street takes its extension with it.
    """
    sign = -1.0 if layout["flip"] else 1.0
    bands = _junction_bands(layout, cache, sel)
    byid = {c["id"]: c for c in sel}

    def point(r, s, o, between=False, o_far=None):
        # a point s along row r's line, o across it; between rows, half way from
        # row r's line at o to row r + 1's at o_far
        line = CV.row_line(layout, r, cache)
        pos, ang = CV._at(line, min(max(s, 0.0), line[3]))
        v = np.array([-math.sin(ang), math.cos(ang)]) * sign
        if not between:
            return pos + v * o
        lb = CV.row_line(layout, r + 1, cache)
        pb, ab = CV._at(lb, min(max(CV.carry(line, lb, s), 0.0), lb[3]))
        vb = np.array([-math.sin(ab), math.cos(ab)]) * sign
        return (pos + v * o + pb + vb * o_far) / 2

    def half(c):
        w = width_of(c)
        return max(w - 2 * fit[c["id"]], 0.0) / 2

    # streets across the rows: x spaces joined through junctions only they cross
    xs = sorted([c for c in sel if c["kind"] == "x"], key=lambda c: (c["row"], c["s"][0]))
    nxt = {}
    for jid, (across, along) in bands.items():
        if along:
            continue                                  # a crossing or a T: the street stops here
        j = byid[jid]
        up = [byid[i] for i in across if byid[i]["row"] == j["row"]]
        down = [byid[i] for i in across if byid[i]["row"] == j["row"] + 1]
        if len(up) == 1 and len(down) == 1:
            nxt[up[0]["id"]] = down[0]["id"]
    starts = [c for c in xs if c["id"] not in set(nxt.values())]
    ends = {(c["id"], side): e for e, ok, c, side in ext if ok}
    out = []
    for c in starts:
        run = [c]
        while run[-1]["id"] in nxt:
            run.append(byid[nxt[run[-1]["id"]]])
        pts = []
        for x in run:
            sc = (x["s"][0] + x["s"][1]) / 2
            pts += [point(x["row"], sc, x["o"][0]), point(x["row"], sc, x["o"][1])]
        for side, at in (("front", 0), ("back", -1)):
            e = ends.get((run[at]["id"], side))
            if e:
                A, B, B2, A2 = (np.asarray(q, float) for q in e)
                far = (A2 + B2) / 2
                pts = [far] + pts if side == "front" else pts + [far]
        out.append((np.array(pts), min(half(x) for x in run)))
    # streets along the rows: y spaces joined through junctions only they cross
    ys = sorted([c for c in sel if c["kind"] == "y"], key=lambda c: (c["row"], c["s"][0]))
    nxt = {}
    for jid, (across, along) in bands.items():
        if across:
            continue
        j = byid[jid]
        left = [byid[i] for i in along if abs(byid[i]["s"][1] - j["s"][0]) < 1e-6]
        right = [byid[i] for i in along if abs(byid[i]["s"][0] - j["s"][1]) < 1e-6]
        if len(left) == 1 and len(right) == 1:
            nxt[left[0]["id"]] = right[0]["id"]
    starts = [c for c in ys if c["id"] not in set(nxt.values())]
    for c in starts:
        run = [c]
        while run[-1]["id"] in nxt:
            run.append(byid[nxt[run[-1]["id"]]])
        pts = []
        for y in run:
            (s0, s1), (o0, o1) = y["s"], y["o"]
            n = max(1, int(math.ceil((s1 - s0) / 1.0)))         # a point a metre, for curved rows
            pts += [point(y["row"], s0 + (s1 - s0) * k / n, o0, True, o1) for k in range(n + 1)]
        for side, at in (("left", 0), ("right", -1)):
            e = ends.get((run[at]["id"], side))
            if e:
                A, B, B2, A2 = (np.asarray(q, float) for q in e)
                far = (A2 + B2) / 2
                pts = [far] + pts if side == "left" else pts + [far]
        out.append((np.array(pts), min(half(y) for y in run)))
    return [(p, h) for p, h in out if len(p) >= 2 and h > 0]


def _narrowed(poly, f):
    """An extension's road: its strip with a sidewalk in from either side (poly: A, B, B + d, A + d)."""
    A, B, B2, A2 = (np.asarray(q, float) for q in poly)
    L = float(np.hypot(*(B - A)))
    if L - 2 * f <= 1e-6:
        return None
    u = (B - A) / L * f
    return [tuple(A + u), tuple(B - u), tuple(B2 - u), tuple(A2 + u)]


def _edge_extensions(layout, cache, sel_cells, all_cells, road, mpp, objects_area):
    """
    Drawn cells on the edge of the placement, continued straight on to the
    nearest street: (polygon in metres, reached, the cell, its side: front,
    back, left or right) per open edge. One that would cross an object
    (objects_area: shapely, metres) stops at the edge instead. cache: the
    rows' lines (app/curves.py row_line).
    """
    from shapely.geometry import Polygon
    rows = layout["rows"]
    sign = -1.0 if layout["flip"] else 1.0
    H, W = road.shape

    def frame(r, s):
        line = CV.row_line(layout, r, cache)
        pos, ang = CV._at(line, min(max(s, 0.0), line[3]))
        t = np.array([math.cos(ang), math.sin(ang)])
        return pos, t, np.array([-t[1], t[0]]) * sign       # along, and across (towards the back)

    def far(r, s, o):
        # across the space between rows r and r + 1: the far side is on the next row's line
        pos, _, v = frame(r + 1, CV.carry(CV.row_line(layout, r, cache), CV.row_line(layout, r + 1, cache), s))
        return pos + v * o

    def is_road(q):
        x, y = int(math.floor(q[0] / mpp)), int(math.floor(q[1] / mpp))
        return 0 <= x < W and 0 <= y < H and bool(road[y, x])

    edges = []
    last = len(rows) - 1
    for c in sel_cells:
        s0, s1 = c["s"]
        o0, o1 = c["o"]
        r = c["row"]
        if c["kind"] == "x":
            v = frame(r, (s0 + s1) / 2)[2]                    # straight on, square to the row
            pa, pb = frame(r, s0), frame(r, s1)
            if r == 0:              # the front row: open towards the front
                edges.append((pa[0] + pa[2] * o0, pb[0] + pb[2] * o0, -v, c, "front"))
            if r == last:           # the back row: open towards the back
                edges.append((pa[0] + pa[2] * o1, pb[0] + pb[2] * o1, v, c, "back"))
        else:
            # the space between two rows is open at its ends
            pair = c["id"][1:].split("-")[0]
            same = [x for x in all_cells if x["kind"] in "yj" and x["id"][1:].split("-")[0] == pair]
            lo, hi = min(x["s"][0] for x in same), max(x["s"][1] for x in same)
            if abs(s0 - lo) < 1e-6:
                pos, t, v = frame(r, s0)
                edges.append((pos + v * o0, far(r, s0, o1), -t, c, "left"))
            if abs(s1 - hi) < 1e-6:
                pos, t, v = frame(r, s1)
                edges.append((pos + v * o0, far(r, s1, o1), t, c, "right"))
    out = []
    for A, B, d, c, side in edges:
        mid = (A + B) / 2
        hit = None
        for k in range(1, int(REACH_M / 0.5) + 1):
            if is_road(mid + d * (k * 0.5)):
                hit = k * 0.5
                break
        if hit is None:
            out.append((None, False, c, side))
            continue
        far = hit + 1.0                                       # a little into the street, so they join
        poly = [tuple(A), tuple(B), tuple(B + d * far), tuple(A + d * far)]
        if objects_area is not None and Polygon(poly).intersection(objects_area).area > 0.5:
            out.append((None, False, c, side))                # it would run through an object
            continue
        out.append((poly, True, c, side))
    return out


def inner_streets(road, mpp, placements, objects, packages):
    """
    road: the street mask (bool, H x W), mpp: metres per mask pixel.
    Returns the road to add (bool H x W), where inner streets carry no
    markings (bool H x W), a report per placement, and per placement its
    road as exact shapes and its streets' centrelines, in mask pixels:
    {"road": shapely shape, "lines": [(points, half width)], "markings": bool}.
    """
    H, W = road.shape
    add = np.zeros((H, W), bool)
    nomark = np.zeros((H, W), bool)
    report, shapes = [], []
    k = max(1, int(math.ceil(mpp / FINE_M - 1e-9)))
    fr = mpp / k                                              # metres per fine pixel
    # every placement's objects (and plants), streets or not: nothing may run through them
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    all_feet = []
    for p in placements:
        try:
            for oid, cc, a, cid, sc in PL.copies_of(p, objects, packages, mpp)[0]:
                _, _, w, d = PL.sized(objects[oid]["meta"])
                all_feet.append([tuple(q) for q in PL.footprint(cc, a, w * sc, d * sc)])
        except Exception:
            continue
    objects_area = unary_union([Polygon(f) for f in all_feet]) if all_feet else None
    for idx, p in enumerate(placements):
        st = p.get("streets") or {}
        chosen = set(st.get("cells") or [])
        if not chosen:
            continue
        name = (objects.get(p.get("object"), {}).get("meta", {}).get("name")
                or "package %s" % packages.get(p.get("package"), {}).get("name", ""))
        try:
            spots, layout = PL.copies_of(p, objects, packages, mpp)
        except Exception as e:
            report.append({"placement": idx + 1, "name": name, "problem": str(e)})
            continue
        cache = {}                                            # the rows' lines
        all_cells = CV.cells(layout, cache=cache)
        sel = [c for c in all_cells if c["id"] in chosen]
        if not sel:
            report.append({"placement": idx + 1, "name": name, "problem": "its drawn spaces no longer exist"})
            continue
        sw = float(st.get("sidewalk_m", 2.0))
        radius = min(float(st.get("corner_m", 4.0)), 3.0 * sw)     # rounder would reach the objects
        # each drawn space with its sidewalk: as set, or narrowed to fit the space
        fit = {c["id"]: round(fitted_sidewalk(sw, width_of(c)), 3) for c in sel}
        polys = [CV.cell_polygon(layout, c, 0.5, cache) for c in sel]
        ext = _edge_extensions(layout, cache, sel, all_cells, road, mpp, objects_area)
        polys += [e for e, ok, c, _ in ext if ok]
        # the road: straight strips down the drawn spaces and their extensions
        lanes = _road_strips(layout, cache, sel, fit)
        lanes += [q for q in (_narrowed(e, fit[c["id"]]) for e, ok, c, _ in ext if ok) if q]
        fmin = min(fit.values()) if fit else sw

        # the window: the drawn streets with room around them, on the mask's pixel grid
        pts = np.array([q for poly in polys for q in poly]) / mpp
        margin = (sw + radius + 3.0) / mpp
        x0 = int(max(0, math.floor(pts[:, 0].min() - margin)))
        y0 = int(max(0, math.floor(pts[:, 1].min() - margin)))
        x1 = int(min(W, math.ceil(pts[:, 0].max() + margin)))
        y1 = int(min(H, math.ceil(pts[:, 1].max() + margin)))
        if x1 <= x0 or y1 <= y0:
            continue
        fw, fh = (x1 - x0) * k, (y1 - y0) * k

        def raster(polygons):
            img = Image.new("L", (fw, fh), 0)
            dr = ImageDraw.Draw(img)
            for poly in polygons:
                dr.polygon([((x / mpp - x0) * k - 0.5, (y / mpp - y0) * k - 0.5) for x, y in poly], fill=1)
            return np.array(img, bool)

        O = raster(all_feet)                                  # every placement's objects
        C = raster(polys) & ~O                                # the drawn streets and their extensions,
                                                              # never over an object (placements may overlap)
        M = np.repeat(np.repeat(road[y0:y1, x0:x1], k, 0), k, 1)   # the streets already there
        # 2. the road: the straight strips, kept clear of any other placement's objects
        R = raster(lanes) & C
        if O.any():
            R &= ndi.distance_transform_edt(~O) * fr >= 0.5 * fmin
        # 3. corners rounded to the radius: the street network closed by it,
        # added only near the drawn streets and never near the objects
        if radius > 0 and R.any():
            net = R | M
            grown = ndi.distance_transform_edt(~net) * fr <= radius
            closed = (ndi.distance_transform_edt(grown) * fr >= radius) | net
            near = ndi.distance_transform_edt(~C) * fr <= radius + sw
            clear = (ndi.distance_transform_edt(~O) * fr >= 0.5 * sw) if O.any() else True
            R |= closed & ~net & near & clear
        # specks of island between the streets, too small to stand on, become road
        lab, n = ndi.label(~(R | M))
        if n:
            sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * fr * fr
            edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
            nearC = ndi.distance_transform_edt(~C) * fr <= sw + radius + 1.0
            touched = set(np.unique(lab[nearC]))
            hasobj = set(np.unique(lab[O])) if O.any() else set()
            speck = [i + 1 for i in range(n) if sizes[i] < SPECK_M2 and (i + 1) not in edge
                     and (i + 1) in touched and (i + 1) not in hasobj]
            if speck:
                R |= np.isin(lab, speck)
        # the same road as exact shapes for the 3D model: the strips as drawn, and the
        # rounded corners and filled specks traced from the grid
        # (grown by a millimetre and back, so strips that meet edge to edge are one shape)
        lane_shape = unary_union([Polygon(q).buffer(1e-3, join_style=2) for q in lanes if len(q) >= 3]).buffer(
            -1e-3, join_style=2)
        if objects_area is not None:
            lane_shape = lane_shape.difference(objects_area.buffer(0.5 * fmin))
        # only real corners and specks: slivers a grid pixel thin along the strips' edges are steps.
        # Corners against the streets already there are left to the 3D model, which rounds
        # every junction's corners along its own kerb line
        extra = ndi.binary_opening(R & ~raster(lanes), structure=np.ones((2, 2), bool))
        if extra.any() and M.any():
            lab_e, n_e = ndi.label(extra)
            touch = np.unique(lab_e[ndi.binary_dilation(M, iterations=2) & extra])
            extra &= ~np.isin(lab_e, touch[touch > 0])
        pieces = [lane_shape]
        if extra.any():
            from app.model3d import outline_polygons
            from shapely.affinity import affine_transform
            for g in outline_polygons(extra.astype(float)):
                # fine pixel i is the point ((i + 0.5) / k + x0) * mpp
                g = affine_transform(g, [fr, 0, 0, fr, (0.5 / k + x0) * mpp, (0.5 / k + y0) * mpp])
                pieces.append(g.buffer(0.25 * fr))
        shape = unary_union(pieces).buffer(0)
        from shapely.affinity import scale as sscale
        from shapely.geometry import box
        shape = sscale(shape, 1 / mpp, 1 / mpp, origin=(0, 0)).intersection(box(0, 0, W, H))
        lines = [(pts / mpp, hw / mpp) for pts, hw in _centrelines(layout, cache, sel, fit, ext)]
        shapes.append({"placement": idx + 1, "road": shape, "lines": lines, "markings": bool(st.get("markings", True))})
        # 4. back to the mask's pixels
        block = R.reshape(y1 - y0, k, x1 - x0, k).mean(axis=(1, 3)) >= 0.5
        add[y0:y1, x0:x1] |= block
        if not st.get("markings", True):
            nomark[y0:y1, x0:x1] |= ndi.binary_dilation(block, iterations=1)
        # spaces too narrow for a road between two full sidewalks: theirs were narrowed
        narrowed = [fit[c["id"]] for c in sel if c["kind"] != "j" and fit[c["id"]] < sw]
        report.append({"placement": idx + 1, "name": name, "cells": len(sel),
                       "connected": sum(1 for _, ok, _, _ in ext if ok), "dead_ends": sum(1 for _, ok, _, _ in ext if not ok),
                       "narrowed": len(narrowed), "sidewalk_m": sw, "sidewalk_min": round(min(narrowed), 2) if narrowed else sw,
                       "road_m2": round(float(block.sum()) * mpp * mpp, 1)})
    return add & ~road, nomark, report, shapes
