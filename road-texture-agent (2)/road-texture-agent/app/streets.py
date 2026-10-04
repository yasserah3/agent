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
  2. the road is that area kept a sidewalk's width away from everything that is
     not street (the objects, the spaces not drawn, the block around), so a
     sidewalk runs between road and objects and wraps round each object's
     corner on a curve. Where full sidewalks would leave less road than a
     lane (3 m), the sidewalks are narrowed: the road keeps a lane, or half
     of a space narrower than two lanes, so every drawn street shows;
  3. the corners are rounded further to the corner radius, never closer to an
     object than half a sidewalk; specks of island left between streets, too
     small to stand on, become road;
  4. back to the mask's own size, a pixel being road when at least half of it is.
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


def _edge_extensions(layout, cache, sel_cells, all_cells, road, mpp, objects_area):
    """
    Drawn cells on the edge of the placement, continued straight on to the
    nearest street: (polygon in metres, reached, the cell) per open edge. One
    that would cross an object (objects_area: shapely, metres) stops at the
    edge instead. cache: the rows' lines (app/curves.py row_line).
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
                edges.append((pa[0] + pa[2] * o0, pb[0] + pb[2] * o0, -v, c))
            if r == last:           # the back row: open towards the back
                edges.append((pa[0] + pa[2] * o1, pb[0] + pb[2] * o1, v, c))
        else:
            # the space between two rows is open at its ends
            pair = c["id"][1:].split("-")[0]
            same = [x for x in all_cells if x["kind"] in "yj" and x["id"][1:].split("-")[0] == pair]
            lo, hi = min(x["s"][0] for x in same), max(x["s"][1] for x in same)
            if abs(s0 - lo) < 1e-6:
                pos, t, v = frame(r, s0)
                edges.append((pos + v * o0, far(r, s0, o1), -t, c))
            if abs(s1 - hi) < 1e-6:
                pos, t, v = frame(r, s1)
                edges.append((pos + v * o0, far(r, s1, o1), t, c))
    out = []
    for A, B, d, c in edges:
        mid = (A + B) / 2
        hit = None
        for k in range(1, int(REACH_M / 0.5) + 1):
            if is_road(mid + d * (k * 0.5)):
                hit = k * 0.5
                break
        if hit is None:
            out.append((None, False, c))
            continue
        far = hit + 1.0                                       # a little into the street, so they join
        poly = [tuple(A), tuple(B), tuple(B + d * far), tuple(A + d * far)]
        if objects_area is not None and Polygon(poly).intersection(objects_area).area > 0.5:
            out.append((None, False, c))                      # it would run through an object
            continue
        out.append((poly, True, c))
    return out


def inner_streets(road, mpp, placements, objects, packages):
    """
    road: the street mask (bool, H x W), mpp: metres per mask pixel.
    Returns the road to add (bool H x W), where inner streets carry no
    markings (bool H x W), and a report per placement.
    """
    H, W = road.shape
    add = np.zeros((H, W), bool)
    nomark = np.zeros((H, W), bool)
    report = []
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
        polys = [(CV.cell_polygon(layout, c, 0.5, cache), fit[c["id"]]) for c in sel]
        ext = _edge_extensions(layout, cache, sel, all_cells, road, mpp, objects_area)
        polys += [(e, fit[c["id"]]) for e, ok, c in ext if ok]   # an extension, as the space it continues

        # the window: the drawn streets with room around them, on the mask's pixel grid
        pts = np.array([q for poly, _ in polys for q in poly]) / mpp
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
        # the drawn streets and their extensions, never over an object (placements
        # may overlap), by the sidewalk they get
        parts = {w: raster([poly for poly, f in polys if f == w]) & ~O for w in sorted({f for _, f in polys})}
        C = np.zeros((fh, fw), bool)
        for part in parts.values():
            C |= part
        M = np.repeat(np.repeat(road[y0:y1, x0:x1], k, 0), k, 1)   # the streets already there
        # 2. a sidewalk's width away from everything that is not street
        inside = ndi.distance_transform_edt(C | M) * fr
        R = np.zeros_like(C)
        for w, part in parts.items():
            R |= part & (inside >= w)
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
        # 4. back to the mask's pixels
        block = R.reshape(y1 - y0, k, x1 - x0, k).mean(axis=(1, 3)) >= 0.5
        add[y0:y1, x0:x1] |= block
        if not st.get("markings", True):
            nomark[y0:y1, x0:x1] |= ndi.binary_dilation(block, iterations=1)
        # spaces too narrow for a road between two full sidewalks: theirs were narrowed
        narrowed = [fit[c["id"]] for c in sel if c["kind"] != "j" and fit[c["id"]] < sw]
        report.append({"placement": idx + 1, "name": name, "cells": len(sel),
                       "connected": sum(1 for _, ok, _ in ext if ok), "dead_ends": sum(1 for _, ok, _ in ext if not ok),
                       "narrowed": len(narrowed), "sidewalk_m": sw, "sidewalk_min": round(min(narrowed), 2) if narrowed else sw,
                       "road_m2": round(float(block.sum()) * mpp * mpp, 1)})
    return add & ~road, nomark, report
