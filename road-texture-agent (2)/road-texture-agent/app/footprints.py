"""
The buildings of an area made from coordinates, read from its street map.

Google's street map draws every building from zoom 17 up: a flat fill, grey in the
classic colours, grey or cream in the 2024 colours (cream: shops, halls, stands and other
public buildings, as the satellite shows them), buildings wall to wall parted by a thin
line. So the street map Universal Maps Downloader gives for the alpha (map type 66) holds
the area's buildings as well; another map type can be named for them instead (the
Coordinates settings' "Buildings map").

Reading them:
  1. the pixels of a building's fill (within TOL of one of the palette's fills for the
     map's colours, coords.street_style), off the road;
  2. each fill's pieces on their own (a cream building wall to wall with a grey one is
     two), thin joins taken off (a pixel eroded, then grown back over its own fill), so
     two buildings wall to wall come apart along the line between them;
  3. each piece of at least MIN_M2 a building: its outline grown by the line's half width
     (OUTLINE_M, but never past halfway to the next building), and the smallest rectangle
     round it. When the building fills that rectangle (RECT_FILL) and the rectangle covers
     no other building, it stands as that rectangle; else (an L, a T, a row of houses that
     did not come apart, a curved hall) it stands as its parts: the largest rectangle inside
     its outline, then the largest inside what is left, ... while a part is at least
     PART_MIN_M a side and PART_SHARE of the largest, so no rectangle reaches over another
     building or the street;
  4. each rectangle's width and depth in metres, its angle, its front: the side that
     reaches a street soonest looking straight out from it;
  5. kept in the area's folder: buildings.json (each building's middle in pixels and in
     longitude and latitude, its width across its front, its depth, where its front faces,
     its area, its outline) and buildings_preview.png.

"a" is where a building's front faces, in degrees clockwise from the top of the map, as an
automatic placement's objects (app/autoplace.py): the front's direction is (sin a, -cos a)
in image axes (x right, y down); its width runs along (cos a, sin a).
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

VERSION = 2                                   # buildings.json of an older version are read again
FILLS = {                                     # each style's building fills (RGB)
    "classic": [(232, 234, 237)],
    "new": [(233, 234, 239), (250, 245, 235)],   # as on a real street map (Riyadh, zoom 20)
}
NAMES = {"classic": ["grey"], "new": ["grey", "cream"]}
TOL = 6                                       # a fill pixel: every channel within this of a fill
MIN_M2 = 8.0                                  # smaller pieces are not buildings (a shed, a label's dot)
OUTLINE_M = 0.25                              # half the line round a building: it reaches the line's middle
RECT_FILL = 0.85                              # a building filling this much of its smallest rectangle is that rectangle
PART_MIN_M = 2.0                              # a part narrower than this is a jag of the outline, not a part
PART_SHARE = 0.15                             # nor is one under this share of the largest part (a sliver)
MAX_PARTS = 12
GRID = 100                                    # cells across a building's longest side when it is parted
PREVIEW_PX = 1600
COLOURS = {"grey": (230, 120, 20), "cream": (20, 150, 160)}   # the preview's rectangles, by fill


def palette(pal, style):
    """The fills (and their names) for a style: the palette's "buildings" entry, else FILLS."""
    b = (pal or {}).get("buildings") or {}
    fills = [tuple(int(v) for v in c) for c in (b.get(style) or FILLS.get(style) or FILLS["classic"])]
    names = b.get(style + "_names") or NAMES.get(style) or ["building"] * len(fills)
    names = list(names) + [f"fill {i + 1}" for i in range(len(names), len(fills))]
    return fills, names, int(b.get("tol", TOL)), float(b.get("min_m2", MIN_M2))


def fill_mask(rgb, fills, tol):
    """Which fill each pixel is (1, 2, ...), 0 for none."""
    out = np.zeros(rgb.shape[:2], np.uint8)
    for i, c in enumerate(fills, start=1):
        hit = out == 0
        for ch in range(3):                           # |v - c| <= tol, a channel at a time
            v = rgb[..., ch]
            hit &= (v >= max(0, c[ch] - tol)) & (v <= min(255, c[ch] + tol))
        out[hit] = i
    return out


def pieces(kind, n_fills):
    """
    Each building's pixels numbered (int32, 0 none): each fill's pieces on their own, a pixel
    eroded so thin joins come off, then each grown back over the rim of its own fill.
    """
    core = np.zeros(kind.shape, bool)
    for k in range(1, n_fills + 1):
        core |= ndi.binary_erosion(kind == k, iterations=1)
    lab, n = ndi.label(core)
    if n:
        own = np.zeros(n + 1, np.uint8)
        own[lab[core]] = kind[core]
        grown = ndi.grey_dilation(lab, size=(3, 3))
        rim = (kind > 0) & ~core & (grown > 0)
        rim &= kind == own[grown]
        lab[rim] = grown[rim]
    return lab, n


def _nearest_road(road, step):
    """The distance (pixels) to the road from every cell of a coarse grid (step pixels)."""
    small = road[::step, ::step]
    if not small.any():
        return None
    return ndi.distance_transform_edt(~small) * step


REACH_M = 60.0                                # a street looked for this far straight out from a side


def _front(rect, dist, step, mpp):
    """
    Which side of a building's rectangle is its front, and how far its street is (pixels):
    the side that reaches a street soonest looking straight out from it (from its middle and
    its quarter points), else the side whose middle is nearest a street in any direction.
    """
    if dist is None:
        return 0, 0.0
    c = rect.mean(axis=0)
    hs, ws = dist.shape
    at = lambda pts: dist[np.clip(np.round(pts[:, 1] / step).astype(int), 0, hs - 1),
                          np.clip(np.round(pts[:, 0] / step).astype(int), 0, ws - 1)]
    ts = np.arange(0.0, REACH_M / mpp, max(1.0, 0.5 / mpp))
    straight, near = [], []
    for k in range(4):
        p, q = rect[k], rect[(k + 1) % 4]
        mid = (p + q) / 2
        n = mid - c
        n = n / max(1e-9, float(np.hypot(*n)))
        hits = []
        for s in (0.25, 0.5, 0.75):
            start = p + (q - p) * s
            d = at(start[None, :] + n[None, :] * ts[:, None])
            on = np.flatnonzero(d <= step)
            if len(on):
                hits.append(ts[on[0]])
        straight.append(min(hits) if hits else np.inf)
        near.append(float(at(mid[None, :])[0]))
    k = int(np.argmin(straight)) if np.isfinite(min(straight)) else int(np.argmin(near))
    return k, float(straight[k] if np.isfinite(straight[k]) else near[k])


def _area(rect):
    x, y = rect[:, 0], rect[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _raster(pts, shape, off=(0, 0)):
    """A ring of points (image pixels, pixel middles at +0.5) filled as a bool mask of shape (h, w) whose corner is off."""
    h, w = shape
    im = Image.new("1", (w, h), 0)
    ImageDraw.Draw(im).polygon([(float(x) - off[0] - 0.5, float(y) - off[1] - 0.5) for x, y in pts], fill=1)
    return np.array(im, bool)


def _covers(lab, rect, i):
    """How many pixels of other buildings a rectangle reaches over."""
    H, W = lab.shape
    x0, y0 = max(0, int(math.floor(rect[:, 0].min()))), max(0, int(math.floor(rect[:, 1].min())))
    x1, y1 = min(W, int(math.ceil(rect[:, 0].max())) + 1), min(H, int(math.ceil(rect[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    v = lab[y0:y1, x0:x1][_raster(rect, (y1 - y0, x1 - x0), (x0, y0))]
    return int(((v > 0) & (v != i)).sum())


def _largest(grid, min_h, min_w):
    """The largest rectangle of True cells at least min_h x min_w: (area, r0, r1, c0, c1), area 0 for none."""
    h, w = grid.shape
    heights = np.zeros(w, np.int64)
    best = (0, 0, 0, 0, 0)
    for r in range(h):
        heights = np.where(grid[r], heights + 1, 0)
        hl = heights.tolist() + [0]
        stack = []                                   # (start column, height), heights rising
        for c, cur in enumerate(hl):
            start = c
            while stack and stack[-1][1] >= cur:
                s0, hh = stack.pop()
                if hh >= min_h and c - s0 >= min_w and hh * (c - s0) > best[0]:
                    best = (hh * (c - s0), r - hh + 1, r + 1, s0, c)
                start = s0
            stack.append((start, cur))
    return best


def _parts(poly, rect, min_px2, min_side):
    """
    A building that is not one rectangle as rectangles inside its outline (shapely polygon,
    pixels), along its smallest rectangle's sides: the largest, then the largest of what is
    left, ... while one is at least min_side a side, min_px2 in area and PART_SHARE of the
    largest (a trapezoid's sliver is not a part).
    """
    o = rect[0]
    u, v = rect[1] - o, rect[3] - o
    L1, L2 = float(np.hypot(*u)), float(np.hypot(*v))
    if L1 < 1e-6 or L2 < 1e-6:
        return []
    e1, e2 = u / L1, v / L2
    s = max(1.0, max(L1, L2) / GRID)
    nu, nv = max(1, int(math.ceil(L1 / s))), max(1, int(math.ceil(L2 / s)))
    bx0, by0, bx1, by1 = poly.bounds
    ox, oy = int(math.floor(bx0)) - 1, int(math.floor(by0)) - 1
    w, h = int(math.ceil(bx1)) - ox + 2, int(math.ceil(by1)) - oy + 2
    shift = np.array([ox, oy], float)
    m = _raster(np.asarray(poly.exterior.coords), (h, w), shift)
    for ring in poly.interiors:                       # a courtyard is not the building
        m &= ~_raster(np.asarray(ring.coords), (h, w), shift)
    a = (np.arange(nu) + 0.5) * s
    b = (np.arange(nv) + 0.5) * s
    ix = np.floor(o[0] + a[None, :] * e1[0] + b[:, None] * e2[0] - ox).astype(int)
    iy = np.floor(o[1] + a[None, :] * e1[1] + b[:, None] * e2[1] - oy).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    grid = np.zeros((nv, nu), bool)
    grid[ok] = m[iy[ok], ix[ok]]
    need = min_px2 / (s * s)
    side = max(1, int(math.ceil(min_side / s)))
    out = []
    while len(out) < MAX_PARTS:
        area, r0, r1, c0, c1 = _largest(grid, side, side)
        if not area or area < need:
            break
        grid[r0:r1, c0:c1] = False
        need = max(need, PART_SHARE * area) if not out else need
        u0, u1, v0, v1 = c0 * s, min(L1, c1 * s), r0 * s, min(L2, r1 * s)
        out.append(np.array([o + e1 * u0 + e2 * v0, o + e1 * u1 + e2 * v0, o + e1 * u1 + e2 * v1, o + e1 * u0 + e2 * v1]))
    return out


def _outline(piece, x0, y0, tol_px):
    """A piece's outline (shapely polygon, image pixels), along its pixels' edges, straightened."""
    from shapely.geometry import Polygon
    from skimage import measure
    pad = np.pad(piece, 1).astype(np.float32)
    rings = measure.find_contours(pad, 0.5)
    if not rings:
        return None
    ring = measure.approximate_polygon(max(rings, key=len), tolerance=tol_px)
    if len(ring) < 4:
        return None
    poly = Polygon([(c + x0 - 1 + 0.5, r + y0 - 1 + 0.5) for r, c in ring])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def find(rgb, road, mpp, style, pal=None):
    """
    The buildings of a street map (H x W x 3 uint8), its road (bool H x W, the alpha), at mpp
    metres a pixel, drawn in style ("classic" or "new"). Returns (buildings, notes); each
    building {"id", "cx", "cy" (its rectangle's middle, pixels), "w_m" (across its front),
    "d_m" (front to back), "a" (where its front faces, degrees), "area_m2", "rect" (four
    corners, pixels), "poly" (its outline, pixels: the whole building's, for each of its
    parts), "fill" (its fill's name), "piece" (its number on the map: parts of one building
    share it), "parts" (how many), "edge", "road_m" (its front's distance to the nearest road)}.
    """
    fills, names, tol, min_m2 = palette(pal, style)
    kind = fill_mask(rgb, fills, tol)
    road = np.asarray(road, bool)
    kind[ndi.binary_dilation(road, iterations=1)] = 0
    H, W = kind.shape
    lab, _ = pieces(kind, len(fills))
    px_m2 = mpp * mpp
    step = max(1, int(round(0.5 / mpp)))
    dist = _nearest_road(road, step)
    tol_px = max(0.75, 0.3 / mpp)
    reach = OUTLINE_M / mpp
    pad = int(math.ceil(2 * reach)) + 2
    out, small, thin, split = [], 0, 0, 0
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        piece = lab[sl] == i
        if piece.sum() * px_m2 < min_m2:
            small += 1
            continue
        y0, x0 = sl[0].start, sl[1].start
        poly = _outline(piece, x0, y0, tol_px)
        if poly is None:
            continue
        # grown by the line's half, but not past halfway to the next building
        wy0, wx0 = max(0, y0 - pad), max(0, x0 - pad)
        win = lab[wy0:min(H, sl[0].stop + pad), wx0:min(W, sl[1].stop + pad)]
        other = (win > 0) & (win != i)
        g = reach
        if other.any():
            gap = float(ndi.distance_transform_edt(win != i)[other].min()) - 1.0
            g = min(reach, max(0.0, gap / 2.0))
        pg = poly.buffer(g, join_style=2) if g > 0.05 else poly
        if pg.geom_type != "Polygon":
            pg = max(pg.geoms, key=lambda q: q.area)
        rect = np.asarray(pg.minimum_rotated_rectangle.exterior.coords)[:4]
        whole = _area(rect)
        if pg.area >= RECT_FILL * whole and _covers(lab, rect, i) <= 0.02 * whole:
            parts = [rect]
        else:
            parts = _parts(pg, rect, min_m2 / px_m2, PART_MIN_M / mpp)
            if not parts:
                thin += 1                            # no part 2 m wide: a wall, a canopy's edge
                continue
            split += 1
        votes = np.bincount(kind[sl][piece], minlength=len(fills) + 1)[1:]
        fill = names[int(np.argmax(votes))] if votes.sum() else names[0]
        cut = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop >= H or sl[1].stop >= W
        outline = np.round(np.asarray(pg.exterior.coords)[:-1], 1).tolist()
        for r in parts:
            front, best = _front(r, dist, step, mpp)
            p, q = r[front], r[(front + 1) % 4]
            c = r.mean(axis=0)
            w = float(np.hypot(*(q - p)))
            d = float(np.hypot(*(r[(front + 2) % 4] - q)))
            # the front's outward direction: from the middle through the front side's middle
            f = (p + q) / 2 - c
            f /= max(1e-9, float(np.hypot(*f)))
            a = math.degrees(math.atan2(f[0], -f[1])) % 360.0
            # cut by the map's edge: only part of it is on the map (a part of it: if it reaches the edge)
            edge = cut and (len(parts) == 1 or bool(r[:, 0].min() < 1.5 or r[:, 1].min() < 1.5
                                                     or r[:, 0].max() > W - 1.5 or r[:, 1].max() > H - 1.5))
            out.append({"id": len(out) + 1, "cx": round(float(c[0]), 2), "cy": round(float(c[1]), 2),
                        "w_m": round(w * mpp, 2), "d_m": round(d * mpp, 2), "a": round(a, 2),
                        "area_m2": round((float(pg.area) if len(parts) == 1 else _area(r)) * px_m2, 1),
                        "rect": np.round(r, 2).tolist(), "poly": outline, "fill": fill, "piece": i,
                        "parts": len(parts), "edge": bool(edge),
                        "road_m": round(best * mpp, 2) if dist is not None else None})
    by_fill = {}
    for b in out:
        by_fill[b["fill"]] = by_fill.get(b["fill"], 0) + 1
    return out, {"buildings": len(out), "at_edge": sum(1 for b in out if b["edge"]), "too_small": small,
                 "too_thin": thin, "parted": split, "by_fill": by_fill, "style": style,
                 "fills": [list(c) for c in fills]}


def georef(buildings, extent, size):
    """Each building's middle in longitude and latitude (the area's extent in Web Mercator metres, its size in pixels)."""
    from app.coords import unproject
    x0, y0, x1, y1 = extent
    W, H = size
    for b in buildings:
        X = x0 + (b["cx"]) / W * (x1 - x0)
        Y = y1 - (b["cy"]) / H * (y1 - y0)
        lon, lat = unproject(X, Y)
        b["lon"], b["lat"] = round(lon, 7), round(lat, 7)
    return buildings


def preview(street, buildings, path, max_px=PREVIEW_PX):
    """
    The street map, paler, with each building's rectangle (orange: grey buildings, teal:
    cream ones, grey: cut by the map's edge) and a tick on its front.
    """
    im = Image.fromarray(np.asarray(street)).convert("RGB")
    k = min(1.0, max_px / max(im.size))
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.LANCZOS)
    im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.35)
    dr = ImageDraw.Draw(im)
    for b in buildings:
        r = [(x * k, y * k) for x, y in b["rect"]]
        dr.polygon(r, outline=(150, 150, 150) if b.get("edge") else COLOURS.get(b.get("fill"), COLOURS["grey"]), width=2)
        a = math.radians(b["a"])
        f = (math.sin(a), -math.cos(a))
        cx, cy = b["cx"] * k, b["cy"] * k
        # the front tick: from the middle to the front side, in the rectangle's own pixels
        fr = [(rx - cx) * f[0] + (ry - cy) * f[1] for rx, ry in r]
        tip = max(fr)
        dr.line([(cx, cy), (cx + f[0] * tip, cy + f[1] * tip)], fill=(200, 40, 40), width=2)
    im.save(path)


def read_area(folder, area, street_rgb=None, road=None, pal=None, style=None):
    """
    An area's buildings, read from its street map (or the buildings map it names) and saved
    in its folder (buildings.json, buildings_preview.png). Returns the saved document.
    """
    from app import coords as CO
    folder = Path(folder)
    src = folder / ("buildings_map.png" if (folder / "buildings_map.png").exists() else "street.png")
    if street_rgb is None:
        street_rgb = np.asarray(Image.open(src).convert("RGB"))
    if road is None:
        road = np.asarray(Image.open(folder / "alpha.png").convert("L")) > 127
    pal = pal if pal is not None else CO.palette(folder.parent)
    style = style or CO.street_style(street_rgb, pal)
    mpp = float(area["mpp"])
    buildings, notes = find(street_rgb, road, mpp, style, pal)
    if area.get("extent"):
        georef(buildings, area["extent"], (street_rgb.shape[1], street_rgb.shape[0]))
    doc = {"version": VERSION, "key": area.get("key"), "mpp": mpp, "size": [int(street_rgb.shape[1]), int(street_rgb.shape[0])],
           "source": src.name, "notes": notes, "buildings": buildings}
    tmp = folder / ".buildings.tmp"
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    tmp.replace(folder / "buildings.json")
    preview(street_rgb, buildings, folder / "buildings_preview.png")
    return doc


def load(folder):
    """An area's saved buildings, or None (none yet, or read by an older version: read again)."""
    try:
        doc = json.loads((Path(folder) / "buildings.json").read_text())
    except (OSError, ValueError):
        return None
    return doc if doc.get("version", 1) >= VERSION else None
