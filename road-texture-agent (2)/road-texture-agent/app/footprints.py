"""
The buildings of an area made from coordinates, read from its street map.

Google's street map draws every building from zoom 17 up: a flat fill with a thin
darker outline, grey in the classic colours, grey or cream in the 2024 colours. So the
street map Universal Maps Downloader gives for the alpha (map type 66) holds the area's
buildings as well; another map type can be named for them instead (the Coordinates
settings' "Buildings map").

Reading them:
  1. the pixels of a building's fill (within TOL of one of the palette's fills for the
     map's colours, coords.street_style), off the road;
  2. thin joins taken off (a pixel eroded, then grown back inside the fill), so two
     buildings wall to wall come apart along their outline even where it is blurred;
  3. each piece of at least MIN_M2 a building: its outline (the fill grown by the outline's
     half width), the smallest rectangle round it (its width and depth in metres, its
     angle), its front: the side of that rectangle nearest a street;
  4. kept in the area's folder: buildings.json (each building's middle in pixels and in
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

FILLS = {                                     # each style's building fills (RGB)
    "classic": [(232, 234, 237)],
    "new": [(230, 232, 235), (252, 248, 236)],
}
NAMES = {"classic": ["grey"], "new": ["grey", "cream"]}
TOL = 6                                       # a fill pixel: every channel within this of a fill
MIN_M2 = 8.0                                  # smaller pieces are not buildings (a shed, a label's dot)
OUTLINE_M = 0.25                              # half the outline's width: the building reaches its middle
PREVIEW_PX = 1600


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
    a = rgb.astype(np.int16)
    for i, c in enumerate(fills, start=1):
        hit = (np.abs(a - np.array(c, np.int16)).max(axis=2) <= tol) & (out == 0)
        out[hit] = i
    return out


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


def find(rgb, road, mpp, style, pal=None):
    """
    The buildings of a street map (H x W x 3 uint8), its road (bool H x W, the alpha), at mpp
    metres a pixel, drawn in style ("classic" or "new"). Returns (buildings, notes); each
    building {"id", "cx", "cy" (its rectangle's middle, pixels), "w_m" (across its front),
    "d_m" (front to back), "a" (where its front faces, degrees), "area_m2", "rect" (four
    corners, pixels), "poly" (its outline, pixels), "fill" (its fill's name), "road_m" (its
    front's distance to the nearest road)}.
    """
    from shapely.geometry import Polygon
    from skimage import measure
    fills, names, tol, min_m2 = palette(pal, style)
    kind = fill_mask(rgb, fills, tol)
    road = np.asarray(road, bool)
    kind[ndi.binary_dilation(road, iterations=1)] = 0
    fill = kind > 0
    # thin joins off: a pixel eroded, the pieces numbered, then each grown back over its fill
    core = ndi.binary_erosion(fill, iterations=1)
    lab, n = ndi.label(core)
    if n:
        gap, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(fill & (gap <= 1.5), lab[iy, ix], 0)
        del gap, iy, ix
    px_m2 = mpp * mpp
    step = max(1, int(round(0.5 / mpp)))
    dist = _nearest_road(road, step)
    grow = max(0.5, OUTLINE_M / mpp)
    tol_px = max(0.75, 0.3 / mpp)
    out, small = [], 0
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        piece = lab[sl] == i
        if piece.sum() * px_m2 < min_m2:
            small += 1
            continue
        y0, x0 = sl[0].start, sl[1].start
        pad = np.pad(piece, 1).astype(np.float32)
        rings = measure.find_contours(pad, 0.5)
        if not rings:
            continue
        ring = max(rings, key=len)
        ring = measure.approximate_polygon(ring, tolerance=tol_px)
        if len(ring) < 4:
            continue
        poly = Polygon([(c + x0 - 1 + 0.5, r + y0 - 1 + 0.5) for r, c in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        poly = poly.buffer(grow, join_style=2)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
        front, best = _front(rect, dist, step, mpp)
        p, q = rect[front], rect[(front + 1) % 4]
        c = rect.mean(axis=0)
        along = q - p
        w = float(np.hypot(*along))
        d = float(np.hypot(*(rect[(front + 2) % 4] - q)))
        # the front's outward direction: from the middle through the front side's middle
        f = (p + q) / 2 - c
        f /= max(1e-9, float(np.hypot(*f)))
        a = math.degrees(math.atan2(f[0], -f[1])) % 360.0
        votes = np.bincount(kind[sl][piece], minlength=len(fills) + 1)[1:]
        # cut by the map's edge: only part of it is on the map
        edge = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop >= lab.shape[0] or sl[1].stop >= lab.shape[1]
        out.append({"id": len(out) + 1, "cx": round(float(c[0]), 2), "cy": round(float(c[1]), 2),
                    "w_m": round(w * mpp, 2), "d_m": round(d * mpp, 2), "a": round(a, 2),
                    "area_m2": round(float(poly.area) * px_m2, 1),
                    "rect": np.round(rect, 2).tolist(),
                    "poly": np.round(np.asarray(poly.exterior.coords)[:-1], 1).tolist(),
                    "fill": names[int(np.argmax(votes))] if votes.sum() else names[0], "edge": bool(edge),
                    "road_m": round(best * mpp, 2) if dist is not None else None})
    return out, {"buildings": len(out), "at_edge": sum(1 for b in out if b["edge"]), "too_small": small, "style": style,
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
    """The street map, paler, with each building's rectangle in orange (grey: cut by the map's edge) and a tick on its front."""
    im = Image.fromarray(np.asarray(street)).convert("RGB")
    k = min(1.0, max_px / max(im.size))
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.LANCZOS)
    im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.35)
    dr = ImageDraw.Draw(im)
    for b in buildings:
        r = [(x * k, y * k) for x, y in b["rect"]]
        dr.polygon(r, outline=(150, 150, 150) if b.get("edge") else (230, 120, 20), width=2)
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
    doc = {"version": 1, "key": area.get("key"), "mpp": mpp, "size": [int(street_rgb.shape[1]), int(street_rgb.shape[0])],
           "source": src.name, "notes": notes, "buildings": buildings}
    tmp = folder / ".buildings.tmp"
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    tmp.replace(folder / "buildings.json")
    preview(street_rgb, buildings, folder / "buildings_preview.png")
    return doc


def load(folder):
    """An area's saved buildings, or None."""
    try:
        return json.loads((Path(folder) / "buildings.json").read_text())
    except (OSError, ValueError):
        return None
