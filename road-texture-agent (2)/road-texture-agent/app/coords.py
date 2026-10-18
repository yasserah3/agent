"""
Coordinates instead of an alpha map (the Generate tab's Input): an area given by its
left and right longitude and its top and bottom latitude.

For an area and a zoom (20: 0.136 m a pixel in Riyadh) it downloads, through
Universal Maps Downloader (app/osm/satellite.py, from the OSM tool), the satellite
map (Google Satellite, id 1) and the street map without labels (Google Street Map,
id 66); cuts both to the same Web Mercator pixel grid, the satellite's own pixels
(no resampling of it); finds the satellite's missing tiles (blank squares where a
tile did not come); turns the street map's road colours into the alpha (white the
road, black the rest), and draws the OSM roads of the Saudi PBF over the area (its
road index made once, or the OSM tool's own one used). The alpha goes on as any
loaded street mask, with its metres a pixel set from the zoom and the latitude.

The street map's roads are drawn by Google at map widths, close to but not exactly
the asphalt: the satellite is used for the details (the analysis, app/analysis of
the next step), not to move the alpha's edges. At zoom 17 and up Google draws the
buildings too, some over the streets' edges; so the street map is also downloaded at
zoom 16 (STREETS_ZOOM, a few tiles), where it draws none, and the streets are taken
from that one (low_streets); the buildings are still read from the area's own zoom.

Every area keeps its files in workspace/coords/<key>/ (key: the bounds and zoom),
so asking for the same area again downloads nothing.
"""

import errno
import hashlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app import progress as prog

Image.MAX_IMAGE_PIXELS = None

RADIUS = 6378137.0
TILE_PX = 256
SATELLITE_ID = 1               # Google Satellite in Universal Maps Downloader
STREET_ID = 66                 # Google Street Map (without labels)
MAX_PIXELS = 80_000_000        # one area's crop at most (as the OSM tool's exports)
MAX_TILES = 1600               # one downloader job at most (app/osm/satellite.py)
STREETS_ZOOM = 16              # the street map again at this zoom for the streets: Google draws no buildings on it
PREVIEW_PX = 1024

# the street map's road colours (Google's map style, PNG tiles so the colours are exact):
# the white of the streets, and the yellow to orange of the main roads and highways.
# Kept in workspace/coords/palette.json once written, so they can be tuned on real maps.
PALETTE = {
    "white_min": 252,               # every channel at least this: a street's white fill
    "yellow": {"r_min": 235, "g_min": 170, "b_max": 205, "r_minus_b": 40},
    "hole_m2": 25.0,                # holes in a road smaller than this filled (one-way arrows)
    "speck_m2": 10.0,               # road pieces smaller than this dropped
    "osm_reach_m": 15.0,            # with the PBF: road pieces with no OSM road this near dropped
    # Google's 2024 colours: streets blue-grey (188, 200, 212), main roads a darker blue-grey, land
    # light grey; "style": "auto" works out which a map is drawn in, "classic" or "new" sets it.
    # thin_m, thin_len_m: blue-grey lines narrower than thin_m and at least thin_len_m long are
    # footpaths, parking lanes and service passages, not streets (thin_m 0: kept)
    "style": "auto",
    # straight: the streets' edges straightened (Google's square caps, the notches of buildings
    # drawn over a street's edge, the steps where its pieces join), steps up to straight_share of
    # the streets' width deep (between straight_min_m and straight_max_m; straight_share 0: kept)
    # under_buildings: the streets under the buildings drawn over their edges filled back (0: left out)
    "new": {"b_minus_r": 8, "r_min": 120, "r_max": 214, "open_m": 0.6, "thin_m": 2.0, "thin_len_m": 6.0,
            "straight_share": 0.35, "straight_min_m": 1.0, "straight_max_m": 3.0, "under_buildings": 1},
}

COORDS_PLAN = [("index", 2.0), ("satellite", 10.0), ("street", 6.0), ("street_low", 0.5), ("buildings_map", 1.0), ("crop", 1.0),
               ("missing", 0.5), ("alpha", 1.0), ("osm", 1.0), ("buildings", 1.0), ("previews", 0.5)]


def ground_mpp(lat, zoom):
    """Metres on the ground a pixel of a Web Mercator tile image at this zoom and latitude."""
    return 2 * math.pi * RADIUS / TILE_PX / 2 ** zoom * math.cos(math.radians(lat))


def check_bounds(b):
    """(left, right, top, bottom) as floats, checked; ValueError with a plain message."""
    try:
        l, r, t, bt = (float(b[k]) for k in ("left", "right", "top", "bottom")) if isinstance(b, dict) else map(float, b)
    except (TypeError, ValueError, KeyError):
        raise ValueError("enter the four coordinates as numbers (decimal degrees)")
    if not all(math.isfinite(v) for v in (l, r, t, bt)):
        raise ValueError("enter the four coordinates as numbers (decimal degrees)")
    if not -180 <= l < r <= 180:
        raise ValueError("Left longitude must be less than Right longitude (-180 to 180)")
    if not -85.05 <= bt < t <= 85.05:
        raise ValueError("Bottom latitude must be less than Top latitude")
    return l, r, t, bt


def project(lon, lat):
    return RADIUS * math.radians(lon), RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def unproject(x, y):
    return math.degrees(x / RADIUS), math.degrees(2 * math.atan(math.exp(y / RADIUS)) - math.pi / 2)


def estimate(bounds, zoom):
    """What an area comes to at a zoom: its size on the ground, in pixels and in tiles, and whether it fits."""
    l, r, t, b = check_bounds(bounds)
    zoom = int(zoom)
    if not 15 <= zoom <= 22:
        raise ValueError("zoom 15 to 22")
    x0, y0 = project(l, b)
    x1, y1 = project(r, t)
    px = 2 * math.pi * RADIUS / TILE_PX / 2 ** zoom          # a pixel in Web Mercator metres
    w, h = int(math.ceil((x1 - x0) / px)), int(math.ceil((y1 - y0) / px))
    lat = (t + b) / 2
    mpp = ground_mpp(lat, zoom)
    step = px * TILE_PX
    tx = int(math.ceil((x1 + math.pi * RADIUS) / step)) - int(math.floor((x0 + math.pi * RADIUS) / step))
    ty = int(math.ceil((math.pi * RADIUS - y0) / step)) - int(math.floor((math.pi * RADIUS - y1) / step))
    tiles = max(1, tx) * max(1, ty)
    problems = []
    if w * h > MAX_PIXELS:
        problems.append(f"{w * h / 1e6:.0f} million pixels: at most {MAX_PIXELS // 1_000_000} million in one area "
                        f"(a smaller area or a lower zoom)")
    if tiles > MAX_TILES:
        problems.append(f"{tiles:,} tiles: at most {MAX_TILES:,} in one download (a smaller area or a lower zoom)")
    if w < 64 or h < 64:
        problems.append("under 64 pixels a side: a bigger area or a higher zoom")
    return {"bounds": [l, r, t, b], "zoom": zoom, "mpp": mpp, "width_px": w, "height_px": h,
            "width_m": w * mpp, "height_m": h * mpp, "km2": w * h * mpp * mpp / 1e6, "tiles": tiles,
            "ok": not problems, "problems": problems}


def area_key(bounds, zoom):
    l, r, t, b = check_bounds(bounds)
    return hashlib.sha1(json.dumps([round(l, 7), round(r, 7), round(t, 7), round(b, 7), int(zoom)]).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ settings

def osm_tool_dir():
    """Where the OSM tool keeps its preferences and road indexes on this computer."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    return Path(base) / "OSMOverlayTool"


def load_settings(root):
    """The downloader's path, the PBF's path and the zoom (saved once; the OSM tool's downloader path as a start)."""
    s = {"downloader": "", "pbf": "", "zoom": 20, "buildings_map": 0, "streets_zoom": STREETS_ZOOM}
    try:
        s.update(json.loads((Path(root) / "settings.json").read_text()))
    except (OSError, ValueError):
        pass
    if not s.get("downloader"):
        try:
            saved = json.loads((osm_tool_dir() / "satellite_settings.json").read_text())
            if str(saved.get("downloader", "")).lower().endswith("downloader.exe"):
                s["downloader"], s["downloader_from"] = saved["downloader"], "the OSM tool's settings"
        except (OSError, ValueError, AttributeError):
            pass
    if not s.get("pbf"):
        for f in sorted((osm_tool_dir() / "cache").glob("*.sqlite")) if (osm_tool_dir() / "cache").is_dir() else []:
            src = _index_source(f)
            if src and Path(src).exists():
                s["pbf"], s["pbf_from"] = src, "the OSM tool's road index"
                break
    return s


def save_settings(root, s):
    keep = {k: s[k] for k in ("downloader", "pbf", "zoom", "buildings_map", "streets_zoom") if k in s}
    Path(root).mkdir(parents=True, exist_ok=True)
    tmp = Path(root) / f".settings.{uuid.uuid4().hex[:6]}.tmp"
    tmp.write_text(json.dumps(keep, indent=2))
    os.replace(tmp, Path(root) / "settings.json")
    return load_settings(root)


def palette(root):
    p = dict(PALETTE)
    try:
        p.update(json.loads((Path(root) / "palette.json").read_text()))
    except (OSError, ValueError):
        pass
    return p


def _index_source(path):
    import sqlite3
    from contextlib import closing
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            info = json.loads(db.execute("SELECT value FROM metadata WHERE key='info'").fetchone()[0])
        return info["source"]["path"] if info.get("complete") else None
    except Exception:
        return None


def road_index(pbf, root, note=lambda s: None, cancelled=lambda: False):
    """The PBF's road index: the OSM tool's when it has a complete one for this file, else made once here."""
    from app.osm import core
    path, signature = core.cache_path(pbf, osm_tool_dir() / "cache")
    if core.cache_valid(path, signature):
        note("using the OSM tool's road index of this PBF")
        return path
    try:
        import google.protobuf  # noqa: F401  (the PBF reader's)
    except ImportError:
        raise ValueError("reading the PBF needs the protobuf package: in the program's folder run "
                         "python -m pip install -r requirements.txt, then start the server again")
    note("making the road index of the PBF (once; the first time it reads the whole file, several minutes)")
    try:
        return core.build_index(pbf, Path(root) / "osm_index", progress=note, cancelled=cancelled)
    except core.Cancelled as e:
        raise Cancelled(str(e))


# ------------------------------------------------------------------ the street map's roads

def _classic_road(rgb, pal):
    """Google's classic colours: the streets' white, the main roads' yellow to orange."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    wm = int(pal["white_min"])
    road = (r >= wm) & (g >= wm) & (b >= wm)
    y = pal["yellow"]
    road |= (r >= y["r_min"]) & (g >= y["g_min"]) & (b <= y["b_max"]) & (r - b >= y["r_minus_b"])
    return road


def _new_colour(rgb, pal):
    """Google's 2024 colours: the roads' blue-grey (bluer than red, not as light as the land or the buildings)."""
    p = {**PALETTE["new"], **(pal.get("new") or {})}
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return (b - r >= int(p["b_minus_r"])) & (r >= int(p["r_min"])) & (r <= int(p["r_max"])) & (b >= g) & (g >= r - 2)


def street_style(rgb, pal=PALETTE):
    """
    Which colours a street map is drawn in: "classic" (white streets, yellow main roads) or
    "new" (Google's 2024 colours: blue-grey streets), from its colours, unless the palette
    sets "style".
    """
    forced = pal.get("style", "auto")
    if forced in ("classic", "new"):
        return forced
    sample = np.ascontiguousarray(rgb[::4, ::4])
    white, blue = float(_classic_road(sample, pal).mean()), float(_new_colour(sample, pal).mean())
    return "new" if blue > max(0.02, 2.0 * white) else "classic"


def thin_lines(road, mpp, width_m, length_m, strip=1024):
    """
    The road's thin lines (bool, H x W): its parts narrower than width_m (where no disk that
    wide fits) running at least length_m (the corner of two streets, also too narrow for the
    disk, is a small piece, not a line). None when width_m is under 3 pixels (a map too
    coarse to tell a footpath from a street). Worked in strips of rows, the disk's reach
    added round each, so the memory stays small on a big map.
    """
    r = float(width_m) / mpp / 2
    if width_m <= 0 or 2 * r < 3:
        return None
    h = road.shape[0]
    m = int(math.ceil(2 * r)) + 2
    thin = np.zeros(road.shape, bool)
    for y0 in range(0, h, strip):
        a, b = max(0, y0 - m), min(h, y0 + strip + m)
        part = road[a:b]
        if not part.any():
            continue
        inner = ndi.distance_transform_edt(part) > r          # where a disk of radius r fits: its middles
        opened = ndi.distance_transform_edt(~inner) <= r if inner.any() else np.zeros(part.shape, bool)
        thin[y0:y0 + strip] = (part & ~opened)[y0 - a:y0 - a + min(strip, h - y0)]
    lab, n = ndi.label(thin, structure=np.ones((3, 3), bool))
    if not n:
        return thin
    sl = ndi.find_objects(lab)
    span = np.array([math.hypot(s[0].stop - s[0].start, s[1].stop - s[1].start) for s in sl]) * mpp
    line = np.zeros(n + 1, bool)
    line[1:] = span >= float(length_m)
    return line[lab]


# ------------------------------------------------------------------ the streets' edges straightened

STRAIGHT_NEAR_M = 0.25        # a point this near a piece's line is on it (and never under 1.5 pixels)
STRAIGHT_SHARE = 0.5          # a piece is straightened when at least this share of its points are on its line
STRAIGHT_CURVE_PX = 0.6       # ... and they do not bow away from it more than this: a curve, kept
STRAIGHT_CORNER_DEG = 30.0    # a corner straightened only where the two edges meet at this angle or more
STRAIGHT_RAMP = 0.35          # ... and its points off the line jump to a step's depth: if more than this share
                              # of them are only just off it (under twice the near distance), it drifts: a curve


def _steplike(r, near):
    """Whether the points off a line (offsets r) are a step's: most of them well off it, not drifting away."""
    off = np.abs(r) > near
    if not off.any():
        return True
    return float((off & (np.abs(r) < 2 * near)).sum()) / float(off.sum()) <= STRAIGHT_RAMP


def _dp(P, t):
    """Douglas-Peucker on an open polyline (n x 2): the indices of the points kept."""
    n = len(P)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        A, B = P[a], P[b]
        seg = P[a + 1:b]
        ab = B - A
        L = math.hypot(ab[0], ab[1])
        d = (np.hypot(seg[:, 0] - A[0], seg[:, 1] - A[1]) if L < 1e-9
             else np.abs(ab[0] * (seg[:, 1] - A[1]) - ab[1] * (seg[:, 0] - A[0])) / L)
        k = int(np.argmax(d))
        if d[k] > t:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.flatnonzero(keep)


def _piece_line(seg, near):
    """The line most of a piece's points lie on (within near): (point, unit direction, each point's offset)."""
    n = len(seg)
    qs = [int(round(q * (n - 1))) for q in (0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0)]
    best = None
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            p, q = seg[qs[i]], seg[qs[j]]
            v = q - p
            L = math.hypot(v[0], v[1])
            if L < 1e-6:
                continue
            v = v / L
            k = int((np.abs(v[0] * (seg[:, 1] - p[1]) - v[1] * (seg[:, 0] - p[0])) <= near).sum())
            if best is None or k > best[0]:
                best = (k, p, v)
    if best is None:
        return None
    _, p, v = best
    for _ in range(2):                      # then the least-squares line of the points on it
        r = v[0] * (seg[:, 1] - p[1]) - v[1] * (seg[:, 0] - p[0])
        on = seg[np.abs(r) <= near]
        if len(on) < 2:
            break
        c = on.mean(axis=0)
        _, _, vt = np.linalg.svd(on - c, full_matrices=False)
        p, v = c, vt[0] * (1 if np.dot(vt[0], v) >= 0 else -1)
    return p, v, v[0] * (seg[:, 1] - p[1]) - v[1] * (seg[:, 0] - p[0])


def _pieces(seg, t, near, Lmin, out):
    """
    A stretch of an edge (its points, in order) sorted into pieces, appended to out as
    [kind, points, line point, line direction, straight already]: "line" when it is a
    straight edge with steps no deeper than t, both its ends on its line and not bowing like
    a curve; "raw" otherwise. One whose end runs off its line (it turns) is split where it
    turns; a long one that fits badly, where it fits worst.
    """
    A, B = seg[0], seg[-1]
    L = math.hypot(B[0] - A[0], B[1] - A[1])
    if L >= Lmin and len(seg) >= 4:
        fit = _piece_line(seg, near)
        if fit is not None:
            p, v, r = fit
            on = np.abs(r) <= near
            # both ends on the line: a step comes back to it; a bend (the street turning) does not
            ok = on.mean() >= STRAIGHT_SHARE and np.abs(r).max() <= t and on[0] and on[-1] and _steplike(r, near)
            exact = ok and bool(np.abs(r).max() <= near)
            if ok and not exact and on.sum() >= 6:
                s = (seg - p) @ v
                span = float(s.max() - s.min())
                ok = abs(np.polyfit(s[on], r[on], 2)[0]) * span * span / 4 <= max(STRAIGHT_CURVE_PX, 0.5 * near)
            if ok:
                out.append(["line", seg, p, v, exact])
                return
            m = None
            if not on[-1] and on.any():
                m = int(np.flatnonzero(on)[-1])          # it turns at its end: from the last point on the line
            elif not on[0] and on.any():
                m = int(np.flatnonzero(on)[0])
            elif L >= 2 * Lmin:
                m = int(np.argmax(np.abs(r)))
            if m is not None and 2 <= m <= len(seg) - 3:
                _pieces(seg[:m + 1], t, near, Lmin, out)
                _pieces(seg[m:], t, near, Lmin, out)
                return
    out.append(["raw", seg, None, None, False])


def _cross(p1, v1, p2, v2, min_deg=15.0):
    """Where two lines meet, or None when they are within min_deg of parallel."""
    det = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(det) < math.sin(math.radians(min_deg)):
        return None
    d = p2 - p1
    return p1 + v1 * ((d[0] * v2[1] - d[1] * v2[0]) / det)


def _square(seg, dirs, near, deg=15.0):
    """The share of a stretch's length running along or across the given directions (a cap or a notch is square; a curve sweeps)."""
    w = max(2, int(round(2 * near)))
    if len(seg) <= w:
        return 1.0
    d = seg[w:] - seg[:-w]
    L = np.hypot(d[:, 0], d[:, 1])
    ok = L > 1e-9
    if not ok.any():
        return 1.0
    u = d[ok] / L[ok, None]
    c = math.cos(math.radians(deg))
    along = np.zeros(len(u), bool)
    for v in dirs:
        along |= np.abs(u @ v) >= c
    return float(L[ok][along].sum() / L[ok].sum())


def _corners(pcs, t, near):
    """
    Short raw stretches between two straight edges (one stretch or a few in a row), marked
    in place. Where the edges meet at an angle (Google's square cap or a notch at a junction's
    corner): "corner", with the point where their lines meet, when the stretch's points all
    lie within t of the two lines, that point is near it, and it is square, not a curve (a
    rounded corner stays). Where they are one line (a step cut in two): "bridge", the edge
    carried straight across, when the stretch lies within t of it.
    """
    n = len(pcs)
    if n < 3 or all(pc[0] == "raw" for pc in pcs):
        return
    start = next(j for j in range(n) if pcs[j][0] == "line")
    j = 0
    while j < n:
        jj = (start + j) % n
        if pcs[jj][0] != "raw":
            j += 1
            continue
        run = []
        while j < n and pcs[(start + j) % n][0] == "raw":
            run.append((start + j) % n)
            j += 1
        i, k = (run[0] - 1) % n, (run[-1] + 1) % n
        if pcs[i][0] != "line" or pcs[k][0] != "line" or i == k:
            continue
        seg = np.vstack([pcs[r][1] for r in run])
        p1, v1, p2, v2 = pcs[i][2], pcs[i][3], pcs[k][2], pcs[k][3]
        d1 = np.abs(v1[0] * (seg[:, 1] - p1[1]) - v1[1] * (seg[:, 0] - p1[0]))
        d2 = np.abs(v2[0] * (seg[:, 1] - p2[1]) - v2[1] * (seg[:, 0] - p2[0]))
        X = _cross(p1, v1, p2, v2, STRAIGHT_CORNER_DEG)
        if X is None:
            if _cross(p1, v1, p2, v2) is not None:
                continue                         # a shallow turn (a curve meeting a straight): kept
            # one line: the second edge's line on the first's, and the stretch within t of it
            gap = abs(v1[0] * (p2[1] - p1[1]) - v1[1] * (p2[0] - p1[0]))
            if gap <= 2 * near and d1.max() <= t and _steplike(d1, near):
                for r in run:
                    pcs[r] = ["bridge", pcs[r][1], None, None, False]
            continue
        if np.minimum(d1, d2).max() > t or np.hypot(seg[:, 0] - X[0], seg[:, 1] - X[1]).min() > t:
            continue
        # judged on the part off both lines: a cap's or a notch's sides run along or across them, a
        # rounded corner's arc sweeps round
        off = np.flatnonzero(np.minimum(d1, d2) > near)
        perp = lambda v: np.array([-v[1], v[0]])
        if len(off) and _square(seg[max(0, off[0] - 1):off[-1] + 2], (v1, perp(v1), v2, perp(v2)), near) < 0.75:
            continue
        for q, r in enumerate(run):
            pcs[r] = ["corner", pcs[r][1], X if q == 0 else None, None, False]


def _assemble(pcs):
    """The ring's new points from its pieces, and whether anything changed."""
    out, changed = [], False
    n = len(pcs)
    for j, (kind, seg, p, v, exact) in enumerate(pcs):
        prev = pcs[(j - 1) % n][0]
        if kind == "corner":
            if p is not None:
                out.append(p)
            changed = True
        elif kind == "bridge":
            changed = True
        elif kind == "raw" or exact:
            out.extend(seg[:-1] if not (exact and prev == "corner") else seg[1:-1])
        else:
            A, B = seg[0], seg[-1]
            Ap, Bp = p + v * np.dot(A - p, v), p + v * np.dot(B - p, v)
            if prev != "corner":
                out.append(A)
            if prev == "corner" or math.hypot(*(Ap - A)) > 0.5:
                out.append(Ap)
            nxt = pcs[(j + 1) % n][0]
            if nxt in ("corner", "bridge") or math.hypot(*(Bp - B)) > 0.5:
                out.append(Bp)
            changed = True
    return out, changed


def _one_line(seg, t, near, Lmin):
    """Whether a stretch of an edge is one straight line with steps no deeper than t (both its ends on it)."""
    if len(seg) < 4 or math.hypot(*(seg[-1] - seg[0])) < Lmin:
        return False
    fit = _piece_line(seg, near)
    if fit is None:
        return False
    on = np.abs(fit[2]) <= near
    return bool(on[0] and on[-1] and on.mean() >= STRAIGHT_SHARE and np.abs(fit[2]).max() <= t
                and _steplike(fit[2], near))


def _ring_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def scanfill(rings, shape, origin=(0, 0)):
    """
    The even-odd fill of closed rings ((n, 2) arrays of x, y, pixel middles at whole
    numbers): bool (h, w), its top left at origin. A pixel is in when its middle is, so the
    outlines traced from a mask fill back to exactly that mask.
    """
    h, w = shape
    ox, oy = origin
    x0 = np.concatenate([np.asarray(R, float)[:, 0] for R in rings]) - ox
    y0 = np.concatenate([np.asarray(R, float)[:, 1] for R in rings]) - oy
    x1 = np.concatenate([np.roll(np.asarray(R, float)[:, 0], -1) for R in rings]) - ox
    y1 = np.concatenate([np.roll(np.asarray(R, float)[:, 1], -1) for R in rings]) - oy
    lo = np.clip(np.ceil(np.minimum(y0, y1)), 0, h)          # each edge crosses rows lo .. hi - 1
    hi = np.clip(np.ceil(np.maximum(y0, y1)), 0, h)
    n = (hi - lo).astype(np.int64)
    keep = np.flatnonzero(n > 0)
    if not len(keep):
        return np.zeros(shape, bool)
    e = np.repeat(keep, n[keep])
    first = np.repeat(np.cumsum(n[keep]) - n[keep], n[keep])
    row = (lo[e] + (np.arange(len(e)) - first)).astype(np.int64)
    x = x0[e] + (row - y0[e]) / (y1[e] - y0[e]) * (x1[e] - x0[e])
    col = np.clip(np.ceil(x), 0, w).astype(np.int64)
    flip = np.zeros((h, w + 1), np.uint8)
    np.bitwise_xor.at(flip, (row, col), 1)
    return np.bitwise_xor.accumulate(flip, axis=1)[:, :w].astype(bool)


def straighten_edges(road, mpp, share=0.35, min_m=1.0, max_m=3.0):
    """
    The streets' edges straightened (bool H x W in, a new one out, and notes). Google's
    street map has steps along its streets: square caps where side streets end against a
    road, notches where buildings are drawn over a street's edge, steps where a street's
    pieces join. Each edge, traced between its pixels, is cut into pieces (Douglas-Peucker,
    steps up to t deep in one piece; t: share of the streets' width, between min_m and
    max_m); a piece whose points mostly lie on one line (at least half within
    STRAIGHT_NEAR_M) and that does not bow like a curve becomes that line; a gentle bend is
    split where it fits worst and its halves tried. Corners, junctions, curves (a
    roundabout), and anything deeper than t (a side street, a dead end) stay. Only the
    pixels between an edge and its straight line change; an outline that would lose half its
    area (a thin gap between two carriageways) is left as it was.
    """
    from skimage import measure
    H, W = road.shape
    edge = int((road ^ ndi.binary_erosion(road)).sum())
    width_px = 2.0 * float(road.sum()) / max(1, edge)
    t = min(max_m, max(min_m, share * width_px * mpp)) / mpp
    notes = {"straight_width_m": round(width_px * mpp, 1), "straight_depth_m": round(t * mpp, 2)}
    if share <= 0 or t < 3 or not road.any():
        return road, notes
    near = max(1.5, STRAIGHT_NEAR_M / mpp)
    Lmin = 2.5 * t
    out = road.copy()
    n_px = n_rings = 0
    for ring in measure.find_contours(np.pad(road, 1).astype(np.uint8), 0.5):
        if len(ring) < 8 or not np.allclose(ring[0], ring[-1]):
            continue
        P = ring[:-1, ::-1] - 1.0                              # (x, y), pixel middles at whole numbers
        far = int(np.argmax(np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])))
        Q = np.vstack([P, P[:1]])
        idx = list(np.union1d(_dp(Q[:far + 1], t), far + _dp(Q[far:], t)))
        # a cut that fell on a step's own corner: the two pieces joined when, together, they are one
        # straight edge (both ends on its line); a corner or a turn stays
        k = 1
        while k < len(idx) - 1:
            if _one_line(Q[idx[k - 1]:idx[k + 1] + 1], t, near, Lmin):
                del idx[k]
            else:
                k += 1
        pcs = []
        for a, b in zip(idx[:-1], idx[1:]):
            _pieces(Q[a:b + 1], t, near, Lmin, pcs)
        _corners(pcs, t, near)
        new, changed = _assemble(pcs)
        if not changed:
            continue
        N = np.array(new)
        a0 = _ring_area(P)
        if abs(_ring_area(N) - a0) > 0.5 * abs(a0):
            continue
        x0 = max(0, int(math.floor(min(P[:, 0].min(), N[:, 0].min()))) - 1)
        y0 = max(0, int(math.floor(min(P[:, 1].min(), N[:, 1].min()))) - 1)
        x1 = min(W, int(math.ceil(max(P[:, 0].max(), N[:, 0].max()))) + 2)
        y1 = min(H, int(math.ceil(max(P[:, 1].max(), N[:, 1].max()))) + 2)
        flip = scanfill([P], (y1 - y0, x1 - x0), (x0, y0)) ^ scanfill([N], (y1 - y0, x1 - x0), (x0, y0))
        out[y0:y1, x0:x1] ^= flip
        n_px += int(flip.sum())
        n_rings += 1
    notes.update(straightened_m2=round(n_px * mpp * mpp, 1), straightened_outlines=n_rings)
    return out, notes


# ------------------------------------------------------------------ the streets under the buildings drawn over them

UNDER_SHARE = 0.6             # the edge where it leaves its line: at least this share of it along a building
UNDER_DEPTH = 0.8             # ... and no deeper than this share of the street's width
UNDER_BEND_DEG = 4.0          # an edge on both sides of a building: its two lines within this angle of each other
UNDER_CORNER_DEG = 30.0       # two streets' edges meeting at a corner: at this angle or more (and 180 less it)


def _line_fit(seg):
    """The least-squares line of a straight piece's points: (point, unit direction along the piece)."""
    c = seg.mean(axis=0)
    v = np.linalg.svd(seg - c, full_matrices=False)[2][0]
    return c, (v if np.dot(v, seg[-1] - seg[0]) >= 0 else -v)


def _offsets(P, p, v):
    """Each point's distance from the line (p, v), + to its left."""
    return v[0] * (P[:, 1] - p[1]) - v[1] * (P[:, 0] - p[0])


def _road_left(road, seg, v, reach):
    """Whether the road lies to the left of an edge's piece (a few of its points looked past, the most of them)."""
    H, W = road.shape
    n = np.array([-v[1], v[0]])
    votes = 0
    for q in (0.25, 0.5, 0.75):
        x, y = seg[int(q * (len(seg) - 1))] + n * reach
        xi, yi = int(round(x)), int(round(y))
        votes += 1 if 0 <= xi < W and 0 <= yi < H and road[yi, xi] else -1
    return votes > 0


def _reach(road, p, n, most):
    """How far (pixels) the road runs from p along n (the street's width, from its edge)."""
    H, W = road.shape
    for k in range(1, int(most)):
        x, y = int(round(p[0] + n[0] * k)), int(round(p[1] + n[1] * k))
        if not (0 <= x < W and 0 <= y < H) or not road[y, x]:
            return k
    return int(most)


def under_buildings(road, bld, mpp, anchor_m=4.0, max_m=150.0, max_depth_m=15.0):
    """
    The streets under the buildings Google draws over them (bool H x W in, a new one out,
    and notes). At zoom 20 the street map draws buildings over streets' edges (the shops
    along a main road, a building over a junction's corner): the road under them is not
    drawn, so the alpha has bites; zoomed out, Google draws no buildings and the street
    shows whole. Each road edge, traced between its pixels, is cut into straight pieces
    (Douglas-Peucker); where the edge leaves a straight piece (at least anchor_m long) into
    the road, along a building (bld: the buildings' fill pixels) for at least UNDER_SHARE
    of the way, and comes back to the same line (another piece on it), the road between is
    filled up to that line; where it comes to another street's edge instead (a building over
    a junction's corner), up to the two lines, round the corner where they meet. Bites up to
    max_m long and UNDER_DEPTH of the street's width deep (and max_depth_m); only pixels are
    added, and an edge with no building over it is left as it was.
    """
    from skimage import measure
    H, W = road.shape
    notes = {"under_buildings": 0, "under_buildings_m2": 0.0}
    if not road.any() or not bld.any():
        return road, notes
    near = max(1.5, STRAIGHT_NEAR_M / mpp)
    tol = max(2.0, 0.3 / mpp)                 # two pieces on one line: this near it (and 2% of the gap between)
    out_tol = max(2.0, 0.5 / mpp)             # the edge between them may stray this far out of the road
    A = anchor_m / mpp
    cos_bend, cos_corner = math.cos(math.radians(UNDER_BEND_DEG)), math.cos(math.radians(UNDER_CORNER_DEG))
    over = ndi.binary_dilation(bld, iterations=max(1, int(round(0.4 / mpp)))) & ~road    # a building and its outline
    out = road.copy()
    n_px = n_bites = 0
    for ring in measure.find_contours(np.pad(road, 1).astype(np.uint8), 0.5):
        if len(ring) < 16 or not np.allclose(ring[0], ring[-1]):
            continue
        P = ring[:-1, ::-1] - 1.0                              # (x, y), pixel middles at whole numbers
        n = len(P)
        x0, y0 = np.clip(np.floor(P[:, 0]).astype(int), 0, W - 1), np.clip(np.floor(P[:, 1]).astype(int), 0, H - 1)
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
        cov = over[y0, x0] | over[y0, x1] | over[y1, x0] | over[y1, x1]
        if not cov.any() or cov.all():
            continue
        s0 = int(np.flatnonzero(~cov)[0])                      # the ring from a point with no building on it
        P, cov = np.roll(P, -s0, axis=0), np.roll(cov, -s0)
        far = int(np.argmax(np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])))
        Q = np.vstack([P, P[:1]])
        C2 = np.concatenate([cov, cov, cov[:1]])
        idx = np.union1d(_dp(Q[:far + 1], near), far + _dp(Q[far:], near))
        lines = []
        for a, b in zip(idx[:-1], idx[1:]):
            if math.hypot(*(Q[b] - Q[a])) >= A:
                p, v = _line_fit(Q[a:b + 1])
                lines.append((a, b, p, v, 1.0 if _road_left(road, Q[a:b + 1], v, 3.0) else -1.0,
                              float(C2[a:b + 1].mean())))       # how much of it runs along a building
        if len(lines) < 2:
            continue
        L2 = lines + [(a + n, b + n, p, v, s, g) for a, b, p, v, s, g in lines]
        Q2 = np.vstack([P, P, P[:1]])
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q2, axis=0).T))])
        done = -1
        for li in range(len(lines)):
            a, b, p, v, side, _ = lines[li]
            if b <= done:
                continue
            nr = np.array([-v[1], v[0]]) * side                # into the road
            deep = min(max_depth_m / mpp, UNDER_DEPTH * _reach(road, Q[(a + b) // 2] + nr, nr, 60.0 / mpp))
            # the edge's next piece back on this line (past the buildings over it), or the next clear edge
            # at a corner; the pieces along a building on the way (its sides, its front) passed by
            for lk in range(li + 1, li + len(lines)):
                a2, b2, p2, v2, side2, along2 = L2[lk]
                gap = arc[a2] - arc[b]
                if gap > max_m / mpp:
                    break
                c = float(np.dot(v, v2))
                kind = None
                if c >= cos_bend:
                    off = max(np.abs(_offsets(Q2[[a2, b2]], p, v)).max(), np.abs(_offsets(Q2[[a, b]], p2, v2)).max())
                    if off <= tol + 0.02 * gap:
                        kind = "line"
                if kind is None:
                    if along2 >= 0.5:
                        continue
                    if not -cos_corner <= c <= cos_corner:
                        break                                   # a clear edge going on elsewhere: no bite
                    kind = "corner"
                seg = Q2[b:a2 + 1]
                if ((seg[:, 0] < 0) | (seg[:, 1] < 0) | (seg[:, 0] > W - 1) | (seg[:, 1] > H - 1)).any():
                    break                                       # it runs off the map: not judged
                d = _offsets(seg, p, v) * side                  # + into the road
                poly = clip = None
                if kind == "line":
                    # a building (or a row of them) over the edge between: filled up to the line
                    inside = d > near
                    if (d.min() >= -out_tol and d.max() <= deep and inside.sum() >= 3
                            and C2[b:a2 + 1][inside].mean() >= UNDER_SHARE):
                        poly = seg
                else:
                    # a building over the junction's corner: filled up to both lines, round to where they meet
                    X = _cross(p, v, p2, v2, UNDER_CORNER_DEG)
                    e = np.maximum(d, _offsets(seg, p2, v2) * side2)
                    inside = e > near
                    if X is not None and inside.sum() >= 3:
                        bend = np.hypot(seg[:, 0] - X[0], seg[:, 1] - X[1]) <= deep    # a rounded corner's arc
                        if (e.max() <= deep and np.all((e >= -out_tol) | bend)
                                and C2[b:a2 + 1][inside].mean() >= UNDER_SHARE):
                            poly = np.vstack([seg, X[None]])
                            clip = (p, v, side, p2, v2, side2)
                if poly is not None:
                    bx0, by0 = max(0, int(math.floor(poly[:, 0].min())) - 1), max(0, int(math.floor(poly[:, 1].min())) - 1)
                    bx1, by1 = min(W, int(math.ceil(poly[:, 0].max())) + 2), min(H, int(math.ceil(poly[:, 1].max())) + 2)
                    f = scanfill([poly], (by1 - by0, bx1 - bx0), (bx0, by0)) & ~out[by0:by1, bx0:bx1]
                    if clip is not None:                # the corner's rounding kept: the road's side of either line
                        gy, gx = np.mgrid[by0:by1, bx0:bx1]
                        g = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
                        pa, va, sa, pb, vb, sb = clip
                        f &= ((_offsets(g, pa, va) * sa >= 0) | (_offsets(g, pb, vb) * sb >= 0)).reshape(f.shape)
                    # what is filled is mostly the building itself (a curve along a building's wall is not)
                    if f.any() and over[by0:by1, bx0:bx1][f].mean() >= 0.5:
                        out[by0:by1, bx0:bx1] |= f
                        n_px += int(f.sum())
                        n_bites += 1
                        done = a2
                break
    notes.update(under_buildings=n_bites, under_buildings_m2=round(n_px * mpp * mpp, 1))
    return out, notes


def _tidy(road, mpp, pal, osm_lines, notes):
    """
    An alpha's small holes filled (the one-way arrows drawn on a street) and its small specks
    dropped; with osm_lines (bool, the OSM roads drawn 1 px wide) the road pieces with no OSM
    road within osm_reach_m dropped too. Counts in notes.
    """
    px_m2 = mpp * mpp
    # holes: what is not road and does not reach the edge, small
    holes = ~road
    lab, n = ndi.label(holes)
    if n:
        sizes = ndi.sum(holes, lab, index=np.arange(1, n + 1))
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        small = (sizes * px_m2 < float(pal["hole_m2"]))
        small[edge[edge > 0] - 1] = False
        fill = np.zeros(n + 1, bool)
        fill[1:] = small
        road |= fill[lab]
        notes["holes_filled"] = int(small.sum())
    lab, n = ndi.label(road)
    if n:
        sizes = ndi.sum(road, lab, index=np.arange(1, n + 1))
        keep = sizes * px_m2 >= float(pal["speck_m2"])
        if osm_lines is not None and osm_lines.any():
            near = ndi.binary_dilation(osm_lines, iterations=max(1, int(round(float(pal["osm_reach_m"]) / mpp))))
            touched = np.zeros(n + 1, bool)
            touched[np.unique(lab[near & road])] = True
            far = keep & ~touched[1:]
            notes["far_from_osm"] = int(far.sum())
            notes["far_from_osm_m2"] = round(float(sizes[far].sum() * px_m2), 1)
            keep &= touched[1:]
        notes["specks_dropped"] = int((sizes * px_m2 < float(pal["speck_m2"])).sum())
        k = np.zeros(n + 1, bool)
        k[1:] = keep
        road = k[lab]
    return road


def street_alpha(rgb, mpp, pal=PALETTE, osm_lines=None):
    """
    The road (bool, H x W) in a street map (H x W x 3 uint8): its white and its yellow to
    orange (Google's classic colours) or its blue-grey (the 2024 colours, street_style:
    the building outlines, as blue as the roads but a pixel or two wide, taken off, and the
    thin lines: footpaths, parking lanes, service passages; the streets under the buildings
    drawn over their edges filled back, and the streets' edges straightened), small holes
    filled (the one-way arrows drawn on a street), small specks dropped; with
    osm_lines (bool, the OSM roads drawn 1 px wide) the road pieces with no OSM road within
    osm_reach_m dropped too. Returns (road, notes).
    """
    style = street_style(rgb, pal)
    if style == "new":
        p = {**PALETTE["new"], **(pal.get("new") or {})}
        road = _new_colour(rgb, pal)
        k = max(1, int(round(float(p["open_m"]) / mpp / 2)))
        road = ndi.binary_opening(road, structure=np.ones((2 * k + 1, 2 * k + 1), bool))
        thin = thin_lines(road, mpp, float(p["thin_m"]), float(p["thin_len_m"]))
    else:
        road = _classic_road(rgb, pal)
        thin = None
    notes = {"coloured_px": int(road.sum()), "style": style}
    px_m2 = mpp * mpp
    if thin is not None and thin.any():
        road &= ~thin
        notes["thin_lines"] = int(ndi.label(thin, structure=np.ones((3, 3), bool))[1])
        notes["thin_m2"] = round(float(thin.sum()) * px_m2, 1)
    road = _tidy(road, mpp, pal, osm_lines, notes)
    if style == "new":
        p = {**PALETTE["new"], **(pal.get("new") or {})}
        if float(p.get("under_buildings", 1)):
            from app import footprints as FP
            fills, _, tol, _ = FP.palette(pal, "new")
            road, un = under_buildings(road, FP.fill_mask(rgb, fills, tol) > 0, mpp)
            notes.update(un)
        road, sn = straighten_edges(road, mpp, float(p["straight_share"]), float(p["straight_min_m"]),
                                    float(p["straight_max_m"]))
        notes.update(sn)
    notes["road_px"] = int(road.sum())
    return road, notes


# ------------------------------------------------------------------ the streets from the street map at a low zoom

def _low_colours(rgb, road_at, land_at):
    """
    The low zoom map's street colours (up to 4, each at least 8% of the pixels under the
    area's own streets: a main road may be drawn in another shade) and its land's colour (the
    commonest far from any street), or None when there are not enough of either.
    """
    on, off = rgb[road_at].astype(np.int32), rgb[land_at].astype(np.int32)
    if len(on) < 20 or len(off) < 20:
        return None
    cols, counts = np.unique(off, axis=0, return_counts=True)
    land = cols[int(np.argmax(counts))].astype(np.float64)
    cols, counts = np.unique(on, axis=0, return_counts=True)
    roads = []
    for k in np.argsort(-counts):
        c = cols[k].astype(np.float64)
        near = [r for r in roads if np.abs(r[0] - c).max() <= 8]
        if near:
            near[0][1] += counts[k]
        elif len(roads) < 8:
            roads.append([c, int(counts[k])])
    total = float(len(on))
    roads = [r[0] for r in sorted(roads, key=lambda r: -r[1]) if r[1] >= 0.08 * total and np.abs(r[0] - land).max() >= 10][:4]
    return (roads, land) if roads else None


def low_streets(img, geo, extent, size, road, mpp, pal=PALETTE, osm_lines=None):
    """
    The streets of the street map downloaded at a low zoom (STREETS_ZOOM, where Google draws
    no buildings, so no street is cut by one), on the area's grid (extent, size). Its colours
    are learnt from road (the alpha of the area's own street map: Google draws the low zoom in
    other shades): the colours under its streets and the land's far from them. Each low zoom
    pixel's share of street is where its colour lies between the land's and a street's (the
    anti-aliased pixels at an edge say where in them the edge falls); the shares are
    stretched smoothly onto the grid and cut at a half, so the edges come out smooth and in
    place, not in 2 m steps. Streets the low zoom leaves out (the smallest) are taken from
    road, holes filled, specks dropped, and the edges straightened. Returns (road, notes), or
    (None, notes) when no street colour is found.
    """
    rgb = np.asarray(img.convert("RGB"))
    sh, sw = rgb.shape[:2]
    ex0, ey0, ex1, ey1 = geo["extent"]
    sx, sy = (ex1 - ex0) / geo["source_size"][0], (ey1 - ey0) / geo["source_size"][1]
    x0, y0, x1, y1 = extent
    w, h = size
    px = (x1 - x0) / w
    notes = {"streets_zoom_mpp": round(sx / px * mpp, 2)}
    # where each low zoom pixel's middle falls on a coarse copy of the area's alpha (4 m a cell at most)
    f = max(1, int(round(min(4.0, max(0.5, sx / px * mpp / 2)) / mpp)))
    small = road[::f, ::f]
    inside = ndi.distance_transform_edt(small) * f * mpp
    outside = ndi.distance_transform_edt(~small) * f * mpp
    gx = np.floor((ex0 + (np.arange(sw) + 0.5) * sx - x0) / px / f).astype(int)
    gy = np.floor((y1 - (ey1 - (np.arange(sh) + 0.5) * sy)) / px / f).astype(int)
    ok = (gx >= 0) & (gx < small.shape[1])
    okr = (gy >= 0) & (gy < small.shape[0])
    road_at = np.zeros((sh, sw), bool)
    land_at = np.zeros((sh, sw), bool)
    sub = np.ix_(np.flatnonzero(okr), np.flatnonzero(ok))
    cell = (gy[okr][:, None], gx[ok][None, :])
    road_at[sub] = inside[cell] >= 1.5                       # well inside a street
    for away in (15.0, 8.0, 4.0):                            # well away from any (a close-built town: less far)
        land_at[sub] = outside[cell] >= away
        if land_at.sum() >= 20:
            break
    found = _low_colours(rgb, road_at, land_at)
    if found is None:
        notes["streets_low"] = "no street colour found"
        return None, notes
    roads, land = found
    notes["streets_low_colours"] = [[int(v) for v in c] for c in roads]
    notes["streets_low_land"] = [int(v) for v in land]
    c = rgb.astype(np.float32) - land.astype(np.float32)
    share = np.zeros((sh, sw), np.float32)
    for col in roads:
        d = (col - land).astype(np.float32)
        dd = float(d @ d)
        t = (c @ d) / dd
        off = np.sqrt(np.maximum(0.0, (c * c).sum(axis=2) - t * t * dd))
        share = np.maximum(share, np.where(off <= max(8.0, 0.2 * math.sqrt(dd)), np.clip(t, 0, 1), 0).astype(np.float32))
    grid = np.asarray(_onto(Image.fromarray(share, "F"), geo, extent, size, Image.Resampling.BICUBIC))
    low = grid > 0.5
    del grid
    if not low.any():
        notes["streets_low"] = "no street on the low zoom map"
        return None, notes
    # the area's own streets that the low zoom leaves out: what lies 3 m and more from its streets, with the
    # stretch that joins it to them
    g = max(1, int(round(0.5 / mpp)))
    lo = low[::g, ::g]
    far = road[::g, ::g][:lo.shape[0], :lo.shape[1]] & ~lo & (ndi.distance_transform_edt(~lo) * g * mpp >= 3.0)
    added = 0
    if far.any():
        reach = ndi.distance_transform_edt(~far) * g * mpp <= 4.0
        yy, xx = np.nonzero(road & ~low)
        take = reach[np.minimum(yy // g, reach.shape[0] - 1), np.minimum(xx // g, reach.shape[1] - 1)]
        low[yy[take], xx[take]] = True
        added = int(take.sum())
        notes["streets_from_area_m2"] = round(added * mpp * mpp, 1)
    both, either = int((low & road).sum()), int((low | road).sum())
    notes["streets_agree"] = round(both / max(1, either), 3)
    notes["coloured_px"] = int(low.sum())
    low = _tidy(low, mpp, pal, osm_lines, notes)
    p = {**PALETTE["new"], **(pal.get("new") or {})}
    low, sn = straighten_edges(low, mpp, float(p["straight_share"]), float(p["straight_min_m"]), float(p["straight_max_m"]))
    notes.update(sn)
    notes["road_px"] = int(low.sum())
    return low, notes


# ------------------------------------------------------------------ rasters

def read_geo(path):
    """A downloaded raster's georeferencing (app/osm/satellite.py): crs, extent (x0, y0, x1, y1), size."""
    from app.osm import satellite as SAT
    with Image.open(path) as im:
        geo = SAT.raster_geo(im, Path(path))
    if not geo:
        raise ValueError(f"{Path(path).name} has no georeferencing the program can read (a GeoTIFF in metres, "
                         f"or a world file with a .prj): export it from the Map Combiner as TIFF in metres")
    if geo["crs"] != 3857:
        raise ValueError(f"{Path(path).name} is not in Web Mercator metres (EPSG:3857)")
    return geo


def _write(img, path, optional=False, **kw):
    """
    Save one of the area's pictures. Windows will not write over a file another program
    has open (an image viewer, Paint: "[Errno 22] Invalid argument" or "Permission
    denied"): then a plain message, or for a preview (optional) just False.
    """
    try:
        img.save(path, **kw)
        return True
    except OSError as e:
        if isinstance(e, PermissionError) or e.errno in (errno.EINVAL, errno.EACCES):
            if optional:
                return False
            raise ValueError(f"{Path(path).name} could not be written: it is probably open in another program "
                             f"(Paint, Photos, an image viewer). Close it there and press Download and make the alpha "
                             f"again ({e.strerror or e})") from e
        raise


def _kept(png, sources, size):
    """A picture cut before from the same downloads (newer than they are, the same size): read, not written again."""
    try:
        if png.stat().st_mtime < max(Path(s).stat().st_mtime for s in sources):
            return None
        with Image.open(png) as im:
            if im.size != tuple(size):
                return None
            return im.convert("RGB")
    except OSError:
        return None


def crop_box(geo, bounds):
    """The source pixels (x0, y0, x1, y1) covering the area, snapped to the source's pixel grid, and their extent."""
    l, r, t, b = bounds
    ex0, ey0, ex1, ey1 = geo["extent"]
    sw, sh = geo["source_size"]
    sx, sy = (ex1 - ex0) / sw, (ey1 - ey0) / sh
    x0, y0 = project(l, b)
    x1, y1 = project(r, t)
    c0, c1 = int(math.floor((x0 - ex0) / sx + 1e-6)), int(math.ceil((x1 - ex0) / sx - 1e-6))
    r0, r1 = int(math.floor((ey1 - y1) / sy + 1e-6)), int(math.ceil((ey1 - y0) / sy - 1e-6))
    if c0 < 0 or r0 < 0 or c1 > sw or r1 > sh:
        raise ValueError("the downloaded map does not cover the whole area (the downloader returned less): "
                         "download again, or check the downloader's log in the area's folder")
    return (c0, r0, c1, r1), (ex0 + c0 * sx, ey1 - r1 * sy, ex0 + c1 * sx, ey1 - r0 * sy)


def onto_grid(path, geo, extent, size, resample=Image.Resampling.NEAREST):
    """Another raster (the street map) cut to the satellite crop's grid: extent (x0, y0, x1, y1), size (w, h)."""
    with Image.open(path) as im:
        return _onto(im.convert("RGB"), geo, extent, size, resample)


def _onto(img, geo, extent, size, resample):
    """A georeferenced picture (geo: its extent and size) cut and scaled to the area's grid."""
    ex0, ey0, ex1, ey1 = geo["extent"]
    sw, sh = geo["source_size"]
    sx, sy = (ex1 - ex0) / sw, (ey1 - ey0) / sh
    x0, y0, x1, y1 = extent
    w, h = size
    if x0 < ex0 - sx or y0 < ey0 - sy or x1 > ex1 + sx or y1 > ey1 + sy:
        raise ValueError("the street map does not cover the whole area: download it again")
    a, e = (x1 - x0) / w / sx, (y1 - y0) / h / sy
    # Pillow takes each output pixel's middle to the source and samples it there (pixel middles at +0.5 on
    # both sides): the grid's edge, in source pixels, is the whole offset (a map stretched 16 times, as the
    # zoom 16 street map, lands where it should)
    c, f = (x0 - ex0) / sx, (ey1 - y1) / sy
    return img.transform((w, h), Image.Transform.AFFINE, (a, 0, c, 0, e, f), resample)


def missing_tiles(rgb, extent, zoom):
    """
    The satellite's tiles that did not come: whole tile squares (of the downloader's
    256 px tiles, on the Web Mercator tile grid) of one flat colour, white or black
    (96% of the tile within 8 levels, JPEG's ringing allowed). A white roof is not a
    whole 35 m tile of one flat colour. Returns [[x0, y0, x1, y1]] in
    the crop's pixels.
    """
    h, w = rgb.shape[:2]
    x0, _, x1, y1 = extent
    px = (x1 - x0) / w
    step = 2 * math.pi * RADIUS / 2 ** zoom
    k0 = int(math.floor((x0 + math.pi * RADIUS) / step))
    j0 = int(math.floor((math.pi * RADIUS - y1) / step))
    out = []
    grey = rgb.astype(np.int16)
    for j in range(j0, j0 + h // TILE_PX + 2):
        for k in range(k0, k0 + w // TILE_PX + 2):
            tx0 = int(round((k * step - math.pi * RADIUS - x0) / px))
            ty0 = int(round((y1 - (math.pi * RADIUS - j * step)) / px))
            a0, b0 = max(0, tx0), max(0, ty0)
            a1, b1 = min(w, tx0 + TILE_PX), min(h, ty0 + TILE_PX)
            if a1 - a0 < 32 or b1 - b0 < 32:
                continue
            block = grey[b0:b1, a0:a1].mean(axis=2)
            lo, med, hi = np.percentile(block, [2, 50, 98])
            # one flat colour over the whole tile (JPEG's faint ringing allowed): white or black
            if hi - lo <= 8 and (med >= 240 or med <= 15):
                out.append([a0, b0, a1, b1])
    return out


def world_file(extent, size):
    x0, y0, x1, y1 = extent
    w, h = size
    sx, sy = (x1 - x0) / w, (y1 - y0) / h
    return "\n".join(f"{v:.14f}" for v in (sx, 0, 0, -sy, x0 + sx / 2, y1 - sy / 2)) + "\n"


# ------------------------------------------------------------------ the job

class Cancelled(Exception):
    pass


_jobs = {}
_lock = threading.Lock()           # one area at a time: one downloader at a time


def job_state(jid):
    j = _jobs.get(jid)
    if not j:
        return None
    snap = prog.of(jid) or {}
    return {k: j[k] for k in ("id", "key", "state", "error", "result", "log")} | {
        "stage": snap.get("stage"), "done": snap.get("done"), "left": snap.get("left"), "elapsed": snap.get("elapsed")}


def cancel(jid):
    j = _jobs.get(jid)
    if j and j["state"] in ("queued", "running"):
        j["cancel"] = True
        return True
    return False


def start(bounds, zoom, settings, root, register, jid=None):
    """
    Starts an area's job in a thread: download (or reuse), crop, alpha, OSM roads,
    previews; register(alpha_path, info) makes the alpha a street mask and returns its
    record. Returns the job's id at once (GET it for its state).
    """
    est = estimate(bounds, zoom)
    if not est["ok"]:
        raise ValueError("; ".join(est["problems"]))
    if not settings.get("downloader"):
        raise ValueError("set the path of Universal Maps Downloader's downloader.exe first")
    jid = jid or uuid.uuid4().hex[:12]
    job = {"id": jid, "key": area_key(bounds, zoom), "state": "queued", "error": None, "result": None,
           "log": [], "cancel": False}
    _jobs[jid] = job
    threading.Thread(target=_run, args=(job, est, dict(settings), Path(root), register), daemon=True).start()
    return jid


def _run(job, est, settings, root, register):
    with _lock, prog.job(job["id"], "Coordinates", COORDS_PLAN):
        job["state"] = "running"
        try:
            job["result"] = build_area(est, settings, root, register, job)
            job["state"] = "done"
            # then the satellite analysis, in the background (app/sat_analysis.py)
            try:
                from app import sat_analysis as SA
                key = job["result"]["area"]["key"]
                area = json.loads((root / key / "area.json").read_text())
                job["result"]["analysis"] = {"job": SA.start(root / key, area)}
            except Exception as e:                   # the alpha stands without it
                job["log"].append(f"the satellite analysis could not start: {e}")
        except Cancelled as e:
            job["state"], job["error"] = "cancelled", str(e) or "stopped"
        except Exception as e:                       # the page shows it as it is
            job["state"], job["error"] = "failed", str(e)


def build_area(est, settings, root, register, job=None):
    from app.osm import satellite as SAT
    job = job if job is not None else {"log": [], "cancel": False}
    say = lambda s: (job["log"].append(s), prog.note(s))
    cancelled = lambda: bool(job.get("cancel"))
    bounds, zoom = est["bounds"], est["zoom"]
    key = area_key(bounds, zoom)
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    rec_file = folder / "area.json"
    area = {}
    try:
        area = json.loads(rec_file.read_text())
    except (OSError, ValueError):
        pass
    area.update(key=key, bounds=bounds, zoom=zoom, mpp=est["mpp"], tiles=est["tiles"])

    def save():
        tmp = folder / f".area.{uuid.uuid4().hex[:6]}.tmp"
        tmp.write_text(json.dumps(area, indent=2))
        os.replace(tmp, rec_file)

    # the PBF's road index first (it may take a while the first time), so a wrong path shows at once
    index = None
    prog.stage("index", "the PBF's road index")
    if settings.get("pbf"):
        if not Path(settings["pbf"]).is_file():
            raise ValueError(f"the PBF file is not there: {settings['pbf']}")
        index = road_index(settings["pbf"], root, note=say, cancelled=cancelled)
    else:
        say("no PBF set: the alpha without the OSM check, and no OSM roads drawn")

    def download(stage, map_id, name, label, at=None):
        at = at or zoom
        n = est["tiles"] if at == zoom else SAT.tile_estimate(tuple(bounds), at)
        prog.stage(stage, f"downloading the {label} ({n:,} tiles)")
        have = area.get(stage)
        if have and Path(have).is_file():
            say(f"the {label}: already downloaded")
            return Path(have)
        if cancelled():
            raise Cancelled("stopped")
        try:
            image, _ = SAT.download(settings["downloader"], folder / "downloads", f"{name}_{key}", map_id, at,
                                    tuple(bounds), progress=say, cancelled=cancelled)
        except SAT.Cancelled as e:
            raise Cancelled(str(e))
        area[stage] = str(image)
        save()
        return Path(image)

    sat_path = download("satellite", SATELLITE_ID, "sat", "satellite map")
    street_path = download("street", STREET_ID, "street", "street map")
    # the streets from the street map at a low zoom (no buildings drawn over them); 0: from the area's own
    sz = int(settings.get("streets_zoom", STREETS_ZOOM) or 0)
    low_path = None
    if 0 < sz < zoom:
        if area.get("street_low_zoom") != sz:
            area.pop("street_low", None)               # another zoom than the one downloaded before
        try:
            low_path = download("street_low", STREET_ID, f"street{sz}", f"street map at zoom {sz} (for the streets)", sz)
            area["street_low_zoom"] = sz
        except Cancelled:
            raise
        except Exception as e:                         # noqa: BLE001 - the streets then come from the area's own map
            say(f"the street map at zoom {sz} could not be downloaded ({e}): the streets from zoom {zoom}'s")
    # the buildings: read from the street map, or from another map type set for them
    bmap = int(settings.get("buildings_map") or 0)
    if area.get("buildings_map_type") != bmap:
        area.pop("buildings_map", None)              # another map type than the one downloaded before
    bmap_path = download("buildings_map", bmap, f"bld{bmap}", f"buildings map (type {bmap})") if bmap > 0 else None
    area["buildings_map_type"] = bmap

    prog.stage("crop", "cutting both maps to the area")
    sat_geo = read_geo(sat_path)
    box, ext = crop_box(sat_geo, bounds)
    size = (box[2] - box[0], box[3] - box[1])
    # cut before from these same downloads (the area asked for again): kept as they are, so a
    # picture open in another program (Paint, Photos) does not stop the alpha being made again
    same = len(area.get("extent") or []) == 4 and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                                                       for a, b in zip(area["extent"], ext))
    sat = _kept(folder / "satellite.png", [sat_path], size) if same else None
    street = _kept(folder / "street.png", [street_path], size) if sat is not None else None
    if sat is not None and street is not None:
        say("the satellite and street maps: cut before from the same downloads, kept")
    else:
        with Image.open(sat_path) as im:
            sat = im.convert("RGB").crop(box)
        street = onto_grid(street_path, read_geo(street_path), ext, size)
        _write(sat, folder / "satellite.png")
        _write(street, folder / "street.png")
    if bmap_path is not None:
        if _kept(folder / "buildings_map.png", [bmap_path], size) is None:
            _write(onto_grid(bmap_path, read_geo(bmap_path), ext, size), folder / "buildings_map.png")
    else:
        try:
            (folder / "buildings_map.png").unlink(missing_ok=True)
        except OSError as e:
            raise ValueError("buildings_map.png could not be deleted: it is probably open in another program; "
                             f"close it there and press Download and make the alpha again ({e.strerror or e})") from e
    world = world_file(ext, size)
    for stem in ("satellite", "street", "alpha", "overlay"):
        (folder / f"{stem}.pgw").write_text(world)
        (folder / f"{stem}.prj").write_text(_wkt())
    area.update(extent=list(ext), width=size[0], height=size[1])
    lat = (bounds[2] + bounds[3]) / 2
    mpp = (ext[2] - ext[0]) / size[0] * math.cos(math.radians(lat))
    area["mpp"] = mpp

    prog.stage("missing", "looking for missing tiles")
    sat_a = np.asarray(sat)
    gaps = missing_tiles(sat_a, ext, zoom)
    area["missing_tiles"] = gaps
    if gaps:
        say(f"{len(gaps)} satellite tile(s) missing (flat white or black squares): the analysis counts them as unseen")

    osm_lines = None
    roads = []
    prog.stage("osm", "the OSM roads of the PBF")
    if index is not None:
        from app.osm import core
        roads = core.query_roads(index, tuple(bounds), include_minor=True)
        lines = core.render(roads, tuple(bounds), size[0], size[1], line_width=1, transparent=True)
        osm_lines = np.asarray(lines)[..., 3] > 0
        over = core.render(roads, tuple(bounds), size[0], size[1], line_width=max(2, int(round(1.0 / mpp))), transparent=True)
        _write(over, folder / "overlay.png")
        _geojson(roads, folder / "roads.geojson")
        area["osm_roads"] = len(roads)
        say(f"{len(roads)} OSM road ways in the area")

    prog.stage("alpha", "the alpha from the street map's roads")
    road, notes = street_alpha(np.asarray(street), mpp, palette(root), osm_lines)
    road_area = road                                   # the area's own map's streets: its buildings are read off them
    notes["streets_from"] = zoom
    if low_path is not None:
        with Image.open(low_path) as im:
            low, ln = low_streets(im, read_geo(low_path), ext, size, road, mpp, palette(root), osm_lines)
        if low is not None and low.any():
            road = low
            notes = {"style": notes["style"], "streets_from": sz, **ln, "area_alpha": notes}
            say(f"the streets from the street map at zoom {sz} (no buildings drawn over them): "
                f"{round(100 * ln.get('streets_agree', 0))}% the same as zoom {zoom}'s")
        else:
            say(f"the street map at zoom {sz}: {ln.get('streets_low', 'no streets read')}; the streets from zoom {zoom}'s")
        _low_preview(low_path, folder)
    alpha_path = folder / "alpha.png"
    alpha = (road * 255).astype(np.uint8)
    try:                                       # the same as before: left as it is (it may be open elsewhere)
        with Image.open(alpha_path) as im:
            unchanged = im.mode == "L" and im.size == (alpha.shape[1], alpha.shape[0]) and np.array_equal(np.asarray(im), alpha)
    except OSError:
        unchanged = False
    if not unchanged:
        _write(Image.fromarray(alpha, "L"), alpha_path)
    area["alpha"] = notes
    if notes.get("thin_lines"):
        say(f"{notes['thin_lines']} thin line(s) left out of the alpha ({notes['thin_m2']:.0f} m²): "
            "footpaths, parking lanes, service passages, under 2 m wide")
    if notes.get("under_buildings") and notes.get("streets_from") == zoom:
        say(f"{notes['under_buildings']} place(s) where Google draws a building over a street: the street filled back "
            f"under it ({notes['under_buildings_m2']:.0f} m²)")
    if osm_lines is not None and osm_lines.any():
        on = osm_lines & road
        area["osm_on_road"] = round(float(on.sum()) / max(1, int(osm_lines.sum())), 3)
    if not road.any():
        raise ValueError("no road found on the street map: is it the street map without labels (id 66)? "
                         "Its colours may differ from the ones expected (workspace/coords/palette.json)")

    prog.stage("buildings", "the buildings on the street map")
    from app import footprints as FP
    try:
        bdoc = FP.read_area(folder, area, None if bmap_path is not None else np.asarray(street), road_area, palette(root))
        area["buildings"] = dict(bdoc["notes"], source=bdoc["source"])
        say(f"{len(bdoc['buildings'])} buildings on the {'buildings' if bmap_path is not None else 'street'} map")
    except Exception as e:                    # noqa: BLE001 - the alpha stands without them
        area["buildings"] = {"buildings": 0, "error": f"{type(e).__name__}: {e}"}
        say(f"the buildings could not be read: {e}")

    prog.stage("previews", "previews")
    _previews(folder, sat, street, road, osm_lines, gaps)
    save()
    info = {"key": key, "bounds": bounds, "zoom": zoom, "mpp": mpp, "extent": list(ext),
            "width": size[0], "height": size[1], "missing_tiles": len(gaps), "osm_roads": area.get("osm_roads"),
            "osm_on_road": area.get("osm_on_road"), "alpha": notes, "buildings": area.get("buildings")}
    rec = register(alpha_path, info)
    area["mask_id"] = rec.get("id")
    save()
    say(f"alpha ready: {size[0]} x {size[1]} px, {mpp:.3f} m a pixel")
    return {"area": info, "mask": rec}


def _wkt():
    from app.osm import core
    return core.WKT


def _geojson(roads, path):
    feats = []
    for rd in roads:
        coords = [[list(unproject(*p)) for p in s] for s in rd["segments"]]
        feats.append({"type": "Feature", "properties": {"osm_id": rd["id"], "highway": rd["highway"]},
                      "geometry": {"type": "MultiLineString", "coordinates": coords}})
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def _low_preview(path, folder):
    """The low zoom street map as it came (a few hundred pixels), for the page."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            k = min(1.0, PREVIEW_PX / max(im.size))
            _write(im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.LANCZOS),
                   folder / "street_low_preview.png", optional=True)
    except OSError:
        pass


def _previews(folder, sat, street, road, osm_lines, gaps):
    """Small pictures for the page: each map, the alpha, and the alpha's edge over the satellite."""
    k = min(1.0, PREVIEW_PX / max(sat.size))
    ps = (max(1, round(sat.width * k)), max(1, round(sat.height * k)))
    # a preview open in another program is left as it was
    _write(sat.resize(ps, Image.Resampling.LANCZOS), folder / "satellite_preview.jpg", optional=True, quality=88)
    _write(street.resize(ps, Image.Resampling.LANCZOS), folder / "street_preview.png", optional=True)
    al = Image.fromarray((road * 255).astype(np.uint8), "L")
    _write(al.resize(ps, Image.Resampling.BOX), folder / "alpha_preview.png", optional=True)
    # the check: the satellite, the alpha's edge in amber, the OSM roads in cyan, missing tiles in red
    edge = road ^ ndi.binary_erosion(road, iterations=max(2, int(round(2.5 / k))))     # about 2.5 px in the preview
    view = np.asarray(sat).copy()
    view[edge] = (255, 176, 32)
    if osm_lines is not None:
        view[ndi.binary_dilation(osm_lines, iterations=max(1, int(round(0.6 / k))))] = (0, 220, 240)
    for x0, y0, x1, y1 in gaps:
        view[y0:y1, x0:x0 + 3] = view[y0:y1, x1 - 3:x1] = (230, 40, 40)
        view[y0:y0 + 3, x0:x1] = view[y1 - 3:y1, x0:x1] = (230, 40, 40)
    _write(Image.fromarray(view).resize(ps, Image.Resampling.LANCZOS), folder / "check_preview.jpg", optional=True, quality=88)


def forget(root, key):
    """An area's folder removed (its downloads with it)."""
    folder = Path(root) / key
    if folder.is_dir() and len(key) == 12 and all(c in "0123456789abcdef" for c in key):
        shutil.rmtree(folder)
