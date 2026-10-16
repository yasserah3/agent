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
the next step), not to move the alpha's edges.

Every area keeps its files in workspace/coords/<key>/ (key: the bounds and zoom),
so asking for the same area again downloads nothing.
"""

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
    # light grey; "style": "auto" works out which a map is drawn in, "classic" or "new" sets it
    "style": "auto",
    "new": {"b_minus_r": 8, "r_min": 120, "r_max": 214, "open_m": 0.6},
}

COORDS_PLAN = [("index", 2.0), ("satellite", 10.0), ("street", 6.0), ("buildings_map", 1.0), ("crop", 1.0),
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
    s = {"downloader": "", "pbf": "", "zoom": 20, "buildings_map": 0}
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
    keep = {k: s[k] for k in ("downloader", "pbf", "zoom", "buildings_map") if k in s}
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


def street_alpha(rgb, mpp, pal=PALETTE, osm_lines=None):
    """
    The road (bool, H x W) in a street map (H x W x 3 uint8): its white and its yellow to
    orange (Google's classic colours) or its blue-grey (the 2024 colours, street_style:
    the building outlines, as blue as the roads but a pixel or two wide, taken off), small
    holes filled (the one-way arrows drawn on a street), small specks dropped; with
    osm_lines (bool, the OSM roads drawn 1 px wide) the road pieces with no OSM road within
    osm_reach_m dropped too. Returns (road, notes).
    """
    style = street_style(rgb, pal)
    if style == "new":
        p = {**PALETTE["new"], **(pal.get("new") or {})}
        road = _new_colour(rgb, pal)
        k = max(1, int(round(float(p["open_m"]) / mpp / 2)))
        road = ndi.binary_opening(road, structure=np.ones((2 * k + 1, 2 * k + 1), bool))
    else:
        road = _classic_road(rgb, pal)
    notes = {"coloured_px": int(road.sum()), "style": style}
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
    notes["road_px"] = int(road.sum())
    return road, notes


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
    ex0, ey0, ex1, ey1 = geo["extent"]
    sw, sh = geo["source_size"]
    sx, sy = (ex1 - ex0) / sw, (ey1 - ey0) / sh
    x0, y0, x1, y1 = extent
    w, h = size
    if x0 < ex0 - sx or y0 < ey0 - sy or x1 > ex1 + sx or y1 > ey1 + sy:
        raise ValueError("the street map does not cover the whole area: download it again")
    a, e = (x1 - x0) / w / sx, (y1 - y0) / h / sy
    # Pillow samples at the output pixel's corner: half a source pixel's worth of scale moved to its centre
    c, f = (x0 - ex0) / sx + (a - 1) / 2, (ey1 - y1) / sy + (e - 1) / 2
    with Image.open(path) as im:
        return im.convert("RGB").transform((w, h), Image.Transform.AFFINE, (a, 0, c, 0, e, f), resample)


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

    def download(stage, map_id, name, label):
        prog.stage(stage, f"downloading the {label} ({est['tiles']:,} tiles)")
        have = area.get(stage)
        if have and Path(have).is_file():
            say(f"the {label}: already downloaded")
            return Path(have)
        if cancelled():
            raise Cancelled("stopped")
        try:
            image, _ = SAT.download(settings["downloader"], folder / "downloads", f"{name}_{key}", map_id, zoom,
                                    tuple(bounds), progress=say, cancelled=cancelled)
        except SAT.Cancelled as e:
            raise Cancelled(str(e))
        area[stage] = str(image)
        save()
        return Path(image)

    sat_path = download("satellite", SATELLITE_ID, "sat", "satellite map")
    street_path = download("street", STREET_ID, "street", "street map")
    # the buildings: read from the street map, or from another map type set for them
    bmap = int(settings.get("buildings_map") or 0)
    if area.get("buildings_map_type") != bmap:
        area.pop("buildings_map", None)              # another map type than the one downloaded before
    bmap_path = download("buildings_map", bmap, f"bld{bmap}", f"buildings map (type {bmap})") if bmap > 0 else None
    area["buildings_map_type"] = bmap

    prog.stage("crop", "cutting both maps to the area")
    sat_geo = read_geo(sat_path)
    box, ext = crop_box(sat_geo, bounds)
    with Image.open(sat_path) as im:
        sat = im.convert("RGB").crop(box)
    size = sat.size
    street = onto_grid(street_path, read_geo(street_path), ext, size)
    sat.save(folder / "satellite.png")
    street.save(folder / "street.png")
    if bmap_path is not None:
        onto_grid(bmap_path, read_geo(bmap_path), ext, size).save(folder / "buildings_map.png")
    else:
        (folder / "buildings_map.png").unlink(missing_ok=True)
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
        over.save(folder / "overlay.png")
        _geojson(roads, folder / "roads.geojson")
        area["osm_roads"] = len(roads)
        say(f"{len(roads)} OSM road ways in the area")

    prog.stage("alpha", "the alpha from the street map's roads")
    road, notes = street_alpha(np.asarray(street), mpp, palette(root), osm_lines)
    alpha_path = folder / "alpha.png"
    Image.fromarray((road * 255).astype(np.uint8), "L").save(alpha_path)
    area["alpha"] = notes
    if osm_lines is not None and osm_lines.any():
        on = osm_lines & road
        area["osm_on_road"] = round(float(on.sum()) / max(1, int(osm_lines.sum())), 3)
    if not road.any():
        raise ValueError("no road found on the street map: is it the street map without labels (id 66)? "
                         "Its colours may differ from the ones expected (workspace/coords/palette.json)")

    prog.stage("buildings", "the buildings on the street map")
    from app import footprints as FP
    try:
        bdoc = FP.read_area(folder, area, None if bmap_path is not None else np.asarray(street), road, palette(root))
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


def _previews(folder, sat, street, road, osm_lines, gaps):
    """Small pictures for the page: each map, the alpha, and the alpha's edge over the satellite."""
    k = min(1.0, PREVIEW_PX / max(sat.size))
    ps = (max(1, round(sat.width * k)), max(1, round(sat.height * k)))
    sat.resize(ps, Image.Resampling.LANCZOS).save(folder / "satellite_preview.jpg", quality=88)
    street.resize(ps, Image.Resampling.LANCZOS).save(folder / "street_preview.png")
    al = Image.fromarray((road * 255).astype(np.uint8), "L")
    al.resize(ps, Image.Resampling.BOX).save(folder / "alpha_preview.png")
    # the check: the satellite, the alpha's edge in amber, the OSM roads in cyan, missing tiles in red
    edge = road ^ ndi.binary_erosion(road, iterations=max(2, int(round(2.5 / k))))     # about 2.5 px in the preview
    view = np.asarray(sat).copy()
    view[edge] = (255, 176, 32)
    if osm_lines is not None:
        view[ndi.binary_dilation(osm_lines, iterations=max(1, int(round(0.6 / k))))] = (0, 220, 240)
    for x0, y0, x1, y1 in gaps:
        view[y0:y1, x0:x0 + 3] = view[y0:y1, x1 - 3:x1] = (230, 40, 40)
        view[y0:y0 + 3, x0:x1] = view[y1 - 3:y1, x0:x1] = (230, 40, 40)
    Image.fromarray(view).resize(ps, Image.Resampling.LANCZOS).save(folder / "check_preview.jpg", quality=88)


def forget(root, key):
    """An area's folder removed (its downloads with it)."""
    folder = Path(root) / key
    if folder.is_dir() and len(key) == 12 and all(c in "0123456789abcdef" for c in key):
        shutil.rmtree(folder)
