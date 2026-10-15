"""
The satellite analysis of a Coordinates area (app/coords.py): the satellite map looked
at in square tiles, inside the alpha's roads. It starts by itself once the alpha is
made, in the background, while you generate the texture.

For every tile (512 px, with 20 m of its neighbours around it to see what crosses
its edge; what it finds belongs to the tile its centre is in):

- **The paint on the road**: what is brighter than the asphalt around it (a white
  top-hat: the picture less its opening by a 2.6 m square, so anything narrower than
  2.6 m stands out), white or yellow, near the alpha's roads. Each mark is measured in
  metres from its moments (length, width, how much of its box it fills) and its angle
  to the road (the road's direction there, from the alpha), and named by its shape:

  | Name | Shape |
  | --- | --- |
  | lane line (dash / solid) | at most 0.35 m wide, 4 times as long, along the road; solid from 9 m |
  | zebra crossing | 3 or more stripes 0.3 to 0.85 m wide, 1.5 to 7 m long, along the road, side by side 0.7 to 1.6 m apart |
  | stop line | a bar across the road, at least 1.5 m long, 4 times as long as wide |
  | cyclist crossing | two such lines across, 1.5 to 2.5 m apart |
  | give way line | 3 or more short blocks (0.15 to 0.9 m²) in a row across the road |
  | arrow | 2 to 7 m long along the road, 0.35 to 2.2 m wide, filling under 72% of its box; the way it points
    from its head (more paint there), its kind (straight, left, right, two or three of them) from the arrow
    picture that covers it best |
  | parking bay lines | thin lines 2.5 to 7 m long, across or slanted, within 7 m of the road's edge, 2 or more side by side 2 to 3.6 m apart |
  | car | 1.3 m or more wide, 2.8 to 7.5 m long, filling most of its box (bright or dark): set aside |
  | unknown paint | anything else |

- **How well the road is seen there**, its confidence: 100 to start, less for what
  hides or blurs it, each drop kept with its reason:

  | Reason | Drop |
  | --- | --- |
  | missing tiles (no picture) | 100 x the road's share in them |
  | shadows (road darker than 55% of its usual) | 60 x share |
  | trees over the road (green) | 60 x share |
  | cars over the road | 20 x share |
  | the street map's road shows no asphalt on the satellite | 70 x (share - 10%) |
  | a blurred picture (little fine detail on the road) | up to 25 |

  A second figure, the markings' confidence, also drops when the paint is faint
  (its contrast under 25 levels: up to 20) and when the tile is mostly cars. A tile
  with no road has no confidence (none to give). The figures are a heuristic of
  version VERSION (stored with them), not a probability; they are calibrated when
  checked examples are saved to the dataset.

The tiles are worked by worker processes (python -m app.sat_analysis --worker, the
3D patches' pool), on the pictures saved as arrays in the area's analysis folder.
Results: analysis/tiles.json (each tile: its box, state, confidence and reasons, its
counts), analysis/findings.json (every mark: its name, place in pixels and degrees,
angle, size, contrast, score), and two pictures (the findings over the satellite,
the confidence of every tile).
"""

import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

VERSION = 4                  # 2: an arrow's way (toward) and its box's middle; a zebra's span from its outer stripes
                             # 3: an arrow's kind (straight, left, right, ...) from its shape
                             # 4: a solid line's shape (pts: its middle every metre), so curved lines are followed
TILE = 512
CONTEXT_M = 20.0           # each tile sees this much of its neighbours (a lane line's dashes repeat every 9 to 12 m)
OPEN_M = 2.6                 # the opening's square: marks narrower than this stand out
PAINT_MIN = 25.0             # paint: at least this much brighter than the asphalt around it (and 4.5 x its spread)
MARK_MIN_M2 = 0.08           # smaller bright specks are not counted (the picture's noise)
PAINT_REACH_M = 1.5          # paint looked for this far outside the alpha's road (edge lines, map widths)

CLASS_COLOURS = {
    "lane_dash": (80, 200, 255), "lane_solid": (40, 140, 255), "zebra": (255, 230, 40),
    "stop_line": (255, 70, 70), "cycle_crossing": (60, 220, 120), "giveway": (255, 150, 40),
    "arrow": (230, 90, 255), "parking_bay": (0, 255, 200), "car": (150, 150, 150), "unknown": (255, 255, 255),
}
CLASS_LABELS = {
    "lane_dash": "lane line (dashed)", "lane_solid": "lane line (solid)", "zebra": "zebra crossing",
    "stop_line": "stop line", "cycle_crossing": "cyclist crossing", "giveway": "give way line",
    "arrow": "arrow", "parking_bay": "parking bay lines", "car": "car", "unknown": "unknown paint",
}


# ------------------------------------------------------------------ the tiles

def tile_grid(w, h, tile=TILE, mpp=0.136):
    """The area's tiles: id r<row>c<col>, core [x0, y0, x1, y1] (what it owns), ctx (with its context)."""
    CONTEXT = int(round(CONTEXT_M / mpp))
    out = []
    for r, y0 in enumerate(range(0, h, tile)):
        for c, x0 in enumerate(range(0, w, tile)):
            x1, y1 = min(w, x0 + tile), min(h, y0 + tile)
            out.append({"id": f"r{r}c{c}", "row": r, "col": c, "core": [x0, y0, x1, y1],
                        "ctx": [max(0, x0 - CONTEXT), max(0, y0 - CONTEXT), min(w, x1 + CONTEXT), min(h, y1 + CONTEXT)]})
    return out


def road_direction(road, mpp):
    """
    The road's direction everywhere on it, at about 0.5 m a cell: the angle (radians, 0 to
    pi, image axes: x right, y down) and how sure (0 to 1: 1 on a straight road, low in
    a junction), from the structure of the distance to the road's edge.
    """
    f = max(1, int(round(0.5 / mpp)))
    small = road[::f, ::f].astype(bool)
    d = ndi.gaussian_filter(ndi.distance_transform_edt(small), 1.0)
    gy, gx = np.gradient(d)
    s = max(1.0, 1.5 / (mpp * f))
    jxx, jyy, jxy = (ndi.gaussian_filter(a, s) for a in (gx * gx, gy * gy, gx * gy))
    across = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    theta = np.mod(across + np.pi / 2, np.pi).astype(np.float32)
    coh = (np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)).astype(np.float32)
    return theta, np.clip(coh, 0, 1), f


def prepare(folder, area):
    """The pictures as arrays the workers read (memory-mapped): satellite, road, missing tiles, road direction."""
    folder = Path(folder)
    out = folder / "analysis"
    out.mkdir(exist_ok=True)
    sat = np.asarray(Image.open(folder / "satellite.png").convert("RGB"))
    road = np.asarray(Image.open(folder / "alpha.png").convert("L")) > 127
    h, w = road.shape
    missing = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in area.get("missing_tiles") or []:
        missing[y0:y1, x0:x1] = True
    theta, coh, f = road_direction(road, float(area["mpp"]))
    edge_m = ndi.distance_transform_edt(road[::max(1, int(round(0.5 / area["mpp"])))]) * f * float(area["mpp"])
    for name, a in (("sat", sat), ("road", road), ("missing", missing), ("theta", theta), ("coh", coh),
                    ("edge_m", edge_m.astype(np.float32))):
        np.save(out / f"{name}.npy", a)
    # the road's usual look, for every tile's checks: its median brightness and colour, and their spread
    lum = sat.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    on = road & ~missing
    if on.sum() > 50_000:
        idx = np.flatnonzero(on.ravel())
        idx = idx[np.random.default_rng(1).choice(len(idx), 50_000, replace=False)]
    else:
        idx = np.flatnonzero(on.ravel())
    lv = lum.ravel()[idx] if len(idx) else np.array([100.0])
    rgb = sat.reshape(-1, 3)[idx].astype(np.float32) if len(idx) else np.array([[100.0, 100, 100]])
    stats = {"lum_median": float(np.median(lv)), "lum_mad": float(np.median(np.abs(lv - np.median(lv)))) + 1.0,
             "rgb_median": [float(v) for v in np.median(rgb, axis=0)], "dir_cell": f}
    (out / "stats.json").write_text(json.dumps(stats))
    return {"w": w, "h": h, "stats": stats}


# ------------------------------------------------------------------ one tile

def _components(mask):
    """Labels and, for every component, its pixel count, centroid and second moments."""
    lab, n = ndi.label(mask)
    if not n:
        return lab, 0, None
    ys, xs = np.nonzero(lab)
    k = lab[ys, xs] - 1
    cnt = np.bincount(k, minlength=n).astype(np.float64)
    mx, my = np.bincount(k, xs, n) / cnt, np.bincount(k, ys, n) / cnt
    dx, dy = xs - mx[k], ys - my[k]
    cxx = np.bincount(k, dx * dx, n) / cnt + 1 / 12
    cyy = np.bincount(k, dy * dy, n) / cnt + 1 / 12
    cxy = np.bincount(k, dx * dy, n) / cnt
    return lab, n, {"cnt": cnt, "mx": mx, "my": my, "cxx": cxx, "cyy": cyy, "cxy": cxy, "k": k, "xs": xs, "ys": ys}


def _shape(m, mpp):
    """Length and width (metres, as a rectangle with the same moments), angle of the long side, fill of its box."""
    tr = m["cxx"] + m["cyy"]
    det = m["cxx"] * m["cyy"] - m["cxy"] ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-6)
    length, width = np.sqrt(12 * l1) * mpp, np.sqrt(12 * l2) * mpp
    ang = 0.5 * np.arctan2(2 * m["cxy"], m["cxx"] - m["cyy"])
    area = m["cnt"] * mpp * mpp
    fill = area / np.maximum(length * width, 1e-6)
    return length, width, np.mod(ang, np.pi), area, fill


def analyse_tile(task):
    """One tile: its findings (owned by it) and its confidence. Reads the area's arrays (task['dir'])."""
    d = Path(task["dir"])
    mpp, st = float(task["mpp"]), task["stats"]
    X0, Y0, X1, Y1 = task["ctx"]
    cx0, cy0, cx1, cy1 = task["core"]
    sat = np.load(d / "sat.npy", mmap_mode="r")[Y0:Y1, X0:X1].astype(np.float32)
    road = np.load(d / "road.npy", mmap_mode="r")[Y0:Y1, X0:X1]
    missing = np.load(d / "missing.npy", mmap_mode="r")[Y0:Y1, X0:X1]
    theta = np.load(d / "theta.npy", mmap_mode="r")
    coh = np.load(d / "coh.npy", mmap_mode="r")
    edge_m = np.load(d / "edge_m.npy", mmap_mode="r")
    f = int(st["dir_cell"])
    core = np.zeros(road.shape, bool)
    core[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0] = True
    road_core = road & core
    n_road = int(road_core.sum())
    res = {"id": task["id"], "core": task["core"], "road_px": n_road, "findings": [], "state": "done", "version": VERSION}
    if n_road < 200:
        res.update(state="no_road", confidence=None, confidence_marks=None, reasons=[])
        return res
    lum = sat @ np.array([0.299, 0.587, 0.114], np.float32)
    mx, mn = sat.max(axis=2), sat.min(axis=2)
    sat_s = (mx - mn) / np.maximum(mx, 1)
    hue = _hue(sat)
    k = max(3, int(round(OPEN_M / mpp)) | 1)
    bright = lum - ndi.grey_opening(lum, size=(k, k))
    dark = ndi.grey_closing(lum, size=(k, k)) - lum
    near = ndi.binary_dilation(road, iterations=max(1, int(round(PAINT_REACH_M / mpp)))) & ~missing
    m_l, mad = st["lum_median"], st["lum_mad"]
    thr = max(PAINT_MIN, 4.5 * mad)
    yellow = (hue > 35) & (hue < 65) & (sat_s > 0.45) & (lum > 110)      # yellow paint, not the sand's beige
    paint = near & (bright > thr) & (lum > m_l + 15) & ((sat_s < 0.22) | yellow)
    paint = ndi.binary_opening(paint, iterations=1) if mpp < 0.1 else paint
    darks = near & (dark > max(30.0, 4 * mad)) & (lum < m_l - 25)
    rgb_med = np.array(st["rgb_median"], np.float32)
    inside = ndi.binary_erosion(road, iterations=max(1, int(round(1.0 / mpp)))) & ~missing
    coloured = inside & (sat_s > 0.35) & ~yellow & (np.sqrt(((sat - rgb_med) ** 2).sum(axis=2)) > 60)

    finds = []
    min_px = max(3, int(MARK_MIN_M2 / (mpp * mpp)))

    def owned(x, y):
        return cx0 <= x < cx1 and cy0 <= y < cy1

    def road_dir(x, y):
        j, i = min(theta.shape[0] - 1, int(y) // f), min(theta.shape[1] - 1, int(x) // f)
        return float(theta[j, i]), float(coh[j, i]), float(edge_m[j, i]) if edge_m.shape == theta.shape else 99.0

    lab, n, m = _components(paint)
    if n:
        # a long blob that is not a line (a line and an arrow blurred into one): its parts under
        # 0.4 m wide taken off, so the line and what touched it come apart
        length, width, _, _, fill = _shape(m, mpp)
        merged = np.flatnonzero((length >= 9) & ((width > 0.6) | (fill < 0.5)))
        if len(merged):
            sel = np.isin(lab, merged + 1)
            thick = ndi.binary_opening(sel, structure=np.ones((3, 3), bool), iterations=max(1, int(round(0.2 / mpp))))
            paint = (paint & ~sel) | thick | (sel & ~ndi.binary_dilation(thick, iterations=1))
            lab, n, m = _components(paint)
    cars = np.zeros(road.shape, bool)
    parts = []
    if n:
        length, width, ang, area, fill = _shape(m, mpp)
        contrast = ndi.mean(bright, lab, np.arange(1, n + 1))
        yel = ndi.mean(yellow, lab, np.arange(1, n + 1))
        for i in range(n):
            if m["cnt"][i] < min_px:
                continue
            x, y = m["mx"][i] + X0, m["my"][i] + Y0
            th, ch, em = road_dir(x, y)
            phi = abs((ang[i] - th + np.pi / 2) % np.pi - np.pi / 2)          # 0 along the road, pi/2 across
            L, W, A, F = float(length[i]), float(width[i]), float(area[i]), float(fill[i])
            el = L / max(W, 1e-3)
            cls = _classify(L, W, A, F, el, math.degrees(phi), em)
            if cls == "car":
                cars |= lab == i + 1
            dashed = _dashed(m, i, bright, X0, Y0, ang[i]) if cls == "bar_across" else 0.0
            toward, mid = _pointing(m, i, ang[i]) if cls in ("bar_along", "arrow") else (None, 0.0)
            parts.append({"i": i, "cls": cls, "x": x, "y": y, "L": L, "W": W, "A": A, "F": F, "el": el,
                          "ang": float(ang[i]), "phi": math.degrees(phi), "th": th, "coh": ch, "edge_m": em,
                          "contrast": float(contrast[i]), "yellow": float(yel[i]) > 0.5, "dashed": dashed,
                          "toward": toward, "mid": mid})
            if toward is not None or cls == "lane_solid":
                sel = m["k"] == i
                # its pixels: an arrow's kind from its shape, a solid line's path
                parts[-1]["_px"] = (m["xs"][sel] + X0, m["ys"][sel] + Y0)
    # dark bodies and coloured bodies the size of a car
    for mask in (darks, coloured):
        lab2, n2, m2 = _components(mask)
        if not n2:
            continue
        length, width, ang, area, fill = _shape(m2, mpp)
        for i in range(n2):
            L, W, F = float(length[i]), float(width[i]), float(fill[i])
            if 1.3 <= W <= 2.8 and 2.8 <= L <= 7.5 and F >= 0.7:
                cars |= lab2 == i + 1
                x, y = m2["mx"][i] + X0, m2["my"][i] + Y0
                parts.append({"i": -1, "cls": "car", "x": x, "y": y, "L": L, "W": W, "A": float(area[i]), "F": F,
                              "el": L / max(W, 1e-3), "ang": float(ang[i]), "phi": 0.0, "th": 0.0, "coh": 0.0,
                              "edge_m": 99.0, "contrast": 0.0, "yellow": False, "dashed": 0.0, "toward": None, "mid": 0.0})
    finds = _group(parts, mpp)
    res["findings"] = [fd for fd in finds if owned(fd["x"], fd["y"])]
    if task.get("debug"):
        res["parts"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if not k.startswith("_")}
                        for p in parts if owned(p["x"], p["y"])]

    # ---- how well the road is seen
    rc = road_core
    nr = float(rc.sum())
    f_missing = float((missing & rc).sum()) / nr
    seen = rc & ~missing
    ns = max(1.0, float(seen.sum()))
    shadow = seen & (lum < 0.55 * m_l)
    green = seen & (sat[..., 1] > sat[..., 0] + 10) & (sat[..., 1] > sat[..., 2] + 6) & (sat_s > 0.15)
    car_px = seen & ndi.binary_dilation(cars, iterations=1)
    dist = np.sqrt(((sat - rgb_med) ** 2).sum(axis=2))
    other = seen & ~shadow & ~green & ~car_px & ~paint & (dist > 70)
    f_shadow, f_green = float(shadow.sum()) / ns, float(green.sum()) / ns
    f_cars, f_other = float(car_px.sum()) / ns, float(other.sum()) / ns
    lap = np.abs(ndi.laplace(ndi.gaussian_filter(lum, 0.5)))
    detail = float(np.median(lap[seen & ~paint])) if (seen & ~paint).any() else 0.0
    reasons = []

    def drop(why, amount):
        if amount >= 0.5:
            reasons.append({"why": why, "minus": round(amount, 1)})

    drop(f"missing tiles over {f_missing:.0%} of the road", 100 * f_missing)
    drop(f"shadow over {f_shadow:.0%} of the road", 60 * f_shadow * (1 - f_missing))
    drop(f"trees over {f_green:.0%} of the road", 60 * f_green * (1 - f_missing))
    drop(f"cars over {f_cars:.0%} of the road", 20 * f_cars * (1 - f_missing))
    drop(f"no asphalt seen on {f_other:.0%} of the street map's road", 70 * max(0.0, f_other - 0.10) * (1 - f_missing))
    blur = 25 * float(np.clip((0.8 - detail) / 0.8, 0, 1))
    drop(f"a blurred picture (fine detail {detail:.2f})", blur * (1 - f_missing))
    conf = max(0.0, 100.0 - sum(r["minus"] for r in reasons))
    paint_c = [fd["contrast"] for fd in res["findings"] if fd["cls"] not in ("car",)]
    marks_reasons = list(reasons)
    if paint_c and float(np.median(paint_c)) < 25:
        a = 20 * (25 - float(np.median(paint_c))) / 25
        marks_reasons.append({"why": f"faint paint (contrast {np.median(paint_c):.0f})", "minus": round(a, 1)})
    conf_m = max(0.0, 100.0 - sum(r["minus"] for r in marks_reasons))
    res.update(confidence=round(conf, 1), confidence_marks=round(conf_m, 1), reasons=reasons,
               reasons_marks=marks_reasons[len(reasons):],
               seen={"missing": round(f_missing, 3), "shadow": round(f_shadow, 3), "trees": round(f_green, 3),
                     "cars": round(f_cars, 3), "not_asphalt": round(f_other, 3), "detail": round(detail, 3)})
    return res


def _hue(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return np.where(mx - mn < 1e-6, -1.0, h)


def _dashed(m, i, bright, X0, Y0, ang):
    """How broken a mark is along its length: the spread of its brightness along it over its mean (0 solid)."""
    sel = m["k"] == i
    xs, ys = m["xs"][sel], m["ys"][sel]
    if len(xs) < 12:
        return 0.0
    t = (xs - m["mx"][i]) * math.cos(ang) + (ys - m["my"][i]) * math.sin(ang)
    b = np.round(t - t.min()).astype(int)
    prof = np.bincount(b, bright[ys, xs]) / np.maximum(np.bincount(b), 1)
    cut = max(1, len(prof) // 10)
    prof = prof[cut:-cut] if len(prof) > 2 * cut + 4 else prof
    return float(np.std(prof) / max(np.mean(prof), 1e-6))


def _pointing(m, i, ang):
    """
    Which way a mark along the road points, and how far its box's middle lies from its
    centre of mass (pixels, along it): an arrow's head carries more paint than its stem,
    so its centre of mass lies toward the head. ((x, y) unit, offset).
    """
    sel = m["k"] == i
    xs, ys = m["xs"][sel], m["ys"][sel]
    if len(xs) < 8:
        return None, 0.0
    c, s_ = math.cos(ang), math.sin(ang)
    t = (xs - m["mx"][i]) * c + (ys - m["my"][i]) * s_
    mid = float((t.min() + t.max()) / 2)
    sign = -1.0 if mid > 0 else 1.0                  # the middle lies on the tail's side of the centre of mass
    return [round(c * sign, 4), round(s_ * sign, 4)], mid


ARROW_KINDS = ("S", "L", "R", "SL", "SR", "LR", "SLR")
S_FIRST = 0.05              # a straight arrow unless another kind's picture covers the mark this much better
_TEMPLATES = {}


def _templates():
    """The arrow pictures the road markings lay (app/roadmarks.py): each kind's polygons and box (metres)."""
    if not _TEMPLATES:
        from app import roadmarks as RM
        for k in ARROW_KINDS:
            g = RM._arrow_shape(k)
            polys = [list(q.exterior.coords) for q in getattr(g, "geoms", [g]) if q.geom_type == "Polygon"]
            _TEMPLATES[k] = (polys, g.bounds)
    return _TEMPLATES


def _arrow_kind(xs, ys, toward):
    """
    Which arrow a mark is (straight on, left, right, or two or three of them): each
    arrow picture the road markings lay (app/roadmarks.py) laid over the mark's box,
    its head the way the mark points (turned up to 20 degrees either way, a lone turn
    arrow leans), as thick as the mark's stem, and the one sharing most of its pixels
    with the mark (their overlap over their union); straight on unless another is
    better by S_FIRST. (kind, that share, the next best's).
    """
    if toward is None or len(xs) < 12:
        return None, 0.0, 0.0
    SS = 4
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    cx, cy = xs.mean(), ys.mean()
    ox, oy = int(xs.min()) - 3, int(ys.min()) - 3
    w, h = int(xs.max()) - ox + 4, int(ys.max()) - oy + 4
    obs = np.zeros((h, w), bool)
    obs[(ys - oy).astype(int), (xs - ox).astype(int)] = True
    best = {}
    for dth in (-20, -10, 0, 10, 20):
        a = math.atan2(toward[1], toward[0]) + math.radians(dth)
        tx, ty = math.cos(a), math.sin(a)
        rx, ry = -ty, tx                                     # the driver's right, in the picture
        A = (xs - cx) * tx + (ys - cy) * ty
        R = (xs - cx) * rx + (ys - cy) * ry
        a0, a1, r0, r1 = A.min() - 0.5, A.max() + 0.5, R.min() - 0.5, R.max() + 0.5
        tail = int((A < a0 + 0.3 * (a1 - a0)).sum())
        stem_px = tail / max(1.0, 0.3 * (a1 - a0))            # the mark's stem, pixels wide
        for k, (polys, (bx0, by0, bx1, by1)) in _templates().items():
            sx, sy = (r1 - r0) / max(1e-6, bx1 - bx0), (a1 - a0) / max(1e-6, by1 - by0)
            im = Image.new("L", (w * SS, h * SS), 0)
            d = ImageDraw.Draw(im)
            for poly in polys:
                pts = []
                for x, y in poly:
                    s_, t_ = r0 + (x - bx0) * sx, a0 + (y - by0) * sy
                    X, Y = cx + t_ * tx + s_ * rx, cy + t_ * ty + s_ * ry
                    pts.append(((X - ox + 0.5) * SS, (Y - oy + 0.5) * SS))
                d.polygon(pts, fill=255)
            tpl = np.asarray(im.reduce(SS)) >= 90
            grow = int(round(max(0.0, stem_px - 0.15 * sx) / 2))
            if grow:
                tpl = ndi.binary_dilation(tpl, iterations=grow)
            iou = float((tpl & obs).sum()) / max(1.0, float((tpl | obs).sum()))
            best[k] = max(best.get(k, 0.0), iou)
    order = sorted(best.items(), key=lambda kv: -kv[1])
    if order[0][0] != "S" and best["S"] >= order[0][1] - S_FIRST:
        return "S", round(best["S"], 3), round(order[0][1], 3)    # the commonest, unless another is clearly better
    return order[0][0], round(order[0][1], 3), round(order[1][1], 3)


def _polyline(xs, ys, ang, mpp, step_m=1.0):
    """
    A long mark's middle line, [[x, y], ...] every step_m along it: its pixels in slices
    across its long side (a line along a curving street bends; each slice's middle).
    """
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    if len(xs) < 4:
        return [[round(float(xs.mean()), 1), round(float(ys.mean()), 1)]] if len(xs) else []
    c, s_ = math.cos(ang), math.sin(ang)
    t = (xs - xs.mean()) * c + (ys - ys.mean()) * s_
    k = np.floor((t - t.min()) / max(step_m / mpp, 1.0)).astype(int)
    n = np.bincount(k)
    mx, my = np.bincount(k, xs) / np.maximum(n, 1), np.bincount(k, ys) / np.maximum(n, 1)
    keep = n > 0
    pts = np.column_stack([mx[keep], my[keep]])
    # its two ends: the pixels furthest along it, so the line is as long as the paint
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    pts = np.vstack([[xs[i0], ys[i0]], pts, [xs[i1], ys[i1]]])
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def _classify(L, W, A, F, el, phi, edge_m):
    """
    A mark's first name from its size (metres), fill, length to width, angle to the road
    (degrees) and distance from the road's edge; _group gives the last (a bar along the
    road is a zebra's stripe, a lane dash or an arrow by what is beside it).
    """
    if W >= 1.3 and 2.8 <= L <= 7.5 and F >= 0.7:
        return "car"
    if L >= 9 and W <= 0.6 and phi <= 20:
        return "lane_solid"
    if 2.0 <= L <= 7 and 0.7 <= W <= 2.3 and F <= 0.82 and (phi <= 30 or el < 1.8):
        return "arrow"                      # a shaft and a head: wide, and far from filling its box
    if W <= 0.9 and L >= 0.8 and el >= 2.2 and phi <= 25:
        return "bar_along"
    if W <= 0.3 and 2.5 <= L <= 7 and 30 <= phi < 55 and edge_m <= 7:
        return "parking_line"
    if L >= 1.5 and el >= 2.2 and phi >= 55 and W <= 1.5:
        return "bar_across"
    if 0.12 <= A <= 0.9 and el < 2.5 and L <= 1.2:
        return "block"
    return "unknown"


def _group(parts, mpp):
    """
    The last names: bars along the road side by side across it (3 or more) are a zebra;
    the rest in a row along it (a lane line's dashes) are lane dashes, and a lone wide one
    an arrow; bars across are a give way line when broken or wide, a cyclist crossing
    when two alike lie 1.5 to 2.6 m apart, else a stop line (parking lines when short and
    at the edge); blocks in a row across are a give way line; parking lines side by side
    are bays.
    """
    out = []

    def base(p):
        return {"angle": round(math.degrees(p["ang"]), 1), "length_m": round(p["L"], 2), "width_m": round(p["W"], 2),
                "contrast": round(p["contrast"], 1), "yellow": p["yellow"], "road_angle": round(math.degrees(p["th"]), 1),
                "road_sure": round(p["coh"], 2)}

    def score(p):
        return min(1.0, max(0.3, p["contrast"] / 50)) * (0.6 + 0.4 * p["coh"])

    def emit(p, cls, s=None, **extra):
        x, y = float(p["x"]), float(p["y"])
        if cls == "lane_solid" and p.get("_px") is not None:
            extra = dict(extra, pts=_polyline(p["_px"][0], p["_px"][1], p["ang"], mpp))
        if cls == "arrow" and p.get("toward"):
            # an arrow at its box's middle (its centre of mass lies toward the head), pointing as its head does
            x, y = x + p["mid"] * math.cos(p["ang"]), y + p["mid"] * math.sin(p["ang"])
            extra = dict(extra, toward=p["toward"])
            if p.get("_px") is not None:
                kind, match, _ = _arrow_kind(p["_px"][0], p["_px"][1], p["toward"])
                if kind:
                    extra.update(kind=kind, kind_match=match)
        out.append(dict(base(p), cls=cls, x=x, y=y, n=1, score=round(float(score(p) if s is None else s), 3), **extra))

    def emit_group(g, cls, **extra):
        out.append(dict(base(g[0]), cls=cls, x=float(np.mean([p["x"] for p in g])), y=float(np.mean([p["y"] for p in g])),
                        n=len(g), score=round(float(np.mean([score(p) for p in g])), 3), **extra))

    def clusters(items, ok):
        parent = list(range(len(items)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if ok(items[a], items[b]):
                    parent[root(a)] = root(b)
        groups = {}
        for i in range(len(items)):
            groups.setdefault(root(i), []).append(items[i])
        return list(groups.values())

    def offsets(a, b, own=True):
        """
        How far apart two marks are across a's long side and along it (metres): in its own
        axis (a bar's own direction, surer than the road's at a junction), or the road's.
        """
        t = a["ang"] if own else a["th"]
        ux, uy = math.cos(t), math.sin(t)
        dx, dy = (b["x"] - a["x"]) * mpp, (b["y"] - a["y"]) * mpp
        return abs(-uy * dx + ux * dy), abs(ux * dx + uy * dy)

    def parallel(a, b, deg):
        return abs(((a["ang"] - b["ang"] + np.pi / 2) % np.pi) - np.pi / 2) < math.radians(deg)

    by = {}
    for p in parts:
        by.setdefault(p["cls"], []).append(p)
    # bars along the road: zebras, then lane dashes in rows (the lane lines' own width), then arrows
    bars = by.get("bar_along", [])
    rest = []
    for g in clusters(bars, lambda a, b: parallel(a, b, 15) and (lambda o: 0.55 <= o[0] <= 1.8 and o[1] <= 1.0)(offsets(a, b))):
        if len(g) >= 3:
            # its span across the road: from the outer stripe to the outer stripe
            c, s_ = math.cos(g[0]["ang"]), math.sin(g[0]["ang"])
            t = [(-(p["x"] - g[0]["x"]) * s_ + (p["y"] - g[0]["y"]) * c) * mpp for p in g]
            span = max(t) - min(t) + float(np.median([p["W"] for p in g]))
            emit_group(g, "zebra", across_m=round(span, 2), stripe_m=round(float(np.median([p["L"] for p in g])), 2))
        else:
            rest += g
    chains = clusters(rest, lambda a, b: parallel(a, b, 12) and abs(a["W"] - b["W"]) <= 0.1
                      and (lambda o: o[0] <= 0.45 and 1.5 <= o[1] <= 16)(offsets(a, b)))
    long_ = [p["W"] for g in chains if len(g) >= 3 for p in g]
    lane_w = float(np.median(long_)) if long_ else 0.45
    for g in chains:
        wide = float(np.median([p["W"] for p in g])) >= lane_w + 0.07
        for p in g:
            if wide or (len(g) == 1 and p["F"] <= 0.8):
                emit(p, "arrow", score(p) * (0.9 if len(g) == 1 else 0.7))
            else:
                emit(p, "lane_dash", score(p) * (1.0 if len(g) >= 2 else 0.6))
    # bars across the road: the pieces on one line joined, then named
    across = by.get("bar_across", [])
    lines_ = []
    for g in clusters(across, lambda a, b: parallel(a, b, 10) and abs(a["contrast"] - b["contrast"]) <= 20
                      and (lambda o: o[0] <= 0.6 and o[1] <= (a["L"] + b["L"]) / 2 + 1.5)(offsets(a, b))):
        q = dict(max(g, key=lambda p: p["L"]))
        q["x"], q["y"] = float(np.mean([p["x"] for p in g])), float(np.mean([p["y"] for p in g]))
        q["L"] = float(sum(p["L"] for p in g)); q["n"] = len(g)
        q["dashed"] = max(p["dashed"] for p in g) if len(g) == 1 else max(0.3, max(p["dashed"] for p in g))
        lines_.append(q)
    # two thin lines alike, 1.5 to 2.6 m apart along the road: a cyclist crossing (first: a thin faint
    # line varies along its length, and would read as broken)
    used = set()
    for i, a in enumerate(lines_):
        for j in range(i + 1, len(lines_)):
            b = lines_[j]
            if i in used or j in used or a["W"] > 0.5 or b["W"] > 0.5:
                continue
            ac, al = offsets(a, b)                  # across a's own line: along the road
            if 1.5 <= ac <= 2.6 and al <= max(a["L"], b["L"]) / 2 and abs(a["W"] - b["W"]) <= 0.15 \
                    and abs(a["contrast"] - b["contrast"]) <= 15:
                used.update((i, j))
                emit_group([dict(a, L=max(a["L"], b["L"])), b], "cycle_crossing", track_m=round(ac, 2))
    for i, p in enumerate(lines_):
        if i in used:
            continue
        if p["dashed"] >= (0.25 if p["W"] < 0.45 else 0.18) or p["W"] >= 0.8:
            emit(p, "giveway")
        else:
            emit(p, "stop_line")
    # give way: blocks in a row across the road
    for g in clusters(by.get("block", []), lambda a, b: (lambda o: o[0] <= 1.4 and o[1] <= 0.6)(offsets(a, b, own=False))):
        if len(g) >= 3:
            emit_group(g, "giveway", across_m=round(max(offsets(g[0], p, own=False)[0] for p in g), 2))
    # parking: thin lines at the edge, side by side 2 to 3.6 m apart
    for g in clusters(by.get("parking_line", []), lambda a, b: parallel(a, b, 12)
                      and 1.9 <= math.hypot(a["x"] - b["x"], a["y"] - b["y"]) * mpp <= 3.7):
        if len(g) >= 2:
            emit_group(g, "parking_bay", bays=len(g) - 1)
    for cls in ("lane_solid", "arrow", "car", "unknown"):
        for p in by.get(cls, []):
            emit(p, cls)
    return out


# ------------------------------------------------------------------ the whole area

_pool = None
_pool_lock = threading.Lock()
_runs = {}                  # area key -> {"state", "tiles", "done", "total", "job", "error", "started", "summary"}


def pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            from app import patches as PT
            _pool = PT.Pool(n=max(1, min(4, (os.cpu_count() or 2) - 1)), module="app.sat_analysis")
        return _pool


def state(key):
    return _runs.get(key)


def start(folder, area, jid=None):
    """Starts the analysis of an area in the background (once at a time per area); returns its job id."""
    key = area["key"]
    r = _runs.get(key)
    if r and r["state"] in ("preparing", "running"):
        return r["job"]
    jid = jid or uuid.uuid4().hex[:12]
    _runs[key] = {"state": "preparing", "tiles": {}, "done": 0, "total": 0, "job": jid, "error": None,
                  "started": time.time(), "summary": None}
    threading.Thread(target=_run, args=(Path(folder), dict(area), jid), daemon=True).start()
    return jid


def _run(folder, area, jid):
    from app import progress as prog
    key = area["key"]
    run = _runs[key]
    with prog.job(jid, "Satellite analysis", [("prepare", 1.0), ("tiles", 12.0), ("pictures", 1.0)]):
        try:
            prog.stage("prepare", "preparing the satellite and the alpha")
            info = prepare(folder, area)
            grid = tile_grid(info["w"], info["h"], mpp=float(area["mpp"]))
            run["total"] = len(grid)
            run["grid"] = [{"id": t["id"], "core": t["core"], "row": t["row"], "col": t["col"]} for t in grid]
            run["state"] = "running"
            prog.stage("tiles", f"analysing {len(grid)} tiles")
            done = threading.Event()
            lock = threading.Lock()
            results = {}

            def take(t):
                def fn(res):
                    with lock:
                        if not res.get("ok"):
                            res = {"id": t["id"], "core": t["core"], "state": "failed", "error": res.get("error"),
                                   "findings": [], "confidence": None, "reasons": []}
                        else:
                            res = res["info"]
                        results[t["id"]] = res
                        run["tiles"][t["id"]] = {k: res.get(k) for k in ("state", "confidence", "confidence_marks",
                                                                          "reasons", "road_px")} | {
                            "n": len([f for f in res.get("findings", []) if f["cls"] != "car"])}
                        run["done"] = len(results)
                        prog.part(len(results) / len(grid))
                        if len(results) == len(grid):
                            done.set()
                return fn
            p = pool()
            # the tiles nearest the middle first: what a view shows first
            cx, cy = info["w"] / 2, info["h"] / 2
            order = sorted(grid, key=lambda t: (t["core"][0] + t["core"][2] - 2 * cx) ** 2 + (t["core"][1] + t["core"][3] - 2 * cy) ** 2)
            for t in order:
                task = {"kind": "tile", "dir": str(folder / "analysis"), "mpp": area["mpp"], "stats": info["stats"],
                        "id": t["id"], "core": t["core"], "ctx": t["ctx"]}
                p.submit({"key": f"sat:{key}:{t['id']}", "fn": take(t), "task": task})
            while not done.wait(1.0):
                pass
            prog.stage("pictures", "the findings and confidence pictures")
            summary = finish(folder, area, grid, results)
            run.update(state="done", summary=summary)
        except Exception as e:
            import traceback
            traceback.print_exc()
            run.update(state="failed", error=f"{type(e).__name__}: {e}")


def finish(folder, area, grid, results):
    """Writes tiles.json and findings.json, draws the two pictures, returns the summary."""
    out = Path(folder) / "analysis"
    mpp = float(area["mpp"])
    x0, y0, x1, y1 = area["extent"]
    w, h = int(area["width"]), int(area["height"])
    lat = (area["bounds"][2] + area["bounds"][3]) / 2

    def lonlat(x, y):
        X, Y = x0 + (x + 0.5) / w * (x1 - x0), y1 - (y + 0.5) / h * (y1 - y0)
        R = 6378137.0
        return round(math.degrees(X / R), 7), round(math.degrees(2 * math.atan(math.exp(Y / R)) - math.pi / 2), 7)

    findings, tiles = [], []
    for t in grid:
        r = results.get(t["id"], {})
        for k, fd in enumerate(r.get("findings", [])):
            lon, la = lonlat(fd["x"], fd["y"])
            findings.append(dict(fd, id=f"{t['id']}_{k}", tile=t["id"], x=round(fd["x"], 1), y=round(fd["y"], 1),
                                 lon=lon, lat=la))
        tiles.append({"id": t["id"], "row": t["row"], "col": t["col"], "core": t["core"],
                      **{k: r.get(k) for k in ("state", "confidence", "confidence_marks", "reasons", "reasons_marks",
                                               "seen", "road_px", "error")},
                      "counts": _counts(r.get("findings", []))})
    rated = [t for t in tiles if t["confidence"] is not None]
    weights = np.array([t["road_px"] for t in rated], float)
    avg = float(np.average([t["confidence"] for t in rated], weights=weights)) if rated else None
    avg_m = float(np.average([t["confidence_marks"] for t in rated], weights=weights)) if rated else None
    summary = {"version": VERSION, "tiles": len(tiles), "rated": len(rated), "confidence": None if avg is None else round(avg, 1),
               "confidence_marks": None if avg_m is None else round(avg_m, 1), "counts": _counts(findings),
               "low_tiles": sum(1 for t in rated if t["confidence"] < 60), "tile_px": TILE, "mpp": mpp}
    (out / "tiles.json").write_text(json.dumps({"summary": summary, "tiles": tiles}))
    (out / "findings.json").write_text(json.dumps({"version": VERSION, "mpp": mpp, "findings": findings}))
    draw(Path(folder), tiles, findings, mpp)
    return summary


def _counts(findings):
    c = {}
    for fd in findings:
        c[fd["cls"]] = c.get(fd["cls"], 0) + 1
    return c


def _font(px):
    for f in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(folder, tiles, findings, mpp, size=1600):
    """The findings over the satellite (each name its colour), and every tile's confidence."""
    sat = Image.open(folder / "satellite.png").convert("RGB")
    k = min(1.0, size / max(sat.size))
    ps = (max(1, round(sat.width * k)), max(1, round(sat.height * k)))
    base = sat.resize(ps, Image.Resampling.LANCZOS)
    # findings: a box the mark's size and angle, in its colour
    im = base.copy()
    d = ImageDraw.Draw(im)
    for fd in findings:
        if fd["cls"] == "car":
            continue
        c = CLASS_COLOURS.get(fd["cls"], (255, 255, 255))
        L = max(fd.get("length_m", 1), 1.0) / mpp * k
        W = max(fd.get("width_m", 0.3), 0.6) / mpp * k
        if fd["cls"] in ("zebra", "giveway"):
            W = max(W, fd.get("across_m", 2) / mpp * k)
        a = math.radians(fd.get("angle", 0))
        ux, uy = math.cos(a), math.sin(a)
        vx, vy = -uy, ux
        cx, cy = fd["x"] * k, fd["y"] * k
        pts = [(cx + sx * ux * L / 2 + sy * vx * W / 2, cy + sx * uy * L / 2 + sy * vy * W / 2)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        d.polygon(pts, outline=c, width=2)
    im.save(folder / "analysis" / "findings_preview.jpg", quality=88)
    # confidence: the satellite dimmed, every tile coloured red to green with its figure
    im = Image.blend(base, Image.new("RGB", ps, (20, 22, 26)), 0.55)
    over = Image.new("RGBA", ps, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = _font(max(10, int(TILE * k / 4.5)))
    for t in tiles:
        x0, y0, x1, y1 = (v * k for v in t["core"])
        c = t["confidence"]
        if c is None:
            d.rectangle((x0, y0, x1, y1), outline=(90, 90, 90, 160))
            continue
        col = _ramp(c / 100)
        d.rectangle((x0, y0, x1, y1), fill=col + (90,), outline=col + (230,), width=2)
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), f"{c:.0f}", fill=(255, 255, 255, 255), font=font, anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    im = Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")
    im.save(folder / "analysis" / "confidence_preview.jpg", quality=88)


def _ramp(v):
    """Red (0) to amber (0.6) to green (1)."""
    v = max(0.0, min(1.0, v))
    a, b, c = (220, 50, 50), (240, 170, 30), (60, 200, 90)
    if v < 0.6:
        t = v / 0.6
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    t = (v - 0.6) / 0.4
    return tuple(int(b[i] + (c[i] - b[i]) * t) for i in range(3))


def _worker():
    """A worker's loop: one tile a line in, one answer a line out (as app/patches.py's)."""
    out = sys.stdout
    sys.stdout = sys.stderr
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            ans = {"ok": True, "info": analyse_tile(json.loads(line))}
        except Exception as e:
            import traceback
            traceback.print_exc()
            ans = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        ans["seconds"] = round(time.time() - t0, 2)
        out.write(json.dumps(ans, default=float) + "\n")
        out.flush()


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
