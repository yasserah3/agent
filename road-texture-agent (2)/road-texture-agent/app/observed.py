"""
Coordinates step 3: the road markings the satellite saw, laid where it saw them.

The satellite analysis (app/sat_analysis.py) names the markings it finds in each
tile and rates each tile (100% falling where it could not see well: missing
pictures, shadows, trees, cars, faint paint). Here its findings meet the road
marking presets (app/roadmarks.py):

- in a tile rated at or above the threshold, each crossing, stop line, give way
  line, cyclist crossing and lane arrow the satellite saw is laid where it saw it,
  turned as it lies and as long and wide as it measured, in the preset's paint
  (stripe and gap widths, line widths: the layer's values, by the Saudi rules when
  those are on); the rules lay none of those kinds there, unless "fill" asks them
  to lay what the satellite did not see;
- in a tile below the threshold the rules lay as before: that tile only.

A rule marking lying on one the satellite saw is never laid twice (a crossing on
the line between two tiles). The lane lines stop at the markings laid now: at a
junction whose tile the satellite saw, where its crossings and lines end, and they
are cut where a crossing the satellite saw lies across them.

Pictures (imported decals), the STOP word and the yellow box are not read from the
satellite: pictures and boxes are laid by their layers as before; a STOP word in a
seen tile only before a stop line the satellite saw.

The lane lines (read_lanes): on each street the satellite saw well, the split of its
lanes (within what its width allows, app/lanes.py) whose lines are the ones painted
there; they join the Lanes tool's choices (merge_picks: yours win).
"""
import json
import math
from pathlib import Path

CLASS_PRESET = {"zebra": "zebra", "stop_line": "stop", "giveway": "giveway", "cycle_crossing": "cycle",
                "arrow": "arrows"}
SEEN = ("zebra", "stop", "giveway", "cycle", "arrows")     # the presets the satellite reads
ACROSS = ("zebra", "stop", "giveway", "cycle")             # across the road: the lane lines stop at them
THRESHOLD = 60.0            # a tile rated below this (%): the rules lay its markings
MIN_SCORE = 0.35            # a finding fainter than this is not laid
NEAR_M = 4.0                # fill: a rule marking this near one of its kind the satellite saw is that one
KIND_M = 6.0                # an arrow's kind (straight, left, ...) from the rule arrow this near, pointing its way
KINDS = ("S", "L", "R", "SL", "SR", "LR", "SLR")         # (when the analysis did not read it: before version 3)
WORD_M = 8.0                # a STOP word in a seen tile: only this near a stop line the satellite saw
CLUSTER_GAP_M = 6.0         # a junction's markings: each this near the one before, out from its edge
CELL_M = 20.0


def settings_path(artifacts, gid):
    return Path(artifacts) / f"gen_{gid}_observed.json"


def check_settings(raw):
    """
    {"on", "threshold" (0 to 100), "fill", "key", "satellite_only"} as the page sends
    them, checked. satellite_only: the presets Place decals added for what the satellite
    saw (the layers had none of them): the rules lay none of those, below the threshold too.
    """
    raw = raw if isinstance(raw, dict) else {}
    try:
        t = float(raw.get("threshold", THRESHOLD))
    except (TypeError, ValueError):
        t = THRESHOLD
    t = THRESHOLD if not math.isfinite(t) else min(100.0, max(0.0, t))
    only = [p for p in (raw.get("satellite_only") or []) if p in SEEN]
    return {"on": bool(raw.get("on", True)), "threshold": round(t, 1), "fill": bool(raw.get("fill", False)),
            "key": str(raw.get("key") or ""), "satellite_only": sorted(set(only))}


def load_settings(artifacts, gid):
    f = settings_path(artifacts, gid)
    if not f.exists():
        return None
    try:
        return check_settings(json.loads(f.read_text()))
    except (OSError, ValueError):
        return None


def load_analysis(folder):
    """The finished analysis of an area folder: {"tiles": {id: tile}, "findings", "summary", "size"}, or None."""
    a = Path(folder) / "analysis"
    try:
        t = json.loads((a / "tiles.json").read_text())
        f = json.loads((a / "findings.json").read_text())
    except (OSError, ValueError):
        return None
    tiles = {x["id"]: x for x in t.get("tiles") or []}
    if not tiles:
        return None
    w = max(x["core"][2] for x in tiles.values())
    h = max(x["core"][3] for x in tiles.values())
    return {"tiles": tiles, "findings": f.get("findings") or [], "summary": t.get("summary") or {},
            "size": (w, h), "tile_px": int((t.get("summary") or {}).get("tile_px") or 512)}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _q(v, step):
    return round(round(v / step) * step, 3)


class _Cells:
    """Placements in cells of the map: each looks only at those near it."""

    def __init__(self, mpp):
        self.mpp, self.cell, self.cells, self.far = mpp, max(1.0, CELL_M / max(mpp, 1e-6)), {}, 0.0

    def add(self, p):
        self.cells.setdefault((int(p["x"] // self.cell), int(p["y"] // self.cell)), []).append(p)
        self.far = max(self.far, math.hypot(p["w_m"], p["h_m"]) / self.mpp)

    def near(self, x, y, reach_px):
        r = int(math.ceil((reach_px + self.far) / self.cell))
        ci, cj = int(x // self.cell), int(y // self.cell)
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                yield from self.cells.get((i, j), ())


def merge(placements, ends, layers, analysis, mpp, shape, cfg, colours, threshold=THRESHOLD, fill=False,
          satellite_only=None):
    """
    The rule placements (pictures and presets, as app/server.py's _lay_decals has
    them) and the lines' ends, with what the satellite saw: (placements, ends,
    report). layers: the decal layers (checked); analysis: load_analysis's; mpp:
    the mask's metres per pixel; shape: the mask's (H, W); cfg: the lines'
    settings (the Saudi rules, the drive side); colours: the markings' colours;
    satellite_only: presets the rules do not lay anywhere (check_settings).
    """
    from app import decals as DC, roadmarks as RM
    H, W = shape
    sw, sh = analysis["size"]
    kx, ky = W / max(1, sw), H / max(1, sh)                   # the mask's pixels per satellite pixel (1 when the same)
    tpx = analysis["tile_px"]
    tiles = analysis["tiles"]
    px = max(mpp, 1e-6)
    saudi = cfg.get("standard") == "saudi"
    speed = cfg.get("posted_speed", cfg.get("speed", 60))

    def rating(x, y):
        """The tile a mask point lies in, and how sure the satellite is there (None: not rated)."""
        tid = f"r{int((y / ky) // tpx)}c{int((x / kx) // tpx)}"
        t = tiles.get(tid) or {}
        c = t.get("confidence_marks", t.get("confidence"))
        return tid, (None if c is None else float(c))

    def seen(c):
        return c is not None and c >= threshold

    # each preset's first layer that is on, with the values its markings are painted at
    first = {}
    for li, L in enumerate(layers):
        p = L.get("preset")
        if p and not L.get("off") and p not in first:
            v = dict(L["values"])
            if saudi:
                from app.saudi_rules import preset_values
                v = preset_values(p, v, speed)
            first[p] = (li, dict(L, values=v))
    refs = {}
    if saudi:
        from app.saudi_rules import REFERENCES
        refs = REFERENCES
    preset_of = {i: L.get("preset") for i, L in enumerate(layers)}
    rule_arrows = [p for p in placements if p.get("rm") and preset_of.get(p["layer"]) == "arrows"]
    rep = {"threshold": threshold, "fill": fill, "observed": {}, "rules": {}, "rules_dropped": 0,
           "skipped": {"faint": 0, "low_tile": 0, "overlap": 0}, "no_layer": {}, "not_read": {},
           "tiles": {"seen": sum(1 for t in tiles.values() if seen(t.get("confidence_marks", t.get("confidence")))),
                     "rules": sum(1 for t in tiles.values() if not seen(t.get("confidence_marks", t.get("confidence"))))}}

    # ---- what the satellite saw, as preset markings
    cand = []
    for fd in analysis["findings"]:
        preset = CLASS_PRESET.get(fd.get("cls"))
        if preset is None:
            if fd.get("cls") in ("parking_bay", "lane_dash", "lane_solid"):
                rep["not_read"][fd["cls"]] = rep["not_read"].get(fd["cls"], 0) + 1
            continue
        if preset not in first:
            rep["no_layer"][preset] = rep["no_layer"].get(preset, 0) + 1
            continue
        if float(fd.get("score", 0)) < MIN_SCORE:
            rep["skipped"]["faint"] += 1
            continue
        x, y = float(fd["x"]) * kx, float(fd["y"]) * ky
        tid, conf = rating(x, y)
        if not seen(conf):
            rep["skipped"]["low_tile"] += 1
            continue
        li, L = first[preset]
        v = L["values"]
        a = math.radians(float(fd.get("angle", 0.0)))
        ax, ay = math.cos(a), math.sin(a)                      # the mark's long side
        values, at = v, "intersection"
        if preset == "zebra":
            # its stripes lie along the road: the way in is theirs; as many stripes as span the road it measured
            ux, uy = ax, ay
            per = v["stripe_m"] + v["gap_m"]
            span = float(fd.get("across_m") or fd.get("n", 3) * per)
            n = max(2, int(round((span + v["gap_m"]) / per)))
            w = n * v["stripe_m"] + (n - 1) * v["gap_m"]
            lo, hi = (2.0, 6.0) if saudi else (1.0, 10.0)
            h = _q(_clamp(float(fd.get("stripe_m") or fd.get("length_m") or v["length_m"]), lo, hi), 0.25)
            variant = f"n{n}"
        elif preset == "stop":
            ux, uy = -ay, ax                                   # a line across the road: the way in is square to it
            double = bool(L["opts"].get("double"))
            w = max(0.5, float(fd.get("length_m", 2.0)))
            h = 2 * v["width_m"] + v["gap_m"] if double else v["width_m"]
            variant = "double" if double else "single"
        elif preset == "giveway":
            if fd.get("across_m"):                             # blocks in a row: the road's way, their row's span
                r = math.radians(float(fd.get("road_angle", 0.0)))
                ux, uy, span = math.cos(r), math.sin(r), float(fd["across_m"]) + 0.5
            else:
                ux, uy, span = -ay, ax, float(fd.get("length_m", 2.0))
            n, w = RM.giveway_size(span, v)
            h = 2 * v["width_m"] + v["spacing_m"]
            variant = f"n{n}"
        elif preset == "cycle":
            ux, uy = -ay, ax
            track = _q(_clamp(float(fd.get("track_m") or v["spacing_m"]), 0.5, 6.0), 0.25)
            values = dict(v, spacing_m=track)
            w = max(0.5, float(fd.get("length_m", 2.0)))
            h = track + v["width_m"]
            variant = "c"
        else:                                                  # arrows: pointing the way its head does
            t = fd.get("toward")
            ux, uy = (float(t[0]), float(t[1])) if t else (ax, ay)
            # its kind: as the satellite read its shape; else as the layer sets it, else the rule arrow's near it
            kind = fd.get("kind") if fd.get("kind") in KINDS else L["opts"].get("type", "auto")
            if kind == "auto":
                kind, best = "S", KIND_M / px
                for q in rule_arrows:
                    d = math.hypot(q["x"] - x, q["y"] - y)
                    if d < best and q["ux"] * ux + q["uy"] * uy > 0.5:
                        kind, best = q["rm"], d
            w0, h0 = RM.arrow_size(kind, v)
            h = _q(_clamp(float(fd.get("length_m", h0)), 0.7 * h0, 1.5 * h0), 0.25)
            w = w0 * h / h0
            variant, at = kind, "edge"
        n_ = math.hypot(ux, uy) or 1.0
        p = {"layer": li, "decal": RM.variant_id(li, dict(L, values=values), variant, colours), "rm": variant,
             "x": round(x, 2), "y": round(y, 2), "ux": round(ux / n_, 5), "uy": round(uy / n_, 5),
             "w_m": round(float(w), 3), "h_m": round(float(h), 3), "at": at, "rm_values": dict(values),
             "source": "satellite", "finding": fd.get("id"), "tile": tid, "tile_confidence": round(conf, 1),
             "score": round(float(fd.get("score", 0)), 3)}
        if saudi:
            p["rule_ref"] = refs.get(preset, "Custom decal geometry; no verified SHC dimension")
        cand.append((float(fd.get("score", 0)), preset, p))

    # the surest first; one lying on another already laid is left out
    laid, cells = [], _Cells(px)
    for _, preset, p in sorted(cand, key=lambda c: -c[0]):
        c = DC._corners(p, px, 0.98)
        if any(DC._overlap(c, DC._corners(q, px, 0.98)) for q in cells.near(p["x"], p["y"], math.hypot(p["w_m"], p["h_m"]) / px)):
            rep["skipped"]["overlap"] += 1
            continue
        p["_preset"] = preset
        cells.add(p)
        laid.append(p)
        rep["observed"][preset] = rep["observed"].get(preset, 0) + 1

    # ---- the rules' markings: kept below the threshold (and, with fill, where the satellite saw none of their kind)
    kept = []
    for p in placements:
        preset = preset_of.get(p["layer"]) if p.get("rm") else None
        if preset not in SEEN and preset != "stopword":
            kept.append(p)                                     # pictures, yellow boxes: as their layers lay them
            continue
        if preset in (satellite_only or ()):
            rep["rules_dropped"] += 1
            continue
        tid, conf = rating(p["x"], p["y"])
        c = DC._corners(p, px, 0.98)
        reach = math.hypot(p["w_m"], p["h_m"]) / px
        if any(DC._overlap(c, DC._corners(q, px, 0.98)) for q in cells.near(p["x"], p["y"], reach)):
            rep["rules_dropped"] += 1                          # it lies on one the satellite saw
            continue
        if seen(conf):
            if preset == "stopword":
                ok = any(q["_preset"] == "stop" and math.hypot(q["x"] - p["x"], q["y"] - p["y"]) * px <= WORD_M
                         for q in cells.near(p["x"], p["y"], WORD_M / px))
            else:
                ok = fill and not any(q["_preset"] == preset and math.hypot(q["x"] - p["x"], q["y"] - p["y"]) * px
                                      <= NEAR_M + max(p["w_m"], p["h_m"]) / 2
                                      for q in cells.near(p["x"], p["y"], (NEAR_M + max(p["w_m"], p["h_m"])) / px))
            if not ok:
                rep["rules_dropped"] += 1
                continue
        q = dict(p, source="rule", tile=tid, tile_confidence=None if conf is None else round(conf, 1))
        kept.append(q)
        rep["rules"][preset] = rep["rules"].get(preset, 0) + 1

    # ---- where the lane lines stop
    across = [p for p in laid if p["_preset"] in ACROSS]
    out_ends, i = [], 0
    moved = dropped = 0
    ends = list(ends or [])
    while i < len(ends):
        e = ends[i]
        pair = (e.get("reserve") and i + 1 < len(ends) and "ox" in ends[i + 1] and not ends[i + 1].get("chain"))
        if not pair:
            out_ends.append(e)
            i += 1
            continue
        r, end = e, ends[i + 1]
        i += 2
        _, conf = rating(r["reserve"]["cx"], r["reserve"]["cy"])
        if not seen(conf):
            out_ends += [r, end]
            continue
        # the crossings and lines the satellite saw at this junction edge: out from it, one after another
        ox, oy, reach = end["ox"], end["oy"], min(float(end.get("reach_m", 30.0)), 40.0)
        found = []
        for p in across:
            dx, dy = p["x"] - end["x"], p["y"] - end["y"]
            fwd, side = (dx * ox + dy * oy) * px, abs(-dx * oy + dy * ox) * px
            if -1.0 <= fwd <= reach and side <= float(end.get("half_m", 6.0)) + 1.0 and abs(p["ux"] * ox + p["uy"] * oy) >= 0.8:
                found.append((fwd - p["h_m"] / 2, fwd + p["h_m"] / 2))
        far = 0.0
        for near_, far_ in sorted(found):
            if near_ > far + CLUSTER_GAP_M:
                break
            far = max(far, far_)
        if far <= 0:
            if fill:
                out_ends += [r, end]                           # the rules' markings stay there
            else:
                dropped += 1                                   # none seen: the lines run to the junction's edge
            continue
        if fill:
            far = max(far, float(end["m"]))
        moved += 1
        cx, cy = end["x"] + ox * far / 2 / px, end["y"] + oy * far / 2 / px
        out_ends.append({"x": cx, "y": cy, "reserve": dict(r["reserve"], cx=cx, cy=cy, h=(far + 0.4) / px)})
        out_ends.append(dict(end, m=round(far, 3)))
    # every crossing and line the satellite saw cuts the lane lines it lies across
    for p in across:
        out_ends.append({"x": p["x"], "y": p["y"], "observed": True,
                         "reserve": {"cx": p["x"], "cy": p["y"], "w": p["w_m"] / px, "h": (p["h_m"] + 0.4) / px,
                                     "angle": math.degrees(math.atan2(p["ux"], -p["uy"]))}})
    rep["ends"] = {"moved": moved, "dropped": dropped}
    for p in laid:
        p.pop("_preset", None)
    rep["laid"] = len(laid)
    return kept + laid, out_ends, rep


# ------------------------------------------------------------------ the lane lines
LINE_REACH_M = 0.45         # a painted line this near where a lane split puts one is that line
END_M = 8.0                 # a street's lines read this far from its ends (junctions' lines cross there)
KERB_M = 1.0                # lines this near the kerb are edge lines: not counted
SHIFT_M = 1.2               # the street map may lie this far off the satellite: the lines looked for shifted so far
SEEN_SHARE = 0.6            # a street read when this much of it lies in tiles at or above the threshold
NONE_MIN_M = 25.0           # "no lines" only on a street at least this long (a dashed line repeats every 9 to 12 m)


def read_lanes(streets, analysis, mpp, shape, cfg, threshold=THRESHOLD, fill=False):
    """
    Each street's lanes as the satellite shows them, for the Lanes tool's choices:
    ([{"pt", "sides", "src": "satellite"}], report). streets: the generation's
    streets (its streets file: id, line (x, y mask pixels, the canonical way),
    width_m). On a street the satellite saw well (SEEN_SHARE of it in tiles at or
    above the threshold), the painted lines along it (dashes and solid lines, not
    the edge lines at the kerbs) are set against every split the street can hold:
    each split's lines where app/lanes.py's layout puts them (equal lanes, a raised
    island from 22 m), the street map allowed to lie up to SHIFT_M off; the split
    whose lines are painted, with no painted line left over, wins (ties: two ways
    over one, unless all its arrows point one way; then the split whose centre line
    is painted solid where the guide puts a solid one; then the even split). A
    street seen with no lines at all gets none (unless fill: the rules' lanes).
    Only splits that differ from the rules' even split are returned.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from app import lanes as LN
    H, W = shape
    sw, sh = analysis["size"]
    kx, ky = W / max(1, sw), H / max(1, sh)
    tpx, tiles = analysis["tile_px"], analysis["tiles"]
    px = max(mpp, 1e-6)
    rules, hw = cfg.get("rules", "simple"), cfg.get("highway")
    drive = cfg.get("drive", "right")
    min_m = float(cfg.get("min_marking_m", 6.0))
    rep = {"read": 0, "same": 0, "changed": 0, "none": 0, "not_seen": 0, "narrow": 0, "short": 0, "streets": []}

    def conf(x, y):
        t = tiles.get(f"r{int((y / ky) // tpx)}c{int((x / kx) // tpx)}") or {}
        c = t.get("confidence_marks", t.get("confidence"))
        return None if c is None else float(c)

    # every street's centreline, its tangents and distance along it
    geo = {}
    pts, labs, idx = [], [], []
    for s in streets:
        P = _dense(np.asarray(s.get("line") or [], float).reshape(-1, 2), max(1.0, 0.5 / px))
        if len(P) < 4:
            continue
        T = np.gradient(P, axis=0)
        k = max(1, min(15, len(P) // 6))
        T = np.vstack([np.convolve(T[:, 0], np.ones(2 * k + 1) / (2 * k + 1), "same"),
                       np.convolve(T[:, 1], np.ones(2 * k + 1) / (2 * k + 1), "same")]).T
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]) * px
        geo[s["id"]] = (P, T, d, float(s["width_m"]))
        pts.append(P)
        labs.append(np.full(len(P), s["id"]))
        idx.append(np.arange(len(P)))
    if not geo:
        return [], rep
    tree = cKDTree(np.concatenate(pts))
    labs, idx = np.concatenate(labs), np.concatenate(idx)

    # the painted lines along the streets, and the arrows (which way the lanes run)
    lines, arrows = {}, {}
    for fd in analysis["findings"]:
        cls = fd.get("cls")
        if cls not in ("lane_dash", "lane_solid", "arrow"):
            continue
        x, y = float(fd["x"]) * kx, float(fd["y"]) * ky
        dist, j = tree.query([x, y])
        sid, i = int(labs[j]), int(idx[j])
        P, T, d, w_m = geo[sid]
        if dist * px > w_m / 2 + 1.0 or d[i] < END_M or d[i] > d[-1] - END_M:
            continue
        tx, ty = T[i]
        off = ((x - P[i, 0]) * -ty + (y - P[i, 1]) * tx) * px      # across, metres: + to (-ty, tx), side 2
        if cls == "arrow":
            t = fd.get("toward")
            if t:
                arrows.setdefault(sid, []).append((off, float(t[0]) * tx + float(t[1]) * ty))
            continue
        a = math.radians(float(fd.get("angle", 0.0)))
        if abs(math.cos(a) * tx + math.sin(a) * ty) < math.cos(math.radians(15)):
            continue                                                   # not along the street
        lines.setdefault(sid, []).append((off, float(fd.get("length_m", 1.0)), cls == "lane_solid"))

    picks = []
    for s in streets:
        sid = s["id"]
        if sid not in geo:
            continue
        P, T, d, w_m = geo[sid]
        if w_m + 1e-6 < min_m:
            rep["narrow"] += 1
            continue
        # seen well enough? a sample every 5 m
        step = max(1, int(round(5.0 / px)))
        cs = [conf(x, y) for x, y in P[::step]]
        if sum(1 for c in cs if c is not None and c >= threshold) < SEEN_SHARE * len(cs):
            rep["not_seen"] += 1
            continue
        usable = max(0.0, d[-1] - 2 * END_M)
        half = w_m / 2
        obs = [(o, l, sol) for o, l, sol in lines.get(sid, []) if abs(o) <= half + SHIFT_M]
        need = max(1.5, 0.12 * usable)                                  # a line is there: this much of it painted
        even = LN.layout(w_m, None, rules=rules, hw=hw)["sides"]
        if not obs:
            if fill:
                continue
            if usable < NONE_MIN_M:
                rep["short"] += 1
                continue
            picks.append({"pt": _mid(P), "sides": [0, 0], "src": "satellite"})
            rep["none"] += 1
            rep["read"] += 1
            rep["streets"].append({"id": sid, "sides": [0, 0], "rules": list(even), "lines": 0})
            continue
        lim = LN.limit(w_m)
        cands = [(a, b) for a in range(lim + 1) for b in range(lim + 1 - a) if a + b >= 2] + [(0, 0)]
        lays = {c: [(x, k) for x, k in LN.layout(w_m, c, rules=rules, hw=hw)["lines"] if abs(x) <= half - KERB_M]
                for c in cands}
        score = {}                                                    # each split: its best (score, shift, lines)
        for shift in np.arange(-SHIFT_M, SHIFT_M + 1e-6, 0.1):
            inner = sorted((x - shift, l, sol) for x, l, sol in obs if abs(x - shift) <= half - KERB_M)
            # the painted lines, gathered: each a place across and how much of it is painted
            clusters = []
            for x, l, sol in inner:
                if clusters and x - clusters[-1]["x1"] <= 0.5:
                    c = clusters[-1]
                    c["x1"], c["len"], c["sx"] = x, c["len"] + l, c["sx"] + x * l
                    c["solid"] += l if sol else 0.0
                else:
                    clusters.append({"x1": x, "len": l, "solid": l if sol else 0.0, "sx": x * l})
            real = [dict(c, x=c["sx"] / c["len"]) for c in clusters if c["len"] >= need]
            for c in cands:
                hit, used = 0, set()
                for x, k in lays[c]:
                    near = [n for n, r in enumerate(real) if abs(r["x"] - x) <= LINE_REACH_M and n not in used]
                    if near:
                        used.add(min(near, key=lambda n: abs(real[n]["x"] - x)))
                        hit += 1
                # each line where the split puts one and painted: +1; where none is painted: -1; a painted
                # line the split has no place for: -2
                sc = 2 * hit - len(lays[c]) - 2 * (len(real) - len(used))
                if c not in score or (sc, -abs(shift)) > (score[c][0], -abs(score[c][1])):
                    score[c] = (sc, round(float(shift), 1), len(real))
        top = max(v[0] for v in score.values())
        ties = [c for c in cands if score[c][0] == top]
        pick = ties[0]
        if len(ties) > 1:
            # the same lines: which way the lanes run (arrows on both halves, all one way: one-way), where the
            # centre is (a solid line where the guide puts one), else the even split
            arr = [(x, t) for x, t in arrows.get(sid, []) if abs(t) > 0.7]
            halves = {x > 0 for x, _ in arr if abs(x) > 0.5}
            one_way = len(arr) >= 2 and len(halves) == 2 and len({t > 0 for _, t in arr}) == 1
            N = sum(ties[0])
            if one_way:
                # driving on the right, the lanes running along the street (+) are side 2's
                plus = arr[0][1] > 0
                one = (0, N) if (plus if drive == "right" else not plus) else (N, 0)
            two = [c for c in ties if c[0] >= 1 and c[1] >= 1]
            if one_way and one in ties:
                pick = one
            elif two:
                sh = score[two[0]][1]

                def kind_fit(c):
                    return sum(1 for x, k in lays[c] if k in LN.SOLID_KINDS
                               and any(abs(o - sh - x) <= LINE_REACH_M and sol for o, _, sol in obs))

                both = len({t > 0 for _, t in arr}) == 2

                def ways_fit(c):
                    # driving on the right, side 1's lanes (left of the centre) run against the street (-),
                    # side 2's along it (+): the arrows on the right side of the centre for their way (only
                    # when the arrows show both ways; one closer than 0.5 m to the centre tells nothing)
                    if not both:
                        return 0
                    mid = LN.lane_spans(LN.layout(w_m, c, rules=rules, hw=hw))["centre"]
                    if mid is None:
                        return 0
                    lo, hi = mid if isinstance(mid, tuple) else (mid, mid)
                    right = drive == "right"
                    return sum(1 for x, t in arr if x - sh < lo - 0.5 and (t < 0) == right
                               or x - sh > hi + 0.5 and (t > 0) == right)
                pick = max(two, key=lambda c: (ways_fit(c), kind_fit(c), -abs(c[0] - c[1]), c == tuple(even)))
        best = (score[pick][0], pick, 0, score[pick][1], score[pick][2])
        rep["read"] += 1
        rep["streets"].append({"id": sid, "sides": list(pick), "rules": list(even), "lines": best[4], "shift_m": best[3]})
        if tuple(pick) == tuple(even):
            rep["same"] += 1
            continue
        rep["changed"] += 1
        if pick == (0, 0):
            rep["none"] += 1
        picks.append({"pt": _mid(P), "sides": list(pick), "src": "satellite"})
    return picks, rep


def _dense(P, step):
    """A polyline with a point every step (its own points kept): streets files keep straight streets as their two ends."""
    import numpy as np
    if len(P) < 2:
        return P
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.vstack(out)


def _mid(P):
    """A street's middle point, on its centreline (mask pixels, rounded as the Lanes tool keeps them)."""
    m = P[len(P) // 2]
    return [round(float(m[0]), 1), round(float(m[1]), 1)]


def merge_picks(saved, satellite):
    """
    The Lanes tool's choices with the satellite's: the satellite's first and yours
    after (a later choice wins a street picked twice: yours over the satellite's);
    the satellite's from before taken away.
    """
    mine = [p for p in saved or [] if p.get("src") != "satellite"]
    return list(satellite or []) + mine
