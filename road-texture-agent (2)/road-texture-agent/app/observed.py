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
            "size": (w, h), "tile_px": int((t.get("summary") or {}).get("tile_px") or 512), "folder": Path(folder)}


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
NONE_MIN_M = 25.0           # a street read only when this much of it lies clear of its ends (dashes repeat every 9 to 12 m)
SECTION_M = 40.0            # a street's lines read in sections this long (half overlapping): each its own offset
TRACE_REACH_M = 0.6         # a line followed through the paint this near where it should be
GAP_M = 20.0                # no paint for this long along a line (two dash cycles): the line stops there and on after
FOLLOW_M = 10.0             # a line followed by where its last this many metres head


def read_lanes(streets, analysis, mpp, shape, cfg, threshold=THRESHOLD, fill=False):
    """
    Each street's lanes as the satellite shows them, and the path of each of its lines,
    for the Lanes tool's choices: ([{"pt", "sides", "src": "satellite", "trace"}],
    report). streets: the generation's streets (its streets file: id, line (x, y mask
    pixels, the canonical way), width_m).

    On a street the satellite saw well (SEEN_SHARE of it in tiles at or above the
    threshold), the painted lines along it (dashes and solid lines, not the edge lines at
    the kerbs) are read in sections SECTION_M long: in each, every split the street can
    hold has its lines laid where app/lanes.py puts them (equal lanes, a raised island
    from 22 m), the street map allowed to lie up to SHIFT_M off the paint there, and set
    against the painted lines (a line where the split puts one and painted +1, where none
    is painted -1, a painted line the split has no place for -2). The split with the
    most over all sections wins (ties: two ways over one unless all its arrows point one
    way; which side has more lanes from the arrows' ways; a solid centre line where the
    guide puts one; the even split). A street seen with no lines at all gets none
    (unless fill: the rules' lanes).

    Then every line painted along it is followed (_trace), however many there are (a
    lane added or ending part way is a line that starts or stops): it bends, drifts and
    weaves as the paint does, stops where the paint stops for GAP_M or more (not where
    the road is only hidden: shadow, trees, a car), and within 15 m of the street's end
    runs on to it (the junction's markings stop it). Each is kept with the street's lane
    choice ("trace") and drawn where it is painted (app/lanes.py traced_offsets), dashed
    or solid as the satellite shows it, in the dash, gap, width and colour the Road
    markings panel sets. The kerb edge lines are left to the rules.
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
    rep = {"read": 0, "same": 0, "changed": 0, "none": 0, "not_seen": 0, "narrow": 0, "short": 0, "traced": 0,
           "streets": []}

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
        geo[s["id"]] = (P, T, d, float(s["width_m"]), cKDTree(P))
        pts.append(P)
        labs.append(np.full(len(P), s["id"]))
        idx.append(np.arange(len(P)))
    if not geo:
        return [], rep
    tree = cKDTree(np.concatenate(pts))
    labs, idx = np.concatenate(labs), np.concatenate(idx)

    # the painted lines along the streets, as samples (along, across, metres of paint, solid), and the arrows
    samples, arrows = {}, {}
    for fd in analysis["findings"]:
        cls = fd.get("cls")
        if cls not in ("lane_dash", "lane_solid", "arrow"):
            continue
        x, y = float(fd["x"]) * kx, float(fd["y"]) * ky
        dist, j = tree.query([x, y])
        sid, i = int(labs[j]), int(idx[j])
        P, T, d, w_m, stree = geo[sid]
        if dist * px > w_m / 2 + 1.0 or d[i] < END_M or d[i] > d[-1] - END_M:
            continue
        tx, ty = T[i]
        is_line = cls != "arrow"
        if cls == "arrow":
            t = fd.get("toward")
            if t:
                off = ((x - P[i, 0]) * -ty + (y - P[i, 1]) * tx) * px
                arrows.setdefault(sid, []).append((off, float(t[0]) * tx + float(t[1]) * ty))
            # a mark a dash's size the analysis took for an arrow (a dash or two on their own): offered to the
            # lines' following as paint they may take, never counted as a line by itself
            if not (1.5 <= float(fd.get("length_m", 0)) <= 4.5 and float(fd.get("width_m", 9)) <= 0.6):
                continue
        a = math.radians(float(fd.get("angle", 0.0)))
        if abs(math.cos(a) * tx + math.sin(a) * ty) < math.cos(math.radians(15)):
            continue                                                   # not along the street
        L = float(fd.get("length_m", 1.0))
        if fd.get("pts") and len(fd["pts"]) >= 2:                      # a solid line's own path (analysis 4)
            line = np.asarray(fd["pts"], float) * [kx, ky]
        else:                                                          # a dash: its two ends
            h = L / 2 / px
            line = np.array([[x - math.cos(a) * h, y - math.sin(a) * h], [x + math.cos(a) * h, y + math.sin(a) * h]])
        Q = _dense(line, max(1.0, 0.5 / px))
        _, ii = stree.query(Q)
        dd = Q - P[ii]
        ss = d[ii] + (dd * T[ii]).sum(1) * px
        oo = (dd[:, 0] * -T[ii, 1] + dd[:, 1] * T[ii, 0]) * px          # across, metres: + to (-ty, tx), side 2
        wgt = L / len(Q)
        samples.setdefault(sid, []).extend((float(a_), float(b_), wgt, cls == "lane_solid", is_line)
                                           for a_, b_ in zip(ss, oo))

    hidden = _hidden_at(analysis, kx, ky, px)

    def judge(obs, cands, lays, half, need):
        """Each split's best (score, shift) against these samples (across, metres of paint, solid)."""
        out = {}
        for shift in np.arange(-SHIFT_M, SHIFT_M + 1e-6, 0.15):
            inner = sorted((x - shift, l, sol) for x, l, sol in obs if abs(x - shift) <= half - KERB_M)
            clusters = []
            for x, l, sol in inner:
                if clusters and x - clusters[-1]["x1"] <= 0.5:
                    c = clusters[-1]
                    c["x1"], c["len"], c["sx"] = x, c["len"] + l, c["sx"] + x * l
                    c["solid"] += l if sol else 0.0
                else:
                    clusters.append({"x1": x, "len": l, "solid": l if sol else 0.0, "sx": x * l})
            real = [c["sx"] / c["len"] for c in clusters if c["len"] >= need]
            for c in cands:
                hit, used = 0, set()
                for x, _ in lays[c]:
                    near = [n for n, r in enumerate(real) if abs(r - x) <= LINE_REACH_M and n not in used]
                    if near:
                        used.add(min(near, key=lambda n: abs(real[n] - x)))
                        hit += 1
                sc = 2 * hit - len(lays[c]) - 2 * (len(real) - len(used))
                if c not in out or (sc, -abs(shift)) > (out[c][0], -abs(out[c][1])):
                    out[c] = (sc, round(float(shift), 2), len(real))
        return out

    picks = []
    for s in streets:
        sid = s["id"]
        if sid not in geo:
            continue
        P, T, d, w_m, _ = geo[sid]
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
        allp = [q for q in samples.get(sid, []) if abs(q[1]) <= half + SHIFT_M]
        smp = [q[:4] for q in allp if q[4]]                            # the lines' own paint: counted
        even = LN.layout(w_m, None, rules=rules, hw=hw)["sides"]
        if usable < NONE_MIN_M:
            rep["short"] += 1                                          # too little of it to read: the rules
            continue
        if not smp:
            if fill:
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
        # section by section: each split's score, and how far the street map lies off the paint there
        lo, hi = END_M, max(END_M + 1e-6, d[-1] - END_M)
        n_sec = max(1, int(math.ceil((hi - lo) / (SECTION_M / 2))) - 1) if hi - lo > SECTION_M else 1
        centres = [lo + (hi - lo) / 2] if n_sec == 1 else list(np.linspace(lo + SECTION_M / 2, hi - SECTION_M / 2, n_sec))
        win = min(SECTION_M, hi - lo)
        need = max(1.5, 0.12 * win)
        total, secs = {c: 0 for c in cands}, []
        for sc_ in centres:
            obs = [(o, l, sol) for a_, o, l, sol in smp if abs(a_ - sc_) <= win / 2]
            if sum(l for _, l, _ in obs) < need:
                continue
            res = judge(obs, cands, lays, half, need)
            secs.append((sc_, res))
            for c in cands:
                total[c] += res[c][0]
        if not secs:
            rep["short"] += 1
            continue
        top = max(total.values())
        ties = [c for c in cands if total[c] == top]
        pick = ties[0]
        if len(ties) > 1:
            # the same lines: which way the lanes run (arrows on both halves, all one way: one-way), where the
            # centre is (the arrows' ways, a solid line where the guide puts one), else the even split
            arr = [(x, t) for x, t in arrows.get(sid, []) if abs(t) > 0.7]
            halves = {x > 0 for x, _ in arr if abs(x) > 0.5}
            one_way = len(arr) >= 2 and len(halves) == 2 and len({t > 0 for _, t in arr}) == 1
            N = sum(ties[0])
            one = None
            if one_way:
                # driving on the right, the lanes running along the street (+) are side 2's
                plus = arr[0][1] > 0
                one = (0, N) if (plus if drive == "right" else not plus) else (N, 0)
            two = [c for c in ties if c[0] >= 1 and c[1] >= 1]
            if one_way and one in ties:
                pick = one
            elif two:
                sh = float(np.median([r[two[0]][1] for _, r in secs]))
                both = len({t > 0 for _, t in arr}) == 2

                def kind_fit(c):
                    return sum(1 for x, k in lays[c] if k in LN.SOLID_KINDS
                               and any(abs(o - sh - x) <= LINE_REACH_M and sol for _, o, _, sol in smp))

                def ways_fit(c):
                    # driving on the right, side 1's lanes (left of the centre) run against the street (-),
                    # side 2's along it (+): the arrows on the right side of the centre for their way (only
                    # when the arrows show both ways; one closer than 0.5 m to the centre tells nothing)
                    if not both:
                        return 0
                    mid = LN.lane_spans(LN.layout(w_m, c, rules=rules, hw=hw))["centre"]
                    if mid is None:
                        return 0
                    lo_, hi_ = mid if isinstance(mid, tuple) else (mid, mid)
                    right = drive == "right"
                    return sum(1 for x, t in arr if x - sh < lo_ - 0.5 and (t < 0) == right
                               or x - sh > hi_ + 0.5 and (t > 0) == right)
                pick = max(two, key=lambda c: (ways_fit(c), kind_fit(c), -abs(c[0] - c[1]), c == tuple(even)))
        # how far the street map lies off the paint, along the street (each section's, for the split chosen)
        sc_at = np.array([c for c, _ in secs])
        sh_at = np.array([r[pick][1] for _, r in secs])
        if len(sh_at) >= 3:
            sh_at = np.array([np.median(sh_at[max(0, i - 1):i + 2]) for i in range(len(sh_at))])
        traces = _trace(allp, half, P, T, d, px, hidden)
        n_lines = len(lays[pick])
        rep["read"] += 1
        rep["traced"] += 1 if traces else 0
        rep["streets"].append({"id": sid, "sides": list(pick), "rules": list(even), "lines": n_lines,
                               "traced": len(traces), "drift_m": round(float(np.max(np.abs(sh_at))), 2)})
        same = tuple(pick) == tuple(even)
        rep["same" if same else "changed"] += 1
        if pick == (0, 0):
            rep["none"] += 1
        if same and not traces:
            continue
        q = {"pt": _mid(P), "sides": list(pick), "src": "satellite"}
        if traces:
            q["trace"] = traces
        picks.append(q)
    return picks, rep


def _hidden_at(analysis, kx, ky, px):
    """
    How much of a path (N x 2 mask pixels) the satellite cannot see the road on: in
    shadow (darker than 55% of the road's usual), under trees (green), or under a car it
    found (within 2.5 m). None when the analysis' pictures are gone.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    a = analysis.get("folder")
    try:
        sat = np.load(Path(a) / "analysis" / "sat.npy", mmap_mode="r")
        lum_med = float(json.loads((Path(a) / "analysis" / "stats.json").read_text())["lum_median"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    cars = np.array([[float(f["x"]) * kx, float(f["y"]) * ky] for f in analysis["findings"] if f.get("cls") == "car"])
    ctree = cKDTree(cars) if len(cars) else None
    H, W = sat.shape[:2]

    def hidden(q):
        q = np.asarray(q, float)
        if not len(q):
            return 0.0
        xs = np.clip((q[:, 0] / kx).astype(int), 0, W - 1)
        ys = np.clip((q[:, 1] / ky).astype(int), 0, H - 1)
        rgb = np.asarray(sat[ys, xs], np.float32)
        lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        out = (lum < 0.55 * lum_med) | ((rgb[:, 1] > rgb[:, 0] + 10) & (rgb[:, 1] > rgb[:, 2] + 6))
        if ctree is not None:
            out |= ctree.query(q)[0] * px <= 2.5
        return float(out.mean())
    return hidden


def _trace(smp, half, P, T, d, px, hidden=None):
    """
    Every painted line along a street followed (as many as are painted, whatever the
    split; a lane added or ending part way is one that starts or stops): from the 10 m
    where the paint not yet followed is thickest, step by step forwards and back, each next
    piece of paint taken when it lies near where the last FOLLOW_M of the line heads (so it
    bends, drifts and weaves as the paint does); then smoothed into its path. The edge
    lines at the kerbs are left to the rules. smp: the paint samples (along, across,
    metres, solid). [{"pts": [[x, y], ...], "gaps": [[[x, y], [x, y]], ...]}] (mask
    pixels, across the street in order; gaps: where no paint runs for GAP_M or more, from
    where it stops to where it starts again: left unpainted). Within 15 m of an end of
    the street's read part a line runs on to the street's end (the junction's markings
    stop it there).
    """
    import numpy as np
    if not smp:
        return []
    S = np.array([q[0] for q in smp])
    O = np.array([q[1] for q in smp])
    LINE = np.array([q[4] if len(q) > 4 else True for q in smp], bool)
    SOLID = np.array([bool(q[3]) for q in smp], bool)
    order = np.argsort(S)
    S, O, LINE, SOLID = S[order], O[order], LINE[order], SOLID[order]
    taken = np.zeros(len(S), bool)
    out, found = [], []
    for _ in range(40):
        free = np.flatnonzero(~taken & LINE)                        # a line starts from lines' paint
        if len(free) < 6:
            break
        # the seed: the 10 m along and half metre across with the most paint not yet followed
        key = (np.floor(S[free] / 10.0).astype(int) * 100000 + np.round(O[free] / 0.5).astype(int))
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        if cnt.max() < 6:
            break
        seed = free[inv == int(np.argmax(cnt))]
        o0 = float(np.median(O[seed]))
        seed = free[(np.abs(S[free] - np.median(S[seed])) <= 5.0) & (np.abs(O[free] - o0) <= 0.35)]
        c0 = float(np.median(S[seed]))
        acc = np.zeros(len(S), bool)
        acc[seed] = True
        for step in (1, -1):
            idx = np.flatnonzero(~taken & ~acc)
            idx = idx[S[idx] > c0] if step > 0 else idx[S[idx] < c0][::-1]
            front = c0
            for i in idx:
                back = acc & (np.abs(S - front) <= FOLLOW_M) & ((S <= front) if step > 0 else (S >= front))
                if back.sum() >= 3 and np.ptp(S[back]) > 2.0:
                    b, a = np.polyfit(S[back], O[back], 1)
                    b = float(np.clip(b, -0.15, 0.15))                    # a line drifts slowly
                    pred = a + b * S[i]
                else:
                    near = acc & (np.abs(S - front) <= FOLLOW_M)
                    pred = float(np.median(O[near])) if near.any() else o0
                gap = abs(S[i] - front)
                if abs(O[i] - pred) <= 0.35 + 0.03 * min(gap, 20.0):    # further from its last paint: less sure
                    acc[i] = True
                    front = S[i]
        taken |= acc
        taken[seed] = True
        if acc.sum() >= 4:
            found.append(acc)
    # pieces of one line (it was lost for a while: worn paint, a car over it, a sharp bend): joined when they
    # never lie apart where both are painted, and one runs on from the other (within 12 m, where it heads)
    def one_line(a, b):
        sa, oa, sb, ob = S[a], O[a], S[b], O[b]
        j = np.searchsorted(sa, sb)
        for k in (np.clip(j - 1, 0, len(sa) - 1), np.clip(j, 0, len(sa) - 1)):
            both = np.abs(sa[k] - sb) <= 1.0
            if np.any(both & (np.abs(oa[k] - ob) > 0.4)):
                return False                                           # side by side: two lines
        for end in (sb.min(), sb.max()):
            na = np.abs(sa - end) <= 12.0
            nb = np.abs(sb - end) <= 6.0
            if na.any() and abs(float(np.median(oa[na])) - float(np.median(ob[nb]))) <= 0.6:
                return True
        return False
    found.sort(key=lambda a: -int(a.sum()))
    joined = []
    for acc in found:
        for j, prev in enumerate(joined):
            if one_line(prev, acc):
                joined[j] = prev | acc
                break
        else:
            joined.append(acc)
    for acc in joined:
        s, o = S[acc], O[acc]
        if len(s) < 4 or s[-1] - s[0] < 10.0 or abs(float(np.median(o))) > half - KERB_M or LINE[acc].mean() < 0.3:
            continue                                                   # too short to be sure, an edge line, or arrows
        # a station every 2 m: the median of its paint within 3 m, then smoothed lightly
        st = np.arange(s[0], s[-1] + 1e-6, 2.0)
        med = np.array([np.median(o[np.abs(s - t) <= 3.0]) if np.any(np.abs(s - t) <= 3.0) else np.nan for t in st])
        ok = ~np.isnan(med)
        med = np.interp(st, st[ok], med[ok])
        prof = np.convolve(np.pad(med, 1, mode="edge"), np.ones(3) / 3, "valid")
        # to the street's ends when the paint reaches near them (the markings or the setback stop it there)
        if st[0] <= END_M + 15.0:
            st, prof = np.concatenate([[0.0], st]), np.concatenate([[prof[0]], prof])
        if st[-1] >= d[-1] - END_M - 15.0:
            st, prof = np.concatenate([st, [d[-1]]]), np.concatenate([prof, [prof[-1]]])

        def at(q, st=st, prof=prof):
            q = np.atleast_1d(np.asarray(q, float))
            cx, cy = np.interp(q, d, P[:, 0]), np.interp(q, d, P[:, 1])
            tx, ty = np.interp(q, d, T[:, 0]), np.interp(q, d, T[:, 1])
            n = np.maximum(np.hypot(tx, ty), 1e-9)
            o_ = np.interp(q, st, prof)
            return np.column_stack([cx + (-ty / n) * o_ / px, cy + (tx / n) * o_ / px])
        tr = {"pts": _simplify(at(np.arange(st[0], st[-1] + 1e-6, 1.0) if st[-1] - st[0] > 1 else st), 0.03 / px),
              # how much of its own paint is solid (not broken): dashed or solid as the satellite shows it
              "solid": round(float(SOLID[acc & LINE].mean()) if (acc & LINE).any() else 0.0, 2)}
        # where the paint stops for a while (a junction the street runs through, a lane that ends); not where
        # the road is hidden (shadow, trees, a car): the paint may run on under them
        gaps = [(s[i] + 1.0, s[i + 1] - 1.0) for i in range(len(s) - 1) if s[i + 1] - s[i] >= GAP_M]
        if hidden is not None:
            gaps = [(a_, b_) for a_, b_ in gaps if hidden(at(np.arange(a_, b_, 0.5))) < 0.3]
        if gaps:
            tr["gaps"] = [[[round(float(v), 1) for v in q] for q in at([a_, b_])] for a_, b_ in gaps]
        out.append((float(np.median(o)), tr))
    return [tr for _, tr in sorted(out, key=lambda q: q[0])]


def _simplify(line, tol):
    """A path with only the points it needs to stay within tol of itself (rounded as choices keep them)."""
    try:
        from shapely.geometry import LineString
        q = list(LineString(line).simplify(tol).coords)
    except Exception:
        q = [tuple(p) for p in line]
    return [[round(float(x), 1), round(float(y), 1)] for x, y in q]


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
