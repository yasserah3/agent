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
