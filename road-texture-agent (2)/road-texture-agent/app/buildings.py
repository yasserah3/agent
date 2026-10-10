"""
Buildings made in the Dynamic creation tab: a stack of levels, each a box for now.

A building is {"id", "name", "preset", "levels": [...], "object"?}: its levels from
the ground up, each {"kind": "floor" or "roof", "h": height, "rect": {"x", "z", "w",
"l"} or None (not drawn yet)}, in metres, Y up: x and w across (X), z and l front to
back (Z), the front of the building towards -Z (the top of the map at rotation 0,
where an object's +Y points). A level stands on the levels below it: its base is the
sum of their heights, drawn or not. Levels are named by where they are: the first
is the ground floor, then the first floor, the second, and so on; the roof is always
the last level, and there is one at most.

"object": the object layer made from it (Use as object), kept so that sending it
again updates that layer and every placement of it.
"""

import numpy as np

ORDINALS = ["Ground", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
            "Tenth", "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth",
            "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth"]
MAX_LEVELS = 200
# the levels' colours, as in the tab: the ground floor, the floors above (in turn), the roof
GROUND_RGB = (0.80, 0.69, 0.55)
FLOOR_RGB = [(0.86, 0.85, 0.82), (0.76, 0.79, 0.83)]
ROOF_RGB = (0.45, 0.47, 0.50)


def level_name(levels, i):
    """A level's name from its place: Ground floor, First floor, ..., Roof."""
    if levels[i].get("kind") == "roof":
        return "Roof"
    return ("%s floor" % ORDINALS[i]) if i < len(ORDINALS) else "Floor %d" % i


def _num(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if v != v else min(hi, max(lo, v))


def check(raw):
    """A building as sent by the page, checked: sizes in range, at most one roof, and it last."""
    raw = raw if isinstance(raw, dict) else {}
    levels = []
    for lv in (raw.get("levels") or [])[:MAX_LEVELS]:
        if not isinstance(lv, dict):
            continue
        kind = "roof" if lv.get("kind") == "roof" else "floor"
        out = {"kind": kind, "h": round(_num(lv.get("h"), 0.1, 100.0, 3.0), 3)}
        r = lv.get("rect")
        if isinstance(r, dict):
            out["rect"] = {"x": round(_num(r.get("x"), -5000, 5000, 0.0), 3), "z": round(_num(r.get("z"), -5000, 5000, 0.0), 3),
                           "w": round(_num(r.get("w"), 0.5, 2000, 10.0), 3), "l": round(_num(r.get("l"), 0.5, 2000, 10.0), 3)}
        else:
            out["rect"] = None
        levels.append(out)
    roofs = [lv for lv in levels if lv["kind"] == "roof"]
    levels = [lv for lv in levels if lv["kind"] != "roof"] + roofs[-1:]     # one roof, and on top
    name = str(raw.get("name") or "Building").strip()[:80] or "Building"
    out = {"name": name, "preset": str(raw.get("preset") or "block")[:40], "levels": levels}
    if raw.get("object"):
        out["object"] = str(raw["object"])[:32]
    return out


def boxes(b):
    """The drawn levels as boxes: (name, kind, colour, x0, x1, y0, y1, z0, z1), from the ground up."""
    out, base = [], 0.0
    levels = b["levels"]
    for i, lv in enumerate(levels):
        r = lv.get("rect")
        if r:
            colour = (ROOF_RGB if lv["kind"] == "roof" else GROUND_RGB if i == 0 else FLOOR_RGB[(i - 1) % 2])
            out.append((level_name(levels, i), lv["kind"], colour, r["x"] - r["w"] / 2, r["x"] + r["w"] / 2,
                        base, base + lv["h"], r["z"] - r["l"] / 2, r["z"] + r["l"] / 2))
        base += lv["h"]
    return out


def parts(b):
    """
    The building as an object's parts (app/objects.py): a box for each drawn level,
    flat-shaded (four corners for each face, so every face keeps its own normal).
    """
    from app import objects as OB
    out = []
    for name, kind, colour, x0, x1, y0, y1, z0, z1 in boxes(b):
        c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
        # each face's corners counter-clockwise seen from outside
        faces = [(4, 5, 6, 7), (1, 0, 3, 2), (5, 1, 2, 6), (0, 4, 7, 3), (3, 7, 6, 2), (0, 1, 5, 4)]
        pos = np.array([c[k] for f in faces for k in f])
        tri = np.array([[4 * n, 4 * n + 1, 4 * n + 2, 4 * n, 4 * n + 2, 4 * n + 3] for n in range(6)]).reshape(-1, 3)
        out.append(OB._part(pos, tri, colour=tuple(colour) + (1.0,), name=name.replace(" ", "_")))
    return out
