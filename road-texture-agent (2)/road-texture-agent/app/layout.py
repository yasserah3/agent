"""
Layout from the dataset: an object on every building of an area made from coordinates.

The buildings are read from the area's street map (app/footprints.py): each a rectangle W
wide across its front and D deep, its front to its nearest street. The objects are the
ones ticked in the Layout window (imported, or made in Dynamic creation), each w wide
across its own front (its X) and d deep (its Y), at its own scale.

Best fit by shape: each building takes the object whose proportions are nearest its own,
    score = |ln((W / D) / (w / d))| + SIZE_WEIGHT * |ln((W D) / (w d))|,
so a long narrow building takes a long narrow object; among objects of the same
proportions, the nearest in size. The lowest score wins (ties: the first ticked).

Fill the outline: the object's own width is stretched to W and its depth to D, its
height kept (the street map has no heights). It stands in the rectangle's middle, its
front (+Y) to the building's front, as an automatic placement's objects
(app/autoplace.py): "a" is where the front faces, degrees clockwise from the top of the
map.

Buildings cut by the map's edge are left out (only part of them is on the map), and so
are those under MIN_SIDE_M on a side.
"""

import math

SIZE_WEIGHT = 0.15
MIN_SIDE_M = 2.0


def score(W, D, w, d):
    return abs(math.log((W / D) / (w / d))) + SIZE_WEIGHT * abs(math.log((W * D) / (w * d)))


def lay(buildings, kinds, skip_edge=True):
    """
    buildings: an area's (app/footprints.py: cx, cy pixels, w_m, d_m, a, edge); kinds:
    [(object id, w, d)] in metres, the object's own width across its front and depth, at
    its scale. Returns (items, note): items {"id": "b<n>", "x", "y" (pixels), "a", "object",
    "w", "d" (metres)}, note {"buildings", "laid", "edge", "small", "by_object": {id: n}}.
    """
    kinds = [(oid, float(w), float(d)) for oid, w, d in kinds if w > 0 and d > 0]
    items, edge, small, by = [], 0, 0, {}
    for b in buildings:
        if skip_edge and b.get("edge"):
            edge += 1
            continue
        W, D = float(b["w_m"]), float(b["d_m"])
        if min(W, D) < MIN_SIDE_M:
            small += 1
            continue
        if not kinds:
            continue
        oid = min(kinds, key=lambda k: score(W, D, k[1], k[2]))[0]
        by[oid] = by.get(oid, 0) + 1
        items.append({"id": f"b{b['id']}", "x": b["cx"], "y": b["cy"], "a": b["a"], "object": oid,
                      "w": round(W, 3), "d": round(D, 3)})
    return items, {"buildings": len(buildings), "laid": len(items), "edge": edge, "small": small, "by_object": by}


def settings(raw):
    """A layout placement's own record: the area it was made for, the objects ticked."""
    raw = raw if isinstance(raw, dict) else {}
    key = str(raw.get("key") or "")
    objs = [str(o) for o in raw.get("objects") or [] if isinstance(o, str) and len(o) == 12][:200]
    return {"key": key if len(key) == 12 else "", "objects": objs}


def item_list(raw):
    """A layout's items as kept: each building's object, middle (pixels), front (degrees) and size (metres)."""
    out = []
    for it in raw or []:
        try:
            out.append({"id": str(it["id"])[:12], "x": round(float(it["x"]), 3), "y": round(float(it["y"]), 3),
                        "a": round(float(it["a"]) % 360.0, 3), "object": str(it["object"])[:12],
                        "w": round(max(0.1, float(it["w"])), 3), "d": round(max(0.1, float(it["d"])), 3)})
        except (KeyError, TypeError, ValueError):
            continue
    return out[:200000]
