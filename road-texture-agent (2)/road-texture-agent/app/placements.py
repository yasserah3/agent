"""
Where the copies of a placement go, in one place for everything that needs it:
the 3D export, and the inner streets drawn between them.

A placement is one object or a package, as a rectangle (rows along its middle
line, mirrored or not) or a curve (one or more curve lines), with even or
random spaces and its objects aligned (rows turned round, row ends facing out),
and, for foliage, a random scale, turn and offset per copy (app/curves.py).
Each copy has an
id, "row-column" from the top left, and can be given its own extra rotation in
the placement's "turns": {id: degrees, clockwise on the map}. A turned copy
takes its turned outline in its row, so the spaces around it stay as set.
The ids in the placement's "empty" (Draw spaced) are left out: their spots stay
empty and every other copy stays where it is.

An automatic placement ("auto", app/autoplace.py) has no rectangle: its copies are
the objects it laid on the islands ("items"), ids "island-number", each turned to
face its street; turns and empties apply to them the same way.
"""

import math

import numpy as np

from app import curves as CV


def sized(meta):
    """An object's scale, quarter turn, and footprint: w along its row, d across."""
    k = float(meta.get("scale", 1.0))
    turn = int(meta.get("turn", 0)) % 4
    w, d = meta["width_m"] * k, meta["depth_m"] * k
    return (k, turn) + ((d, w) if turn % 2 else (w, d))    # a quarter turn swaps width and depth


def spaces_of(p):
    return p["spaces"] if (p.get("spaces") or {}).get("on") else None


def curve_points(p, mpp_mask):
    """A curve's line in metres, or its curve lines, the first and the others (same number of points)."""
    first = np.asarray(p["path"], float) * mpp_mask
    more = [np.asarray(q, float) * mpp_mask for q in (p.get("lines") or []) if len(q) == len(p["path"])]
    return np.array([first] + more) if more else first


def copies_of(p, objects, packages, mpp_mask):
    """
    A placement's copies and layout, in metres in image axes (x right, y down).

    objects: {id: {"meta": ...}}, packages: {id: package}. Returns a list of
    (object id, centre, angle, copy id, scale), the angle (radians) being the
    direction of the copy's own X with its extra rotation, and the layout
    (app/curves.py), whose distances and offsets are in metres too. The
    angle includes the alignment's turns (rows turned round, row ends), and,
    with a random transform (foliage), the centre its random offset, the angle
    its random turn, and scale its random scale (1 without).
    """
    if isinstance(p.get("auto"), dict):
        # an automatic placement (app/autoplace.py): its objects as laid on the islands, each
        # with its X along its side and its front to the street, with its own extra turn; empties
        # out. "a" is where its front (+Y, the arrow on the map) faces: a quarter-turned object's
        # frame is turned back by its turn, so its +Y still faces the street
        turns, empty = p.get("turns") or {}, set(p.get("empty") or [])
        items = p.get("items") or []
        quarter = lambda oid: 90.0 * (int(objects[oid]["meta"].get("turn", 0)) % 4)
        out = [(it["object"], np.array([it["x"], it["y"]], float) * mpp_mask,
                math.radians(float(it["a"]) - quarter(it["object"]) + float(turns.get(it["id"], 0.0))), it["id"], 1.0)
               for it in items if it["object"] in objects and it["id"] not in empty]
        return out, {"ids": [it["id"] for it in items]}
    layout, spaces, align, jitter = {}, spaces_of(p), p.get("align"), p.get("jitter")
    if "package" in p:
        pk = packages[p["package"]]
        slots = [sl for sl in pk["slots"] if sl["object"] in objects]
        sizes = [sized(objects[sl["object"]]["meta"]) for sl in slots]
        flip = p.get("flip", False)
        if p.get("path"):
            pts = curve_points(p, mpp_mask)
        else:
            a = math.radians(p["angle"])
            u = np.array([math.cos(a), math.sin(a)])
            c, half = np.array([p["cx"], p["cy"]]) * mpp_mask, p["length"] * mpp_mask / 2
            pts = np.array([c - u * half, c + u * half])        # a rectangle is a straight line
            if p.get("mirror"):
                pts, flip = pts[::-1], True                     # laid out from the other end, facing the same way
        spots, _, _ = CV.package_copies(pts, [(z[2], z[3], sl.get("weight", 1.0)) for z, sl in zip(sizes, slots)],
                                        p["gap_x"], p["gap_y"], p["ny"], flip, p["seed"], spaces, layout,
                                        p.get("turns"), align, jitter)
        spots = [(slots[si]["object"], np.array([x, y]), ang) for x, y, ang, si in spots]
    else:
        k, turn, w, d = sized(objects[p["object"]]["meta"])
        if p.get("path"):
            spots, _, _ = CV.copies(curve_points(p, mpp_mask), w, d,
                                    p["gap_x"], p["gap_y"], p["ny"], p.get("flip", False), spaces, layout, p.get("turns"),
                                    align, jitter)
        else:
            spots, _, _ = CV.grid_copies(np.array([p["cx"], p["cy"]]) * mpp_mask, math.radians(p["angle"]), w, d,
                                         p["nx"], p["ny"], p["gap_x"], p["gap_y"], spaces, layout, p.get("turns"),
                                         bool(p.get("mirror")), align, jitter)
        spots = [(p["object"], np.array([x, y]), ang) for x, y, ang in spots]
    turns = p.get("turns") or {}
    empty = set(p.get("empty") or [])               # Draw spaced: left out, their spots empty
    out = []
    for (oid, cc, a), cid in zip(spots, layout["ids"]):
        if cid in empty:
            continue
        s = 1.0
        if jitter:
            # its random offset across its own row's frame, then its random turn
            r, i = (int(x) - 1 for x in cid.split("-"))
            s, rot, off, th = CV.jitter_of(jitter, r, i)
            x_, v_ = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
            cc = cc + off * (math.cos(th) * x_ + math.sin(th) * v_)
            a = a + math.radians(rot)
        out.append((oid, cc, a + math.radians(float(turns.get(cid, 0.0))), cid, s))
    return out, layout


def footprint(cc, angle, w, d):
    """A copy's four corners, turned with it."""
    u = np.array([math.cos(angle), math.sin(angle)])
    v = np.array([-u[1], u[0]])
    return [cc + u * sx * w / 2 + v * sy * d / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
