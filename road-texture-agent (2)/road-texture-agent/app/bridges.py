"""
Bridges: lifting a road smoothly over another.

A bridge is a red rectangle on the map. The road running through it lengthwise
is the bridge; everything else stays on the ground. Starting from the rectangle,
the road is followed outward through junctions, always taking the straightest
way on, for the ramp length on each side. Its height along it:

    ground ... ramp up (ramp length) ... full height across the rectangle ...
    ramp down (ramp length) ... ground

The ramp is a smooth S-curve, flat where it leaves the ground and flat where it
reaches full height, so there is no kink at either end. Its steepest point is
1.5 times the average slope: 5 m over 120 m averages 4.2% and peaks at 6.3%.

Side streets that meet the bridge road partway up a ramp get a short ramp of
their own, so they meet it at its height instead of at a step.
"""

import math

import numpy as np

from app.quadmesh import BRIDGE_OWNER, UNDER_OWNER, inside_rect

SIDE_RAMP_M = 30.0          # side streets climb to meet a ramp over this distance


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _street_path(mesh, sid, which):
    """A street's centreline (mask pixels), starting from end `which`."""
    pts = mesh.streets[sid]["pts"]
    return pts if which == 0 else pts[::-1]


def _junction_at(mesh, sid, which):
    for jid, j in mesh.jinfo.items():
        if (sid, which) in j["mouths"]:
            return jid
    return None


def _follow(mesh, sid, which, needed_px, scale, max_turn_deg=35.0):
    """
    Follow the road outward from a street end, through junctions, taking the
    straightest way on each time, until far enough. Returns the path, the
    streets used, the junctions passed and the side streets met at each.
    """
    path = [_street_path(mesh, sid, which)]
    used, passed = [sid], []
    length = float(np.sum(np.linalg.norm(np.diff(path[0], axis=0), axis=1)))
    cur_sid, cur_which = sid, which
    while length < needed_px:
        far = 1 - cur_which
        jid = _junction_at(mesh, cur_sid, far)
        if jid is None or mesh.jinfo[jid]["kind"] != "patch":
            break
        seg = path[-1]
        arrive = seg[-1] - seg[max(0, len(seg) - 4)]
        arrive /= (np.linalg.norm(arrive) or 1.0)
        best = None
        for (s2, w2) in mesh.jinfo[jid]["mouths"]:
            if s2 == cur_sid or s2 in used:
                continue
            p2 = _street_path(mesh, s2, w2)
            leave = p2[min(3, len(p2) - 1)] - p2[0]
            leave /= (np.linalg.norm(leave) or 1.0)
            ang = math.degrees(math.acos(float(np.clip(arrive @ leave, -1, 1))))
            if ang <= max_turn_deg and (best is None or ang < best[0]):
                best = (ang, s2, w2)
        if best is None:
            break
        C = np.array(mesh.jinfo[jid]["C"], float) / scale
        sides = [m for m in mesh.jinfo[jid]["mouths"] if m[0] not in (cur_sid, best[1])]
        passed.append((jid, sides))
        _, s2, w2 = best
        nxt = _street_path(mesh, s2, w2)
        path.append(C[None, :])
        path.append(nxt)
        length += float(np.linalg.norm(C - seg[-1]) + np.linalg.norm(nxt[0] - C)
                        + np.sum(np.linalg.norm(np.diff(nxt, axis=0), axis=1)))
        used.append(s2)
        cur_sid, cur_which = s2, w2
    return np.vstack(path), used, passed


def plan(mesh, bridges, scale, mpp, height_m=5.0, ramp_m=120.0):
    """
    For each bridge: the road it lifts, its height profile, and what it passes.

    Returns a list of plans; each can lift points and says which parts of the
    mesh belong to it.
    """
    plans = []
    for bi, b in enumerate(bridges):
        a = math.radians(b["angle"])
        u = np.array([math.cos(a), math.sin(a)])
        need = ramp_m / mpp + b["length"]
        conn = [(oid, c) for oid, c in mesh.connectors.items()
                if oid != "_warnings" and c["bridge"] and c["index"] == bi]
        warnings = []
        owners = set()
        if conn:
            oid, c = conn[0]
            owners.add(oid)
            (sa, wa), (sb, wb) = c["ends"]
            pa, ua, ja = _follow(mesh, sa, wa, need, scale)
            pb, ub, jb = _follow(mesh, sb, wb, need, scale)
            line = np.vstack([pa[::-1], pb])
            used, passed = ua + ub, ja + jb
        else:
            # no crossing inside: the bridge carries a road over something the
            # mask does not show, such as water. Take the street lying along it
            best = None
            for sid, st in mesh.streets.items():
                inside = inside_rect(st["pts"], b)
                if inside.sum() < 2:
                    continue
                q = st["pts"][inside]
                d = q[-1] - q[0]
                align = abs(float(d @ u)) / (np.linalg.norm(d) or 1.0)
                if align > 0.8 and (best is None or inside.sum() > best[0]):
                    best = (inside.sum(), sid)
            if best is None:
                plans.append({"index": bi, "ok": False,
                              "warnings": ["no road runs lengthwise through this rectangle"]})
                continue
            sid = best[1]
            pa, ua, ja = _follow(mesh, sid, 1, need, scale)          # outward past its start
            pb, ub, jb = _follow(mesh, sid, 0, need, scale)          # outward past its end
            line = np.vstack([pa[::-1], pb[len(mesh.streets[sid]["pts"]):]])
            used, passed = list(dict.fromkeys(ua + ub)), ja + jb

        owners.update(used)
        owners.update(-jid for jid, _ in passed)
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)]) * mpp                  # metres
        inside = inside_rect(line, b)
        if not inside.any():
            plans.append({"index": bi, "ok": False, "warnings": ["the road never enters the rectangle"]})
            continue
        s_in, s_out = float(arc[inside].min()), float(arc[inside].max())
        before, after = s_in, float(arc[-1]) - s_out
        r_before, r_after = min(ramp_m, before), min(ramp_m, after)
        for name, have in (("start", r_before), ("end", r_after)):
            if have < ramp_m - 1:
                warnings.append(f"the road only runs {have:.0f} m past the {name} of the bridge, "
                                f"so that ramp is {have:.0f} m long instead of {ramp_m:.0f} m")
        slopes = [1.5 * height_m / r for r in (r_before, r_after) if r > 0]
        steepest = max(slopes) if slopes else float("inf")
        if steepest > 0.08:
            warnings.append(f"steepest point of a ramp is {steepest * 100:.1f}%, steeper than roads "
                            f"usually allow (about 8%): lengthen the ramp or move the bridge")
        if passed:
            n_side = sum(len(s) for _, s in passed)
            if n_side:
                warnings.append(f"{n_side} side street(s) meet a ramp and get a short ramp of their own")

        plans.append({
            "index": bi, "ok": True, "line": line, "arc": arc, "s_in": s_in, "s_out": s_out,
            "r_before": r_before, "r_after": r_after, "height": height_m, "owners": owners,
            "passed": passed, "steepest_pct": round(steepest * 100, 2) if slopes else None,
            "warnings": warnings, "streets": used, "mpp": mpp,
        })
    return plans


def profile(p, s):
    """Height (metres) at distance s (metres) along the bridge road."""
    s = np.asarray(s, float)
    h = np.zeros_like(s)
    up = (s < p["s_in"])
    if p["r_before"] > 0:
        h = np.where(up, p["height"] * smooth((s - (p["s_in"] - p["r_before"])) / p["r_before"]), h)
    down = (s > p["s_out"])
    if p["r_after"] > 0:
        h = np.where(down, p["height"] * smooth(((p["s_out"] + p["r_after"]) - s) / p["r_after"]), h)
    h = np.where(~up & ~down, p["height"], h)
    return h


def project(p, pts_mask):
    """Distance along the bridge road (metres) of the nearest point on it."""
    L = p["line"]
    a, b = L[:-1], L[1:]
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    P = np.asarray(pts_mask, float)
    t = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(axis=2) / ab2[None], 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    d2 = ((P[:, None, :] - closest) ** 2).sum(axis=2)
    k = d2.argmin(axis=1)
    seg_len = np.sqrt(ab2)
    return p["arc"][k] + t[np.arange(len(P)), k] * seg_len[k] * p["mpp"]


def lifts(mesh, plans, scale):
    """
    How far to raise every vertex of the mesh, in metres.

    Anything belonging to a bridge road is raised by its profile; side streets
    meeting a ramp are raised to meet it, falling back to the ground over a
    short distance.
    """
    n = len(mesh.v)
    V = np.array(mesh.v, float) / scale
    L = np.zeros(n)
    assigned = np.zeros(n, bool)

    def owned(owner_set):
        ids = set()
        for q, o in zip(mesh.quads, mesh.owner):
            if o in owner_set:
                ids.update(q)
        for q, o in zip(mesh.sw_quads, mesh.sw_owner):
            if o in owner_set:
                ids.update(q)
        for q, o in zip(mesh.kerb_quads, mesh.kerb_owner):
            if o in owner_set:
                ids.update(q)
        return np.array(sorted(ids), int)

    for p in plans:
        if not p.get("ok"):
            continue
        ids = owned(p["owners"])
        if len(ids):
            h = profile(p, project(p, V[ids]))
            L[ids] = np.maximum(L[ids], h)
            assigned[ids] = True
        # side streets meeting a ramp: rise to meet it
        for jid, sides in p["passed"]:
            j = mesh.jinfo[jid]
            C = np.array(j["C"], float) / scale
            hj = float(profile(p, project(p, C[None]))[0])
            if hj <= 0.01:
                continue
            side_ids = owned({sid for sid, _ in sides})
            side_ids = side_ids[~assigned[side_ids]]
            if len(side_ids):
                d = (np.linalg.norm(V[side_ids] - C, axis=1) - j["r"] / scale) * p["mpp"]
                L[side_ids] = np.maximum(L[side_ids], hj * smooth(1 - d / SIDE_RAMP_M))
    return L


def dash_lift(mesh, plans, scale, owner, pts_texture):
    """Lift for dash corner points, by the same rules as the road they lie on."""
    P = np.asarray(pts_texture, float) / scale
    out = np.zeros(len(P))
    for p in plans:
        if p.get("ok") and owner in p["owners"]:
            out = np.maximum(out, profile(p, project(p, P)))
    return out
