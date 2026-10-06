"""
Decals: pictures laid on the streets (arrows, crossings, manhole covers,
stains, patches). Each imported picture is a layer with where it goes:

- Place randomly: along the streets, never inside a junction, about one every
  "every_m" metres of street, at a random point across the road, lying along
  it (facing either way; with "spin", any way).
- and with one of these, at the junctions instead:
  - Place at junctions: at the edges of the junctions (where a street enters
    one), a random share of them ("chance", %).
  - At all edges: at every edge of every junction.
  - Even edges: at two opposite edges of every junction (the two arms most in
    line: the through road of a T, one of the two roads of a crossroads).

At a junction edge a decal lies across the street just outside the junction
(or "back_m" metres further back), its top towards the junction (an arrow
painted before a junction points into it); "fit" scales it to the road's
width there. A picture's top is its front;
its width goes across the road, its length along it, in proportion.

The places come from the generation's mask (app/junctions.py: its junctions,
their arms and the streets' centre lines), in mask pixels, so the texture and
the 3D model have the decals in the same places. Decals over a bridge's road
are left out (the deck lifts it).
"""
import io
import json
import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GAP_M = 0.5            # between a junction's edge and a decal laid there
LIFT_SEPARATE = 0.015  # metres above the road: a decal as its own object (the dashes are at 0.01)
LIFT_PAINTED = 0.006   # a decal that is part of the road object


def layer_list(raw, known):
    """Decal layers as the page sends them, checked; known: the decal ids there are pictures for."""
    out = []
    for l in (raw or [])[:64]:
        if not isinstance(l, dict) or str(l.get("decal")) not in known:
            continue
        num = lambda k, lo, hi, d: min(hi, max(lo, float(l.get(k, d) if l.get(k) not in (None, "") else d)))
        out.append({"decal": str(l["decal"]), "random": bool(l.get("random", True)),
                    "junctions": bool(l.get("junctions")), "all_edges": bool(l.get("all_edges")),
                    "even_edges": bool(l.get("even_edges")), "fit": bool(l.get("fit")), "spin": bool(l.get("spin")),
                    "width_m": num("width_m", 0.1, 60.0, 3.0), "every_m": num("every_m", 2.0, 5000.0, 40.0),
                    "chance": num("chance", 0.0, 100.0, 50.0), "back_m": num("back_m", 0.0, 500.0, 0.0)})
    return out


def _in_bridge(x, y, bridges):
    if not bridges:
        return False
    from app import quadmesh as QM
    return any(QM.inside_rect(np.array([[x, y]]), b)[0] for b in bridges)


def _edge(j, arm, det):
    """
    A junction edge: where the arm's street centre line leaves the junction (its
    point nearest the junction's circle) and the street's own direction there,
    away from the junction (from that street's centre line, not from the
    junction's centre, which need not lie on it). (x, y, out x, out y), or None.
    """
    cx, cy, r = j["cx"], j["cy"], j["r"]
    s = arm.get("segment")
    pts = det["segment_pixels"][s] if s else None
    if pts is None or len(pts) < 3:
        t = math.radians(arm["angle"])
        ox, oy = math.cos(t), -math.sin(t)
        return (cx + ox * r, cy + oy * r, ox, oy) if not arm.get("off_image") else None
    d = np.hypot(pts[:, 1] - cx, pts[:, 0] - cy)
    k = int(np.argmin(np.abs(d - r)))
    ey, ex = float(pts[k, 0]), float(pts[k, 1])
    near = pts[(d >= d[k] - 1) & (d <= d[k] + max(8.0, 3 * float(arm["width_px"])))]
    if len(near) >= 3:
        c = near - near.mean(axis=0)
        ev = np.linalg.eigh(c.T @ c)[1][:, -1]
        ox, oy = float(ev[1]), float(ev[0])
    else:
        ox, oy = ex - cx, ey - cy
    n = math.hypot(ox, oy) or 1.0
    ox, oy = ox / n, oy / n
    if ox * (ex - cx) + oy * (ey - cy) < 0:                    # pointing out of the junction
        ox, oy = -ox, -oy
    return ex, ey, ox, oy


def place(gray, mpp, layers, sizes, seed=7, bridges=None):
    """
    Where every decal goes: [{"layer", "decal", "x", "y" (mask pixels, its
    centre), "ux", "uy" (the way its top faces, in image axes), "w_m", "h_m",
    "at" ("street" or "edge")}]. sizes: {decal id: (width px, height px)}.
    """
    from app import junctions as J
    det = J.detect(np.asarray(gray))
    road, dt = det["road"], det["dt"]
    H, W = road.shape
    out = []
    for li, L in enumerate(layers):
        if not L["random"]:
            continue
        pw, ph = sizes[L["decal"]]
        aspect = ph / max(1, pw)
        rng = np.random.default_rng([int(seed), 3101, li])
        mode = "all" if L["all_edges"] else "even" if L["even_edges"] else "chance" if L["junctions"] else None
        if mode:
            for j in det["junctions"]:
                if j["type"] == "interchange (flagged)":
                    continue
                arms = list(j["arms"])
                if mode == "even":
                    pairs = []
                    for a in range(len(arms)):
                        for b in range(a + 1, len(arms)):
                            d = abs(((arms[a]["angle"] - arms[b]["angle"]) % 360) - 180)
                            if d <= 35:
                                pairs.append((d, a, b))
                    if not pairs:
                        continue
                    best = min(p[0] for p in pairs)
                    near = [p for p in pairs if p[0] <= best + 10]  # a crossroads: either road, at random
                    _, a, b = near[int(rng.integers(len(near)))]
                    arms = [arms[a], arms[b]]
                for arm in arms:
                    if mode == "chance" and rng.uniform(0, 100) >= L["chance"]:
                        continue
                    edge = _edge(j, arm, det)
                    if edge is None:
                        continue
                    ex, ey, ox, oy = edge                             # where the street leaves it, and the way out
                    width_m = max(0.5, float(arm["width_px"]) * mpp)
                    w = width_m * 0.96 if L["fit"] else L["width_m"]
                    h = w * aspect
                    far = L["back_m"] + h / 2 + GAP_M                     # back from the edge, into the street
                    cx, cy = ex + ox * far / mpp, ey + oy * far / mpp
                    xi, yi = int(round(cx)), int(round(cy))
                    if not (0 <= xi < W and 0 <= yi < H) or not road[yi, xi] or _in_bridge(cx, cy, bridges):
                        continue
                    # a street between two junctions close together: one decal, not one from each end
                    if any(q["layer"] == li and math.hypot(q["x"] - cx, q["y"] - cy) * mpp < max(w, h) for q in out):
                        continue
                    out.append({"layer": li, "decal": L["decal"], "x": round(cx, 2), "y": round(cy, 2),
                                "ux": round(-ox, 5), "uy": round(-oy, 5), "w_m": round(w, 3), "h_m": round(h, 3), "at": "edge"})
            continue
        # along the streets: each street's centre line, about one decal every every_m metres
        w, h = L["width_m"], L["width_m"] * aspect
        taken = []
        for pts in det["segment_pixels"][1:]:
            if pts is None or len(pts) < 3:
                continue
            length_m = len(pts) * mpp * 1.1                          # centre-line pixels, some diagonal
            if length_m < h * 2:
                continue
            n = int(rng.poisson(length_m / L["every_m"]))
            if not n:
                continue
            for k in rng.choice(len(pts), size=min(n, len(pts)), replace=False):
                y, x = pts[k]
                hw_m = float(dt[y, x]) * mpp
                if w > 2 * hw_m * 1.05:
                    continue                                             # wider than the road here
                near = pts[np.hypot(pts[:, 0] - y, pts[:, 1] - x) <= 6]
                if len(near) < 3:
                    continue
                c = near - near.mean(axis=0)
                ev = np.linalg.eigh(c.T @ c)[1][:, -1]                   # the street's direction (row, col)
                dx, dy = float(ev[1]), float(ev[0])
                if L["spin"]:
                    a = rng.uniform(0, 2 * math.pi)
                    ux, uy = math.cos(a), math.sin(a)
                else:
                    s = 1.0 if rng.random() < 0.5 else -1.0
                    ux, uy = dx * s, dy * s
                room = max(0.0, hw_m - w / 2 - 0.2)
                off = rng.uniform(-room, room) / mpp
                cx, cy = x - dy * off, y + dx * off                      # across the road
                if any(math.hypot(cx - px, cy - py) * mpp < max(w, h) * 1.2 for px, py in taken):
                    continue
                xi, yi = int(round(cx)), int(round(cy))
                if not (0 <= xi < W and 0 <= yi < H) or not road[yi, xi] or _in_bridge(cx, cy, bridges):
                    continue
                taken.append((cx, cy))
                out.append({"layer": li, "decal": L["decal"], "x": round(float(cx), 2), "y": round(float(cy), 2),
                            "ux": round(ux, 5), "uy": round(uy, 5), "w_m": round(w, 3), "h_m": round(h, 3), "at": "street"})
    return out


# ------------------------------------------------------------------ the texture
def paint(shape, placements, images, mpp_out, scale, coverage=None):
    """
    The decals as a picture of the texture's size (RGBA, float 0-1, H x W x 4),
    each turned and sized as placed; only on the road (coverage).
    images: {decal id: PIL RGBA image}.
    """
    H, W = shape
    acc = np.zeros((H, W, 4), np.float32)
    for p in placements:
        img = images.get(p["decal"])
        if img is None:
            continue
        w = max(1, int(round(p["w_m"] / mpp_out)))
        h = max(1, int(round(p["h_m"] / mpp_out)))
        im = img.resize((w, h), Image.LANCZOS if w < img.width else Image.BICUBIC)
        # turned so its top (0, -1) faces (ux, uy): PIL turns anticlockwise as seen
        deg = math.degrees(math.atan2(-p["ux"], -p["uy"]))
        im = im.rotate(deg, resample=Image.BICUBIC, expand=True)
        a = np.asarray(im, np.float32) / 255.0
        cx, cy = p["x"] * scale, p["y"] * scale
        x0, y0 = int(round(cx - im.width / 2)), int(round(cy - im.height / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + im.width), min(H, y0 + im.height)
        if xa >= xb or ya >= yb:
            continue
        src = a[ya - y0:yb - y0, xa - x0:xb - x0]
        al = src[..., 3:4]
        if coverage is not None:
            al = al * coverage[ya:yb, xa:xb, None]
        dst = acc[ya:yb, xa:xb]
        # premultiplied "over"
        dst[..., :3] = src[..., :3] * al + dst[..., :3] * (1 - al)
        dst[..., 3:4] = al + dst[..., 3:4] * (1 - al)
    return acc


def composite(base_rgb, decal_rgba):
    """The road texture with the decals painted in (uint8 RGB)."""
    a = decal_rgba[..., 3:4]
    rgb = decal_rgba[..., :3] / np.maximum(a, 1e-6)
    out = base_rgb.astype(np.float32) / 255.0 * (1 - a) + rgb * a
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ the 3D model
def mesh_prims(placements, images, materials, image_list, W, H, mpp_out, scale, lift, image_index, names=None):
    """
    Each decal as a flat quad lift metres above the road, one primitive per
    picture with a see-through material of its own. W, H: the texture's size
    (the model's ground is W x H texture pixels of mpp_out metres).
    image_index(bytes, mime): the GLB's picture index. Returns primitives.
    """
    by = {}
    for p in placements:
        by.setdefault(p["decal"], []).append(p)
    prims = []
    for did, ps in by.items():
        img = images.get(did)
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, "PNG")
        ti = image_index(buf.getvalue(), "image/png")
        label = "".join(c if c.isalnum() else "_" for c in (names or {}).get(did, did)).strip("_") or did
        materials.append({"name": f"Decal_{label}", "doubleSided": True, "alphaMode": "BLEND",
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": ti}, "metallicFactor": 0.0,
                                                   "roughnessFactor": 0.75}})
        mat = len(materials) - 1
        pos, uv, idx = [], [], []
        for p in ps:
            cx = p["x"] * scale * mpp_out - W * mpp_out / 2
            cz = p["y"] * scale * mpp_out - H * mpp_out / 2
            u = np.array([p["ux"], p["uy"]])                   # its top, in image axes = world X, Z
            r = np.array([-u[1], u[0]])                        # its right
            hw, hh = p["w_m"] / 2, p["h_m"] / 2
            base = len(pos)
            for (sr, su), t in (((-1, 1), (0, 0)), ((1, 1), (1, 0)), ((1, -1), (1, 1)), ((-1, -1), (0, 1))):
                q = np.array([cx, cz]) + r * sr * hw + u * su * hh
                pos.append((q[0], lift, q[1])); uv.append(t)
            # facing up: (0, 1, 0)
            a, b, c, d = base, base + 1, base + 2, base + 3
            P = np.array(pos[-4:])
            ny = np.cross(P[1] - P[0], P[2] - P[0])[1]
            idx += [[a, b, c], [a, c, d]] if ny > 0 else [[a, c, b], [a, d, c]]
        P = np.array(pos)
        prims.append({"positions": P, "normals": np.tile([0.0, 1.0, 0.0], (len(P), 1)), "uv0": np.array(uv, float),
                      "indices": np.array(idx), "material": mat})
    return prims


def load_pictures(index, folder):
    """{decal id: PIL RGBA image} for the decals in index."""
    out = {}
    for d in index:
        p = folder / f"{d['id']}.png"
        if p.exists():
            out[d["id"]] = Image.open(p).convert("RGBA")
    return out


def summary(placements, layers, names):
    """Per layer: how many, along streets or at junction edges."""
    out = []
    for li, L in enumerate(layers):
        mine = [p for p in placements if p["layer"] == li]
        out.append({"layer": li, "name": names.get(L["decal"], L["decal"]), "count": len(mine),
                    "edges": sum(p["at"] == "edge" for p in mine), "streets": sum(p["at"] == "street" for p in mine)})
    return out


def save(placements, path):
    path.write_text(json.dumps({"placements": placements}, separators=(",", ":")))
