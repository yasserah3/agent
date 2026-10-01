"""
A small top-down renderer for checking exported models.

It reads the GLB exactly as saved: positions, texture coordinates, vertex
colours, materials and textures, and draws the model from directly above. It
is not a replacement for Blender or Unreal; it exists so the agent can check
its own output, including whether repeating tiles are visible.
"""

import io
import json
import struct

import numpy as np
from PIL import Image

COMP = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}
SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load_glb(path):
    data = open(path, "rb").read()
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    bin_start = 20 + jlen + 8
    blob = data[bin_start:]

    def acc(i):
        a = gltf["accessors"][i]
        v = gltf["bufferViews"][a["bufferView"]]
        arr = np.frombuffer(blob, COMP[a["componentType"]], a["count"] * SIZE[a["type"]],
                            v.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], SIZE[a["type"]]) if SIZE[a["type"]] > 1 else arr

    images = []
    for im in gltf.get("images", []):
        v = gltf["bufferViews"][im["bufferView"]]
        raw = blob[v["byteOffset"]: v["byteOffset"] + v["byteLength"]]
        images.append(np.array(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.float32))

    prims = []
    for m in gltf["meshes"]:
        for p in m["primitives"]:
            mat = gltf["materials"][p["material"]]
            pbr = mat.get("pbrMetallicRoughness", {})
            tex = pbr.get("baseColorTexture")
            prims.append({
                "mesh": m["name"],
                "pos": acc(p["attributes"]["POSITION"]).astype(np.float64),
                "uv": acc(p["attributes"]["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in p["attributes"] else None,
                "col": acc(p["attributes"]["COLOR_0"]).astype(np.float64) if "COLOR_0" in p["attributes"] else None,
                "idx": acc(p["indices"]).reshape(-1, 3).astype(np.int64),
                "tex": images[gltf["textures"][tex["index"]]["source"]] if tex else None,
                "factor": np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]))[:3] * 255,
            })
    return prims


def render(prims, x0, z0, width_m, height_m, mm_per_px=20.0, use_colours=True, background=(24, 26, 29)):
    """Draw the region [x0, x0+width] x [z0, z0+height], seen from above."""
    k = 1000.0 / mm_per_px                                     # pixels per metre
    W, H = int(width_m * k), int(height_m * k)
    img = np.zeros((H, W, 3), np.float32); img[:] = background
    depth = np.full((H, W), -np.inf, np.float32)
    for p in sorted(prims, key=lambda q: q["mesh"] == "Markings"):   # markings drawn last, on top
        P = p["pos"]
        X = (P[:, 0] - x0) * k; Z = (P[:, 2] - z0) * k
        for tri in p["idx"]:
            xs, zs = X[tri], Z[tri]
            if xs.max() < 0 or xs.min() > W or zs.max() < 0 or zs.min() > H:
                continue
            xa, xb = max(0, int(np.floor(xs.min()))), min(W, int(np.ceil(xs.max())) + 1)
            za, zb = max(0, int(np.floor(zs.min()))), min(H, int(np.ceil(zs.max())) + 1)
            if xa >= xb or za >= zb:
                continue
            gx, gz = np.meshgrid(np.arange(xa, xb) + 0.5, np.arange(za, zb) + 0.5)
            (x1, x2, x3), (z1, z2, z3) = xs, zs
            den = (z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3)
            if abs(den) < 1e-12:
                continue
            w1 = ((z2 - z3) * (gx - x3) + (x3 - x2) * (gz - z3)) / den
            w2 = ((z3 - z1) * (gx - x3) + (x1 - x3) * (gz - z3)) / den
            w3 = 1 - w1 - w2
            inside = (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
            if not inside.any():
                continue
            y = w1 * P[tri[0], 1] + w2 * P[tri[1], 1] + w3 * P[tri[2], 1]
            sub_d = depth[za:zb, xa:xb]
            draw = inside & (y >= sub_d)
            if not draw.any():
                continue
            if p["tex"] is not None and p["uv"] is not None:
                uv = p["uv"][tri]
                u = w1 * uv[0, 0] + w2 * uv[1, 0] + w3 * uv[2, 0]
                v = w1 * uv[0, 1] + w2 * uv[1, 1] + w3 * uv[2, 1]
                th, tw = p["tex"].shape[:2]
                tx = np.mod(np.floor(u * tw).astype(np.int64), tw)      # repeating, as in the file
                ty = np.mod(np.floor(v * th).astype(np.int64), th)
                colour = p["tex"][ty, tx]
            else:
                colour = np.broadcast_to(p["factor"], gx.shape + (3,)).copy()
            if use_colours and p["col"] is not None:
                c = p["col"][tri]
                f = (w1[..., None] * c[0, :3] + w2[..., None] * c[1, :3] + w3[..., None] * c[2, :3])
                colour = colour * f
            sub = img[za:zb, xa:xb]
            sub[draw] = colour[draw]
            sub_d[draw] = y[draw]
    return np.clip(img, 0, 255).astype(np.uint8)


def render_oblique(prims, x0, z0, width_m, depth_m, elevation_deg=30.0, exaggerate=1.0,
                   mm_per_px=50.0, **kw):
    """
    The same renderer, seen at an angle from the +Z side, so heights show.
    Heights can be exaggerated to make a few metres visible over a large area.
    """
    e = np.radians(elevation_deg)
    turned = []
    for p in prims:
        q = dict(p)
        P = p["pos"].copy()
        Y = P[:, 1] * exaggerate
        Z = P[:, 2]
        q["pos"] = np.column_stack([P[:, 0], Y * np.cos(e) + Z * np.sin(e), Z * np.cos(e) - Y * np.sin(e)])
        turned.append(q)
    top = z0 * np.cos(e) - 12 * exaggerate * np.sin(e)
    return render(turned, x0, top, width_m, depth_m * np.cos(e) + 12 * exaggerate * np.sin(e),
                  mm_per_px=mm_per_px, **kw)
