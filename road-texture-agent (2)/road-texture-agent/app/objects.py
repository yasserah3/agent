"""
Objects to scatter on blocks and sidewalks: importing and normalising them.

Every object is brought to the same convention, whatever file it came from:
metres, Y up, its footprint centred on the origin and its lowest point at
height 0. Its footprint (width along X, depth along Z) is then exactly the size
of the rectangle shown on the map.

Front: the object keeps the orientation it has in its file, the way it looks in
the program it came from, including any rotation on its objects. Its front is
the direction of Blender's +Y axis (the green arrow), which arrives here as -Z:
the top of the map at rotation 0. Every conversion on the way is a rotation,
never a mirror, so nothing is flipped. A layer's quarter-turn setting handles
objects modelled facing another way.

GLB and OBJ are read with trimesh. FBX has no ready-made Python reader, so a
small reader for binary FBX is included: meshes, UVs, model transforms, unit
scale, axis settings, material colour and embedded textures. ASCII FBX, rigs and
animation are not read.
"""

import io
import json
import math
import struct
import zlib

import numpy as np
from PIL import Image


# ----------------------------------------------------------------- normalised parts
def _part(pos, faces, uv=None, image=None, colour=None, name="part"):
    return {"pos": np.asarray(pos, float), "faces": np.asarray(faces, np.int64).reshape(-1, 3),
            "uv": None if uv is None else np.asarray(uv, float),
            "image": image, "colour": colour if colour is not None else (0.7, 0.7, 0.7, 1.0),
            "name": name}


# how every import is oriented, for the console
SCENE_FRAME = "the orientation it has in its file"


def _from_trimesh(path):
    import trimesh
    scene = trimesh.load(path, force="scene", process=False)
    parts = []
    # every mesh where the scene puts it, rotations included: the object looks
    # as it did in the program it came from
    meshes = scene.dump(concatenate=False)
    for m in meshes:
        if not hasattr(m, "faces") or len(m.faces) == 0:
            continue
        uv, image, colour = None, None, None
        vis = m.visual
        if getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
            uv = np.array(vis.uv, float)
            mat = getattr(vis, "material", None)
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                image = img.convert("RGB")
            fac = getattr(mat, "baseColorFactor", None)
            if fac is not None:
                colour = tuple(np.asarray(fac, float).ravel()[:4] / (255.0 if np.max(fac) > 1 else 1.0))
            elif getattr(mat, "diffuse", None) is not None:
                colour = tuple(np.asarray(mat.diffuse, float).ravel()[:4] / 255.0)
        elif getattr(vis, "kind", None) == "vertex":
            c = np.asarray(vis.vertex_colors, float)[:, :4].mean(axis=0) / 255.0
            colour = tuple(c)
        elif getattr(vis, "kind", None) == "face":
            c = np.asarray(vis.face_colors, float)[:, :4].mean(axis=0) / 255.0
            colour = tuple(c)
        parts.append(_part(m.vertices, m.faces, uv, image, colour, name=str(getattr(m, "metadata", {}).get("name", "part"))))
    for q in parts:
        q["frame"] = SCENE_FRAME
    return parts


# ----------------------------------------------------------------- binary FBX
class _FBX:
    """A minimal reader for binary FBX: the node tree, enough to find meshes."""

    def __init__(self, data):
        if not data.startswith(b"Kaydara FBX Binary  \x00"):
            raise ValueError("only binary FBX can be read: in Blender, export FBX (it is binary by default)")
        self.d = data
        self.version = struct.unpack("<I", data[23:27])[0]
        self.wide = self.version >= 7500
        self.nodes = []
        pos = 27
        while pos < len(data):
            node, pos = self._node(pos)
            if node is None:
                break
            self.nodes.append(node)

    def _node(self, pos):
        d = self.d
        if self.wide:
            end, nprops, _ = struct.unpack("<QQQ", d[pos:pos + 24]); pos += 24
        else:
            end, nprops, _ = struct.unpack("<III", d[pos:pos + 12]); pos += 12
        nlen = d[pos]; pos += 1
        if end == 0:
            return None, pos
        name = d[pos:pos + nlen].decode("latin-1"); pos += nlen
        props = []
        for _ in range(nprops):
            t = chr(d[pos]); pos += 1
            if t == "Y":
                props.append(struct.unpack("<h", d[pos:pos + 2])[0]); pos += 2
            elif t == "C":
                props.append(bool(d[pos])); pos += 1
            elif t == "I":
                props.append(struct.unpack("<i", d[pos:pos + 4])[0]); pos += 4
            elif t == "F":
                props.append(struct.unpack("<f", d[pos:pos + 4])[0]); pos += 4
            elif t == "D":
                props.append(struct.unpack("<d", d[pos:pos + 8])[0]); pos += 8
            elif t == "L":
                props.append(struct.unpack("<q", d[pos:pos + 8])[0]); pos += 8
            elif t in "SR":
                n = struct.unpack("<I", d[pos:pos + 4])[0]; pos += 4
                raw = d[pos:pos + n]; pos += n
                props.append(raw.decode("utf-8", "replace") if t == "S" else raw)
            elif t in "fdlib":
                n, enc, clen = struct.unpack("<III", d[pos:pos + 12]); pos += 12
                raw = d[pos:pos + clen]; pos += clen
                if enc == 1:
                    raw = zlib.decompress(raw)
                dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[t]
                props.append(np.frombuffer(raw, dt, n).copy())
            else:
                raise ValueError("unexpected FBX property type %r" % t)
        children = []
        while pos < end:
            child, pos = self._node(pos)
            if child is None:
                break
            children.append(child)
        return (name, props, children), end


def _kids(node, name):
    return [c for c in node[2] if c[0] == name]


def _kid(node, name):
    k = _kids(node, name)
    return k[0] if k else None


def _props70(node):
    out = {}
    p70 = _kid(node, "Properties70") if node else None
    for p in (p70[2] if p70 else []):
        if p[0] == "P" and p[1]:
            out[p[1][0]] = p[1][4:]
    return out


def _euler_xyz(deg):
    x, y, z = np.radians(deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _from_fbx(path):
    f = _FBX(open(path, "rb").read())
    top = {n[0]: n for n in f.nodes}
    gs = _props70(top.get("GlobalSettings"))
    unit = float((gs.get("UnitScaleFactor") or [1.0])[0])             # centimetres per file unit
    up, up_s = int((gs.get("UpAxis") or [1])[0]), int((gs.get("UpAxisSign") or [1])[0])
    fr, fr_s = int((gs.get("FrontAxis") or [2])[0]), int((gs.get("FrontAxisSign") or [1])[0])
    co, co_s = int((gs.get("CoordAxis") or [0])[0]), int((gs.get("CoordAxisSign") or [1])[0])
    axis = np.zeros((3, 3))
    axis[0, co] = co_s                                                  # right
    axis[1, up] = up_s                                                  # up
    axis[2, fr] = fr_s                                                  # front
    objs = top.get("Objects")
    conns = top.get("Connections")
    if objs is None:
        raise ValueError("this FBX has no objects")
    by_id = {}
    for n in objs[2]:
        if n[1] and isinstance(n[1][0], int):
            by_id[n[1][0]] = n
    parent_of, children = {}, {}
    for c in (conns[2] if conns else []):
        if c[0] == "C" and len(c[1]) >= 3:
            kind, child, parent = c[1][0], c[1][1], c[1][2]
            prop = c[1][3] if len(c[1]) > 3 else None
            children.setdefault(parent, []).append((child, kind, prop))
            if kind == "OO":
                parent_of.setdefault(child, []).append(parent)

    def world(model_id):
        M = np.eye(4)
        cur = model_id
        seen = set()
        while cur in by_id and by_id[cur][0] == "Model" and cur not in seen:
            seen.add(cur)
            p = _props70(by_id[cur])
            T = np.eye(4)
            S = np.diag(list(p.get("Lcl Scaling", [1, 1, 1])[:3]) + [1])
            R = np.eye(4); R[:3, :3] = _euler_xyz(p.get("Lcl Rotation", [0, 0, 0])[:3])
            T[:3, 3] = p.get("Lcl Translation", [0, 0, 0])[:3]
            M = T @ R @ S @ M
            nxt = [q for q in parent_of.get(cur, []) if q in by_id and by_id[q][0] == "Model"]
            cur = nxt[0] if nxt else None
        return M

    # every mesh is placed where the scene puts it, rotations included, then
    # turned from the file's axes to Y up: the object looks as it did in the
    # program it came from. (Undoing the objects' own rotations instead turned
    # any object whose rotation was not zero, and Blender writes no usable
    # OriginalUpAxis, so its buildings came in lying on their side.)
    parts = []
    for gid, g in by_id.items():
        if g[0] != "Geometry" or len(g[1]) < 3 or g[1][2] != "Mesh":
            continue
        v_node, i_node = _kid(g, "Vertices"), _kid(g, "PolygonVertexIndex")
        if v_node is None or i_node is None:
            continue
        verts = np.asarray(v_node[1][0], float).reshape(-1, 3)
        pvi = np.asarray(i_node[1][0], np.int64)
        # polygons end at a negative index (stored as -(index+1))
        polys, cur = [], []
        for k, idx in enumerate(pvi):
            if idx < 0:
                cur.append((~idx, k)); polys.append(cur); cur = []
            else:
                cur.append((idx, k))
        uv_all = None
        uvl = _kid(g, "LayerElementUV")
        if uvl is not None and _kid(uvl, "UV") is not None:
            uvs = np.asarray(_kid(uvl, "UV")[1][0], float).reshape(-1, 2)
            mapping = (_kid(uvl, "MappingInformationType") or (0, ["ByPolygonVertex"]))[1][0]
            ref = (_kid(uvl, "ReferenceInformationType") or (0, ["Direct"]))[1][0]
            uidx = np.asarray(_kid(uvl, "UVIndex")[1][0], np.int64) if _kid(uvl, "UVIndex") is not None else None
            if mapping == "ByPolygonVertex":
                uv_all = uvs[uidx] if (ref == "IndexToDirect" and uidx is not None) else uvs[:len(pvi)]
            elif mapping in ("ByVertex", "ByVertice", "ByControlPoint"):
                uv_all = ("vertex", uvs[uidx] if (ref == "IndexToDirect" and uidx is not None) else uvs)
        # one vertex per polygon corner, then fan triangles
        P, U, T = [], [], []
        for poly in polys:
            base = len(P)
            for vi, k in poly:
                P.append(verts[vi])
                if isinstance(uv_all, np.ndarray) and k < len(uv_all):
                    U.append(uv_all[k])
                elif isinstance(uv_all, tuple) and vi < len(uv_all[1]):
                    U.append(uv_all[1][vi])
                else:
                    U.append((0.0, 0.0))
            for t in range(1, len(poly) - 1):
                T.append((base, base + t, base + t + 1))
        if not T:
            continue
        P = np.array(P, float)
        models = [q for q in parent_of.get(gid, []) if q in by_id and by_id[q][0] == "Model"]
        M = world(models[0]) if models else np.eye(4)
        P = (M[:3, :3] @ P.T).T + M[:3, 3]
        P = (axis @ P.T).T * unit / 100.0                               # metres, Y up, front -Z
        # material colour and texture, if any
        colour, image = None, None
        mats = [c for c, kind, _ in children.get(models[0], [])
                if c in by_id and by_id[c][0] == "Material"] if models else []
        if mats:
            pm = _props70(by_id[mats[0]])
            dc = pm.get("DiffuseColor") or pm.get("Diffuse")
            if dc:
                colour = (float(dc[0]), float(dc[1]), float(dc[2]), 1.0)
            for tid, kind, prop in children.get(mats[0], []):
                if tid in by_id and by_id[tid][0] == "Texture" and (prop or "").startswith("Diffuse"):
                    for vid, k2, _ in children.get(tid, []):
                        v = by_id.get(vid)
                        if v is not None and v[0] == "Video" and _kid(v, "Content") is not None:
                            raw = _kid(v, "Content")[1][0]
                            if isinstance(raw, (bytes, bytearray)) and len(raw) > 16:
                                try:
                                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                                except Exception:
                                    image = None
        U = np.array(U, float)
        U[:, 1] = 1.0 - U[:, 1]                                         # FBX UV origin is bottom-left
        name = g[1][1].split("\x00")[0] if isinstance(g[1][1], str) else "mesh"
        parts.append(_part(P, T, U if image is not None else None, image, colour, name=name))
    if not parts:
        raise ValueError("no meshes found in this FBX")
    for q in parts:
        q["frame"] = SCENE_FRAME
    return parts


# ----------------------------------------------------------------- normalise and store
def load(path):
    """Read an object file and bring it to metres, Y up, footprint centred, base at 0."""
    ext = str(path).lower().rsplit(".", 1)[-1]
    if ext == "fbx":
        parts = _from_fbx(path)
    elif ext in ("glb", "gltf", "obj"):
        parts = _from_trimesh(path)
    else:
        raise ValueError("unsupported format .%s: use GLB, OBJ or FBX" % ext)
    if not parts:
        raise ValueError("no geometry found in the file")
    # straighten by the object's own shape: find the tightest rectangle around
    # its footprint and turn the object so that rectangle lines up with the axes.
    # This removes a tilt whatever put it there (a rotation in the scene it came
    # from, or geometry modelled at an angle). It never turns more than 45
    # degrees, so the side that was facing the front still does
    straightened = 0.0
    allp = np.vstack([p["pos"] for p in parts])
    try:
        from shapely.geometry import MultiPoint
        rect = MultiPoint([tuple(q) for q in allp[:, [0, 2]]]).convex_hull.minimum_rotated_rectangle
        c = np.array(rect.exterior.coords)
        e = np.diff(c, axis=0)
        longest = e[np.argmax(np.hypot(e[:, 0], e[:, 1]))]
        ang = math.degrees(math.atan2(longest[1], longest[0]))
        ang = (ang + 45.0) % 90.0 - 45.0                          # the smallest turn that squares it
        if abs(ang) > 0.5:
            t = math.radians(-ang)
            ct, st = math.cos(t), math.sin(t)
            for p in parts:
                x, z = p["pos"][:, 0].copy(), p["pos"][:, 2].copy()
                p["pos"][:, 0] = x * ct - z * st
                p["pos"][:, 2] = x * st + z * ct
            straightened = ang
    except Exception:
        pass
    allp = np.vstack([p["pos"] for p in parts])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    shift = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    for p in parts:
        p["pos"] = p["pos"] - shift
    size = hi - lo
    return parts, {"frame": parts[0].get("frame", "its own axes"),
                   "straightened_deg": round(float(straightened), 1),
                   "width_m": round(float(size[0]), 3), "depth_m": round(float(size[2]), 3),
                   "height_m": round(float(size[1]), 3),
                   "triangles": int(sum(len(p["faces"]) for p in parts)), "parts": len(parts)}


def save(parts, folder):
    """Store normalised parts: one npz for geometry, one PNG per textured part."""
    folder.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, []
    for i, p in enumerate(parts):
        arrays["pos_%d" % i] = p["pos"]; arrays["faces_%d" % i] = p["faces"]
        if p["uv"] is not None:
            arrays["uv_%d" % i] = p["uv"]
        img = None
        if p["image"] is not None:
            img = "tex_%d.png" % i
            p["image"].save(folder / img)
        meta.append({"colour": list(p["colour"]), "image": img, "name": p["name"]})
    np.savez_compressed(folder / "geometry.npz", **arrays)
    (folder / "parts.json").write_text(json.dumps(meta))


def load_saved(folder):
    z = np.load(folder / "geometry.npz")
    meta = json.loads((folder / "parts.json").read_text())
    parts = []
    for i, m in enumerate(meta):
        img = Image.open(folder / m["image"]).convert("RGB") if m.get("image") else None
        parts.append(_part(z["pos_%d" % i], z["faces_%d" % i],
                           z["uv_%d" % i] if ("uv_%d" % i) in z.files else None, img,
                           tuple(m["colour"]), m.get("name", "part")))
    return parts
