"""
The 3D model in other formats: FBX (binary, version 7.4, as Blender and the
FBX SDK write it) and OBJ (with its .mtl), and the GLB again, with parts
combined as asked.

Everything is read back from the GLB the model is built as (app/model3d.py's
write_glb_scene): its meshes, where each stands (an object is one mesh placed
many times), its materials and pictures. So every format holds exactly the
same model.

- combine_ground: the streets (Road, and BridgeDeck), sidewalks (Sidewalk) and
  kerbs (Kerb) become one object, "Streets"; the blocks and islands and the
  markings stay objects of their own.
- combine_objects: every placed object (all copies of all objects) becomes one
  object, "Objects", each copy's place baked into its vertices. Off, each copy
  is its own object: in GLB and FBX the copies share one mesh (instances); OBJ
  has no instances, so each copy is written out.

Units are metres, Y up, as in the GLB. FBX and OBJ come in a zip with their
pictures in textures/, which the files refer to by relative path.

- front: which way every placed object's own front faces in the file, as Blender
  shows the axes once imported (Z up): "+y" as made here (Blender's green arrow),
  or "+x" (Unreal Engine's), "-y" or "-x". Each object's mesh is turned so its
  front is along that axis and each copy turned back by as much, so every object
  stands exactly where it stood, facing its street; only its own axes change. The
  streets and the rest have no front and do not move. OBJ keeps no object axes
  (every copy is written in place), so there it changes nothing.
"""
import io
import json
import math
import struct
import zipfile
import zlib

import numpy as np
from PIL import Image

GROUND = ("Road", "Sidewalk", "Kerb", "BridgeDeck")
_COMP = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ------------------------------------------------------------------ reading the GLB
def read_glb(path):
    """
    The model: {"materials": glTF materials, "images": [(bytes, mime)],
    "objects": [{"name", "prims": [{positions, normals, uv0, colors, indices,
    material}], "places": None, or [(translation, quaternion xyzw, scale)]}]}.
    """
    data = open(path, "rb").read()
    jlen = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + jlen])
    blob = data[20 + jlen + 8:]

    def view(i):
        v = gltf["bufferViews"][i]
        return blob[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]

    def acc(i):
        a = gltf["accessors"][i]
        v = gltf["bufferViews"][a["bufferView"]]
        n = _SIZE[a["type"]]
        arr = np.frombuffer(blob, _COMP[a["componentType"]], a["count"] * n, v.get("byteOffset", 0) + a.get("byteOffset", 0))
        return arr.reshape(a["count"], n) if n > 1 else arr

    images = [(view(im["bufferView"]), im.get("mimeType", "image/png")) for im in gltf.get("images", [])]
    tex_image = [t["source"] for t in gltf.get("textures", [])]
    materials = json.loads(json.dumps(gltf.get("materials", [])))
    # texture indices to image indices (this writer makes them one to one, but be sure)
    for m in materials:
        for holder, key in ((m.get("pbrMetallicRoughness", {}), "baseColorTexture"),
                            (m.get("pbrMetallicRoughness", {}), "metallicRoughnessTexture"), (m, "normalTexture")):
            if key in holder:
                holder[key]["index"] = tex_image[holder[key]["index"]]
    places = {}
    for nd in gltf.get("nodes", []):
        if "mesh" not in nd:
            continue
        if "translation" in nd or "rotation" in nd or "scale" in nd:
            sc = nd.get("scale", [1.0, 1.0, 1.0])
            places.setdefault(nd["mesh"], []).append((np.array(nd.get("translation", [0.0, 0.0, 0.0]), float),
                                                      np.array(nd.get("rotation", [0.0, 0.0, 0.0, 1.0]), float),
                                                      float(sc[0])))
        else:
            places.setdefault(nd["mesh"], None)
    objects = []
    for mi, m in enumerate(gltf.get("meshes", [])):
        prims = []
        for p in m["primitives"]:
            at = p["attributes"]
            prims.append({"positions": acc(at["POSITION"]).astype(np.float64),
                          "normals": acc(at["NORMAL"]).astype(np.float64) if "NORMAL" in at else None,
                          "uv0": acc(at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else None,
                          "colors": acc(at["COLOR_0"]).astype(np.float64) if "COLOR_0" in at else None,
                          "indices": acc(p["indices"]).astype(np.int64).reshape(-1, 3),
                          "material": p.get("material", 0)})
        objects.append({"name": m.get("name") or f"Mesh_{mi}", "prims": prims, "places": places.get(mi)})
    return {"materials": materials, "images": images, "objects": objects}


# ------------------------------------------------------------------ the objects' front
# the turn about the up axis (degrees, counter-clockwise seen from above; Y up here, Z up
# in Blender) that takes an object's front from +Y (Blender's, -Z in glTF) to each axis
FRONT_TURN = {"+y": 0.0, "+x": -90.0, "-y": 180.0, "-x": 90.0}


def _qmul(a, b):
    """Quaternions (x, y, z, w) multiplied: a then b in a's frame (the rotation a·b)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def face_front(model, front):
    """
    Every placed object's own front along `front` ("+y", "+x", "-y", "-x"): its mesh turned
    about the up axis, each copy turned back as much, so the world stays exactly as it was.
    """
    th = math.radians(FRONT_TURN.get(front, 0.0))
    if not th:
        return model
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])          # about +Y (up)
    back = np.array([0.0, math.sin(-th / 2), 0.0, math.cos(-th / 2)])
    objs = []
    for o in model["objects"]:
        if o["places"]:
            o = dict(o, prims=[dict(p, positions=p["positions"] @ R.T,
                                    normals=None if p["normals"] is None else p["normals"] @ R.T) for p in o["prims"]],
                     places=[(t, _qmul(q, back), sc) for t, q, sc in o["places"]])
        objs.append(o)
    return dict(model, objects=objs)


# ------------------------------------------------------------------ combining
def _quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _placed(p, place):
    """A primitive moved to a copy's place (rotation, uniform scale, then translation)."""
    t, q, s = place
    R = _quat_matrix(q)
    out = dict(p)
    out["positions"] = p["positions"] @ R.T * s + t
    if p["normals"] is not None:
        out["normals"] = p["normals"] @ R.T
    return out


def _merge(prims):
    """Primitives with the same material joined into one, the order of materials kept."""
    by, order = {}, []
    for p in prims:
        if p["material"] not in by:
            by[p["material"]] = []
            order.append(p["material"])
        by[p["material"]].append(p)
    out = []
    for mat in order:
        group = by[mat]
        if len(group) == 1:
            out.append(group[0])
            continue
        off, idx = 0, []
        for p in group:
            idx.append(p["indices"] + off)
            off += len(p["positions"])

        def cat(key, width):
            if any(p[key] is None for p in group):
                if all(p[key] is None for p in group):
                    return None
                fill = {"colors": 1.0, "uv0": 0.0, "normals": 0.0}[key]
                return np.concatenate([p[key] if p[key] is not None else np.full((len(p["positions"]), width), fill)
                                       for p in group])
            return np.concatenate([p[key] for p in group])
        out.append({"positions": np.concatenate([p["positions"] for p in group]), "normals": cat("normals", 3),
                    "uv0": cat("uv0", 2), "colors": cat("colors", 4), "indices": np.concatenate(idx), "material": mat})
    return out


def combine(model, combine_ground=False, combine_objects=False):
    """The model's objects with the streets, sidewalks and kerbs, and the placed objects, combined as asked."""
    objs, ground, placed = [], [], []
    for o in model["objects"]:
        if combine_ground and o["name"] in GROUND and o["places"] is None:
            ground.extend(o["prims"])
        elif combine_objects and o["places"]:
            for place in o["places"]:
                placed.extend(_placed(p, place) for p in o["prims"])
        else:
            objs.append(o)
    if ground:
        objs.insert(0, {"name": "Streets", "prims": _merge(ground), "places": None})
    if placed:
        objs.append({"name": "Objects", "prims": _merge(placed), "places": None})
    return dict(model, objects=objs)


# ------------------------------------------------------------------ pictures
def _ext(mime):
    return {"image/jpeg": "jpg", "image/png": "png"}.get(mime, "png")


def _texture_files(model):
    """Each picture's file name in textures/, and each material's roughness picture (the green channel)."""
    names = [f"textures/image_{i + 1}.{_ext(m)}" for i, (_, m) in enumerate(model["images"])]
    rough = {}
    for m in model["materials"]:
        r = m.get("pbrMetallicRoughness", {}).get("metallicRoughnessTexture")
        if r is not None and r["index"] not in rough:
            rough[r["index"]] = f"textures/image_{r['index'] + 1}_roughness.jpg"
    return names, rough


def _write_textures(z, model, names, rough):
    for (data, _), name in zip(model["images"], names):
        z.writestr(name, data)
    for i, name in rough.items():
        g = Image.open(io.BytesIO(model["images"][i][0])).convert("RGB").split()[1]
        buf = io.BytesIO()
        g.save(buf, "JPEG", quality=90)
        z.writestr(name, buf.getvalue())


def _safe(name):
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in name) or "material"


# ------------------------------------------------------------------ GLB
def write_glb(model, path):
    from app import model3d as M3
    meshes = []
    for o in model["objects"]:
        m = {"name": o["name"], "primitives": [dict(p, normals=p["normals"] if p["normals"] is not None
                                                     else np.tile([0.0, 1.0, 0.0], (len(p["positions"]), 1)))
                                                for p in o["prims"]]}
        if o["places"]:
            m["instances"] = [(t, q, s) for t, q, s in o["places"]]
        meshes.append(m)
    materials = json.loads(json.dumps(model["materials"]))
    return M3.write_glb_scene(path, meshes, materials, list(model["images"]))


# ------------------------------------------------------------------ OBJ
def write_obj_zip(model, path, stem):
    """An OBJ with its .mtl and pictures, in a zip."""
    names, rough = _texture_files(model)
    mats = [_safe(m.get("name") or f"material_{i + 1}") for i, m in enumerate(model["materials"])]
    seen = {}
    for i, n in enumerate(mats):                         # names must be unique
        if n in seen:
            mats[i] = f"{n}_{i + 1}"
        seen[n] = i
    mtl = io.StringIO()
    mtl.write("# Road Texture Agent: materials (metres, Y up)\n")
    for i, m in enumerate(model["materials"]):
        pbr = m.get("pbrMetallicRoughness", {})
        c = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        mtl.write(f"\nnewmtl {mats[i]}\nKa 0 0 0\nKd {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\nKs 0.04 0.04 0.04\n"
                  f"Ns 50\nd {c[3] if len(c) > 3 else 1.0:.4f}\nillum 2\nPr {pbr.get('roughnessFactor', 1.0):.3f}\nPm 0\n")
        if "baseColorTexture" in pbr:
            mtl.write(f"map_Kd {names[pbr['baseColorTexture']['index']]}\n")
            if m.get("alphaMode") in ("BLEND", "MASK"):
                mtl.write(f"map_d {names[pbr['baseColorTexture']['index']]}\n")
        if "metallicRoughnessTexture" in pbr:
            mtl.write(f"map_Pr {rough[pbr['metallicRoughnessTexture']['index']]}\n")
        if "normalTexture" in m:
            mtl.write(f"map_Bump -bm 1.0 {names[m['normalTexture']['index']]}\n")
    out = io.StringIO()
    out.write(f"# Road Texture Agent: {stem} (metres, Y up)\nmtllib {stem}.mtl\n")
    base = [1, 1, 1]                                     # next v, vt, vn (OBJ counts from 1)

    def put(name, prims):
        out.write(f"o {_safe(name)}\n")
        for p in prims:
            P, N, T, C = p["positions"], p["normals"], p["uv0"], p["colors"]
            n = len(P)
            if C is not None:
                out.write("\n".join(f"v {a:.4f} {b:.4f} {c:.4f} {r:.4f} {g:.4f} {bl:.4f}"
                                    for (a, b, c), (r, g, bl) in zip(P.tolist(), C[:, :3].tolist())) + "\n")
            else:
                out.write("\n".join(f"v {a:.4f} {b:.4f} {c:.4f}" for a, b, c in P.tolist()) + "\n")
            if T is not None:
                out.write("\n".join(f"vt {u:.5f} {1.0 - v:.5f}" for u, v in T.tolist()) + "\n")
            if N is not None:
                out.write("\n".join(f"vn {a:.4f} {b:.4f} {c:.4f}" for a, b, c in N.tolist()) + "\n")
            out.write(f"usemtl {mats[p['material']]}\n")
            I = p["indices"]
            fv = I + base[0]
            if T is not None and N is not None:
                ft, fn = I + base[1], I + base[2]
                out.write("\n".join(f"f {a}/{d}/{g} {b}/{e}/{h} {c}/{f}/{i}" for (a, b, c), (d, e, f), (g, h, i)
                                    in zip(fv.tolist(), ft.tolist(), fn.tolist())) + "\n")
            elif N is not None:
                fn = I + base[2]
                out.write("\n".join(f"f {a}//{d} {b}//{e} {c}//{f}" for (a, b, c), (d, e, f) in zip(fv.tolist(), fn.tolist())) + "\n")
            else:
                out.write("\n".join(f"f {a} {b} {c}" for a, b, c in fv.tolist()) + "\n")
            base[0] += n
            base[1] += n if T is not None else 0
            base[2] += n if N is not None else 0

    for o in model["objects"]:
        if o["places"]:
            for k, place in enumerate(o["places"]):
                put(f"{o['name']}_{k + 1}", [_placed(p, place) for p in o["prims"]])
        else:
            put(o["name"], o["prims"])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(f"{stem}.obj", out.getvalue())
        z.writestr(f"{stem}.mtl", mtl.getvalue())
        _write_textures(z, model, names, rough)
    return path


# ------------------------------------------------------------------ FBX (binary 7.4)
_HEAD = b"Kaydara FBX Binary\x20\x20\x00\x1a\x00"
_FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
_TIME_ID = b"1970-01-01 10:00:00:000"
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_VERSION = 7400


class _Node:
    """One FBX record: a name, its properties (typed), and the records inside it."""

    def __init__(self, name, *props):
        self.name, self.props, self.kids = name.encode() if isinstance(name, str) else name, [], []
        for p in props:
            self.add(p)

    def add(self, p):
        if isinstance(p, tuple):                        # an explicit (type, value)
            self.props.append(p)
        elif isinstance(p, bool):
            self.props.append(("C", p))
        elif isinstance(p, int):
            self.props.append(("L" if abs(p) >= 2 ** 31 else "I", p))
        elif isinstance(p, float):
            self.props.append(("D", p))
        elif isinstance(p, (bytes, str)):
            self.props.append(("S", p))
        elif isinstance(p, np.ndarray):
            self.props.append(({np.dtype(np.float64): "d", np.dtype(np.float32): "f", np.dtype(np.int32): "i",
                                np.dtype(np.int64): "l"}[p.dtype], p))
        else:
            raise TypeError(p)
        return self

    def child(self, name, *props):
        n = _Node(name, *props)
        self.kids.append(n)
        return n

    @staticmethod
    def _prop_bytes(t, v):
        if t == "C":
            return b"C" + struct.pack("<?", v)
        if t == "I":
            return b"I" + struct.pack("<i", v)
        if t == "L":
            return b"L" + struct.pack("<q", v)
        if t == "D":
            return b"D" + struct.pack("<d", v)
        if t == "F":
            return b"F" + struct.pack("<f", v)
        if t in ("S", "R"):
            b = v.encode("utf-8") if isinstance(v, str) else v
            return t.encode() + struct.pack("<I", len(b)) + b
        raw = np.ascontiguousarray(v).astype({"d": "<f8", "f": "<f4", "i": "<i4", "l": "<i8"}[t]).tobytes()
        if len(raw) > 128:
            comp = zlib.compress(raw, 1)
            return t.encode() + struct.pack("<3I", v.size, 1, len(comp)) + comp
        return t.encode() + struct.pack("<3I", v.size, 0, len(raw)) + raw

    def encode(self, offset, last):
        """The record as bytes, starting at offset in the file (its end offset is part of it)."""
        props = b"".join(self._prop_bytes(t, v) for t, v in self.props)
        head_len = 12 + 1 + len(self.name)
        body, pos = [], offset + head_len + len(props)
        for i, k in enumerate(self.kids):
            b = k.encode(pos, i == len(self.kids) - 1)
            body.append(b)
            pos += len(b)
        if self.kids or (not self.props and not last):
            body.append(b"\0" * 13)                     # the end of a nested list
            pos += 13
        return struct.pack("<3I", pos, len(self.props), len(props)) + bytes([len(self.name)]) + self.name + props + b"".join(body)


def _P(parent, name, ptype, label, flags, *values):
    """A Properties70 entry."""
    n = parent.child("P", name, ptype, label, flags)
    for v in values:
        n.add(v)
    return n


def _cls(name, cls):
    return name.encode("utf-8") + b"\x00\x01" + cls.encode()


def _euler_xyz(q):
    """A rotation (quaternion xyzw) as FBX's Euler angles in degrees (rotation order XYZ: X first)."""
    R = _quat_matrix(q)
    # R = Rz Ry Rx
    sy = -R[2, 0]
    if abs(sy) < 0.999999:
        y = math.asin(sy)
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        y = math.copysign(math.pi / 2, sy)
        x = math.atan2(-R[1, 2], R[1, 1])
        z = 0.0
    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def write_fbx_zip(model, path, stem):
    """A binary FBX (7.4) with its pictures, in a zip."""
    names, rough = _texture_files(model)
    uid = iter(range(1000000001, 2 ** 40))
    objects = _Node("Objects")
    conns = []
    counts = {"Geometry": 0, "Model": 0, "Material": 0, "Texture": 0, "Video": 0}

    # pictures and materials
    video_of, mat_ids = {}, []

    def video(path_rel):
        if path_rel in video_of:
            return video_of[path_rel]
        vid = next(uid)
        nm = path_rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        v = objects.child("Video", ("L", vid), _cls(nm, "Video"), "Clip")
        v.child("Type", "Clip")
        p70 = v.child("Properties70")
        _P(p70, "Path", "KString", "XRefUrl", "", path_rel)
        v.child("UseMipMap", 0)
        v.child("Filename", path_rel)
        v.child("RelativeFilename", path_rel)
        counts["Video"] += 1
        video_of[path_rel] = vid
        return vid

    def texture(path_rel, mat_id, slot, label):
        tid = next(uid)
        t = objects.child("Texture", ("L", tid), _cls(label, "Texture"), "")
        t.child("Type", "TextureVideoClip")
        t.child("Version", 202)
        t.child("TextureName", _cls(label, "Texture"))
        p70 = t.child("Properties70")
        _P(p70, "UseMaterial", "bool", "", "", 1)
        _P(p70, "UVSet", "KString", "", "", "UVMap")
        nm = path_rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        t.child("Media", _cls(nm, "Video"))
        t.child("FileName", path_rel)
        t.child("RelativeFilename", path_rel)
        t.child("ModelUVTranslation", 0.0, 0.0)
        t.child("ModelUVScaling", 1.0, 1.0)
        t.child("Texture_Alpha_Source", "None")
        t.child("Cropping", 0, 0, 0, 0)
        counts["Texture"] += 1
        conns.append(("OP", tid, mat_id, slot))
        conns.append(("OO", video(path_rel), tid))

    for i, m in enumerate(model["materials"]):
        mid = next(uid)
        mat_ids.append(mid)
        pbr = m.get("pbrMetallicRoughness", {})
        c = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        nm = m.get("name") or f"material_{i + 1}"
        mm = objects.child("Material", ("L", mid), _cls(nm, "Material"), "")
        mm.child("Version", 102)
        mm.child("ShadingModel", "Phong")
        mm.child("MultiLayer", 0)
        p70 = mm.child("Properties70")
        _P(p70, "DiffuseColor", "Color", "", "A", float(c[0]), float(c[1]), float(c[2]))
        _P(p70, "DiffuseFactor", "Number", "", "A", 1.0)
        _P(p70, "SpecularColor", "Color", "", "A", 0.04, 0.04, 0.04)
        _P(p70, "SpecularFactor", "Number", "", "A", 0.5)
        r = float(pbr.get("roughnessFactor", 1.0))
        _P(p70, "Shininess", "Number", "", "A", float(max(2.0, (1.0 - r) * 100)))
        _P(p70, "ShininessExponent", "Number", "", "A", float(max(2.0, (1.0 - r) * 100)))
        _P(p70, "ReflectionFactor", "Number", "", "A", 0.0)
        _P(p70, "Opacity", "Number", "", "A", float(c[3]) if len(c) > 3 else 1.0)
        counts["Material"] += 1
        if "baseColorTexture" in pbr:
            texture(names[pbr["baseColorTexture"]["index"]], mid, "DiffuseColor", f"{nm}_color")
            if m.get("alphaMode") in ("BLEND", "MASK"):         # see-through (decals): its alpha is the opacity
                texture(names[pbr["baseColorTexture"]["index"]], mid, "TransparentColor", f"{nm}_alpha")
        if "metallicRoughnessTexture" in pbr:
            texture(rough[pbr["metallicRoughnessTexture"]["index"]], mid, "ShininessExponent", f"{nm}_roughness")
        if "normalTexture" in m:
            texture(names[m["normalTexture"]["index"]], mid, "NormalMap", f"{nm}_normal")

    # geometry and models
    for o in model["objects"]:
        prims = o["prims"]
        mats = []
        for p in prims:
            if p["material"] not in mats:
                mats.append(p["material"])
        P = np.concatenate([p["positions"] for p in prims])
        idx, poly_mat, off = [], [], 0
        for p in prims:
            idx.append(p["indices"] + off)
            poly_mat.append(np.full(len(p["indices"]), mats.index(p["material"]), np.int32))
            off += len(p["positions"])
        I = np.concatenate(idx)
        pvi = I.astype(np.int32).copy()
        pvi[:, 2] = -pvi[:, 2] - 1                     # the last corner of each polygon, negated
        gid = next(uid)
        g = objects.child("Geometry", ("L", gid), _cls(o["name"], "Geometry"), "Mesh")
        g.child("Properties70")
        g.child("GeometryVersion", 124)
        g.child("Vertices", P.reshape(-1).astype(np.float64))
        g.child("PolygonVertexIndex", pvi.reshape(-1))
        flat = I.reshape(-1)
        layers = []
        if all(p["normals"] is not None for p in prims):
            N = np.concatenate([p["normals"] for p in prims])
            le = g.child("LayerElementNormal", 0)
            le.child("Version", 101); le.child("Name", "")
            le.child("MappingInformationType", "ByPolygonVertex"); le.child("ReferenceInformationType", "Direct")
            le.child("Normals", N[flat].reshape(-1).astype(np.float64))
            layers.append("LayerElementNormal")
        if any(p["colors"] is not None for p in prims):
            C = np.concatenate([p["colors"] if p["colors"] is not None else np.ones((len(p["positions"]), 4)) for p in prims])
            le = g.child("LayerElementColor", 0)
            le.child("Version", 101); le.child("Name", "Col")
            le.child("MappingInformationType", "ByPolygonVertex"); le.child("ReferenceInformationType", "IndexToDirect")
            le.child("Colors", C.reshape(-1).astype(np.float64))
            le.child("ColorIndex", flat.astype(np.int32))
            layers.append("LayerElementColor")
        if any(p["uv0"] is not None for p in prims):
            T = np.concatenate([p["uv0"] if p["uv0"] is not None else np.zeros((len(p["positions"]), 2)) for p in prims]).copy()
            T[:, 1] = 1.0 - T[:, 1]                    # glTF's pictures start at the top, FBX's at the bottom
            le = g.child("LayerElementUV", 0)
            le.child("Version", 101); le.child("Name", "UVMap")
            le.child("MappingInformationType", "ByPolygonVertex"); le.child("ReferenceInformationType", "IndexToDirect")
            le.child("UV", T.reshape(-1).astype(np.float64))
            le.child("UVIndex", flat.astype(np.int32))
            layers.append("LayerElementUV")
        le = g.child("LayerElementMaterial", 0)
        le.child("Version", 101); le.child("Name", "")
        le.child("MappingInformationType", "ByPolygon" if len(mats) > 1 else "AllSame")
        le.child("ReferenceInformationType", "IndexToDirect")
        le.child("Materials", np.concatenate(poly_mat) if len(mats) > 1 else np.zeros(1, np.int32))
        layers.append("LayerElementMaterial")
        lay = g.child("Layer", 0)
        lay.child("Version", 100)
        for name in layers:
            e = lay.child("LayerElement")
            e.child("Type", name)
            e.child("TypedIndex", 0)
        counts["Geometry"] += 1
        places = o["places"] or [None]
        for k, place in enumerate(places):
            mdl = next(uid)
            label = o["name"] if place is None or len(places) == 1 else f"{o['name']}_{k + 1}"
            n = objects.child("Model", ("L", mdl), _cls(label, "Model"), "Mesh")
            n.child("Version", 232)
            p70 = n.child("Properties70")
            if place is not None:
                t, q, s = place
                _P(p70, "Lcl Translation", "Lcl Translation", "", "A", *[float(v) for v in t])
                _P(p70, "Lcl Rotation", "Lcl Rotation", "", "A", *[float(v) for v in _euler_xyz(q)])
                _P(p70, "Lcl Scaling", "Lcl Scaling", "", "A", float(s), float(s), float(s))
            _P(p70, "DefaultAttributeIndex", "int", "Integer", "", 0)
            _P(p70, "InheritType", "enum", "", "", 1)
            n.child("MultiLayer", 0)
            n.child("MultiTake", 0)
            n.child("Shading", ("C", True))
            n.child("Culling", "CullingOff")
            counts["Model"] += 1
            conns.append(("OO", mdl, 0))
            conns.append(("OO", gid, mdl))
            for mi in mats:
                conns.append(("OO", mat_ids[mi], mdl))

    root = []
    hx = _Node("FBXHeaderExtension")
    hx.child("FBXHeaderVersion", 1003)
    hx.child("FBXVersion", _VERSION)
    hx.child("EncryptionType", 0)
    ts = hx.child("CreationTimeStamp")
    for k, v in (("Version", 1000), ("Year", 1970), ("Month", 1), ("Day", 1), ("Hour", 10), ("Minute", 0),
                 ("Second", 0), ("Millisecond", 0)):
        ts.child(k, v)
    hx.child("Creator", "Road Texture Agent")
    si = hx.child("SceneInfo", _cls("GlobalInfo", "SceneInfo"), "UserData")
    si.child("Type", "UserData")
    si.child("Version", 100)
    md = si.child("MetaData")
    md.child("Version", 100)
    for k in ("Title", "Subject", "Author", "Keywords", "Revision", "Comment"):
        md.child(k, "")
    p70 = si.child("Properties70")
    _P(p70, "DocumentUrl", "KString", "Url", "", f"/{stem}.fbx")
    _P(p70, "SrcDocumentUrl", "KString", "Url", "", f"/{stem}.fbx")
    _P(p70, "Original", "Compound", "", "")
    _P(p70, "Original|ApplicationVendor", "KString", "", "", "Road Texture Agent")
    _P(p70, "Original|ApplicationName", "KString", "", "", "Road Texture Agent")
    _P(p70, "Original|ApplicationVersion", "KString", "", "", "1")
    _P(p70, "Original|DateTime_GMT", "DateTime", "", "", "01/01/1970 00:00:00.000")
    _P(p70, "Original|FileName", "KString", "", "", f"/{stem}.fbx")
    root.append(hx)
    root.append(_Node("FileId", ("R", _FILE_ID)))
    root.append(_Node("CreationTime", ("S", _TIME_ID)))
    root.append(_Node("Creator", "Road Texture Agent"))
    gs = _Node("GlobalSettings")
    gs.child("Version", 1000)
    p70 = gs.child("Properties70")
    # metres, Y up, Z to the front, X to the right: the GLB's own axes
    for k, v in (("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2), ("FrontAxisSign", 1), ("CoordAxis", 0),
                 ("CoordAxisSign", 1), ("OriginalUpAxis", 1), ("OriginalUpAxisSign", 1)):
        _P(p70, k, "int", "Integer", "", v)
    _P(p70, "UnitScaleFactor", "double", "Number", "", 100.0)
    _P(p70, "OriginalUnitScaleFactor", "double", "Number", "", 100.0)
    _P(p70, "AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0)
    _P(p70, "DefaultCamera", "KString", "", "", "Producer Perspective")
    _P(p70, "TimeMode", "enum", "", "", 11)
    _P(p70, "TimeSpanStart", "KTime", "Time", "", ("L", 0))
    _P(p70, "TimeSpanStop", "KTime", "Time", "", ("L", 46186158000))
    _P(p70, "CustomFrameRate", "double", "Number", "", 24.0)
    root.append(gs)
    docs = _Node("Documents")
    docs.child("Count", 1)
    doc = docs.child("Document", ("L", next(uid)), "Scene", "Scene")
    p70 = doc.child("Properties70")
    _P(p70, "SourceObject", "object", "", "")
    _P(p70, "ActiveAnimStackName", "KString", "", "", "")
    doc.child("RootNode", ("L", 0))
    root.append(docs)
    root.append(_Node("References"))
    defs = _Node("Definitions")
    defs.child("Version", 100)
    kinds = [("GlobalSettings", 1)] + [(k, v) for k, v in counts.items() if v]
    defs.child("Count", sum(v for _, v in kinds))
    for k, v in kinds:
        ot = defs.child("ObjectType", k)
        ot.child("Count", v)
    root.append(defs)
    root.append(objects)
    cn = _Node("Connections")
    for c in conns:
        n = cn.child("C", c[0], ("L", c[1]), ("L", c[2]))
        if len(c) > 3:
            n.add(("S", c[3]))
    root.append(cn)
    tk = _Node("Takes")
    tk.child("Current", "")
    root.append(tk)

    buf = bytearray(_HEAD + struct.pack("<I", _VERSION))
    for i, n in enumerate(root):
        buf += n.encode(len(buf), False)
    buf += b"\0" * 13                                   # the end of the top-level list
    buf += _FOOT_ID + b"\0" * 4
    pad = ((len(buf) + 15) & ~15) - len(buf)
    buf += b"\0" * (pad or 16)
    buf += struct.pack("<I", _VERSION) + b"\0" * 120 + _FOOT_MAGIC
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(f"{stem}.fbx", bytes(buf))
        _write_textures(z, model, names, rough)
    return path


def convert(glb_path, out_path, fmt, stem, combine_ground=False, combine_objects=False, front="+y"):
    """The model at glb_path written as fmt ("glb", "fbx" or "obj"), its objects' front along front, combined as asked."""
    model = combine(face_front(read_glb(glb_path), front), combine_ground, combine_objects)
    if fmt == "fbx":
        write_fbx_zip(model, out_path, stem)
    elif fmt == "obj":
        write_obj_zip(model, out_path, stem)
    else:
        write_glb(model, out_path)
    return {"objects": [o["name"] for o in model["objects"]][:50], "count": len(model["objects"]),
            "copies": sum(len(o["places"]) for o in model["objects"] if o["places"])}
