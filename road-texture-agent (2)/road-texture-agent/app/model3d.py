"""
Stage 1 of 3D: a flat road model.

The road outline is traced from the same smooth distance field the texture is
drawn from, so the mesh border and the painted road edge line up exactly. The
outline becomes polygons with holes (the city blocks), is triangulated, placed
in real metres, and textured with the generated image. The result is written as
a GLB file, which Blender and Unreal Engine both open directly.

Axes follow glTF: Y is up, the road lies flat on X and Z, one unit is one metre.
Image right is +X and image down is +Z, so the model reads the same way as the
picture when seen from above. The model is centred on the origin.
"""

import io
import math
import json
import struct

import numpy as np
from PIL import Image
from skimage import measure
import mapbox_earcut as earcut
from shapely.geometry import Polygon

from app.generation import prepare_mask
from app import quadmesh as QM
from scipy import ndimage as ndi

MAX_TEXTURE = 8192          # the largest texture Unreal accepts without extra settings


# ------------------------------------------------------------------ outline
def outline_polygons(sdf_mask_coverage, simplify_px=0.35):
    """
    Trace the road outline where the distance field crosses zero.

    Returns shapely polygons in output-pixel coordinates (x, y), each with its
    holes. Tracing at the zero crossing gives sub-pixel positions, so the
    outline is smooth rather than following pixel steps.
    """
    coverage = sdf_mask_coverage
    # coverage is sdf + 0.5 clipped to 0..1; trace at 0.5, which is sdf = 0.
    # Pad with background so outlines touching the image border still close.
    padded = np.pad(coverage, 1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, 0.5)

    rings = []
    for c in contours:
        if len(c) < 4:
            continue
        xy = np.column_stack([c[:, 1] - 1, c[:, 0] - 1])       # back to image x, y
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 2.0:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        rings.append(poly.exterior.simplify(simplify_px, preserve_topology=True))

    # nest them: a ring inside an odd number of others is a hole
    polys = sorted((Polygon(r) for r in rings), key=lambda p: p.area, reverse=True)
    parents = []
    for i, p in enumerate(polys):
        pt = p.representative_point()
        parent = None
        for j in range(i - 1, -1, -1):             # smallest containing ring wins
            if polys[j].contains(pt):
                parent = j
                break
        parents.append(parent)

    def depth(i):
        d, k = 0, parents[i]
        while k is not None:
            d, k = d + 1, parents[k]
        return d

    out = []
    for i, p in enumerate(polys):
        if depth(i) % 2 == 0:                       # road
            holes = [polys[k].exterior for k in range(len(polys))
                     if parents[k] == i and depth(k) % 2 == 1]
            shape = Polygon(p.exterior, holes)
            if not shape.is_valid:
                shape = shape.buffer(0)
            if shape.geom_type == "MultiPolygon":
                out.extend(shape.geoms)
            elif not shape.is_empty:
                out.append(shape)
    return out


def triangulate(polygons):
    """Triangulate polygons with holes. Returns vertices (N, 2) and triangles (M, 3)."""
    verts, tris, base = [], [], 0
    for poly in polygons:
        ext = np.asarray(poly.exterior.coords)[:-1]
        rings = [ext] + [np.asarray(h.coords)[:-1] for h in poly.interiors]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue
        flat = np.vstack(rings).astype(np.float64)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        idx = earcut.triangulate_float64(flat, ends)
        if len(idx) == 0:
            continue
        verts.append(flat)
        tris.append(idx.reshape(-1, 3) + base)
        base += len(flat)
    if not verts:
        return np.zeros((0, 2)), np.zeros((0, 3), np.uint32)
    return np.vstack(verts), np.vstack(tris).astype(np.uint32)


# --------------------------------------------------------------------- glb
def _pad4(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path, positions, normals, uvs, indices, image_bytes, image_mime, name="Roads"):
    pos = positions.astype(np.float32).tobytes()
    nor = normals.astype(np.float32).tobytes()
    uv = uvs.astype(np.float32).tobytes()
    idx = indices.astype(np.uint32).reshape(-1).tobytes()

    chunks, views, offset = [], [], 0
    for blob, target in ((pos, 34962), (nor, 34962), (uv, 34962), (idx, 34963), (image_bytes, None)):
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
        if target:
            view["target"] = target
        views.append(view)
        padded = _pad4(blob)
        chunks.append(padded)
        offset += len(padded)
    binary = b"".join(chunks)

    gltf = {
        "asset": {"version": "2.0", "generator": "road-texture-agent"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0}]}],
        "materials": [{
            "name": "RoadSurface", "doubleSided": True,
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                     "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"source": 0, "sampler": 0}],
        "images": [{"bufferView": 4, "mimeType": image_mime, "name": "road_texture"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": int(indices.size), "type": "SCALAR"},
        ],
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(binary), 0x004E4942)); f.write(binary)
    return total


# ------------------------------------------------------------------ export
def export_road_glb(mask_path, texture_path, metres_per_pixel, output_scale, out_path):
    """
    Build the road model for one generated texture.

    metres_per_pixel is the mask's scale; output_scale is the size the texture
    was generated at (1, 2 or 4). The outline is traced at the texture's size,
    so the mesh edge sits exactly on the texture's road edge.
    """
    gray = np.array(Image.open(mask_path).convert("L"))
    mask, coverage = prepare_mask(gray, output_scale)
    H, W = mask.shape
    polys = outline_polygons(coverage)
    verts2d, tris = triangulate(polys)
    if len(tris) == 0:
        raise ValueError("no road found in the mask")

    mpp_out = metres_per_pixel / output_scale
    x = verts2d[:, 0] * mpp_out
    z = verts2d[:, 1] * mpp_out
    cx, cz = (W * mpp_out) / 2.0, (H * mpp_out) / 2.0
    positions = np.column_stack([x - cx, np.zeros_like(x), z - cz])
    uvs = np.column_stack([verts2d[:, 0] / W, verts2d[:, 1] / H])

    # every triangle must face up (+Y) so the road is visible from above.
    # Near-flat slivers are dropped first: their direction is decided by
    # rounding, so after conversion to 32-bit they can end up facing down.
    p = positions
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    ny = np.cross(b - a, c - a)[:, 1]
    keep = np.abs(ny) * 0.5 > 1e-3                 # at least a thousandth of a square metre
    tris, ny = tris[keep], ny[keep]
    flip = ny < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    # confirm in the precision the file is written in
    p32 = positions.astype(np.float32).astype(np.float64)
    a, b, c = p32[tris[:, 0]], p32[tris[:, 1]], p32[tris[:, 2]]
    ok = np.cross(b - a, c - a)[:, 1] > 0
    tris = tris[ok]
    normals = np.tile([0.0, 1.0, 0.0], (len(positions), 1))

    # texture: JPEG keeps the file small; capped at the size Unreal accepts
    tex = Image.open(texture_path).convert("RGB")
    if max(tex.size) > MAX_TEXTURE:
        k = MAX_TEXTURE / max(tex.size)
        tex = tex.resize((int(tex.width * k), int(tex.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    tex.save(buf, "JPEG", quality=92)

    size = write_glb(out_path, positions, normals, uvs, tris, buf.getvalue(), "image/jpeg")
    return {
        "vertices": int(len(positions)), "triangles": int(len(tris)),
        "polygons": len(polys),
        "holes": int(sum(len(p.interiors) for p in polys)),
        "size_m": [round(W * mpp_out, 1), round(H * mpp_out, 1)],
        "texture_px": list(tex.size),
        "file_bytes": int(size),
    }


# --------------------------------------------------------------- quad export
def _extend_road_colour(tex_arr, coverage, px):
    """
    Carry the road's colour a few pixels out into the background.

    A straightened mesh edge can sit a pixel outside the painted road edge, and
    would then show a thin dark line of background. Filling the first few
    pixels outside the road with the nearest road colour hides that, without
    changing anything inside the road.
    """
    road = coverage >= 0.5
    if not road.any():
        return tex_arr
    dist, (iy, ix) = ndi.distance_transform_edt(~road, return_indices=True)
    out = tex_arr.copy()
    near = (~road) & (dist <= px)
    out[near] = tex_arr[iy[near], ix[near]]
    return out


def export_road_quads_glb(mask_path, texture_path, metres_per_pixel, output_scale, out_path,
                          straightness=0.7, spacing_m=2.0, optimise=False):
    """
    The road as quads with straightened edges. Each quad is written as two
    triangles split along the same diagonal, so Blender's Triangles to Quads
    (Alt+J) joins every pair back into the quad it came from.
    """
    gray = np.array(Image.open(mask_path).convert("L"))
    mask1, cov1 = prepare_mask(gray, 1)
    mesh, det, rep = QM.build(mask1, output_scale, metres_per_pixel, straightness, spacing_m,
                              coverage=cov1, optimise=optimise)

    _, coverage = prepare_mask(gray, output_scale)
    H, W = coverage.shape

    # interchanges are multi-level and stay as traced triangles for now
    tris_extra = 0
    if rep["flagged"]:
        from shapely.geometry import Point
        polys = outline_polygons(coverage)
        for j in rep["flagged"]:
            disc = Point(j["cx"] * output_scale, j["cy"] * output_scale).buffer(j["r"] * output_scale)
            pieces = []
            for p in polys:
                inter = p.intersection(disc)
                if inter.is_empty:
                    continue
                pieces.extend(inter.geoms if inter.geom_type == "MultiPolygon" else [inter])
            v2, t2 = triangulate([g for g in pieces if g.geom_type == "Polygon"])
            base = len(mesh.v)
            mesh.v.extend(map(tuple, v2))
            mesh.tris.extend([tuple(int(i) + base for i in t) for t in t2])
            tris_extra += len(t2)

    fv, ft = QM.cluster_fill(mesh.clusters, coverage, output_scale, polys=mesh.fill_polys)
    base_fill = len(mesh.v)
    if len(ft):
        mesh.v.extend(map(tuple, fv))
        mesh.tris.extend([tuple(int(i) + base_fill for i in t) for t in ft])
    V = np.array(mesh.v, float)
    mpp_out = metres_per_pixel / output_scale
    positions = np.column_stack([V[:, 0] * mpp_out - W * mpp_out / 2, np.zeros(len(V)),
                                 V[:, 1] * mpp_out - H * mpp_out / 2])
    positions[base_fill:, 1] = -0.003                    # the fill sits just under the road
    uvs = np.column_stack([V[:, 0] / W, V[:, 1] / H])

    # face direction measured in the precision the file is written in, so what
    # is checked here is exactly what Blender and Unreal will see
    p32 = positions.astype(np.float32).astype(np.float64)

    def up(a, b, c):
        return np.cross(p32[b] - p32[a], p32[c] - p32[a])[1]

    tri_list, quads_kept, dropped, other_diag = [], 0, 0, 0
    for q in mesh.quads:
        a, b, c, d = q
        if up(a, b, c) + up(a, c, d) < 0:          # wound the wrong way: reverse the quad
            a, b, c, d = a, d, c, b
        # split along a-c; if that gives a flipped or flat triangle, the quad is
        # bent the other way, so split along b-d instead
        tris_q = _split_quad(up, a, b, c, d, 2e-4)
        if tris_q:
            tri_list.extend(tris_q)
            quads_kept += 1
            other_diag += int(len(tris_q) == 2 and tris_q[0] == (a, b, d))
        else:
            dropped += 1                           # a quad with no area at all
    for t in mesh.tris:
        a, b, c = t
        if up(a, b, c) < 0:
            a, b, c = a, c, b
        if up(a, b, c) > 2e-4:
            tri_list.append((a, b, c))
    tris = np.array(tri_list, np.uint32)
    normals = np.tile([0.0, 1.0, 0.0], (len(positions), 1))

    tex = np.array(Image.open(texture_path).convert("RGB"))
    tex = _extend_road_colour(tex, coverage, px=max(3, 2 * output_scale))
    img = Image.fromarray(tex)
    if max(img.size) > MAX_TEXTURE:
        k = MAX_TEXTURE / max(img.size)
        img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)

    size = write_glb(out_path, positions, normals, uvs, tris, buf.getvalue(), "image/jpeg")
    return {
        "mesh": "quads", "quads": int(quads_kept), "triangles_written": int(len(tris)),
        "extra_triangles": int(len(mesh.tris)), "interchange_triangles": int(tris_extra),
        "streets": rep["streets"], "junctions": rep["junctions_patched"],
        "junctions_skipped": rep["junctions_skipped"],
        "straightness": rep["straightness"], "spacing_m": spacing_m,
        "vertices": int(len(positions)), "size_m": [round(W * mpp_out, 1), round(H * mpp_out, 1)],
        "texture_px": list(img.size), "file_bytes": int(size), "dropped_slivers": dropped,
        "split_other_diagonal": int(other_diag),
    }


# ------------------------------------------------------ tiled export (stage 2)
def write_glb_scene(path, meshes, materials, images):
    """
    A GLB with several meshes, each with several primitives.

    meshes:    [{"name", "primitives": [{"positions","normals","uv0" (opt),"colors" (opt),
                                          "indices","material"}]}]
    materials: glTF material dicts; a texture index refers to `images`
    images:    [(bytes, mime)]
    """
    chunks, views, accessors = [], [], []
    offset = 0

    def add_view(blob, target=None):
        nonlocal offset
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
        if target:
            v["target"] = target
        views.append(v)
        padded = _pad4(blob)
        chunks.append(padded)
        offset += len(padded)
        return len(views) - 1

    def add_acc(arr, comp, typ, target, minmax=False):
        a = np.ascontiguousarray(arr)
        view = add_view(a.tobytes(), target)
        acc = {"bufferView": view, "componentType": comp, "count": int(len(a)), "type": typ}
        if minmax:
            acc["min"] = a.min(axis=0).tolist(); acc["max"] = a.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    gl_meshes, nodes = [], []
    for m in meshes:
        prims = []
        for p in m["primitives"]:
            attrs = {"POSITION": add_acc(p["positions"].astype(np.float32), 5126, "VEC3", 34962, True),
                     "NORMAL": add_acc(p["normals"].astype(np.float32), 5126, "VEC3", 34962)}
            if p.get("uv0") is not None:
                attrs["TEXCOORD_0"] = add_acc(p["uv0"].astype(np.float32), 5126, "VEC2", 34962)
            if p.get("colors") is not None:
                attrs["COLOR_0"] = add_acc(p["colors"].astype(np.float32), 5126, "VEC4", 34962)
            idx = add_acc(p["indices"].astype(np.uint32).reshape(-1), 5125, "SCALAR", 34963)
            prims.append({"attributes": attrs, "indices": idx, "material": p["material"]})
        gl_meshes.append({"name": m["name"], "primitives": prims})
        if m.get("instances"):
            # one copy of the geometry, placed many times: the file stays small
            for k, (tr, q, sc) in enumerate(m["instances"]):
                nodes.append({"mesh": len(gl_meshes) - 1, "name": "%s_%d" % (m["name"], k + 1),
                              "translation": [float(v) for v in tr], "rotation": [float(v) for v in q],
                              "scale": [float(sc)] * 3})
        else:
            nodes.append({"mesh": len(gl_meshes) - 1, "name": m["name"]})

    img_entries = []
    for data, mime in images:
        img_entries.append({"bufferView": add_view(data), "mimeType": mime})

    binary = b"".join(chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "road-texture-agent"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": gl_meshes, "materials": materials,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "textures": [{"source": i, "sampler": 0} for i in range(len(img_entries))],
        "images": img_entries,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views, "accessors": accessors,
    }
    if not img_entries:
        for k in ("samplers", "textures", "images"):
            gltf.pop(k)
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(binary), 0x004E4942)); f.write(binary)
    return total


def variation_factor(result_path, markings_path, coverage, mpp_out, strength=1.0, sigma_m=3.0):
    """
    The large variation layer: where the generated road is lighter or darker
    than usual, as a multiplier around 1.

    Dashes are removed first and filled from the road around them, so they do
    not show up as bright bands; then everything finer than a few metres is
    blurred away, since fine detail is the tile's job.
    """
    lum = np.array(Image.open(result_path).convert("L")).astype(np.float32)
    road = coverage >= 0.5
    paint = np.zeros_like(road)
    if markings_path:
        paint = np.array(Image.open(markings_path).convert("L")) > 20
    good = road & ~paint
    if not good.any():
        return np.ones_like(lum)
    _, (iy, ix) = ndi.distance_transform_edt(~good, return_indices=True)
    filled = lum[iy, ix]
    broad = ndi.gaussian_filter(filled, sigma=max(1.0, sigma_m / mpp_out))
    f = broad / max(float(np.median(broad[good])), 1e-6)
    f = 1.0 + (f - 1.0) * float(strength)
    return np.clip(f, 0.7, 1.3)


def export_road_tiled_glb(mask_path, result_path, markings_path, tileset, metres_per_pixel,
                          output_scale, out_path, straightness=0.7, spacing_m=2.0,
                          variation=1.0, seed=7, dash_cfg=None, paint_rgb=(235, 232, 222),
                          sidewalk=None, bridges=None, bridge_cfg=None, markings="strips",
                          optimise=False, blocks=None, scatter=None):
    """
    The road with repeating material tiles laid along each street, a large
    variation layer as vertex colours, and dashes as their own strips.
    """
    from scipy.ndimage import map_coordinates
    gray = np.array(Image.open(mask_path).convert("L"))
    mask1, cov1 = prepare_mask(gray, 1)
    mesh, det, rep = QM.build(mask1, output_scale, metres_per_pixel, straightness, spacing_m,
                              coverage=cov1, kerb_band=True, dashes=dash_cfg or {},
                              sidewalk=sidewalk if sidewalk and tileset["tiles"].get("sidewalk") else None,
                              bridges=bridges or None, optimise=optimise)
    _, coverage = prepare_mask(gray, output_scale)
    H, W = coverage.shape
    mpp_out = metres_per_pixel / output_scale
    tile_m = float(tileset["settings"]["tile_m"])
    tiles = tileset["tiles"]

    V = np.array(mesh.v, float)
    bcfg = dict({"height_m": 5.0, "ramp_m": 120.0, "deck_m": 1.0}, **(bridge_cfg or {}))
    plans, lift = [], np.zeros(len(V))
    if bridges:
        from app import bridges as BR
        plans = BR.plan(mesh, bridges, output_scale, metres_per_pixel,
                        height_m=bcfg["height_m"], ramp_m=bcfg["ramp_m"])
        lift = BR.lifts(mesh, plans, output_scale)
    world = np.column_stack([V[:, 0] * mpp_out - W * mpp_out / 2, lift,
                             V[:, 1] * mpp_out - H * mpp_out / 2])
    fac = variation_factor(result_path, markings_path, coverage, mpp_out, variation)
    vfac = map_coordinates(fac, [V[:, 1], V[:, 0]], order=1, mode="nearest")
    vfac_raw = vfac.copy()
    # each part's tone as a multiplier on the tile's common tone. Vertices where
    # two parts meet carry both parts' tones between them, so the change fades
    # across a quad instead of stepping at a line
    tones = tileset.get("part_tones") or {}
    base = float(tileset.get("tile_tone") or tones.get("open") or 1.0)
    role_f = {"open": 1.0}
    for role, part in (("kerb", "edge"), ("junction", "junction")):
        t = tones.get(part)
        role_f[role] = float(np.clip(t / base, 0.6, 1.5)) if (t and base) else 1.0
    vtone = np.array([role_f.get(r, 1.0) for r in mesh.role])
    vfac = vfac * vtone

    # each street or junction: its own variant, offset, rotation and flip
    def layout(owner):
        r = np.random.default_rng([seed, owner + 100000])
        return {"variant": int(r.integers(0, 10 ** 6)), "off": r.uniform(0, tile_m, 2),
                "rot": int(r.integers(0, 4)), "flip": bool(r.integers(0, 2))}
    layouts = {}

    def uv_for(vi, owner, part):
        L = layouts.setdefault(owner, layout(owner))
        if part == "junction" or mesh.st[vi] is None or owner <= 0 or owner >= 1_000_000:
            a, b = world[vi, 0], world[vi, 2]              # junctions and crossings: flat layout
        else:
            a, b = mesh.st[vi]                             # streets: along and across, in metres
        u, v = (a + L["off"][0]) / tile_m, (b + L["off"][1]) / tile_m
        for _ in range(L["rot"]):
            u, v = v, -u
        return (-u if L["flip"] else u), v

    p32 = world.astype(np.float32).astype(np.float64)

    def up(a, b, c):
        return np.cross(p32[b] - p32[a], p32[c] - p32[a])[1]

    # painted markings: one texture per dash width, each holding exactly one
    # dash cycle along the street, with the road's own grain
    painted = markings == "painted"
    width_class = {"centres": []}
    marked = {}                  # width class (cm) -> index into the marked textures
    mark_tex = []
    E = None
    if painted:
        dashed = [st for st in mesh.streets.values() if st.get("dash")]
        if dashed:
            cycle = float((dash_cfg or {}).get("cycle_m", 9.0))
            share = float((dash_cfg or {}).get("dash_share", 0.6))
            E = math.ceil(2 * max(st["half_width_m"] for st in dashed) * 1.15 / tile_m) * tile_m
            # at most three line widths, each a group of streets with similar
            # widths: many short pieces would otherwise each want a texture
            ws = np.sort([st["dash"]["width_m"] * 100 for st in dashed])
            groups_w = np.array_split(ws, min(3, len(ws)))
            centres = sorted({max(5, int(round(float(np.median(g)) / 5)) * 5) for g in groups_w if len(g)})
            width_class["centres"] = centres
            widths = centres
            for w_cm in widths:
                img = marked_texture(tiles["open"][0]["path"], tile_m, cycle, share, E, w_cm / 100.0, paint_rgb)
                marked[w_cm] = len(mark_tex)
                mark_tex.append((w_cm, img))

    def mark_class(owner):
        st = mesh.streets.get(owner) if 0 < owner < 1_000_000 else None
        if not painted or not st or not st.get("dash") or not width_class["centres"]:
            return None
        w = st["dash"]["width_m"] * 100
        return min(width_class["centres"], key=lambda c: abs(c - w))

    def in_dashed_run(q, owner):
        """Whether a quad lies where the street's dashes run, so the texture repeats line up."""
        st = mesh.streets[owner]["dash"]
        us = []
        for vi in q:
            if mesh.st[vi] is None:
                return False
            us.append((mesh.st[vi][0] - st["setback_m"]) / st["step_m"])
        return min(us) >= -(1 - st["share"]) + 1e-6 and max(us) <= st["n"] + 1e-6

    def marked_uv(vi, owner):
        st = mesh.streets[owner]["dash"]
        along, across = mesh.st[vi]
        return (along - st["setback_m"]) / st["step_m"], 0.5 + across / E

    # group quads into primitives by part and tile variant
    groups = {}
    for qi, q in enumerate(mesh.quads):
        part, owner = mesh.part[qi], mesh.owner[qi]
        L = layouts.setdefault(owner, layout(owner))
        tile_part = "junction" if part == "junction" and tiles.get("junction") else "open"
        wc = mark_class(owner) if part != "junction" else None
        if wc is not None and wc in marked and in_dashed_run(q, owner):
            groups.setdefault(("marked", wc), []).append((q, owner, part))
            continue
        vk = L["variant"] % len(tiles[tile_part])
        groups.setdefault((tile_part, vk), []).append((q, owner, part))

    images, materials, primitives, key_to_mat = [], [], [], {}
    for (part, vk), quads in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        buf = io.BytesIO()
        if part == "marked":
            mark_tex[marked[vk]][1].save(buf, "JPEG", quality=92)
            name = f"Road_marked_{vk}cm"
        else:
            Image.open(tiles[part][vk]["path"]).convert("RGB").save(buf, "JPEG", quality=92)
            name = f"Road_{'street' if part == 'open' else part}_{vk + 1}"
        images.append((buf.getvalue(), "image/jpeg"))
        materials.append({"name": name, "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1, "texCoord": 0},
                                                   "metallicFactor": 0.0, "roughnessFactor": 0.9}})
        remap, pos, uv, col, idx = {}, [], [], [], []

        def vid(vi, owner, qpart, _marked=(part == "marked")):
            key = (vi, owner)
            if key not in remap:
                remap[key] = len(pos)
                pos.append(world[vi])
                uv.append(marked_uv(vi, owner) if _marked else uv_for(vi, owner, qpart))
                c = float(vfac[vi]); col.append((c, c, c, 1.0))
            return remap[key]

        for q, owner, qpart in quads:
            a, b, c, d = q
            if up(a, b, c) + up(a, c, d) < 0:
                a, b, c, d = a, d, c, b
            for t in _split_quad(up, a, b, c, d, 2e-4):
                idx.append([vid(i, owner, qpart) for i in t])
        if not idx:
            continue
        primitives.append({"positions": np.array(pos), "normals": np.tile([0, 1, 0], (len(pos), 1)),
                           "uv0": np.array(uv), "colors": np.array(col),
                           "indices": np.array(idx), "material": len(materials) - 1})

    road_quads = sum(len(p["indices"]) for p in primitives) // 2   # streets and junctions only

    # tight groups of junctions, junctions at the border and any road nothing
    # else covered: the traced outline, 3 mm below the road, so strips on top
    # win wherever they reach and no hole can open where they do not
    fill_tris = 0
    fv, ft = QM.cluster_fill(mesh.clusters, coverage, output_scale, polys=mesh.fill_polys)
    if len(ft):
        Pf = np.column_stack([fv[:, 0] * mpp_out - W * mpp_out / 2, np.full(len(fv), -0.003),
                              fv[:, 1] * mpp_out - H * mpp_out / 2])
        ny = np.cross(Pf[ft[:, 1]] - Pf[ft[:, 0]], Pf[ft[:, 2]] - Pf[ft[:, 0]])[:, 1]
        ft = np.where((ny < 0)[:, None], ft[:, [0, 2, 1]], ft)
        ft = ft[np.abs(ny) * 0.5 > 1e-4]
        ff = map_coordinates(fac, [fv[:, 1], fv[:, 0]], order=1, mode="nearest")
        buf = io.BytesIO()
        Image.open(tiles["open"][0]["path"]).convert("RGB").save(buf, "JPEG", quality=92)
        images.append((buf.getvalue(), "image/jpeg"))
        materials.append({"name": "Road_fill", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                                   "metallicFactor": 0.0, "roughnessFactor": 0.9}})
        primitives.append({"positions": Pf, "normals": np.tile([0, 1, 0], (len(Pf), 1)),
                           "uv0": np.column_stack([Pf[:, 0] / tile_m, Pf[:, 2] / tile_m]),
                           "colors": np.column_stack([ff, ff, ff, np.ones_like(ff)]),
                           "indices": ft, "material": len(materials) - 1})
        fill_tris = int(len(ft))

    # interchanges: multi-level, so they keep their traced triangles, with the
    # open-road tile laid flat and the same variation layer
    ic_tris = 0
    if rep["flagged"]:
        from shapely.geometry import Point
        polys = outline_polygons(coverage)
        pieces = []
        for j in rep["flagged"]:
            disc = Point(j["cx"] * output_scale, j["cy"] * output_scale).buffer(j["r"] * output_scale)
            for poly in polys:
                inter = poly.intersection(disc)
                if not inter.is_empty:
                    pieces.extend(inter.geoms if inter.geom_type == "MultiPolygon" else [inter])
        v2, t2 = triangulate([g for g in pieces if g.geom_type == "Polygon"])
        if len(t2):
            P = np.column_stack([v2[:, 0] * mpp_out - W * mpp_out / 2, np.zeros(len(v2)),
                                 v2[:, 1] * mpp_out - H * mpp_out / 2])
            ny = np.cross(P[t2[:, 1]] - P[t2[:, 0]], P[t2[:, 2]] - P[t2[:, 0]])[:, 1]
            t2 = np.where((ny < 0)[:, None], t2[:, [0, 2, 1]], t2)
            t2 = t2[np.abs(ny) * 0.5 > 1e-3]
            f = map_coordinates(fac, [v2[:, 1], v2[:, 0]], order=1, mode="nearest")
            buf = io.BytesIO()
            Image.open(tiles["open"][0]["path"]).convert("RGB").save(buf, "JPEG", quality=92)
            images.append((buf.getvalue(), "image/jpeg"))
            materials.append({"name": "Road_interchange", "doubleSided": True,
                              "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.9}})
            primitives.append({"positions": P, "normals": np.tile([0, 1, 0], (len(P), 1)),
                               "uv0": np.column_stack([P[:, 0] / tile_m, P[:, 2] / tile_m]),
                               "colors": np.column_stack([f, f, f, np.ones_like(f)]),
                               "indices": t2, "material": len(materials) - 1})
            ic_tris = int(len(t2))

    meshes = [{"name": "Road", "primitives": primitives}]
    sw_info = _sidewalk_meshes(mesh, world, vfac_raw, tiles, tile_m, seed, images, materials, meshes)
    scatter_info = None
    if scatter:
        stand = (float(blocks.get("height_m", 0.10)) if blocks else
                 float(sidewalk.get("height_m", 0.10)) if sidewalk else 0.0)
        scatter_info = _object_meshes(mesh, scatter, metres_per_pixel, output_scale, mpp_out, W, H,
                                      stand, images, materials, meshes)
    blocks_info = None
    if blocks:
        mask_shape = (coverage.shape[0] // output_scale, coverage.shape[1] // output_scale)
        blocks_info = _block_meshes(mesh, mask_shape, output_scale, mpp_out, W, H,
                                    fac, tiles, tile_m, images, materials, meshes,
                                    height_m=float(blocks.get("height_m", 0.10)),
                                    with_sidewalks=bool(mesh.sw_quads),
                                    cell_m=16.0 if optimise else 8.0)
    deck_info = _deck_edges(mesh, plans, world, tiles, tile_m, bcfg["deck_m"], images, materials, meshes) \
        if plans else None

    # dashes: their own mesh, a centimetre above the road so they never flicker into it
    dpos, dcol, didx = [], [], []
    painted_dashes = 0
    for di, strip in enumerate(mesh.dashes):
        base = len(dpos)
        owner = mesh.dash_owner[di] if di < len(mesh.dash_owner) else None
        if painted and owner is not None and mark_class(owner) in marked:
            painted_dashes += 1
            continue                                        # painted into the road texture
        pts = [(x, y) for (lx, ly, rx, ry) in strip for (x, y) in ((lx, ly), (rx, ry))]
        dl = BR.dash_lift(mesh, plans, output_scale, owner, pts) if plans else np.zeros(len(pts))
        for k_, (x, y) in enumerate(pts):
            dpos.append((x * mpp_out - W * mpp_out / 2, 0.01 + float(dl[k_]), y * mpp_out - H * mpp_out / 2))
            f = float(map_coordinates(fac, [[y], [x]], order=1, mode="nearest")[0])
            c = 1.0 + (f - 1.0) * 0.5                       # paint wears too, but less than asphalt
            dcol.append((c, c, c, 1.0))
        P = np.array(dpos, np.float32).astype(np.float64)     # the precision the file stores
        for k in range(len(strip) - 1):
            a, b, c, d = base + 2 * k, base + 2 * k + 1, base + 2 * k + 3, base + 2 * k + 2
            for t in ((a, b, c), (a, c, d)):
                n_y = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])[1]
                if abs(n_y) < 2e-6:
                    continue                                # no area: a repeated point
                didx.append(t if n_y > 0 else (t[0], t[2], t[1]))
    dash_count = len(mesh.dashes)
    if didx:
        materials.append({"name": "RoadMarkings", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": [c / 255 for c in paint_rgb] + [1.0],
                                                   "metallicFactor": 0.0, "roughnessFactor": 0.6}})
        meshes.append({"name": "Markings", "primitives": [{
            "positions": np.array(dpos), "normals": np.tile([0, 1, 0], (len(dpos), 1)),
            "colors": np.array(dcol), "indices": np.array(didx), "material": len(materials) - 1}]})

    size = write_glb_scene(out_path, meshes, materials, images)

    return {"mesh": "tiled", "quads": int(road_quads), "streets": rep["streets"],
            "junctions": rep["junctions_patched"], "materials": len(materials),
            "dashes": dash_count, "tile_m": tile_m, "mm_per_px": tileset.get("mm_per_px"),
            "variation": variation, "size_m": [round(W * mpp_out, 1), round(H * mpp_out, 1)],
            "file_bytes": int(size), "interchange_triangles": ic_tris, "sidewalk": sw_info,
            "fill_triangles": fill_tris, "groups": rep.get("clusters"), "bare_filled": rep.get("bare_filled"),
            "blocks": blocks_info,
            "scatter": scatter_info,
            "mesh_detail": "optimised" if optimise else "full",
            "rows_full": rep.get("rows_full"), "rows_kept": rep.get("rows_kept"),
            "markings": ("painted" if painted and marked else "strips"),
            "painted_dashes": painted_dashes if painted else 0,
            "marked_textures": [{"width_cm": w, "px": list(im.size)} for w, im in mark_tex],
            "marked_quads": sum(len(v) for k, v in groups.items() if k[0] == "marked"),
            "strip_dashes": len(mesh.dashes) - (painted_dashes if painted else 0),
            "bridges": [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                         for k, v in pl.items() if k in ("index", "ok", "s_in", "s_out", "r_before",
                                                           "r_after", "height", "steepest_pct",
                                                           "warnings", "streets")}
                        for pl in plans] if plans else None,
            "deck": deck_info,
            "layouts": {int(k): v for k, v in layouts.items()}, "_mesh": mesh, "_world": world,
            "_fac": fac}


def repeat_check(glb_path, mesh, world, tile_m, mm_per_px=20.0):
    """
    Measure, on the saved model, whether the tile repeat is visible.

    Along the longest street, the road is rendered from above and averaged
    across its width. A visible repeat shows as a peak in similarity exactly one
    tile length apart, standing above nearby distances. Similarity alone would
    mislead: smooth large-scale variation always resembles itself a few metres
    on, without repeating.
    """
    from app import preview3d as P3
    # the longest street, and a straight stretch of it
    lengths = {}
    for qi, owner in enumerate(mesh.owner):
        if owner > 0:
            for vi in mesh.quads[qi]:
                st = mesh.st[vi]
                if st is not None:
                    lo, hi = lengths.get(owner, (1e9, -1e9))
                    lengths[owner] = (min(lo, st[0]), max(hi, st[0]))
    if not lengths:
        return None
    sid = max(lengths, key=lambda k: lengths[k][1] - lengths[k][0])
    # a line running along the street: its centre line, or in the optimised mesh
    # (which has none) the vertices nearest the centre on one side
    cand = {vi for qi, o in enumerate(mesh.owner) if o == sid for vi in mesh.quads[qi]
            if mesh.st[vi] is not None and mesh.st[vi][1] >= -1e-6}
    by_row = {}
    for vi in cand:
        key = round(mesh.st[vi][0], 3)
        if key not in by_row or mesh.st[vi][1] < mesh.st[by_row[key]][1]:
            by_row[key] = vi
    verts = sorted(by_row.values(), key=lambda vi: mesh.st[vi][0])
    if len(verts) < 3:
        return None
    pts = world[verts][:, [0, 2]]
    mid = len(pts) // 2
    span = max(1, min(len(pts) // 2, int(30 / max(1e-6, np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))))))
    a, b = pts[max(0, mid - span)], pts[min(len(pts) - 1, mid + span)]
    d = b - a
    if np.linalg.norm(d) < 3 * tile_m:
        return None
    horizontal = abs(d[0]) >= abs(d[1])
    # the road surface only: paving is a regular pattern that repeats by design,
    # and would be counted as the tile repeating if it were included
    prims = [p for p in P3.load_glb(glb_path) if p["mesh"] == "Road"]
    x0, z0 = min(a[0], b[0]) - 6, min(a[1], b[1]) - 6
    w, h = abs(d[0]) + 12, abs(d[1]) + 12
    out = {}
    for use in (False, True):
        img = P3.render(prims, x0, z0, w, h, mm_per_px=mm_per_px, use_colours=use).astype(float).mean(axis=2)
        road = (img > 35) & (img < 150)
        if not horizontal:
            img, road = img.T, road.T
        col = np.array([img[road[:, x], x].mean() if road[:, x].sum() > 20 else np.nan
                        for x in range(img.shape[1])])
        p = col[~np.isnan(col)]
        if len(p) < 4 * tile_m * 1000 / mm_per_px:
            return None
        p = p - p.mean()
        # one tile along a street at an angle is shorter than a tile along the axis
        along = abs(d[0] if horizontal else d[1]) / np.linalg.norm(d)
        lag = int(tile_m * along * 1000 / mm_per_px)

        def r(l):
            x, y = p[:-l], p[l:]
            return float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-9))
        out["with_variation" if use else "tiles_only"] = round(r(lag) - np.mean(
            [r(int(lag * f)) for f in (0.8, 0.85, 1.15, 1.2)]), 3)
    out["street_m"] = round(float(np.linalg.norm(d)), 1)
    return out



def _sidewalk_meshes(mesh, world, vfac, tiles, tile_m, seed, images, materials, meshes):
    """
    The sidewalk top (paving, with a kerb stone along its edge) and the vertical
    kerb face down to the road, as two meshes of quads.
    """
    if (not mesh.sw_quads and not getattr(mesh, "sw_tris", None)) or not tiles.get("sidewalk"):
        return None
    Wh = world.copy()
    Wh[:, 1] = world[:, 1] + np.array(mesh.h)                # raised sidewalk, on top of any bridge lift
    p32 = Wh.astype(np.float32).astype(np.float64)

    def up(a, b, c):
        return np.cross(p32[b] - p32[a], p32[c] - p32[a])[1]

    def layout(owner):
        r = np.random.default_rng([seed, owner + 300000])
        # paving is laid square to the kerb: flips are fine, quarter turns would
        # turn bricks across the pavement, so only half turns
        return {"variant": int(r.integers(0, 10 ** 6)), "off": r.uniform(0, tile_m, 2),
                "rot": 2 * int(r.integers(0, 2)), "flip": bool(r.integers(0, 2))}
    lay = {}

    def uv(vi, owner):
        L = lay.setdefault(owner, layout(owner))
        st = mesh.st[vi]
        # a quad must use one kind of coordinate for all its corners: streets
        # measure along the street, corners (owned by a junction) use flat map
        # coordinates, even where they share a vertex with a street
        along_kerb = 0 < owner < 1_000_000 or owner > 3_000_000
        a, b = (st if (st is not None and along_kerb) else (Wh[vi, 0], Wh[vi, 2]))
        u, v = (a + L["off"][0]) / tile_m, (b + L["off"][1]) / tile_m
        for _ in range(L["rot"]):
            u, v = v, -u
        return (-u if L["flip"] else u), v

    def add_tile_material(name, path):
        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, "JPEG", quality=92)
        images.append((buf.getvalue(), "image/jpeg"))
        materials.append({"name": name, "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                                   "metallicFactor": 0.0, "roughnessFactor": 0.85}})
        return len(materials) - 1

    # top surface: paving grouped by variant, kerb stone on its own material
    groups = {}
    for qi, q in enumerate(mesh.sw_quads):
        owner, part = mesh.sw_owner[qi], mesh.sw_part[qi]
        L = lay.setdefault(owner, layout(owner))
        if part == "paving":
            key = ("paving", L["variant"] % len(tiles["sidewalk"]))
        else:
            key = ("kerbstone", 0)
        groups.setdefault(key, []).append((q, owner))

    prims = []
    for (part, vk), quads in sorted(groups.items()):
        if part == "paving":
            mat = add_tile_material(f"Sidewalk_paving_{vk + 1}", tiles["sidewalk"][vk]["path"])
        else:
            kt = (tiles.get("kerbstone") or tiles["sidewalk"])[0]["path"]
            mat = add_tile_material("Kerb_stone", kt)
        remap, pos, uvs, col, idx = {}, [], [], [], []

        def vid(vi, owner):
            key = (vi, owner)
            if key not in remap:
                remap[key] = len(pos)
                pos.append(Wh[vi]); uvs.append(uv(vi, owner))
                c = 1.0 + (float(vfac[vi]) - 1.0) * 0.5      # weathering, gentler than on asphalt
                col.append((c, c, c, 1.0))
            return remap[key]

        for q, owner in quads:
            a, b, c, d = q
            if up(a, b, c) + up(a, c, d) < 0:
                a, b, c, d = a, d, c, b
            for t in _split_quad(up, a, b, c, d, 1e-5):
                idx.append([vid(i, owner) for i in t])
        if idx:
            prims.append({"positions": np.array(pos), "normals": np.tile([0, 1, 0], (len(pos), 1)),
                          "uv0": np.array(uvs), "colors": np.array(col), "indices": np.array(idx),
                          "material": mat})
    # small islands paved completely
    if getattr(mesh, "sw_tris", None):
        pos, uvs, col, idx = [], [], [], []
        remap = {}
        for tri, owner in zip(mesh.sw_tris, mesh.sw_tri_owner):
            a, b, c = tri
            if up(a, b, c) < 0:
                a, b, c = a, c, b
            if up(a, b, c) <= 1e-6:
                continue
            row = []
            for vi in (a, b, c):
                if vi not in remap:
                    remap[vi] = len(pos)
                    pos.append(Wh[vi]); uvs.append((Wh[vi, 0] / tile_m, Wh[vi, 2] / tile_m))
                    cc = 1.0 + (float(vfac[vi]) - 1.0) * 0.5
                    col.append((cc, cc, cc, 1.0))
                row.append(remap[vi])
            idx.append(row)
        if idx:
            mat = add_tile_material("Sidewalk_island", tiles["sidewalk"][0]["path"])
            prims.append({"positions": np.array(pos), "normals": np.tile([0, 1, 0], (len(pos), 1)),
                          "uv0": np.array(uvs), "colors": np.array(col), "indices": np.array(idx),
                          "material": mat})
    if prims:
        meshes.append({"name": "Sidewalk", "primitives": prims})

    # the vertical kerb face, facing the road
    kt = (tiles.get("kerbstone") or tiles["sidewalk"])[0]["path"]
    mat = add_tile_material("Kerb_face", kt)
    pos, nrm, uvs, col, idx = [], [], [], [], []
    for q, facing in zip(mesh.kerb_quads, mesh.kerb_facing):
        b0, b1, t1, t0 = q
        P = p32[[b0, b1, t1, t0]]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        f3 = np.array([facing[0], 0.0, facing[1]])
        order = [0, 1, 2, 3] if np.dot(n, f3) > 0 else [0, 3, 2, 1]
        n = n / np.linalg.norm(n) * (1 if np.dot(n, f3) > 0 else -1)
        base = len(pos)
        for k in order:
            vi = (b0, b1, t1, t0)[k]
            pos.append(Wh[vi]); nrm.append(n)
            uvs.append(((Wh[vi, 0] + Wh[vi, 2]) / tile_m, Wh[vi, 1] / tile_m))
            c = 1.0 + (float(vfac[vi]) - 1.0) * 0.5
            col.append((c, c, c, 1.0))
        idx.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    if idx:
        meshes.append({"name": "Kerb", "primitives": [{
            "positions": np.array(pos), "normals": np.array(nrm), "uv0": np.array(uvs),
            "colors": np.array(col), "indices": np.array(idx), "material": mat}]})

    top = sum(len(p["indices"]) for p in prims) // 2
    return {"top_quads": int(top), "kerb_faces": int(len(idx) // 2),
            "height_m": mesh.sw_cfg["height_m"] if mesh.sw_cfg else None}



def _deck_edges(mesh, plans, world, tiles, tile_m, deck_m, images, materials, meshes):
    """
    Give a raised road some thickness: a side face down each edge and an
    underside, deck_m below the road surface, wherever it is off the ground.
    Without them a bridge is a ribbon that is see-through from the side.
    """
    Wh = world.copy()
    Wh[:, 1] = world[:, 1] + np.array(mesh.h)
    rows_list = []                                   # per row: (left outer, right outer, road left, road right)
    for p in plans:
        if not p.get("ok"):
            continue
        for sid in p["streets"]:
            st = mesh.streets.get(sid)
            if not st:
                continue
            sw = st["sw"]
            rows = []
            for i, r in enumerate(st["rows"]):
                lo = sw.get(1.0, {}).get(i); ro = sw.get(-1.0, {}).get(i)
                rows.append((lo[3] if lo else r[0], ro[3] if ro else r[-1], r[0], r[-1]))
            rows_list.append(rows)
        for oid in p["owners"]:
            c = mesh.connectors.get(oid) if isinstance(oid, int) else None
            if not c or not c["bridge"]:
                continue
            rows = []
            left, right = c["sw"].get(0), c["sw"].get(len(c["rows"][0]) - 1)
            for i, r in enumerate(c["rows"]):
                rows.append((left[i][3] if left and i < len(left) else r[0],
                             right[i][3] if right and i < len(right) else r[-1], r[0], r[-1]))
            rows_list.append(rows)

    pos, nrm, uvs, idx = [], [], [], []
    under_pos, under_idx = [], []

    def face(a, b, c, d, n):
        base = len(pos)
        for q in (a, b, c, d):
            pos.append(q); nrm.append(n)
            uvs.append(((q[0] + q[2]) / tile_m, q[1] / tile_m))
        idx.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    faces = 0
    for rows in rows_list:
        for i in range(len(rows) - 1):
            r0, r1 = rows[i], rows[i + 1]
            y_road = min(world[r0[2], 1], world[r1[2], 1])
            if y_road < 0.05:
                continue
            centre0 = (Wh[r0[2]] + Wh[r0[3]]) / 2
            bottoms = []
            for side in (0, 1):
                t0, t1 = Wh[r0[side]].copy(), Wh[r1[side]].copy()
                b0, b1 = t0.copy(), t1.copy()
                b0[1] = max(0.0, world[r0[2 + side], 1] - deck_m)
                b1[1] = max(0.0, world[r1[2 + side], 1] - deck_m)
                bottoms.append((b0, b1))
                n = np.cross(t1 - t0, b0 - t0)
                out = t0 - centre0; out[1] = 0
                if np.dot(n, out) < 0:
                    face(t0, b0, b1, t1, -n / (np.linalg.norm(n) or 1))
                else:
                    face(t0, t1, b1, b0, n / (np.linalg.norm(n) or 1))
                faces += 1
            # underside, facing down
            (lb0, lb1), (rb0, rb1) = bottoms
            if min(lb0[1], lb1[1], rb0[1], rb1[1]) > 0.01:
                n = np.cross(lb1 - lb0, rb0 - lb0)
                if n[1] > 0:
                    face(lb0, rb0, rb1, lb1, np.array([0.0, -1.0, 0.0]))
                else:
                    face(lb0, lb1, rb1, rb0, np.array([0.0, -1.0, 0.0]))
    if not idx:
        return {"faces": 0}
    kt = (tiles.get("kerbstone") or tiles.get("sidewalk") or tiles["open"])[0]["path"]
    buf = io.BytesIO()
    Image.open(kt).convert("RGB").save(buf, "JPEG", quality=92)
    images.append((buf.getvalue(), "image/jpeg"))
    materials.append({"name": "Bridge_deck", "doubleSided": True,
                      "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                               "metallicFactor": 0.0, "roughnessFactor": 0.9}})
    meshes.append({"name": "BridgeDeck", "primitives": [{
        "positions": np.array(pos), "normals": np.array(nrm), "uv0": np.array(uvs),
        "indices": np.array(idx), "material": len(materials) - 1}]})
    return {"faces": int(len(idx) // 2), "thickness_m": deck_m}



def marked_texture(tile_path, tile_m, cycle_m, share, across_m, width_m, paint_rgb, max_px=2048):
    """
    One dash cycle of road, with the dash painted along its centre.

    Columns run along the street (one full cycle: dash then gap), rows run
    across it, centred on the centre line. The grain is the open-road tile,
    laid a whole number of times each way so the texture joins itself where it
    repeats. The dash cycle is rarely a whole number of tiles, so along the
    street the tiles are stretched by the small amount that makes them fit
    (9 m holds two 4 m tiles stretched by 12%, which cannot be seen in grain).
    """
    tile = np.array(Image.open(tile_path).convert("RGB")).astype(np.float32)
    tp = tile.shape[0]
    k_along = max(1, int(round(cycle_m / tile_m)))
    k_across = max(1, int(round(across_m / tile_m)))
    base = np.tile(tile, (k_across, k_along, 1))                   # rows across, columns along
    h, w = base.shape[:2]
    f = min(1.0, max_px / max(h, w))
    out_w, out_h = max(8, int(w * f)), max(8, int(h * f))
    img = np.array(Image.fromarray(base.clip(0, 255).astype(np.uint8)).resize((out_w, out_h), Image.LANCZOS)).astype(np.float32)

    # the dash: along [0, share) of the cycle, across the centre, soft one-pixel edges
    along_px = out_w / cycle_m                                     # pixels per metre along
    across_px = out_h / across_m
    xs = np.arange(out_w) + 0.5
    ys = np.arange(out_h) + 0.5
    ax = np.clip(np.minimum(xs, share * out_w - xs) + 0.5, 0, 1)   # coverage along
    ay = np.clip(width_m / 2 * across_px - np.abs(ys - out_h / 2) + 0.5, 0, 1)
    a = (ay[:, None] * ax[None, :])[..., None]
    grain = img.mean(axis=2, keepdims=True)
    paint = np.array(paint_rgb, np.float32)[None, None, :] * (0.92 + 0.08 * grain / max(float(grain.mean()), 1e-6))
    img = img * (1 - a) + paint * a
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))



def _split_quad(up, a, b, c, d, eps):
    """
    Two triangles for a quad, facing up. Split along a-c, or along b-d if the
    quad is bent the other way. A quad with two corners in the same place is
    really a triangle (rows fanning round a corner meet at one point): then the
    one half with area is kept, where dropping the whole quad left a hole.
    """
    for t1, t2 in (((a, b, c), (a, c, d)), ((a, b, d), (b, c, d))):
        if up(*t1) > eps and up(*t2) > eps:
            return [t1, t2]
    for t in ((a, b, c), (a, c, d), (a, b, d), (b, c, d)):
        if up(*t) > eps:
            return [t]
    return []



def _block_meshes(mesh, shape_mask, scale, mpp_out, W, H, fac, tiles, tile_m, images, materials, meshes,
                  height_m=0.10, with_sidewalks=True, cell_m=8.0):
    """
    The blocks and islands between roads as flat planes with the sidewalk's
    paving. With sidewalks, each block reaches all the way to the kerb line
    2 mm below the sidewalk top, so the sidewalk covers its edge with no gap and
    the two never share a height (which would flicker). Without sidewalks, the
    block sits at sidewalk height and gets its own kerb face down to the road.
    """
    from scipy.ndimage import map_coordinates
    polys = QM.block_polygons(mesh, shape_mask, scale, mpp_out * scale)
    if not polys or not tiles.get("sidewalk"):
        return {"blocks": 0}
    # blocks stop exactly at the sidewalk's outer edge: the sidewalk (paving,
    # kerb stone, paved islands) is cut out of each block, so the two meet edge
    # to edge instead of one running underneath the other
    overlap_cut = 0.0
    if with_sidewalks:
        sw = _sidewalk_union(mesh)
        if sw is not None and not sw.is_empty:
            cut = []
            for g in polys:
                r = g.difference(sw)
                for part in (r.geoms if hasattr(r, "geoms") else [r]):
                    if part.geom_type == "Polygon" and part.area * (mpp_out ** 2) > 1.0:
                        cut.append(part)
            overlap_cut = sum(g.area for g in polys) - sum(g.area for g in cut)
            polys = cut
    if not polys:
        return {"blocks": 0}
    top = height_m
    # a grid of quads rather than long thin triangles: the variation layer is
    # blended across each face, so a face hundreds of metres long smeared it
    # into diagonal streaks. Small even faces blend it smoothly
    cell_px = cell_m / mpp_out
    v2, t2, n_quads = _grid_mesh(polys, cell_px)
    if not len(t2):
        return {"blocks": 0}
    P = np.column_stack([v2[:, 0] * mpp_out - W * mpp_out / 2, np.full(len(v2), top),
                         v2[:, 1] * mpp_out - H * mpp_out / 2])
    P32 = P.astype(np.float32).astype(np.float64)                 # as the file stores it
    ny = np.cross(P32[t2[:, 1]] - P32[t2[:, 0]], P32[t2[:, 2]] - P32[t2[:, 0]])[:, 1]
    t2 = np.where((ny < 0)[:, None], t2[:, [0, 2, 1]], t2)
    t2 = t2[np.abs(ny) * 0.5 > 1e-4]
    # the variation layer comes from the road image; inside a block there is no
    # road, only the nearest road's colour copied outwards, which makes stripes.
    # Blocks get their own gentle, broad weathering: the layer blurred over
    # about 25 m, at a third of its strength
    from scipy import ndimage as _ndi
    fb = _ndi.gaussian_filter(fac, sigma=max(1.0, 25.0 / mpp_out))
    f = map_coordinates(fb, [np.clip(v2[:, 1], 0, H - 1), np.clip(v2[:, 0], 0, W - 1)], order=1, mode="nearest")
    f = 1.0 + (f - 1.0) * 0.33
    buf = io.BytesIO()
    Image.open(tiles["sidewalk"][0]["path"]).convert("RGB").save(buf, "JPEG", quality=92)
    images.append((buf.getvalue(), "image/jpeg"))
    materials.append({"name": "Block_paving", "doubleSided": True,
                      "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                               "metallicFactor": 0.0, "roughnessFactor": 0.85}})
    prims = [{"positions": P, "normals": np.tile([0, 1, 0], (len(P), 1)),
              "uv0": np.column_stack([P[:, 0] / tile_m, P[:, 2] / tile_m]),
              "colors": np.column_stack([f, f, f, np.ones_like(f)]),
              "indices": t2, "material": len(materials) - 1}]

    faces = 0
    road_edge = getattr(mesh, "kerb_surface", None)
    if road_edge is None or road_edge.is_empty:
        road_edge = QM._road_surface(mesh)
    if polys:
        # the block's edge is the kerb: a vertical face down to the road, facing it
        pos, nrm, uvs, idx = [], [], [], []
        for g in polys:
            for ring in [g.exterior] + list(g.interiors):
                c = np.array(ring.coords)
                on_frame = ((c[:, 0] < 0.6 * scale) | (c[:, 1] < 0.6 * scale) |
                            (c[:, 0] > W - 0.6 * scale) | (c[:, 1] > H - 0.6 * scale))
                for k in range(len(c) - 1):
                    if on_frame[k] and on_frame[k + 1]:
                        continue                                # the image border is not a kerb
                    from shapely.geometry import Point as _Pt
                    mid_pt = _Pt((c[k][0] + c[k + 1][0]) / 2, (c[k][1] + c[k + 1][1]) / 2)
                    if road_edge.distance(mid_pt) > 0.05 / mpp_out:
                        continue                                # beside a sidewalk: no kerb here
                    a2, b2 = c[k], c[k + 1]
                    A = np.array([a2[0] * mpp_out - W * mpp_out / 2, 0.0, a2[1] * mpp_out - H * mpp_out / 2])
                    B = np.array([b2[0] * mpp_out - W * mpp_out / 2, 0.0, b2[1] * mpp_out - H * mpp_out / 2])
                    if np.linalg.norm(B - A) < 1e-3:
                        continue                                # under a millimetre: no face
                    quad = [A, B, B + [0, height_m, 0], A + [0, height_m, 0]]
                    n = np.cross(quad[1] - quad[0], quad[3] - quad[0])
                    mid = (a2 + b2) / 2
                    # the face should look away from the block, towards the road
                    d = np.array([b2[1] - a2[1], -(b2[0] - a2[0])])
                    probe = mid + d / (np.linalg.norm(d) or 1) * 0.5
                    from shapely.geometry import Point
                    outward = not g.contains(Point(*probe))
                    want = np.array([d[0], 0.0, d[1]]) * (1 if outward else -1)
                    if np.dot(n, want) < 0:
                        quad = [quad[1], quad[0], quad[3], quad[2]]
                        n = -n
                    n = n / (np.linalg.norm(n) or 1)
                    base = len(pos)
                    for qv in quad:
                        pos.append(qv); nrm.append(n); uvs.append(((qv[0] + qv[2]) / tile_m, qv[1] / tile_m))
                    idx.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
                    faces += 1
        if idx:
            kt = (tiles.get("kerbstone") or tiles["sidewalk"])[0]["path"]
            buf = io.BytesIO()
            Image.open(kt).convert("RGB").save(buf, "JPEG", quality=92)
            images.append((buf.getvalue(), "image/jpeg"))
            materials.append({"name": "Block_kerb", "doubleSided": True,
                              "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                                       "metallicFactor": 0.0, "roughnessFactor": 0.9}})
            prims.append({"positions": np.array(pos), "normals": np.array(nrm), "uv0": np.array(uvs),
                          "indices": np.array(idx), "material": len(materials) - 1})
    meshes.append({"name": "Blocks", "primitives": prims})
    return {"blocks": len(polys), "triangles": int(len(t2)), "quads": int(n_quads),
            "cell_m": cell_m, "kerb_faces": faces, "height_m": round(top, 3),
            "overlap_removed_m2": round(overlap_cut * mpp_out ** 2, 1)}



def _grid_mesh(polys, cell):
    """
    Cover polygons with one grid of square cells aligned across the whole map.
    Cells fully inside become quads (two triangles sharing a diagonal); cells
    cut by the outline are clipped and split into a few small triangles, so the
    edge follows the outline exactly. Vertices on shared cell corners and edges
    are welded, so there are no cracks between cells.
    """
    import math
    from shapely.geometry import box
    from shapely.prepared import prep
    index, V, T = {}, [], []
    quads = 0

    def vid(x, y):
        key = (round(x, 3), round(y, 3))
        i = index.get(key)
        if i is None:
            i = len(V)
            index[key] = i
            V.append((x, y))
        return i

    for g in polys:
        pg = prep(g)
        minx, miny, maxx, maxy = g.bounds
        for ix in range(int(math.floor(minx / cell)), int(math.ceil(maxx / cell))):
            for iy in range(int(math.floor(miny / cell)), int(math.ceil(maxy / cell))):
                x0, y0 = ix * cell, iy * cell
                b = box(x0, y0, x0 + cell, y0 + cell)
                if not pg.intersects(b):
                    continue
                if pg.contains(b):
                    a, b2, c, d = vid(x0, y0), vid(x0 + cell, y0), vid(x0 + cell, y0 + cell), vid(x0, y0 + cell)
                    T.extend([(a, b2, c), (a, c, d)])
                    quads += 1
                    continue
                piece = g.intersection(b)
                parts = piece.geoms if hasattr(piece, "geoms") else [piece]
                for part in parts:
                    if part.geom_type != "Polygon" or part.area < 1e-6:
                        continue
                    pv, pt = triangulate([part])
                    ids = [vid(float(x), float(y)) for x, y in pv]
                    T.extend((ids[i], ids[j], ids[k]) for i, j, k in pt)
    return np.array(V, float), np.array(T, np.int64).reshape(-1, 3), quads



def _sidewalk_union(mesh):
    """The whole sidewalk, as one shape in texture pixels: paving, kerb stone, paved islands."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    V = mesh.v
    parts = []
    for q in mesh.sw_quads:
        pg = Polygon([V[i] for i in q])
        if not pg.is_valid:
            pg = pg.buffer(0)
        if not pg.is_empty and pg.area > 0:
            parts.append(pg)
    for tri in getattr(mesh, "sw_tris", []):
        pg = Polygon([V[i] for i in tri])
        if pg.is_valid and pg.area > 0:
            parts.append(pg)
    if not parts:
        return None
    # a hair of growth merges quads that share an edge; shrinking back keeps
    # the outer edge exactly where the sidewalk ends
    return unary_union([p.buffer(0.001) for p in parts]).buffer(-0.001)



def _object_meshes(mesh, scatter, mpp_mask, scale, mpp_out, W, H, stand_m, images, materials, meshes):
    """
    Each placement is a grid of copies of one object: nx along the rectangle's
    width, ny along its depth, with the gaps between them. Each copy is an
    instance of the object, standing on the block or sidewalk. Copies whose
    footprint lies on the road are left out: objects stand only on blocks and
    sidewalks.
    """
    from shapely.geometry import Polygon
    from shapely import make_valid
    road = getattr(mesh, "kerb_surface", None)
    if road is None or road.is_empty:
        road = QM._road_surface(mesh)
    road = make_valid(road)                    # a repaired shape: intersections cannot fail on it
    placed, skipped, by_obj = 0, 0, {}
    problems = []
    for p in scatter["placements"]:
      try:
        o = scatter["objects"][p["object"]]
        meta = o["meta"]
        k = float(meta.get("scale", 1.0))
        turn = int(meta.get("turn", 0)) % 4
        w, d = meta["width_m"] * k, meta["depth_m"] * k
        if turn % 2:
            w, d = d, w                                         # a quarter turn swaps width and depth
        a = math.radians(p["angle"])
        u = np.array([math.cos(a), math.sin(a)])                 # along the width, in image axes
        v = np.array([-u[1], u[0]])                               # along the depth
        Lx = p["nx"] * w + (p["nx"] - 1) * p["gap_x"]
        Ly = p["ny"] * d + (p["ny"] - 1) * p["gap_y"]
        c = np.array([p["cx"], p["cy"]]) * mpp_mask              # metres, image axes
        for i in range(p["nx"]):
            for j in range(p["ny"]):
                cc = c + u * (-Lx / 2 + w / 2 + i * (w + p["gap_x"])) + v * (-Ly / 2 + d / 2 + j * (d + p["gap_y"]))
                corners = [cc + u * sx * w / 2 + v * sy * d / 2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                fp = Polygon([(x / mpp_out, y / mpp_out) for x, y in corners])
                if road.intersection(fp).area > 0.02 * fp.area:
                    skipped += 1
                    continue
                # image x right, image y down = world +X, +Z; rotation about the up axis
                tr = (cc[0] - W * mpp_out / 2, stand_m, cc[1] - H * mpp_out / 2)
                # the object keeps the orientation it was modelled in: at rotation 0
                # its front (Blender's +Y) faces the top of the map. Each turn adds a
                # quarter turn clockwise, seen from above
                phi = -(a + turn * math.pi / 2)
                q = (0.0, math.sin(phi / 2), 0.0, math.cos(phi / 2))
                by_obj.setdefault(p["object"], []).append((tr, q, k))
                placed += 1
      except Exception as e:
        problems.append("%s: %s" % (scatter["objects"].get(p["object"], {}).get("meta", {}).get("name", "object"), e))
    for oid, inst in list(by_obj.items()):
      try:
        o = scatter["objects"][oid]
        prims = []
        for part in o["parts"]:
            normals = _vertex_normals(part["pos"], part["faces"])
            if part["image"] is not None and part["uv"] is not None:
                buf = io.BytesIO(); part["image"].save(buf, "PNG")
                images.append((buf.getvalue(), "image/png"))
                mat = {"name": "%s_%s" % (o["meta"]["name"], part["name"]), "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": len(images) - 1},
                                                "metallicFactor": 0.0, "roughnessFactor": 0.8}}
            else:
                col = [float(x) for x in part["colour"]][:4]
                col += [1.0] * (4 - len(col))
                mat = {"name": "%s_%s" % (o["meta"]["name"], part["name"]), "doubleSided": True,
                       "pbrMetallicRoughness": {"baseColorFactor": col, "metallicFactor": 0.0,
                                                "roughnessFactor": 0.8}}
            materials.append(mat)
            prims.append({"positions": part["pos"], "normals": normals,
                          "uv0": part["uv"] if (part["image"] is not None and part["uv"] is not None) else None,
                          "indices": part["faces"], "material": len(materials) - 1})
        meshes.append({"name": o["meta"]["name"], "primitives": prims, "instances": inst})
      except Exception as e:
        placed -= len(inst)
        problems.append("%s: %s" % (scatter["objects"][oid]["meta"].get("name", "object"), e))
    return {"copies": placed, "skipped_on_road": skipped, "objects": len(by_obj), "problems": problems}



def _vertex_normals(pos, faces):
    """Smooth lighting normals: each face's normal, weighted by its area, summed at its corners."""
    pos = np.asarray(pos, float); faces = np.asarray(faces, np.int64)
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    length = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(length > 1e-12, vn / np.maximum(length, 1e-12), np.array([0.0, 1.0, 0.0]))
    return vn
