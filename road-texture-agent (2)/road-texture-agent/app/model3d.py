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
from app import surface as SF
from app import library as LIB
from app import quadmesh as QM
from app import progress as prog
from app import placements as PL
from app import islands as ISL
from app import decals as DC
from app import lanes as LN
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
                          straightness=0.7, spacing_m=2.0, optimise=False, inner=None):
    """
    The road as quads with straightened edges. Each quad is written as two
    triangles split along the same diagonal, so Blender's Triangles to Quads
    (Alt+J) joins every pair back into the quad it came from. inner: the inner
    streets as exact shapes (see export_road_tiled_glb).
    """
    gray = np.array(Image.open(mask_path).convert("L"))
    exact = inner["shapes"] if inner and inner.get("shapes") else None
    mask1, cov1 = prepare_mask(np.array(Image.open(inner["base_mask_path"]).convert("L")) if exact else gray, 1)
    mesh, det, rep = QM.build(mask1, output_scale, metres_per_pixel, straightness, spacing_m,
                              coverage=cov1, optimise=optimise, exact=exact)

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
def _image_index(images, data, mime):
    """The index of this picture in images, added if it is not there yet (shared, not repeated)."""
    for i, (d, m) in enumerate(images):
        if m == mime and d == data:
            return i
    images.append((data, mime))
    return len(images) - 1


def _sidewalk_kind(tile):
    """Paving (a pattern whose joints must line up with the colour), plain concrete, or a library material's kind."""
    tile = tile or {}
    if tile.get("library"):
        e = LIB.entry(tile["library"])
        return e["kind"] if e else "paving"
    return "concrete" if tile.get("method") == "placeholder concrete" else "paving"


def tile_material(name, img, images, materials, size_m, kind, roughness, surface="scan", own=None, mix=None, drift=0.0):
    """
    A material for a repeating tile: its colour, and with surface, the bump
    (normal map) and roughness (app/surface.py): "scan" from a bundled scan of
    the kind where there is one, else from the tile's grain; "grain" always from
    the grain; False: colour only. own: the tile's own (normal, roughness, name,
    height, ao, relief_m) pictures, made with its colour (library materials),
    used unless surface is off. size_m: the width and height the picture
    covers; kind: asphalt, paving or concrete. Returns the material's index.

    With surface, also its ambient occlusion (glTF occlusionTexture: the gaps
    between stones and the pores, in the shade of the sky) and its height map,
    which glTF has no slot for. All three grey maps share one picture, as glTF
    lays them out: occlusion in red, roughness in green, and the height in blue
    (metalness's channel, unused: the metallic factor is 0). The material's
    extras say so: "relief": {"index": the texture, "channel": 2, "depth_m":
    metres from its 0 to its 1}. The 3D tab and the photo render use it for
    parallax (no extra faces); other programs ignore it.

    mix (ground): a second ground material in patches through this one,
    {"tile": its tile record, "amount": 0-0.9, "size_m": patch size, "seed"}:
    its colour, normal and occlusion-roughness-height pictures in the extras
    ("mix": textures, depth_m, amount, size_m, seed, and where it lies: thresh,
    scale, offset), blended by the 3D tab and the photo render where the ground
    pattern (app/islands.py; extras "pattern": its texture, red) is above
    thresh, the higher stones of either showing at the edges. drift: how much
    colour and shine drift over a metre or few (ground; the pattern's green and
    blue). The islands texture reads the same pattern, so they agree. Other
    programs see this material alone.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    pbr = {"baseColorTexture": {"index": _image_index(images, buf.getvalue(), "image/jpeg")},
           "metallicFactor": 0.0, "roughnessFactor": roughness}
    mat = {"name": name, "doubleSided": True, "pbrMetallicRoughness": pbr}
    if surface:
        nrm, rgh, hgt, ao, used, relief = SF.maps(img, size_m, kind, source="grain" if surface == "grain" else "scan", own=own)
        orh = _image_index(images, SF.pack_orh(rgh, ao, hgt), "image/jpeg")
        mat["extras"] = {"surface": used, "relief": {"index": orh, "channel": 2, "depth_m": relief}}
        if mix and mix.get("tile"):
            bimg = Image.open(mix["tile"]["path"])
            bn, br, bh, ba, _, brel = SF.maps(bimg, size_m, "ground", own=LIB.own_maps(mix["tile"]))
            bbuf = io.BytesIO()
            bimg.convert("RGB").save(bbuf, "JPEG", quality=92)
            mat["extras"]["mix"] = {"map": _image_index(images, bbuf.getvalue(), "image/jpeg"),
                                    "normal": _image_index(images, bn, "image/jpeg"),
                                    "orh": _image_index(images, SF.pack_orh(br, ba, bh), "image/jpeg"),
                                    "depth_m": brel, "amount": round(float(mix["amount"]), 3),
                                    "size_m": round(float(mix["size_m"]), 2), "seed": int(mix.get("seed", 0)),
                                    "name": mix.get("name", ""),
                                    **ISL.mix_params(mix["amount"], mix["size_m"], mix.get("seed", 0))}
        if drift > 0:
            mat["extras"]["drift"] = round(float(drift), 3)
        if drift > 0 or "mix" in mat["extras"]:
            mat["extras"]["pattern"] = _image_index(images, ISL.ground_pattern_png(), "image/png")
        mat["normalTexture"] = {"index": _image_index(images, nrm, "image/jpeg")}
        mat["occlusionTexture"] = {"index": orh}
        pbr["metallicRoughnessTexture"] = {"index": orh}
        pbr["roughnessFactor"] = 1.0                      # the map holds it
    materials.append(mat)
    return len(materials) - 1


def _tangents(pos, nrm, uv, idx):
    """
    glTF tangents (x, y, z, w) for a normal map: the tangent along increasing u,
    and w such that cross(normal, tangent) * w runs up the picture (decreasing v),
    as glTF's normal maps expect.
    """
    pos, nrm, uv = (np.asarray(a, np.float64) for a in (pos, nrm, uv))
    idx = np.asarray(idx, np.int64).reshape(-1, 3)
    p0, p1, p2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    t0, t1, t2 = uv[idx[:, 0]], uv[idx[:, 1]], uv[idx[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    du = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r                # dP/du
    dv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r                # dP/dv
    T, B = np.zeros_like(pos), np.zeros_like(pos)
    for k in range(3):
        np.add.at(T, idx[:, k], du)
        np.add.at(B, idx[:, k], dv)
    n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    T = T - n * (T * n).sum(axis=1, keepdims=True)
    ln = np.linalg.norm(T, axis=1, keepdims=True)
    # no usable direction (no area in the picture): any one along the surface
    alt = np.cross(n, np.where(np.abs(n[:, 1:2]) < 0.9, [[0, 1, 0]], [[1, 0, 0]]))
    T = np.where(ln > 1e-12, T / np.maximum(ln, 1e-12), alt / np.maximum(np.linalg.norm(alt, axis=1, keepdims=True), 1e-12))
    w = np.where((np.cross(n, T) * -B).sum(axis=1) < 0, -1.0, 1.0)
    return np.column_stack([T, w])


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
            if p.get("uv0") is not None and "normalTexture" in materials[p["material"]]:
                # the normal map's frame, stored so every viewer reads it the same way
                tan = _tangents(p["positions"], p["normals"], p["uv0"], p["indices"])
                attrs["TANGENT"] = add_acc(tan.astype(np.float32), 5126, "VEC4", 34962)
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


def _copy_material(m, src_images, dst_images, part=None):
    """A material of one scene for another: its pictures put in dst_images (shared, not repeated), tagged with part."""
    import copy
    m = copy.deepcopy(m)

    def pic(i):
        return _image_index(dst_images, *src_images[i])
    pbr = m.get("pbrMetallicRoughness") or {}
    for holder, key in ((pbr, "baseColorTexture"), (pbr, "metallicRoughnessTexture"), (m, "normalTexture"),
                        (m, "occlusionTexture"), (m, "emissiveTexture")):
        if key in holder:
            holder[key]["index"] = pic(holder[key]["index"])
    ex = m.get("extras") or {}
    if "relief" in ex:
        ex["relief"]["index"] = pic(ex["relief"]["index"])
    if "pattern" in ex:
        ex["pattern"] = pic(ex["pattern"])
    for key in ("map", "normal", "orh"):
        if key in (ex.get("mix") or {}):
            ex["mix"][key] = pic(ex["mix"][key])
    if part:
        m["extras"] = dict(ex, part=part)
    return m


def _assemble(base, parts):
    """
    One scene for write_glb_scene, (meshes, materials, images): base's and each
    part's, parts = [(name, (meshes, materials, images))]. A part's materials
    carry its name (extras "part"), so a view can tell its objects from the
    rest and put new ones in their place. A part's mesh with "into" ("front" or
    "back") joins the base's mesh of its name (the road object), its primitives
    first or last, instead of being one of its own.
    """
    meshes = [dict(m, primitives=list(m["primitives"])) for m in base[0]]
    materials, images = list(base[1]), list(base[2])
    for name, (ms, mats, imgs) in parts:
        mat_of = {}

        def mat(i):
            if i not in mat_of:
                materials.append(_copy_material(mats[i], imgs, images, name))
                mat_of[i] = len(materials) - 1
            return mat_of[i]
        for m in ms:
            prims = [dict(p, material=mat(p["material"])) for p in m["primitives"]]
            into = m.get("into")
            dst = next((d for d in meshes if d["name"] == m["name"]), None) if into else None
            if dst is not None:
                dst["primitives"] = prims + dst["primitives"] if into == "front" else dst["primitives"] + prims
            else:
                meshes.append({k: v for k, v in m.items() if k != "into"} | {"primitives": prims})
    return meshes, materials, images


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
                          optimise=False, blocks=None, scatter=None, inner=None, surface="scan", islands=None,
                          decals=None):
    """
    The road with repeating material tiles laid along each street, a large
    variation layer as vertex colours, and dashes as their own strips.
    surface: each tile material also gets a bump and a roughness map
    (app/surface.py): "scan" from a bundled scan where there is one for its kind,
    "grain" from its own grain, False none.
    inner: the inner streets as exact shapes, {"base_mask_path": the mask
    without them, "shapes": app/streets.py's shapes}: the other roads are
    built from the mask without them, and they are laid as they are.
    islands: the island material slots (server.py's _islands_for): the islands
    picked for a slot are laid with its material, block and small paved island.
    decals: the decals laid on the streets (app/decals.py), {"placements",
    "images", "mode"}: "separate" makes them an object of their own (Decals),
    "painted" part of the road object, just above its surface.
    """
    from scipy.ndimage import map_coordinates
    gray = np.array(Image.open(mask_path).convert("L"))
    exact = inner["shapes"] if inner and inner.get("shapes") else None
    mask1, cov1 = prepare_mask(np.array(Image.open(inner["base_mask_path"]).convert("L")) if exact else gray, 1)
    mesh, det, rep = QM.build(mask1, output_scale, metres_per_pixel, straightness, spacing_m,
                              coverage=cov1, kerb_band=True, dashes=dash_cfg or {},
                              sidewalk=sidewalk if sidewalk and tileset["tiles"].get("sidewalk") else None,
                              bridges=bridges or None, optimise=optimise, exact=exact)
    _, coverage = prepare_mask(gray, output_scale)
    H, W = coverage.shape
    mpp_out = metres_per_pixel / output_scale
    tile_m = float(tileset["settings"]["tile_m"])
    tiles = tileset["tiles"]

    prog.stage("road", "road surfaces, materials and colours")
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
    pl = p32.tolist()

    def up(a, b, c):
        # the y part of cross(b - a, c - a), in plain arithmetic: the same operations
        # numpy does, without its overhead on three numbers (called for every triangle)
        A, B, C = pl[a], pl[b], pl[c]
        return (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2])

    painted = markings == "painted"

    def road_prims(images, materials):
        """
        The streets' and junctions' surfaces: their quads grouped into
        primitives by tile variant, each with its material. Painted markings:
        one texture per kind of street (its line width, and its lane lines: how
        many and where across), each holding exactly one dash cycle along the
        street with all its lanes' dashes, on the road's own grain, laid on the
        street's dashed run. Solid lines (a highway's edges) stay strips, so they
        run exactly as far as the texture's lines and the islands do. Returns the
        primitives and what the dashes need to know: which streets are painted.
        """
        mark_of = {}                 # street -> its marked texture's class
        marked = {}                  # class -> index into the marked textures
        mark_tex = []                # (class info, picture)
        mark_own = {}                # class -> the marked texture's own maps (library street material)
        E = None
        mark_size = None
        if painted:
            # a street the markings area cuts keeps its dashes as strips (only those inside it)
            dashed = {sid: st for sid, st in mesh.streets.items()
                      if st.get("dash") and any(k == "dash" for _, k in st["dash"]["lines"])
                      and not st["dash"].get("clipped")}
            if dashed:
                cycle = float((dash_cfg or {}).get("cycle_m", 9.0))
                share = float((dash_cfg or {}).get("dash_share", 0.6))
                E = math.ceil(2 * max(st["half_width_m"] for st in dashed.values()) * 1.15 / tile_m) * tile_m
                mark_size = (cycle, E)                         # what a marked texture covers, along and across
                # at most three line widths, each a group of streets with similar
                # widths: many short pieces would otherwise each want a texture
                ws = np.sort([st["dash"]["width_m"] * 100 for st in dashed.values()])
                groups_w = np.array_split(ws, min(3, len(ws)))
                centres = sorted({max(5, int(round(float(np.median(g)) / 5)) * 5) for g in groups_w if len(g)})
                # then streets of the same line width and number of lines, whose widths are
                # close enough that their lines lie within a few centimetres of each other's
                by = {}
                for sid, st in dashed.items():
                    d = st["dash"]
                    wc = min(centres, key=lambda c: abs(c - d["width_m"] * 100))
                    by.setdefault((wc, d["kind"], sum(1 for _, k in d["lines"] if k == "dash"),
                                   tuple(d["sides"]) if d.get("sides") else None), []).append((d["street_m"], sid))
                tol = 0.6
                while True:
                    classes = []
                    for (wc, kind, nd, sides), items in sorted(by.items(), key=lambda kv: str(kv[0])):
                        items = sorted(items)
                        run = []
                        for w, sid in items + [(math.inf, None)]:
                            if run and (sid is None or (nd > 1 and w - run[0][0] > tol)):
                                classes.append((wc, float(np.median([r[0] for r in run])), [r[1] for r in run], sides))
                                run = []
                            if sid is not None:
                                run.append((w, sid))
                    # many kinds of street: one more try with streets up to 1.2 m apart sharing
                    # (their lines then lie within about 30 cm of where they belong), never wider
                    if len(classes) <= 8 or tol >= 1.2:
                        break
                    tol = 1.2
                for ci, (wc, w_m, sids, sides) in enumerate(classes):
                    xs = [x for x, k in LN.layout(w_m, sides)["lines"] if k == "dash"]
                    img = marked_texture(tiles["open"][0]["path"], tile_m, cycle, share, E, wc / 100.0, paint_rgb, xs)
                    marked[ci] = len(mark_tex)
                    mark_tex.append(({"width_cm": wc, "street_m": round(w_m, 2), "lines": len(xs), "streets": len(sids)},
                                     img))
                    mark_own[ci] = marked_maps(tiles["open"][0], tile_m, cycle, E)
                    for sid in sids:
                        mark_of[sid] = ci

        def mark_class(owner):
            return mark_of.get(owner) if painted and owner is not None and 0 < owner else None

        def canon_along(vi, owner):
            """Metres along the street the canonical way (app/lanes.py), and its across sign."""
            st = mesh.streets[owner]["dash"]
            along, across = mesh.st[vi]
            return (along, across) if st["canon"] else (st["length_m"] - along, -across)

        def in_dashed_run(q, owner):
            """Whether a quad lies where the street's dashes run, so the texture repeats line up."""
            st = mesh.streets[owner]["dash"]
            us = []
            for vi in q:
                if mesh.st[vi] is None:
                    return False
                us.append((canon_along(vi, owner)[0] - st["setback_m"]) / st["step_m"])
            return min(us) >= -(1 - st["share"]) + 1e-6 and max(us) <= st["n"] + 1e-6

        def marked_uv(vi, owner):
            st = mesh.streets[owner]["dash"]
            along, across = canon_along(vi, owner)
            # across: a vertex at -across metres from the centreline lies at x = -across in
            # the lanes' coordinate (app/lanes.py), which the texture's rows follow
            return (along - st["setback_m"]) / st["step_m"], 0.5 - across / E

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

        prims = []
        for (part, vk), quads in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            if part == "marked":
                info_m = mark_tex[marked[vk]][0]
                tile_material(f"Road_marked_{info_m['lines']}x{info_m['width_cm']}cm_{vk + 1}", mark_tex[marked[vk]][1],
                              images, materials,
                              mark_size, "asphalt", 0.9, surface, own=mark_own.get(vk))
            else:
                tile_material(f"Road_{'street' if part == 'open' else part}_{vk + 1}",
                              Image.open(tiles[part][vk]["path"]), images, materials,
                              (tile_m, tile_m), "asphalt", 0.9, surface, own=LIB.own_maps(tiles[part][vk]))
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
            prims.append({"positions": np.array(pos), "normals": np.tile([0, 1, 0], (len(pos), 1)),
                          "uv0": np.array(uv), "colors": np.array(col),
                          "indices": np.array(idx), "material": len(materials) - 1})
        return prims, {"mark_class": mark_class, "marked": marked, "mark_tex": mark_tex,
                       "road_quads": sum(len(p["indices"]) for p in prims) // 2,   # streets and junctions only
                       "marked_quads": sum(len(v) for k, v in groups.items() if k[0] == "marked")}

    images, materials = [], []
    # the streets' surfaces: here, unless painted markings decide them (lane_parts)
    primitives, road_rep = ([], None) if painted else road_prims(images, materials)

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
        # the kerb's tone along the kerb, as the streets have it in their kerb band:
        # full at the kerb, fading out over 1.5 m, so a fill beside a kerb
        # does not show as a lighter or darker line along it
        ks = getattr(mesh, "kerb_surface", None)
        if ks is not None and not ks.is_empty and role_f.get("kerb", 1.0) != 1.0:
            dk = QM.EdgeIndex(ks.boundary).distance(fv[:, :2]) * mpp_out
            ff = ff * (role_f["kerb"] + (1.0 - role_f["kerb"]) * np.clip(dk / 1.5, 0.0, 1.0))
        tile_material("Road_fill", Image.open(tiles["open"][0]["path"]), images, materials,
                      (tile_m, tile_m), "asphalt", 0.9, surface, own=LIB.own_maps(tiles["open"][0]))
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
            tile_material("Road_interchange", Image.open(tiles["open"][0]["path"]), images, materials,
                          (tile_m, tile_m), "asphalt", 0.9, surface, own=LIB.own_maps(tiles["open"][0]))
            primitives.append({"positions": P, "normals": np.tile([0, 1, 0], (len(P), 1)),
                               "uv0": np.column_stack([P[:, 0] / tile_m, P[:, 2] / tile_m]),
                               "colors": np.column_stack([f, f, f, np.ones_like(f)]),
                               "indices": t2, "material": len(materials) - 1})
            ic_tris = int(len(t2))

    meshes = [{"name": "Road", "primitives": primitives}]
    # one paving layout per paved area between roads, shared by its sidewalks,
    # islands and block, so they meet with no jump in pattern or tone
    frames = None
    if (sidewalk or blocks) and tiles.get("sidewalk"):
        regions = QM.block_polygons(mesh, (coverage.shape[0] // output_scale, coverage.shape[1] // output_scale),
                                    output_scale, mpp_out * output_scale, min_area_m2=0.5)
        frames = PavedFrames(regions, mpp_out, W, H, tile_m, len(tiles["sidewalk"]), seed, fac)
    deck_info = _deck_edges(mesh, plans, world, tiles, tile_m, bcfg["deck_m"], images, materials, meshes, surface) \
        if plans else None
    base = (meshes, materials, images)

    # The parts of the model that a change on its own can build again (rebuild): each
    # makes its own meshes, materials and pictures, and the model is the base and
    # them put together (_assemble), its materials tagged with the part's name

    def paving_part(islands_now):
        """Sidewalks and kerbs, blocks and islands: the island material slots decide their paving."""
        imgs, mats, ms = [], [], []
        if frames is not None:
            frames.set_islands(islands_now, output_scale)
        prog.stage("sidewalk_meshes", "sidewalk paving")
        sw_info = _sidewalk_meshes(mesh, world, vfac_raw, tiles, tile_m, seed, imgs, mats, ms, surface, frames)
        blocks_info = None
        if blocks:
            mask_shape = (coverage.shape[0] // output_scale, coverage.shape[1] // output_scale)
            prog.stage("blocks", "blocks and islands")
            blocks_info = _block_meshes(mesh, mask_shape, output_scale, mpp_out, W, H,
                                        fac, tiles, tile_m, imgs, mats, ms,
                                        height_m=float(blocks.get("height_m", 0.10)),
                                        with_sidewalks=bool(mesh.sw_quads),
                                        cell_m=16.0 if optimise else 8.0, surface=surface, frames=frames)
        return (ms, mats, imgs), {"sidewalk": sw_info, "blocks": blocks_info}

    def objects_part(scatter_now):
        """Your objects and plants, and the street lamps (they keep clear of the objects)."""
        imgs, mats, ms = [], [], []
        scatter_info = None
        if scatter_now:
            stand = (float(blocks.get("height_m", 0.10)) if blocks else
                     float(sidewalk.get("height_m", 0.10)) if sidewalk else 0.0)
            prog.stage("objects", "objects")
            scatter_info = _object_meshes(mesh, scatter_now, metres_per_pixel, output_scale, mpp_out, W, H,
                                          stand, imgs, mats, ms)
        feet = scatter_info.pop("_feet", None) if scatter_info else None
        prog.stage("lamps", "street lamps")
        lamps = _lamps(mesh, world, feet, metres_per_pixel, output_scale, mpp_out, W, H)
        return (ms, mats, imgs), {"scatter": scatter_info, "lamps": lamps}

    def decals_part(decals_now):
        """The decals: an object of their own (Decals), or painted: part of the road object, just above it."""
        imgs, mats, ms = [], [], []
        decal_info = None
        if decals_now and decals_now.get("placements"):
            on_road = decals_now.get("mode") == "painted"
            dprims = DC.mesh_prims(decals_now["placements"], decals_now["images"], mats, imgs, W, H, mpp_out,
                                   output_scale, DC.LIFT_PAINTED if on_road else DC.LIFT_SEPARATE,
                                   lambda data, mime: _image_index(imgs, data, mime), decals_now.get("names"))
            if dprims:
                ms.append({"name": "Road", "primitives": dprims, "into": "back"} if on_road
                          else {"name": "Decals", "primitives": dprims})
                decal_info = {"count": len(decals_now["placements"]), "mode": "painted" if on_road else "separate",
                              "pictures": len(dprims)}
        return (ms, mats, imgs), {"decals": decal_info}

    def lanes_part(_=None):
        """
        The parts of the model the streets' lanes decide: with painted markings
        the streets' surfaces (first in the road object), the highways' raised
        islands (Median islands) and the lines laid as strips (Markings).
        """
        prog.stage("lanes", "the streets' lanes: " + ("painted surfaces, " if painted else "") + "islands and lines")
        imgs, mats, ms = [], [], []
        rr = road_rep
        if painted:
            prims, rr = road_prims(imgs, mats)
            ms.append({"name": "Road", "primitives": prims, "into": "front"})
        # dashes: their own mesh, a centimetre above the road so they never flicker into it
        dpos, dcol, didx = [], [], []
        painted_dashes = 0
        for di, strip in enumerate(mesh.dashes):
            base_i = len(dpos)
            owner = mesh.dash_owner[di] if di < len(mesh.dash_owner) else None
            solid = di < len(mesh.dash_solid) and mesh.dash_solid[di]
            if painted and not solid and owner is not None and rr["mark_class"](owner) in rr["marked"]:
                painted_dashes += 1
                continue                                        # painted into the road texture
            pts = [(x, y) for (lx, ly, rx, ry) in strip for (x, y) in ((lx, ly), (rx, ry))]
            dl = BR.dash_lift(mesh, plans, output_scale, owner, pts) if plans else np.zeros(len(pts))
            P_ = np.asarray(pts, float)
            # paint wears too, but less than asphalt
            c_ = 1.0 + (map_coordinates(fac, [P_[:, 1], P_[:, 0]], order=1, mode="nearest") - 1.0) * 0.5
            for k_, (x, y) in enumerate(pts):
                dpos.append((x * mpp_out - W * mpp_out / 2, 0.01 + float(dl[k_]), y * mpp_out - H * mpp_out / 2))
                c = float(c_[k_])
                dcol.append((c, c, c, 1.0))
            # this dash's points at the precision the file stores (only its own: all the
            # dashes so far, again for every dash, grew with the square of their number)
            Pl = np.array(dpos[base_i:], np.float32).astype(np.float64).tolist()
            for k in range(len(strip) - 1):
                a, b, c, d = base_i + 2 * k, base_i + 2 * k + 1, base_i + 2 * k + 3, base_i + 2 * k + 2
                for t in ((a, b, c), (a, c, d)):
                    # the y part of cross(B - A, C - A), as numpy works it out
                    A_, B_, C_ = Pl[t[0] - base_i], Pl[t[1] - base_i], Pl[t[2] - base_i]
                    n_y = (B_[2] - A_[2]) * (C_[0] - A_[0]) - (B_[0] - A_[0]) * (C_[2] - A_[2])
                    if abs(n_y) < 2e-6:
                        continue                                # no area: a repeated point
                    didx.append(t if n_y > 0 else (t[0], t[2], t[1]))
        island_info = _lane_island_mesh(mesh, plans, output_scale, mpp_out, W, H, tiles, tile_m, imgs, mats, ms,
                                        surface)
        if didx:
            mats.append({"name": "RoadMarkings", "doubleSided": True,
                         "pbrMetallicRoughness": {"baseColorFactor": [c / 255 for c in paint_rgb] + [1.0],
                                                  "metallicFactor": 0.0, "roughnessFactor": SF.PAINT_ROUGH}})
            ms.append({"name": "Markings", "primitives": [{
                "positions": np.array(dpos), "normals": np.tile([0, 1, 0], (len(dpos), 1)),
                "colors": np.array(dcol), "indices": np.array(didx), "material": len(mats) - 1}]})
        return (ms, mats, imgs), {
            "quads": int(rr["road_quads"]), "dashes": len(mesh.dashes),
            "markings": ("painted" if painted and rr["marked"] else "strips"),
            "painted_dashes": painted_dashes if painted else 0,
            "marked_textures": [dict(info, px=list(im.size)) for info, im in rr["mark_tex"]],
            "lane_islands": island_info,
            "lanes_cut": sum(1 for st in mesh.streets.values() if (st.get("dash") or {}).get("cut")),
            "marked_quads": rr["marked_quads"],
            "strip_dashes": len(mesh.dashes) - (painted_dashes if painted else 0),
        }

    parts = (("paving", paving_part), ("objects", objects_part), ("decals", decals_part), ("lanes", lanes_part))
    inputs = {"paving": islands, "objects": scatter, "decals": decals, "lanes": None}
    built = {name: fn(inputs[name]) for name, fn in parts}

    out = {"mesh": "tiled", "streets": rep["streets"],
           "junctions": rep["junctions_patched"],
           "tile_m": tile_m, "mm_per_px": tileset.get("mm_per_px"),
           "variation": variation, "size_m": [round(W * mpp_out, 1), round(H * mpp_out, 1)],
           "interchange_triangles": ic_tris,
           "fill_triangles": fill_tris, "groups": rep.get("clusters"), "bare_filled": rep.get("bare_filled"),
           "mesh_detail": "optimised" if optimise else "full",
           "rows_full": rep.get("rows_full"), "rows_kept": rep.get("rows_kept"),
           "bridges": [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in pl.items() if k in ("index", "ok", "s_in", "s_out", "r_before",
                                                          "r_after", "height", "steepest_pct",
                                                          "warnings", "streets")}
                       for pl in plans] if plans else None,
           "deck": deck_info}

    def assemble(names=None):
        """The whole model (names None), or those parts alone (for a view that has the rest)."""
        chosen = [(n, built[n][0]) for n, _ in parts if names is None or n in names]
        return _assemble(base if names is None else ([], [], []), chosen)

    def report(scene, nbytes):
        info = dict(out)
        for name, _ in parts:
            info.update(built[name][1])
        return dict(info, materials=len(scene[1]), file_bytes=int(nbytes),
                    surface=sorted({m["extras"]["surface"] for m in scene[1] if (m.get("extras") or {}).get("surface")}))

    prog.stage("writing", "writing the 3D file")
    scene = assemble()
    size = write_glb_scene(out_path, *scene)

    def rebuild(changes, glb_path, part_path, send=None):
        """
        The model with some of its parts built again from new inputs, changes:
        {part: its input} ("paving": the island slots, "objects": the objects,
        "decals": the decals, "lanes": {"picks": the Lanes tool's choices,
        "area": the markings area}), everything else
        as built: the whole model to glb_path, and to part_path the parts a view
        needs alone (send: their names; None: those built again; nothing written
        when they come to nothing). Returns (report, those parts' names), or None
        when a street gains its first lines or loses all of them with painted
        markings: its road was built for the lines it had, so build it all.
        """
        lanes_changed = None
        if "lanes" in changes:
            ch = changes["lanes"]
            lanes_changed, flipped = (QM.relane(mesh, ch["picks"], ch.get("area")) if isinstance(ch, dict)
                                      else QM.relane(mesh, ch))
            if flipped and painted:
                return None
        for name, fn in parts:
            if name in changes:
                built[name] = fn(changes[name])
        names = [n for n, _ in parts if (n in changes if send is None else n in send)]
        prog.stage("writing", "writing the 3D file")
        whole = assemble()
        nbytes = write_glb_scene(glb_path, *whole)
        piece = assemble(names)
        part_bytes = write_glb_scene(part_path, *piece) if piece[0] else 0
        info = dict(report(whole, nbytes), part_bytes=int(part_bytes))
        if lanes_changed is not None:
            info["lanes_changed"] = sorted(int(c) for c in lanes_changed if c is not None)
        return info, names

    return dict(report(scene, size), layouts={int(k): v for k, v in layouts.items()}, _mesh=mesh, _world=world,
                _fac=fac, _rebuild=rebuild)


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
    prims = P3.load_glb(glb_path, meshes={"Road"})
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



class PavedFrames:
    """
    One paving layout per paved area between roads: its sidewalks, block and
    islands share one pattern grid, turned to the area's main street direction,
    one tile variant and one gentle weathering, so the paving runs on across
    every edge between them with no jump in pattern or tone.

    regions: the areas (shapely polygons in texture pixels, QM.block_polygons);
    fac: the variation layer (texture pixels).
    """

    def __init__(self, regions, mpp_out, W, H, tile_m, n_variants, seed, fac):
        import shapely
        from scipy import ndimage as _ndi
        self.regions, self.mpp, self.W, self.H, self.tile_m = list(regions), mpp_out, W, H, tile_m
        self.seed, self.looks, self.ground = seed, {}, None
        self.near, self.slot_map, self.iscale = None, None, 1
        self.tree = shapely.STRtree(self.regions) if self.regions else None
        self.theta, self.off, self.variant = [], [], []
        for k, g in enumerate(self.regions):
            # the main direction of its kerbs (not the image border): edge directions
            # averaged four-fold, so streets at right angles agree
            c = np.asarray(g.exterior.coords)
            d = np.diff(c, axis=0)
            mid = (c[:-1] + c[1:]) / 2
            inner = ~((mid[:, 0] < 1) | (mid[:, 1] < 1) | (mid[:, 0] > W - 1) | (mid[:, 1] > H - 1))
            ln = np.hypot(d[:, 0], d[:, 1]) * inner
            a = np.arctan2(d[:, 1], d[:, 0])
            th = np.arctan2((ln * np.sin(4 * a)).sum(), (ln * np.cos(4 * a)).sum()) / 4 if ln.sum() > 0 else 0.0
            r = np.random.default_rng([seed, k + 500000])
            self.theta.append(float(th))
            self.off.append(r.uniform(0, tile_m, 2))
            self.variant.append(int(r.integers(0, 10 ** 6)) % max(1, n_variants))
        # weathering: the variation layer blurred over about 25 m, at a third of its
        # strength, the same function everywhere so it is continuous across edges
        self.fb = _ndi.gaussian_filter(fac, sigma=max(1.0, 25.0 / mpp_out))

    def to_px(self, x, z):
        return np.asarray(x) / self.mpp + self.W / 2, np.asarray(z) / self.mpp + self.H / 2

    def region_of(self, x, z):
        """The area each world point (x, z) lies in (the nearest one if none)."""
        import shapely
        px, pz = self.to_px(np.atleast_1d(x), np.atleast_1d(z))
        pts = shapely.points(px, pz)
        out = np.full(len(pts), -1, int)
        if self.tree is None:
            return np.zeros(len(pts), int)
        hit = self.tree.query(pts, predicate="intersects")
        out[hit[0]] = hit[1]
        miss = np.where(out < 0)[0]
        if len(miss):
            near = self.tree.query_nearest(pts[miss], return_distance=False)
            out[miss[near[0]]] = near[1]
        return np.maximum(out, 0)

    def uv(self, k, x, z):
        th, off = (self.theta[k], self.off[k]) if self.regions else (0.0, (0.0, 0.0))
        c, s = np.cos(th), np.sin(th)
        return ((c * x + s * z + off[0]) / self.tile_m, (-s * x + c * z + off[1]) / self.tile_m)

    def variant_of(self, k):
        return self.variant[k] if self.regions else 0

    def set_islands(self, islands, scale):
        """
        The island material slots (server.py's _islands_for): the numbered islands
        (mask pixels, scale times coarser than the texture), each pixel of road
        given its nearest island, so a point on the kerb line still finds one.
        None: no slots (every island laid alike).
        """
        if not islands:
            self.near, self.slot_map, self.iscale, self.looks = None, None, 1, {}
            return
        lab = islands["labels"]
        if lab.any() and (lab == 0).any():
            iy, ix = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
            lab = lab[iy, ix]
        self.near, self.iscale = lab, scale
        self.slot_map = np.full(int(lab.max()) + 1, -1, np.int64)
        for k, s in islands["slot_of"].items():
            if 0 < int(k) < len(self.slot_map):
                self.slot_map[int(k)] = int(s)
        self.looks = islands["looks"]

    def slot_at(self, x, z):
        """The material slot of the island under each world point (x, z); -1: none (the default look)."""
        x = np.atleast_1d(np.asarray(x, np.float64))
        if self.slot_map is None:
            return np.full(len(x), -1, np.int64)
        px, pz = self.to_px(x, np.atleast_1d(z))
        H, W = self.near.shape
        xi = np.clip((px / self.iscale).astype(np.int64), 0, W - 1)
        yi = np.clip((pz / self.iscale).astype(np.int64), 0, H - 1)
        return self.slot_map[self.near[yi, xi]]

    def ground_tone(self, x, z):
        """Broad light and dark over ground (app/islands.py), on top of tone()."""
        if self.ground is None:
            self.ground = ISL.ground_field(self.W * self.mpp, self.H * self.mpp, self.seed)
        f = ISL.field_at(self.ground, np.asarray(x) + self.W * self.mpp / 2, np.asarray(z) + self.H * self.mpp / 2)
        return 1.0 + ISL.GROUND_AMP * f

    def tone(self, x, z):
        from scipy.ndimage import map_coordinates
        px, pz = self.to_px(np.atleast_1d(x), np.atleast_1d(z))
        f = map_coordinates(self.fb, [np.clip(pz, 0, self.H - 1), np.clip(px, 0, self.W - 1)], order=1, mode="nearest")
        return 1.0 + (f - 1.0) * 0.33


GROUND_DRIFT = 0.07     # ground: how far colour and shine drift over a metre or few (0.07: about 7%)


def tile_relief(tile, tile_m, kind):
    """A tile's height map (grey picture) and relief (m), as its material in the model has them."""
    _, _, hgt, _, _, relief = SF.maps(Image.open(tile["path"]), (tile_m, tile_m), kind, own=LIB.own_maps(tile))
    return Image.open(io.BytesIO(hgt)).convert("L"), float(relief)


def _slot_mix(frames, s, vk):
    """An island slot's mix (its second ground material's tile for this variant, amount, patch size), or None."""
    look = frames.looks.get(s) if frames is not None and s >= 0 else None
    mix = look.get("mix") if look else None
    if not mix or not mix.get("tiles"):
        return None
    return {"tile": mix["tiles"][vk % len(mix["tiles"])], "amount": mix["amount"], "size_m": mix["size_m"],
            "seed": mix.get("seed", s), "name": mix.get("name", "")}


def _sidewalk_meshes(mesh, world, vfac, tiles, tile_m, seed, images, materials, meshes, surface="scan", frames=None):
    """
    The sidewalk top (paving, with a kerb stone along its edge) and the vertical
    kerb face down to the road, as two meshes of quads.
    """
    if (not mesh.sw_quads and not getattr(mesh, "sw_tris", None)) or not tiles.get("sidewalk"):
        return None
    Wh = world.copy()
    Wh[:, 1] = world[:, 1] + np.array(mesh.h)                # raised sidewalk, on top of any bridge lift
    p32 = Wh.astype(np.float32).astype(np.float64)
    pl = p32.tolist()

    def up(a, b, c):
        # the y part of cross(b - a, c - a), as numpy works it out (see the road's)
        A, B, C = pl[a], pl[b], pl[c]
        return (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2])

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

    def add_tile_material(name, tile, kind="paving", mix=None):
        return tile_material(name, Image.open(tile["path"]), images, materials, (tile_m, tile_m), kind, 0.85, surface,
                             own=LIB.own_maps(tile), mix=mix, drift=GROUND_DRIFT if kind == "ground" else 0.0)

    # top surface: the paving and the small paved islands laid by their area's
    # layout (frames: one grid, variant and weathering per paved area, shared with
    # the block), the kerb stone along each kerb on its own material
    if frames is None:
        frames = PavedFrames([], 1.0, 0, 0, tile_m, 1, seed, np.ones((1, 1)))
    pieces = [(list(q), mesh.sw_part[qi]) for qi, q in enumerate(mesh.sw_quads)]
    pieces += [(list(t), "island") for t in (getattr(mesh, "sw_tris", None) or [])]
    paved = [i for i, (_, part) in enumerate(pieces) if part != "kerbstone"]
    region = np.zeros(len(pieces), int)
    slot = np.full(len(pieces), -1, np.int64)
    if paved:
        cen = np.array([Wh[pieces[i][0]].mean(axis=0) for i in paved])
        region[paved] = frames.region_of(cen[:, 0], cen[:, 2])
        slot[paved] = frames.slot_at(cen[:, 0], cen[:, 2])
    groups = {}
    for i, (vs, part) in enumerate(pieces):
        if part == "kerbstone":
            groups.setdefault(("kerbstone", -1, 0), []).append((vs, mesh.sw_owner[i], None))
            continue
        k = int(region[i])
        # a small island paved over completely: its island's material (a slot's, or the
        # squares'), when it has one; else, as before, the sidewalk paving
        s = int(slot[i]) if part == "island" else -1
        own = frames.looks[s]["tiles"] if s >= 0 else (tiles.get("block") if part == "island" else None)
        if own:
            groups.setdefault(("island", s, frames.variant_of(k) % len(own)), []).append((vs, None, k))
        else:
            groups.setdefault(("paving", -1, frames.variant_of(k) % len(tiles["sidewalk"])), []).append((vs, None, k))

    prims = []
    tone_all = None
    for (part, s, vk), polys in sorted(groups.items()):
        ground = False
        if part == "paving":
            mat = add_tile_material(f"Sidewalk_paving_{vk + 1}", tiles["sidewalk"][vk],
                                    _sidewalk_kind(tiles["sidewalk"][vk]))
        elif part == "island":
            tile = (frames.looks[s]["tiles"] if s >= 0 else tiles["block"])[vk]
            ground = _sidewalk_kind(tile) == "ground"
            mat = add_tile_material(f"Island_{s + 1}_{vk + 1}" if s >= 0 else f"Block_paving_{vk + 1}", tile,
                                    _sidewalk_kind(tile), mix=_slot_mix(frames, s, vk))
        else:
            mat = add_tile_material("Kerb_stone", (tiles.get("kerbstone") or tiles["sidewalk"])[0], "concrete")
        remap, pos, uvs, col, idx = {}, [], [], [], []
        if tone_all is None and frames.regions:
            tone_all = frames.tone(Wh[:, 0], Wh[:, 2])          # every vertex at once, not one by one

        def vid(vi, owner, k):
            key = (vi, owner, k)
            if key not in remap:
                remap[key] = len(pos)
                pos.append(Wh[vi])
                if k is None:                                  # kerb stone: along its kerb
                    uvs.append(uv(vi, owner))
                    c = 1.0 + (float(vfac[vi]) - 1.0) * 0.5
                else:                                          # paving: its area's layout
                    uvs.append(frames.uv(k, Wh[vi, 0], Wh[vi, 2]))
                    c = float(tone_all[vi]) if frames.regions else 1.0 + (float(vfac[vi]) - 1.0) * 0.5
                    if ground:
                        c *= float(frames.ground_tone(Wh[vi, 0], Wh[vi, 2])[0])
                col.append((c, c, c, 1.0))
            return remap[key]

        for vs, owner, k in polys:
            if len(vs) == 4:
                a, b, c, d = vs
                if up(a, b, c) + up(a, c, d) < 0:
                    a, b, c, d = a, d, c, b
                tris = _split_quad(up, a, b, c, d, 1e-5)
            else:
                a, b, c = vs
                if up(a, b, c) < 0:
                    a, b, c = a, c, b
                tris = [(a, b, c)] if up(a, b, c) > 1e-6 else []
            for t in tris:
                idx.append([vid(i, owner, k) for i in t])
        if idx:
            prims.append({"positions": np.array(pos), "normals": np.tile([0, 1, 0], (len(pos), 1)),
                          "uv0": np.array(uvs), "colors": np.array(col), "indices": np.array(idx),
                          "material": mat})
    if prims:
        meshes.append({"name": "Sidewalk", "primitives": prims})

    # the vertical kerb face, facing the road
    mat = add_tile_material("Kerb_face", (tiles.get("kerbstone") or tiles["sidewalk"])[0], "concrete")
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



def _lane_island_mesh(mesh, plans, scale, mpp_out, W, H, tiles, tile_m, images, materials, meshes, surface="scan"):
    """
    The highways' raised islands (app/lanes.py, mesh.lane_islands): along each,
    a top a kerb's height above the road between its two edges, a kerb face down
    each side and a face at each end, in the kerb stone's concrete. A mesh of
    their own (Median islands). Returns {"count", "length_m"} or None.
    """
    if not mesh.lane_islands:
        return None
    pos, nrm, uv, idx = [], [], [], []
    total_m = 0.0

    def at(p, h):
        return (p[0] * mpp_out - W * mpp_out / 2, h, p[1] * mpp_out - H * mpp_out / 2)

    def face(quad, n, uvs):
        # two triangles of a quad, wound to face n (the material is double-sided, the normal is not)
        base = len(pos)
        pos.extend(quad); nrm.extend([n] * 4); uv.extend(uvs)
        P = np.array(quad)
        for t in ((0, 1, 2), (0, 2, 3)):
            g = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
            if float(np.linalg.norm(g)) < 1e-10:
                continue
            idx.append(tuple(base + i for i in (t if float(g @ np.array(n)) >= 0 else (t[0], t[2], t[1]))))

    count = 0
    for k, isl in enumerate(mesh.lane_islands):
        owner = mesh.island_owner[k] if k < len(mesh.island_owner) else None
        A, B = (np.asarray(e, float) for e in isl["edges"])
        n = min(len(A), len(B))
        if n < 2:
            continue
        A, B = A[:n], B[:n]
        h = float(isl["height_m"])
        if plans:
            from app import bridges as BR
            lift = BR.dash_lift(mesh, plans, scale, owner, np.vstack([A, B]))
        else:
            lift = np.zeros(2 * n)
        la, lb = lift[:n], lift[n:]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff((A + B) / 2, axis=0).T))]) * mpp_out
        wid = float(np.median(np.hypot(*(A - B).T))) * mpp_out
        total_m += float(s[-1])
        count += 1
        for i in range(n - 1):
            u0, u1 = s[i] / tile_m, s[i + 1] / tile_m
            # top
            face([at(A[i], h + la[i]), at(B[i], h + lb[i]), at(B[i + 1], h + lb[i + 1]), at(A[i + 1], h + la[i + 1])],
                 (0.0, 1.0, 0.0), [(u0, 0.0), (u0, wid / tile_m), (u1, wid / tile_m), (u1, 0.0)])
            # a kerb face down each side, facing away from the other side
            for E, F, le in ((A, B, la), (B, A, lb)):
                d = E[i + 1] - E[i]
                out = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-9)
                if float(out @ (E[i] - F[i])) < 0:
                    out = -out
                face([at(E[i], le[i]), at(E[i + 1], le[i + 1]), at(E[i + 1], h + le[i + 1]), at(E[i], h + le[i])],
                     (float(out[0]), 0.0, float(out[1])), [(u0, 0.0), (u1, 0.0), (u1, h / tile_m), (u0, h / tile_m)])
        # its two ends
        for i, j in ((0, 1), (n - 1, n - 2)):
            d = (A[i] + B[i]) / 2 - (A[j] + B[j]) / 2
            d = d / max(float(np.hypot(*d)), 1e-9)
            face([at(A[i], la[i]), at(B[i], lb[i]), at(B[i], h + lb[i]), at(A[i], h + la[i])],
                 (float(d[0]), 0.0, float(d[1])), [(0.0, 0.0), (wid / tile_m, 0.0), (wid / tile_m, h / tile_m),
                                                   (0.0, h / tile_m)])
    if not idx:
        return None
    tile = (tiles.get("kerbstone") or tiles.get("sidewalk") or tiles["open"])[0]
    tile_material("Median_island", Image.open(tile["path"]), images, materials, (tile_m, tile_m), "concrete", 0.85,
                  surface, own=LIB.own_maps(tile))
    P = np.array(pos)
    meshes.append({"name": "Median islands", "primitives": [{
        "positions": P, "normals": np.array(nrm), "uv0": np.array(uv),
        "colors": np.ones((len(P), 4)), "indices": np.array(idx), "material": len(materials) - 1}]})
    return {"count": count, "length_m": round(total_m, 1)}


def _deck_edges(mesh, plans, world, tiles, tile_m, deck_m, images, materials, meshes, surface="scan"):
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
    kt = (tiles.get("kerbstone") or tiles.get("sidewalk") or tiles["open"])[0]
    tile_material("Bridge_deck", Image.open(kt["path"]), images, materials, (tile_m, tile_m), "concrete", 0.9, surface,
                  own=LIB.own_maps(kt))
    meshes.append({"name": "BridgeDeck", "primitives": [{
        "positions": np.array(pos), "normals": np.array(nrm), "uv0": np.array(uvs),
        "indices": np.array(idx), "material": len(materials) - 1}]})
    return {"faces": int(len(idx) // 2), "thickness_m": deck_m}



def marked_maps(tile, tile_m, cycle_m, across_m, max_px=2048):
    """
    For a tile with its own bump and roughness (a library material): those maps
    laid exactly as marked_texture lays its colour (the same repeats, stretch
    and size), without the paint, which app/surface.py adds from the colour.
    (normal, roughness, name, height, ao, relief_m) pictures, or None.
    """
    own = LIB.own_maps(tile)
    if own is None:
        return None
    n = np.asarray(own[0]).astype(np.float32) / 127.5 - 1.0
    r = np.asarray(own[1]).astype(np.float32) / 255.0
    k_along = max(1, int(round(cycle_m / tile_m)))
    k_across = max(1, int(round(across_m / tile_m)))
    h, w = n.shape[0] * k_across, n.shape[1] * k_along
    f = min(1.0, max_px / max(h, w))
    out_w, out_h = max(8, int(w * f)), max(8, int(h * f))
    nn = LIB._resize(np.tile(n, (k_across, k_along, 1)), out_w, out_h)
    rr = LIB._resize(np.tile(r, (k_across, k_along)), out_w, out_h)
    nn = LIB.squeeze_normals(nn, (k_along * tile_m / cycle_m, k_across * tile_m / across_m))
    grey = lambda a: Image.fromarray(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
    extra = [None, None]
    for i, pic in enumerate(own[3:5]):
        if pic is not None:
            a = np.asarray(pic).astype(np.float32) / 255.0
            extra[i] = grey(LIB._resize(np.tile(a, (k_across, k_along)), out_w, out_h))
    return (Image.fromarray(np.clip((nn * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)),
            grey(rr), own[2], extra[0], extra[1], own[5] if extra[0] is not None else 0.0)


_marked = {}                 # marked textures made, by what they were made from: the last few


def marked_texture(tile_path, tile_m, cycle_m, share, across_m, width_m, paint_rgb, lines=(0.0,), max_px=2048):
    """
    One dash cycle of road, with a dash painted along each lane line: lines,
    metres across from the centre (app/lanes.py's x; the centre line alone by
    default).

    Columns run along the street (one full cycle: dash then gap), rows run
    across it, centred on the centre line. The grain is the open-road tile,
    laid a whole number of times each way so the texture joins itself where it
    repeats. The dash cycle is rarely a whole number of tiles, so along the
    street the tiles are stretched by the small amount that makes them fit
    (9 m holds two 4 m tiles stretched by 12%, which cannot be seen in grain).
    """
    import os
    key = (str(tile_path), os.path.getmtime(tile_path), float(tile_m), float(cycle_m), float(share), float(across_m),
           float(width_m), tuple(float(v) for v in paint_rgb), tuple(round(float(x), 4) for x in lines), int(max_px))
    if key in _marked:
        return _marked[key]                                        # the same kind of street as before
    tile = np.array(Image.open(tile_path).convert("RGB")).astype(np.float32)
    tp = tile.shape[0]
    k_along = max(1, int(round(cycle_m / tile_m)))
    k_across = max(1, int(round(across_m / tile_m)))
    base = np.tile(tile, (k_across, k_along, 1))                   # rows across, columns along
    h, w = base.shape[:2]
    f = min(1.0, max_px / max(h, w))
    out_w, out_h = max(8, int(w * f)), max(8, int(h * f))
    img = np.array(Image.fromarray(base.clip(0, 255).astype(np.uint8)).resize((out_w, out_h), Image.LANCZOS)).astype(np.float32)

    # the dashes: along [0, share) of the cycle, across at each line, soft one-pixel edges
    along_px = out_w / cycle_m                                     # pixels per metre along
    across_px = out_h / across_m
    xs = np.arange(out_w) + 0.5
    ys = np.arange(out_h) + 0.5
    ax = np.clip(np.minimum(xs, share * out_w - xs) + 0.5, 0, 1)   # coverage along
    ay = np.zeros(out_h)
    for x in lines:
        yc = out_h / 2 + float(x) * across_px
        ay = np.maximum(ay, np.clip(width_m / 2 * across_px - np.abs(ys - yc) + 0.5, 0, 1))
    a = (ay[:, None] * ax[None, :])[..., None]
    grain = img.mean(axis=2, keepdims=True)
    paint = np.array(paint_rgb, np.float32)[None, None, :] * (0.92 + 0.08 * grain / max(float(grain.mean()), 1e-6))
    img = img * (1 - a) + paint * a
    out = _marked[key] = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    if len(_marked) > 32:
        _marked.pop(next(iter(_marked)))
    return out



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



SLIVER_M = 0.06               # block pieces thinner than this along the kerb are slivers, not blocks


def _block_meshes(mesh, shape_mask, scale, mpp_out, W, H, fac, tiles, tile_m, images, materials, meshes,
                  height_m=0.10, with_sidewalks=True, cell_m=8.0, surface="scan", frames=None):
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
        parts = _sidewalk_parts(mesh)
        if parts:
            import shapely
            from shapely.ops import unary_union
            # grown a hair, as _sidewalk_union does, so pieces with a hairline between them join
            grown = list(shapely.buffer(np.array(parts, object), 0.001, quad_segs=16))   # as geometry.buffer()
            tree = shapely.STRtree(grown)
            cut = []
            for n_g, g in enumerate(polys):
                prog.part(0.5 * n_g / len(polys))
                # the sidewalk near this block, joined and shrunk back
                x0, y0, x1, y1 = g.bounds
                near = tree.query(shapely.box(x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01))
                r = g.difference(unary_union([grown[i] for i in near]).buffer(-0.001)) if len(near) else g
                # hairline slivers left between the kerb line and the sidewalk's kerb (where its
                # rows cut a bend as straight lines) taken off: they stood in front of the kerb
                # with a kerb face of their own, two faces in one place. Square corners kept;
                # the road is filled up to the kerb there instead
                if len(near) and not r.is_empty:
                    e = SLIVER_M / 2 / mpp_out
                    r = r.buffer(-e, join_style=2).buffer(e, join_style=2).intersection(r)
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
    # Paved areas get a gentle, broad weathering instead (PavedFrames.tone), the
    # same as their sidewalks; and each block is laid with its area's layout,
    # so its paving runs on from its sidewalks with no edge
    if frames is None:
        frames = PavedFrames(QM.block_polygons(mesh, shape_mask, scale, mpp_out * scale), mpp_out, W, H, tile_m,
                             len(tiles["sidewalk"]), 7, fac)
    cen = P32[t2].mean(axis=1)
    reg = frames.region_of(cen[:, 0], cen[:, 2])
    prims = []
    # the squares' own material when one is chosen (cobblestone fans, say), else the
    # sidewalks'; an island in a material slot, that slot's
    default = tiles.get("block") or tiles["sidewalk"]
    slots = frames.slot_at(cen[:, 0], cen[:, 2])            # by each triangle: two islands may share an area
    keys = [(int(s), frames.variant_of(int(k)) % len(frames.looks[int(s)]["tiles"] if s >= 0 else default))
            for s, k in zip(slots, reg)]
    used_slots = {}
    for key in sorted(set(keys)):
        s, vk = key
        ptiles = frames.looks[s]["tiles"] if s >= 0 else default
        sel = np.array([kk == key for kk in keys])
        remap, pos, uvs, col, idx = {}, [], [], [], []
        for t, k in zip(t2[sel], reg[sel]):
            row = []
            for vi in t:
                key = (int(vi), int(k))
                if key not in remap:
                    remap[key] = len(pos)
                    pos.append(P[vi]); uvs.append(frames.uv(int(k), P[vi, 0], P[vi, 2]))
                row.append(remap[key])
            idx.append(row)
        pos = np.array(pos)
        f = frames.tone(pos[:, 0], pos[:, 2])
        tile = ptiles[vk]
        kind = _sidewalk_kind(tile)
        if kind == "ground":
            f = f * frames.ground_tone(pos[:, 0], pos[:, 2])
        if s >= 0:
            used_slots[s] = used_slots.get(s, 0) + int(sel.sum())
        tile_material(f"Island_{s + 1}_{vk + 1}" if s >= 0 else f"Block_paving_{vk + 1}", Image.open(tile["path"]),
                      images, materials, (tile_m, tile_m), kind, 0.92 if kind == "ground" else 0.85, surface,
                      own=LIB.own_maps(tile), mix=_slot_mix(frames, s, vk),
                      drift=GROUND_DRIFT if kind == "ground" else 0.0)
        prims.append({"positions": pos, "normals": np.tile([0, 1, 0], (len(pos), 1)), "uv0": np.array(uvs),
                      "colors": np.column_stack([f, f, f, np.ones_like(f)]),
                      "indices": np.array(idx), "material": len(materials) - 1})

    faces = 0
    road_edge = getattr(mesh, "kerb_surface", None)
    if road_edge is None or road_edge.is_empty:
        road_edge = QM._road_surface(mesh)
    if polys:
        # the block's edge is the kerb: a vertical face down to the road, facing it
        import shapely
        pos, nrm, uvs, idx = [], [], [], []
        edge = QM.EdgeIndex.of(road_edge)
        for g in polys:
            shapely.prepare(g)                                  # many inside tests on one block
            for ring in [g.exterior] + list(g.interiors):
                c = np.array(ring.coords)
                on_frame = ((c[:, 0] < 0.6 * scale) | (c[:, 1] < 0.6 * scale) |
                            (c[:, 0] > W - 0.6 * scale) | (c[:, 1] > H - 0.6 * scale))
                d_mid = edge.distance((c[:-1] + c[1:]) / 2) if len(c) > 1 else []
                for k in range(len(c) - 1):
                    if on_frame[k] and on_frame[k + 1]:
                        continue                                # the image border is not a kerb
                    if d_mid[k] > 0.05 / mpp_out:
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
            kt = (tiles.get("kerbstone") or tiles["sidewalk"])[0]
            tile_material("Block_kerb", Image.open(kt["path"]), images, materials, (tile_m, tile_m), "concrete", 0.9, surface,
                          own=LIB.own_maps(kt))
            prims.append({"positions": np.array(pos), "normals": np.array(nrm), "uv0": np.array(uvs),
                          "indices": np.array(idx), "material": len(materials) - 1})
    meshes.append({"name": "Blocks", "primitives": prims})
    return {"blocks": len(polys), "triangles": int(len(t2)), "quads": int(n_quads),
            "cell_m": cell_m, "kerb_faces": faces, "height_m": round(top, 3),
            "island_slots": {int(s): {"name": frames.looks[s]["name"], "triangles": n} for s, n in used_slots.items()},
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

    import shapely
    for n_g, g in enumerate(polys):
        prog.part(0.5 + 0.4 * n_g / max(1, len(polys)))
        shapely.prepare(g)
        minx, miny, maxx, maxy = g.bounds
        # the block's squares, in the same order as one by one, each tested in one
        # call for all of them (one call per square was most of the time)
        cells = [(ix, iy) for ix in range(int(math.floor(minx / cell)), int(math.ceil(maxx / cell)))
                 for iy in range(int(math.floor(miny / cell)), int(math.ceil(maxy / cell)))]
        if not cells:
            continue
        X0 = np.array([ix * cell for ix, _ in cells]); Y0 = np.array([iy * cell for _, iy in cells])
        boxes = shapely.box(X0, Y0, X0 + cell, Y0 + cell)
        hits = shapely.intersects(g, boxes)
        inside = np.zeros(len(cells), bool)
        inside[hits] = shapely.contains(g, boxes[hits])
        cut = {}
        edge = hits & ~inside
        if edge.any():
            cut = dict(zip(np.nonzero(edge)[0].tolist(), shapely.intersection(g, boxes[edge])))
        for n, (ix, iy) in enumerate(cells):
            if not hits[n]:
                continue
            x0, y0 = ix * cell, iy * cell
            if inside[n]:
                a, b2, c, d = vid(x0, y0), vid(x0 + cell, y0), vid(x0 + cell, y0 + cell), vid(x0, y0 + cell)
                T.extend([(a, b2, c), (a, c, d)])
                quads += 1
                continue
            piece = cut[n]
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
    from shapely.ops import unary_union
    parts = _sidewalk_parts(mesh)
    if not parts:
        return None
    # a hair of growth merges quads that share an edge; shrinking back keeps
    # the outer edge exactly where the sidewalk ends
    return unary_union([p.buffer(0.001) for p in parts]).buffer(-0.001)


def _polygons_of(V, rings):
    """Polygons from rows of vertex ids (all the same length), made all at once."""
    import shapely
    R = np.asarray(rings, np.int64)
    if R.ndim != 2 or not len(R):
        from shapely.geometry import Polygon
        return np.array([Polygon([V[i] for i in r]) for r in rings], object)
    return shapely.polygons(np.asarray(V, float)[R])


def _sidewalk_parts(mesh):
    """The sidewalk's quads and triangles as polygons, in texture pixels."""
    import shapely
    parts = []
    if mesh.sw_quads:
        pg = _polygons_of(mesh.v, mesh.sw_quads)
        bad = ~shapely.is_valid(pg)
        if bad.any():
            pg[bad] = shapely.buffer(pg[bad], 0)
        parts.extend(pg[~shapely.is_empty(pg) & (shapely.area(pg) > 0)])
    tris = getattr(mesh, "sw_tris", [])
    if tris:
        pg = _polygons_of(mesh.v, tris)
        parts.extend(pg[shapely.is_valid(pg) & (shapely.area(pg) > 0)])
    return parts



def _object_meshes(mesh, scatter, mpp_mask, scale, mpp_out, W, H, stand_m, images, materials, meshes):
    """
    Each placement is a grid of copies of one object: nx along the rectangle's
    width, ny along its depth, with the gaps between them. A curved placement
    has a path instead: its copies fill the smooth line through the path's
    points, each turned with the curve, in ny rows (app/curves.py). A package
    placement mixes its slots' objects along its line (straight for a
    rectangle, or curved), each by its own width. Each copy is an instance of
    its object, standing on the block or sidewalk. Copies
    whose footprint lies on the road are left out: objects stand only on
    blocks and sidewalks.
    """
    from shapely.geometry import Polygon
    from shapely import make_valid
    road = getattr(mesh, "kerb_surface", None)
    if road is None or road.is_empty:
        road = QM._road_surface(mesh)
    road = make_valid(road)                    # a repaired shape: intersections cannot fail on it
    import shapely
    shapely.prepare(road)                      # most footprints do not touch it: a quick test first
    placed, skipped, by_obj = 0, 0, {}
    problems, feet = [], []
    sized = lambda oid: PL.sized(scatter["objects"][oid]["meta"])
    for p in scatter["placements"]:
      label = (scatter["objects"].get(p.get("object"), {}).get("meta", {}).get("name") or
               "package %s" % scatter.get("packages", {}).get(p.get("package"), {}).get("name", ""))
      try:
        # every copy: its object, centre in metres (image axes), the direction of
        # its X (with its own extra rotation) and its id (app/placements.py)
        spots, _ = PL.copies_of(p, scatter["objects"], scatter.get("packages", {}), mpp_mask)
        for oid, cc, a, cid, sc in spots:
            k, turn, w, d = sized(oid)
            k, w, d = k * sc, w * sc, d * sc                  # a plant's random scale (1 otherwise)
            fp = Polygon([(x / mpp_out, y / mpp_out) for x, y in PL.footprint(cc, a, w, d)])
            if road.intersects(fp) and road.intersection(fp).area > 0.02 * fp.area:
                skipped += 1
                continue
            # image x right, image y down = world +X, +Z; rotation about the up axis
            tr = (cc[0] - W * mpp_out / 2, stand_m, cc[1] - H * mpp_out / 2)
            # the object keeps the orientation it was modelled in: at rotation 0
            # its front (Blender's +Y) faces the top of the map. Each turn adds a
            # quarter turn clockwise, seen from above
            phi = -(a + turn * math.pi / 2)
            q = (0.0, math.sin(phi / 2), 0.0, math.cos(phi / 2))
            by_obj.setdefault(oid, []).append((tr, q, k))
            feet.append(fp)
            placed += 1
      except Exception as e:
        problems.append("%s: %s" % (label, e))
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
    return {"copies": placed, "skipped_on_road": skipped, "objects": len(by_obj), "problems": problems,
            "_feet": feet}



def _lamps(mesh, world, feet, mpp_mask, scale, mpp_out, W, H):
    """
    Street lamps for the 3D view (app/quadmesh.py lamp_spots), clear of the
    objects (feet: their footprints in texture pixels): [x, y, z, dx, dz] each,
    the foot of the pole in metres on the sidewalk's top (raised with a
    bridge) and the way its arm reaches, towards the road. Not in the GLB.
    """
    from scipy.spatial import cKDTree
    from shapely.affinity import scale as sscale
    from shapely.ops import unary_union
    if not getattr(mesh, "sw_runs", None):
        return []
    avoid = None
    if feet:
        avoid = sscale(unary_union(feet), 1.0 / scale, 1.0 / scale, origin=(0, 0))     # to mask pixels
    spots = QM.lamp_spots(mesh, mpp_mask, avoid=avoid)
    if not spots:
        return []
    top = sorted({i for q in mesh.sw_quads for i in q if mesh.h[i] > 0})
    if not top:
        return []
    V = np.array(mesh.v, float)[top] / scale
    tree = cKDTree(V)
    out = []
    for x, y, dx, dy in spots:
        _, j = tree.query((x, y))
        vi = top[j]
        out.append([round(x * mpp_mask - W * mpp_out / 2, 2), round(float(world[vi, 1] + mesh.h[vi]), 3),
                    round(y * mpp_mask - H * mpp_out / 2, 2), round(dx, 3), round(dy, 3)])
    return out


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
