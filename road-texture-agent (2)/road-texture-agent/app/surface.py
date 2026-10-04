"""
Surface detail for the repeating material tiles: a bump map (glTF normal map)
and a roughness map, worked out from each tile's own grain, so the sun and the
street lamps catch the surface, in the 3D tab, the photo render and in Blender
or any glTF viewer.

Three sources:
- the tile's own: a material from the scanned library (app/library.py) comes
  with its own measured bump and roughness, made with its colour so they line
  up exactly.
- scanned over your tiles: the library's default scan for asphalt and for
  concrete, laid over the tile at its real size. Only for surfaces without a
  pattern of their own: the joints of paving must line up with the paving in
  the colour, so paving uses its own grain.
- grain: worked out from the tile's own colour. The bump comes from the fine
  grain only, the pores and stones a few millimetres across: lighter grain
  stands proud, darker sinks. Larger tone changes (stains, patches, wear) are
  not relief and are left out; their edges are softened so a stain does not
  grow a rim. The roughness follows the tone gently (lighter, polished parts a
  little smoother).
Whatever the source, road paint found in the colour is smoother than the road and
slightly raised at its edges.
"""
import hashlib
import io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# relief (m) of a typical grain step, roughness, how far roughness follows the tone,
# and whether bright paint is looked for (painted markings in road textures)
KINDS = {
    "asphalt": {"relief_m": 0.0008, "rough": 0.86, "spread": 0.08, "paint": True},
    "paving": {"relief_m": 0.0004, "rough": 0.78, "spread": 0.06, "paint": False},
    "concrete": {"relief_m": 0.0003, "rough": 0.80, "spread": 0.05, "paint": False},
}
PAINT_ROUGH = 0.55          # road paint: smoother than asphalt
PAINT_THICK_M = 0.0004      # its thickness, a slight step at its edges
GRAIN_M = 0.007             # the grain: features finer than about this
SCAN_KINDS = ("asphalt", "concrete")   # kinds a scan may stand in for (no pattern of their own)

_cache = {}
_scans = {}


def scan_for(kind):
    """
    The library's default scan for a kind of surface (app/scans/library.json,
    the entry marked bump_default), or None: its normal map (OpenGL convention,
    as glTF), roughness, the size it covers in metres and its name.
    """
    if kind not in SCAN_KINDS:
        return None
    from app import library as LIB
    e = LIB.default_for(kind)
    if not e:
        return None
    files = [LIB.DIR / e["normal"], LIB.DIR / e["roughness"]]
    if not all(f.exists() for f in files):
        return None
    stamp = (e["id"], tuple(f.stat().st_mtime for f in files))
    hit = _scans.get(kind)
    if hit and hit["stamp"] == stamp:
        return hit
    nrm = np.asarray(Image.open(files[0]).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    rgh = np.asarray(Image.open(files[1]).convert("L")).astype(np.float32) / 255.0
    if nrm.shape[:2] != rgh.shape:
        rgh = np.asarray(Image.fromarray(rgh).resize((nrm.shape[1], nrm.shape[0]), Image.BILINEAR))
    hit = {"stamp": stamp, "normal": nrm, "rough": rgh, "size_m": [float(v) for v in e["size_m"]],
           "name": e["name"], "source": e.get("source", "")}
    _scans[kind] = hit
    return hit


def _resize(arr, w, h):
    """A float picture (H x W, or H x W x C) resized to w x h, bilinear, channel by channel."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize((w, h), Image.BILINEAR))
    return np.stack([_resize(arr[..., c], w, h) for c in range(arr.shape[2])], axis=-1)


def _scan_layer(scan, size_m, w_px, h_px):
    """
    A scan laid over a tile of size_m: repeated a whole number of times each way
    (so the tile still joins itself), stretched by the little it takes to fit, and
    resampled to the tile's pixels. Returns the slopes (dh/dx, dh/d-row-down) and
    the roughness.
    """
    sw, sh = scan["size_m"]
    nx, ny = max(1, int(round(size_m[0] / sw))), max(1, int(round(size_m[1] / sh)))
    n = _resize(np.tile(scan["normal"], (ny, nx, 1)), w_px, h_px)
    r = _resize(np.tile(scan["rough"], (ny, nx)), w_px, h_px)
    nz = np.maximum(n[..., 2], 0.05)
    # a stretched surface has gentler slopes
    fx, fy = (nx * sw) / size_m[0], (ny * sh) / size_m[1]
    return -n[..., 0] / nz * fx, n[..., 1] / nz * fy, np.clip(r, 0.02, 1.0)


def maps(img, size_m, kind="asphalt", quality=90, source="scan", own=None):
    """
    The normal map and the roughness map of a tile, as JPEG bytes, and what they
    came from ("scan: <name>" or "grain").

    img: the tile (PIL image), repeating seamlessly; size_m: (width, height)
    it covers in metres; kind: one of KINDS; source: "scan" uses a bundled scan
    where there is one for the kind, else the tile's grain; "grain" always the
    grain. own: the tile's own (normal, roughness, name) pictures, made with
    its colour (library materials): used whatever the source. Normal map:
    glTF's convention (+X right, +Y up in the picture, +Z out of the surface).
    Roughness: in the green channel as glTF's metallicRoughness texture wants
    it (grey picture).
    """
    rgb = np.asarray(img.convert("RGB"))
    scan = scan_for(kind) if source == "scan" and own is None else None
    own_key = None
    if own is not None:
        own_n = np.asarray(own[0].convert("RGB"))
        own_r = np.asarray(own[1].convert("L"))
        own_key = (hashlib.sha1(own_n.tobytes() + own_r.tobytes()).hexdigest(), own[2])
    key = (hashlib.sha1(rgb.tobytes()).hexdigest(), rgb.shape, tuple(round(float(s), 4) for s in size_m), kind, quality,
           (scan["name"], scan["stamp"]) if scan else None, own_key)
    if key in _cache:
        return _cache[key]
    k = KINDS[kind]
    h_px, w_px = rgb.shape[:2]
    mx, my = float(size_m[0]) / w_px, float(size_m[1]) / h_px            # metres per pixel, each way
    lum = (rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
    wrap = dict(mode="wrap")

    # paint: well above the road's own tone, found only where asked (road textures)
    paint = np.zeros_like(lum)
    med = float(np.median(lum))
    if k["paint"]:
        paint = np.clip((lum - (med + 0.30)) / 0.10, 0.0, 1.0)
        paint = ndi.gaussian_filter(paint, 0.7, **wrap)
        # under the paint the grain shows only faintly: shift its tone down to the road's
        lum = lum - paint * (float(np.median(lum[paint > 0.5])) - med if (paint > 0.5).any() else 0.0)

    def slopes(height):
        """Slopes (metres per metre) of a height field: along x, and down the rows."""
        return ((np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * mx),
                (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (2 * my))

    # the paint's own edge: a slight step
    pdx, pdy = slopes(paint * PAINT_THICK_M)
    if own is not None:
        # the material's own measured relief and roughness, made with its colour
        if own_n.shape[:2] != (h_px, w_px):
            own_n = np.asarray(Image.fromarray(own_n).resize((w_px, h_px), Image.BILINEAR))
            own_r = np.asarray(Image.fromarray(own_r).resize((w_px, h_px), Image.BILINEAR))
        on = own_n.astype(np.float32) / 127.5 - 1.0
        onz = np.maximum(on[..., 2], 0.05)
        odx, ody = -on[..., 0] / onz, on[..., 1] / onz
        dx, dy_down = odx * (1.0 - 0.8 * paint) + pdx, ody * (1.0 - 0.8 * paint) + pdy
        rough = own_r.astype(np.float32) / 255.0
        used = "scan: " + own[2]
    elif scan:
        # measured relief and roughness, flattened under paint (paint fills the pores)
        sdx, sdy, rough = _scan_layer(scan, size_m, w_px, h_px)
        dx, dy_down = sdx * (1.0 - 0.8 * paint) + pdx, sdy * (1.0 - 0.8 * paint) + pdy
        used = "scan: " + scan["name"]
    else:
        # the fine grain, a band of a pixel or so to GRAIN_M, its large steps softened
        smooth = ndi.gaussian_filter(lum, 0.6, **wrap)
        sig = (max(0.8, GRAIN_M / my), max(0.8, GRAIN_M / mx))            # (rows, columns)
        grain = smooth - ndi.gaussian_filter(smooth, sig, **wrap)
        scale = 1.4826 * float(np.median(np.abs(grain))) + 1e-6
        h = np.tanh(grain / (2.5 * scale)) * 2.5                           # about -2.5 .. 2.5
        gdx, gdy = slopes(h * k["relief_m"] * (1.0 - 0.8 * paint))
        dx, dy_down = gdx + pdx, gdy + pdy
        # roughness following the tone gently
        tone = ndi.gaussian_filter(lum, 1.0, **wrap)
        z = (tone - float(np.median(tone))) / (float(tone.std()) + 1e-6)
        rough = k["rough"] - k["spread"] * np.tanh(z / 1.5)
        used = "grain"

    n = np.stack([-dx, dy_down, np.ones_like(dx)], axis=-1)              # +Y up the picture
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rough = rough * (1.0 - paint) + PAINT_ROUGH * paint                   # paint smoother
    rgh = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out = []
    for arr in (nrm, rgh):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=quality)
        out.append(buf.getvalue())
    out.append(used)
    _cache[key] = tuple(out)
    if len(_cache) > 64:
        _cache.pop(next(iter(_cache)))
    return _cache[key]
