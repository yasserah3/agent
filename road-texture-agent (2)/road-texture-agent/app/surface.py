"""
Surface detail for the repeating material tiles: a bump map (glTF normal map)
and a roughness map, worked out from each tile's own grain, so the sun and the
street lamps catch the surface, in the 3D tab, the photo render and in Blender
or any glTF viewer.

The bump comes from the fine grain only, the pores and stones a few millimetres
across: lighter grain stands proud, darker sinks. Larger tone changes (stains,
patches, wear) are not relief and are left out; their edges are softened so a
stain does not grow a rim. The roughness follows the tone gently (lighter,
polished parts a little smoother) and paint is smoother than the road.
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

_cache = {}


def maps(img, size_m, kind="asphalt", quality=90):
    """
    The normal map and the roughness map of a tile, as JPEG bytes.

    img: the tile (PIL image), repeating seamlessly; size_m: (width, height)
    it covers in metres; kind: one of KINDS. Normal map: glTF's convention
    (+X right, +Y up in the picture, +Z out of the surface). Roughness: in the
    green channel as glTF's metallicRoughness texture wants it (grey picture).
    """
    rgb = np.asarray(img.convert("RGB"))
    key = (hashlib.sha1(rgb.tobytes()).hexdigest(), rgb.shape, tuple(round(float(s), 4) for s in size_m), kind, quality)
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

    # the fine grain, a band of a pixel or so to GRAIN_M, its large steps softened
    smooth = ndi.gaussian_filter(lum, 0.6, **wrap)
    sig = (max(0.8, GRAIN_M / my), max(0.8, GRAIN_M / mx))                # (rows, columns)
    grain = smooth - ndi.gaussian_filter(smooth, sig, **wrap)
    scale = 1.4826 * float(np.median(np.abs(grain))) + 1e-6
    h = np.tanh(grain / (2.5 * scale)) * 2.5                               # about -2.5 .. 2.5
    height = h * k["relief_m"] * (1.0 - 0.8 * paint) + paint * PAINT_THICK_M

    # slopes (metres per metre), rows running down the picture
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * mx)
    dy_down = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (2 * my)
    n = np.stack([-dx, dy_down, np.ones_like(dx)], axis=-1)              # +Y up the picture
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = np.clip((n * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # roughness: following the tone gently, paint smoother
    tone = ndi.gaussian_filter(lum, 1.0, **wrap)
    z = (tone - float(np.median(tone))) / (float(tone.std()) + 1e-6)
    rough = k["rough"] - k["spread"] * np.tanh(z / 1.5)
    rough = rough * (1.0 - paint) + PAINT_ROUGH * paint
    rgh = np.clip(rough * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out = []
    for arr in (nrm, rgh):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=quality)
        out.append(buf.getvalue())
    _cache[key] = tuple(out)
    if len(_cache) > 64:
        _cache.pop(next(iter(_cache)))
    return _cache[key]
