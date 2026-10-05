"""
The scanned material library (app/scans, CC0 scans from Poly Haven): real
street surfaces with their own colour, bump and roughness.

A library material stands in for a part's material tile (streets, sidewalks or
kerbs). Its three pictures are laid over the tile together: repeated a whole
number of times each way, so the tile still joins itself, and stretched by
the little it takes to fit, all three in exactly the same way, so every
stone's bump and roughness sit on that stone. Optionally its colour takes on
the tone of the trained tile it replaces.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

DIR = Path(__file__).with_name("scans")
KINDS_FOR_PART = {"street": ("asphalt", "concrete"),
                  "sidewalk": ("paving", "concrete", "asphalt"),
                  "kerb": ("concrete",),
                  "square": ("paving", "concrete", "asphalt")}


def materials():
    """The library's materials: id, name, kind, size_m, the three files, source, authors, licence."""
    path = DIR / "library.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("materials", [])


def entry(mid):
    return next((e for e in materials() if e["id"] == mid), None)


def default_for(kind):
    """The material whose bump and roughness go over your own tiles of this kind (Surface detail: Scanned)."""
    return next((e for e in materials() if e.get("bump_default") == kind), None)


def thumbnail(mid, px=96):
    """A small picture of a material's colour (JPEG bytes)."""
    import io
    e = entry(mid)
    if not e:
        return None
    im = Image.open(DIR / e["colour"]).convert("RGB")
    im.thumbnail((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _resize(arr, w, h):
    """A float picture (H x W, or H x W x C) resized to w x h, bilinear, channel by channel."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize((w, h), Image.BILINEAR))
    return np.stack([_resize(arr[..., c], w, h) for c in range(arr.shape[2])], axis=-1)


def layout(colour, normal, rough, reps, out_w, out_h):
    """
    The three pictures of a material repeated reps = (across, down) times and
    resampled to out_w x out_h, the same way for all three. Returns RGB uint8,
    the normals (-1..1) and roughness (0..1) as float arrays.
    """
    rx, ry = reps
    c = Image.fromarray(np.tile(colour, (ry, rx, 1))).resize((out_w, out_h), Image.LANCZOS)
    n = _resize(np.tile(normal, (ry, rx, 1)), out_w, out_h)
    r = _resize(np.tile(rough, (ry, rx)), out_w, out_h)
    return np.asarray(c), n, r


def squeeze_normals(n, scale):
    """
    Unit normals of a surface squeezed by scale = (along x, down the rows): a
    surface squeezed into less room keeps its heights, so its slopes steepen
    by the same factor (stretched: below 1, gentler).
    """
    nz = np.maximum(n[..., 2], 0.05)
    dx, dy = -n[..., 0] / nz * scale[0], n[..., 1] / nz * scale[1]
    out = np.stack([-dx, dy, np.ones_like(dx)], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _load(e):
    colour = np.asarray(Image.open(DIR / e["colour"]).convert("RGB"))
    normal = np.asarray(Image.open(DIR / e["normal"]).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    rough = np.asarray(Image.open(DIR / e["roughness"]).convert("L")).astype(np.float32) / 255.0
    return colour, normal, rough


def turn_normals(p, k, flipped):
    """
    A patch of colour (3), bump (3: x right, y up, out) and roughness (1)
    channels that was turned k quarter turns anticlockwise (np.rot90) and then
    maybe flipped left to right: its slopes turned and flipped with it, so
    every stone is still lit from the right side.
    """
    p = p.copy()
    x, y = p[..., 3].copy(), p[..., 4].copy()
    for _ in range(k % 4):
        x, y = -y, x
    if flipped:
        x = -x
    p[..., 3], p[..., 4] = x, y
    return p


def _calm(tile, size):
    """
    app/tiles.py's _calm (large blotches and extreme spots evened out), its
    wide blur done by FFT: exact for a tile that wraps around, and quick.
    """
    from scipy.ndimage import fourier_gaussian
    t = tile.astype(np.float64)
    sigma = size / 12
    broad = np.stack([np.fft.ifft2(fourier_gaussian(np.fft.fft2(t[..., c]), sigma)).real for c in range(t.shape[2])], -1)
    mean = t.reshape(-1, t.shape[2]).mean(axis=0)
    t = t - broad + mean
    lum = t.mean(axis=2)
    mu, sd = lum.mean(), lum.std() + 1e-6
    z = (lum - mu) / sd
    soft = np.tanh(z / 2.5) * 2.5
    return t + ((soft - z) * sd)[:, :, None]


def _synth(e, tile_m, px, variant):
    """
    A fresh 4 m tile of a scanned material, made like the tiles from your
    training (app/tiles.py): patches of the scan, half a metre across, placed
    at random, turned and flipped, blended into a tile that joins itself on
    every edge, then its large blotches and extreme spots evened out. Colour,
    bump and roughness are placed together, patch by patch, so they still line
    up; a turned patch has its slopes turned with it. Each variant is another
    arrangement, so the scan no longer repeats every couple of metres.
    """
    from app import tiles as TL
    colour, normal, rough = _load(e)
    # to the tile's scale (metres per pixel)
    mpp = tile_m / px
    w, h = max(64, int(round(float(e["size_m"][0]) / mpp))), max(64, int(round(float(e["size_m"][1]) / mpp)))
    c = np.asarray(Image.fromarray(colour).resize((w, h), Image.LANCZOS if w > colour.shape[1] else Image.BOX), np.float64)
    n = _resize(normal, w, h).astype(np.float64)
    r = _resize(rough, w, h).astype(np.float64)
    # the scan's broad light and dark (wear, stains), shine and undulation evened out
    # first, over about half a patch: neighbouring patches then match in tone and
    # blend unseen, instead of showing as a quilt. The detail inside (stones, pores,
    # cracks) is kept; large-scale variation comes from the model's weathering layer
    patch = int(px) // 8
    flat = lambda a: a - ndi.gaussian_filter(a, patch / 2, mode="wrap") + a.mean()
    c = np.stack([flat(c[..., i]) for i in range(3)], -1)
    r = flat(r)
    nz = np.maximum(n[..., 2], 0.05)
    sx, sy = flat(n[..., 0] / nz), flat(n[..., 1] / nz)
    n = np.stack([sx, sy, np.ones_like(sx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    src = np.concatenate([c, n, r[..., None]], axis=-1)
    rng = np.random.default_rng([variant, 9001])
    out = TL._seamless_synth(src, int(px), patch, rng, quarter_turns=e.get("quarter_turns", True),
                             turned=turn_normals)
    c = _calm(out[..., :3], int(px))
    n = out[..., 3:6]
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    return c, n, np.clip(out[..., 6], 0.0, 1.0)


def tile_for(mid, tile_m, px, cache_dir, tone=None, variant=None):
    """
    A material tile made from a library material: its colour, normal and
    roughness pictures covering tile_m x tile_m at px x px, made together so
    they line up. tone: the mean colour (0-255 RGB) its colour should take on,
    or None for its own. variant: None lays the scan itself, repeated (paving,
    whose joints must stay on their grid); a number makes a fresh tile from
    the scan's patches, that arrangement (streets: no visible repeat).
    Returns a tile record as the tileset's (path), with normal_path,
    rough_path and the material's name; the pictures are kept in cache_dir.
    """
    e = entry(mid)
    if not e:
        raise ValueError(f"no material {mid} in the library")
    files = [DIR / e[k] for k in ("colour", "normal", "roughness")]
    key = hashlib.sha1(json.dumps([mid, float(tile_m), int(px), [round(float(t), 1) for t in tone] if tone is not None else None,
                                   [f.stat().st_mtime for f in files], variant, e.get("quarter_turns", True), 2]).encode()).hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: cache_dir / f"lib_{mid}_{key}_{k}.png" for k in ("colour", "normal", "rough")}
    rec = {"id": f"lib_{mid}" + (f"_{variant}" if variant is not None else ""), "library": mid, "name": e["name"],
           "method": "library" if variant is None else "library synth", "variant": variant, "covers_m": [tile_m, tile_m],
           "path": str(paths["colour"]), "normal_path": str(paths["normal"]), "rough_path": str(paths["rough"])}
    if all(p.exists() for p in paths.values()):
        return rec
    if variant is not None:
        c, n, r = _synth(e, tile_m, px, variant)
    else:
        colour, normal, rough = _load(e)
        reps = (max(1, int(round(tile_m / e["size_m"][0]))), max(1, int(round(tile_m / e["size_m"][1]))))
        c, n, r = layout(colour, normal, rough, reps, int(px), int(px))
        # reps x the scan's real size fit into tile_m: squeezed (or stretched) by that much
        n = squeeze_normals(n, (reps[0] * e["size_m"][0] / tile_m, reps[1] * e["size_m"][1] / tile_m))
    c = c.astype(np.float32)
    if tone is not None:
        gain = np.clip(np.asarray(tone, np.float32) / np.maximum(c.reshape(-1, 3).mean(axis=0), 1.0), 0.25, 4.0)
        c = c * gain
    # cache files: quick to write, lossless
    Image.fromarray(np.clip(c + 0.5, 0, 255).astype(np.uint8)).save(paths["colour"], compress_level=1)
    Image.fromarray(np.clip((n * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)).save(paths["normal"], compress_level=1)
    Image.fromarray(np.clip(r * 255 + 0.5, 0, 255).astype(np.uint8)).save(paths["rough"], compress_level=1)
    return rec


def own_maps(tile):
    """A tile's own normal and roughness pictures and its name, if it has them (library tiles), else None."""
    if not tile or not tile.get("normal_path"):
        return None
    return (Image.open(tile["normal_path"]).convert("RGB"), Image.open(tile["rough_path"]).convert("L"),
            tile.get("name") or tile.get("library") or "library")


# ------------------------------------------------------------------ material balls
BALL_D_M = 1.0          # the ball's diameter in metres: the material is shown at its real size


def _to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def ball(colour, normal, rough, size_m, px=152, ss=2):
    """
    A material on a ball, as material browsers show it (PNG bytes, transparent
    round it): its colour, with its bump catching a light from the upper left
    and its roughness in the highlight. colour: RGB uint8; normal: the normal
    map (-1..1, OpenGL convention) or None; rough: roughness (0..1) or None;
    size_m: (width, height) the pictures cover in metres. The ball is
    BALL_D_M across, so the material shows at its real size.
    """
    import io
    from scipy.ndimage import map_coordinates
    n = px * ss
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    x = (x + 0.5) / n * 2 - 1
    y = 1 - (y + 0.5) / n * 2
    r2 = x * x + y * y
    inside = r2 < 1.0
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    N0 = np.stack([x, y, z], -1)
    phi, theta = np.arctan2(x, z), np.arcsin(np.clip(y, -1, 1))
    R = BALL_D_M / 2
    th, tw = colour.shape[:2]
    # surface position in metres (along, up), to picture pixels (rows go down)
    cols = (phi * R / size_m[0] * tw) % tw
    rows = (-theta * R / size_m[1] * th) % th
    at = [rows.ravel(), cols.ravel()]

    def sample(img):
        if img.ndim == 2:
            return map_coordinates(img.astype(np.float64), at, order=1, mode="grid-wrap").reshape(n, n)
        return np.stack([sample(img[..., c]) for c in range(img.shape[2])], -1)

    A = _to_linear(sample(colour.astype(np.float64)))
    T = np.stack([np.cos(phi), np.zeros_like(phi), -np.sin(phi)], -1)
    B = np.stack([-np.sin(phi) * np.sin(theta), np.cos(theta), -np.cos(phi) * np.sin(theta)], -1)
    if normal is not None:
        t = sample(normal)
        N = T * t[..., :1] + B * t[..., 1:2] + N0 * t[..., 2:3]
        N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)
    else:
        N = N0
    rgh = np.clip(sample(rough), 0.04, 1.0) if rough is not None else np.full((n, n), 0.8)
    L = np.array([-0.55, 0.62, 0.56]); L /= np.linalg.norm(L)
    V = np.array([0.0, 0.0, 1.0])
    H = (L + V) / np.linalg.norm(L + V)
    nl = np.clip(N @ L, 0, 1)
    nv = np.clip(N @ V, 1e-3, 1)
    nh = np.clip(N @ H, 0, 1)
    a2 = (rgh * rgh) ** 2
    D = a2 / (np.pi * (nh * nh * (a2 - 1) + 1) ** 2)
    k = (rgh + 1) ** 2 / 8
    G = (nl / (nl * (1 - k) + k)) * (nv / (nv * (1 - k) + k))
    F = 0.04 + 0.96 * (1 - float(V @ H)) ** 5
    spec = D * G * F / np.maximum(4 * nl * nv, 1e-4)
    # white light, so the ball shows the material's own colour
    sky = 0.30 * (0.55 + 0.45 * N[..., 1:2])
    key = 2.6
    lit = A * (key * nl[..., None] + sky) + (spec * nl)[..., None] * key
    rgb = _to_srgb(1 - np.exp(-1.35 * lit))
    rgba = np.concatenate([rgb * inside[..., None], inside[..., None].astype(np.float64)], -1)
    # anti-aliased: averaged down from the finer picture (premultiplied, so the edge stays clean)
    rgba = rgba.reshape(px, ss, px, ss, 4).mean(axis=(1, 3))
    out = np.zeros_like(rgba)
    out[..., 3] = rgba[..., 3]
    out[..., :3] = rgba[..., :3] / np.maximum(rgba[..., 3:], 1e-6)
    buf = io.BytesIO()
    Image.fromarray(np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA").save(buf, "PNG")
    return buf.getvalue()


def material_ball(mid, cache_dir, px=152):
    """A library material's ball (PNG bytes), kept in cache_dir."""
    e = entry(mid)
    if not e:
        return None
    files = [DIR / e[k] for k in ("colour", "normal", "roughness")]
    key = hashlib.sha1(json.dumps([mid, px, BALL_D_M, 2, [f.stat().st_mtime for f in files]]).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"ball_{mid}_{key}.png"
    if path.exists():
        return path.read_bytes()
    colour, normal, rough = _load(e)
    data = ball(colour, normal, rough, e["size_m"], px)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data
