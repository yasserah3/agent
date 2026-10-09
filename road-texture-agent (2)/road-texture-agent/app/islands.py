"""
Islands: the areas between the roads (city blocks, traffic islands, squares),
each with a number, so each can be given a material of its own.

They are found when the texture is generated, from the same cleaned mask the
texture is made from (app/junctions.py's clean_mask): every connected area
that is not road and covers at least MIN_M2 is an island. They are numbered
in reading order (by each island's topmost pixel, then leftmost), so the same
mask always numbers them the same way.

A material slot remembers its islands by a point well inside each (mask
pixels), not by number: if the islands change (an inner street splits one),
the point still finds the island it lies in.

The islands texture (texture()) is the counterpart of the road texture: the
same size, the islands laid with their materials at real size and the roads
left transparent, so the two fit together edge to edge.
"""
import functools
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app import progress as prog

MIN_M2 = 4.0            # smaller areas between roads get no number (the 3D model fills only larger)
TEXTURE_VERSION = 1     # how the islands texture is laid: a new one lays every island again
GROUND_AMP = 0.10       # broad light and dark over ground (sand, gravel, earth): +-10%
PAVED_AMP = 0.04        # and over paving and concrete, gentler


# ------------------------------------------------------------------ finding them
def find(gray, mpp, min_m2=MIN_M2):
    """
    The islands of a mask (grey picture, road white) at mpp metres per pixel.
    Returns a label picture (0 road or too small, 1..N the islands) and the
    islands: id, area_m2, pt (a point well inside, where its number is drawn
    and what a slot remembers), r_m (how far pt is from the nearest road),
    bbox [x0, y0, x1, y1] and rings (its outline, and any holes, as lists of
    [x, y]); all in mask pixels, the image from 0 to its width.
    """
    from skimage import measure
    from shapely.geometry import LineString
    from app import junctions as J
    road, _ = J.clean_mask(np.asarray(gray))
    lab, n = ndi.label(~road)
    if n == 0:
        return np.zeros(road.shape, np.int32), []
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area * mpp * mpp >= min_m2
    keep[0] = False
    renumber = np.zeros(n + 1, np.int32)
    renumber[keep] = np.arange(1, int(keep.sum()) + 1)
    lab = renumber[lab]
    out = []
    boxes = ndi.find_objects(lab)
    for k, sl in enumerate(boxes, start=1):
        if sl is None:
            continue
        prog.part(k / max(1, len(boxes)))
        y0, x0 = sl[0].start, sl[1].start
        sub = np.pad(lab[sl] == k, 1)
        dist = ndi.distance_transform_edt(sub)
        r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
        rings = []
        for ring in measure.find_contours(sub.astype(np.float32), 0.5):
            if len(ring) < 4:
                continue
            xy = np.column_stack([ring[:, 1] - 1 + x0 + 0.5, ring[:, 0] - 1 + y0 + 0.5])
            s = np.asarray(LineString(xy).simplify(0.6).coords)
            if len(s) >= 4:
                rings.append(np.round(s, 1).tolist())
        out.append({"id": k, "area_m2": round(float(area[keep][k - 1]) * mpp * mpp, 1),
                    "pt": [round(float(x0 + c - 1 + 0.5), 1), round(float(y0 + r - 1 + 0.5), 1)],
                    "r_m": round(float(dist.max()) * mpp, 1),
                    "bbox": [int(x0), int(y0), int(sl[1].stop), int(sl[0].stop)], "rings": rings})
    return lab, out


def save(lab, islands, mpp, json_path, npz_path):
    """Keep the islands: the list (JSON, for the page) and the label picture (for the 3D model and the texture)."""
    np.savez_compressed(npz_path, labels=lab.astype(np.uint16 if len(islands) < 65535 else np.int32))
    json_path.write_text(json.dumps({"mpp": mpp, "size": [int(lab.shape[1]), int(lab.shape[0])],
                                     "count": len(islands), "islands": islands}, separators=(",", ":")))


def load_labels(npz_path):
    return np.load(npz_path)["labels"].astype(np.int32)


def label_at(lab, x, y, reach=3):
    """The island at mask pixel (x, y), or the most common one within reach pixels; 0 if none."""
    H, W = lab.shape
    xi, yi = int(np.floor(x)), int(np.floor(y))
    if 0 <= xi < W and 0 <= yi < H and lab[yi, xi]:
        return int(lab[yi, xi])
    win = lab[max(0, yi - reach):max(0, yi + reach + 1), max(0, xi - reach):max(0, xi + reach + 1)]
    win = win[win > 0]
    return int(np.bincount(win).argmax()) if win.size else 0


def resolve(slots, lab):
    """
    Which slot each island has: {island id: slot index}, from each slot's picks
    (points in mask pixels). A later slot wins an island picked twice.
    """
    out = {}
    for s, slot in enumerate(slots or []):
        for p in slot.get("picks") or []:
            try:
                k = label_at(lab, float(p[0]), float(p[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if k:
                out[k] = s
    return out


MIX_AMOUNT, MIX_SIZE_M = 0.3, 10.0     # a ground slot's mix when the page leaves it to us ("auto")


def slot_list(raw):
    """
    Material slots as sent by the page, checked: [{"material": id, "picks":
    [[x, y], ...], "mix": {"material": id, "auto" or "none", "amount": 0-0.9,
    "size_m": 2-60}}]. The mix: a second material in patches through a ground
    slot's islands (coarse gravel through fine sand); "auto" is the library's
    partner for the material (mix_with), "none" no mix.
    """
    out = []
    for s in (raw or [])[:64]:
        if not isinstance(s, dict):
            continue
        picks = []
        for p in (s.get("picks") or [])[:100000]:
            try:
                picks.append([round(float(p[0]), 1), round(float(p[1]), 1)])
            except (TypeError, ValueError, IndexError):
                continue
        mix = s.get("mix") if isinstance(s.get("mix"), dict) else {}
        num = lambda k, d, lo, hi: min(hi, max(lo, float(mix.get(k, d)) if isinstance(mix.get(k, d), (int, float)) else d))
        out.append({"material": str(s.get("material") or "same")[:80], "picks": picks,
                    "mix": {"material": str(mix.get("material") or "auto")[:80], "amount": num("amount", MIX_AMOUNT, 0.0, 0.9),
                            "size_m": num("size_m", MIX_SIZE_M, 2.0, 60.0)}})
    return out


# ------------------------------------------------------------------ light and dark over ground
def ground_field(w_m, h_m, seed, cell_m=2.0):
    """
    Broad light and dark as ground shows it from above (drifts of sand, damper
    and drier patches): a smooth random field over the map, mean 0 and spread
    about 0.6, made of features about 3, 10 and 35 m across. Indexed by
    (metres down / cell_m, metres across / cell_m) from the map's top left.
    """
    gw, gh = int(np.ceil(w_m / cell_m)) + 2, int(np.ceil(h_m / cell_m)) + 2
    rng = np.random.default_rng([int(seed), 4242])
    f = np.zeros((gh, gw), np.float64)
    for sigma_m, wgt in ((3.0, 0.3), (10.0, 0.4), (35.0, 0.3)):
        z = ndi.gaussian_filter(rng.standard_normal((gh, gw)), sigma_m / cell_m, mode="wrap")
        f += wgt * z / (z.std() + 1e-9)
    return f.astype(np.float32)


def field_at(field, x_m, y_m, cell_m=2.0):
    """The field at points in metres from the map's top left."""
    return ndi.map_coordinates(field, [np.asarray(y_m, np.float64) / cell_m, np.asarray(x_m, np.float64) / cell_m],
                               order=1, mode="nearest")


# ------------------------------------------------------------------ the ground pattern
# One small picture, the same in every model, that the 3D tab, the photo render, the
# Top view and the islands texture all read for the ground's variety, so all four agree
# on where a slot's Mix lies and how the colour drifts: three smooth random fields (red:
# the Mix's patches; green and blue: the drift), each about normal around 0.5 with
# spread PATTERN_SD, repeating at the picture's edges. Read at a point's world (x, z)
# in metres (the model's: x from the map's left edge less half its width, z likewise
# down) as uv = (x, z) * scale + offset, bilinear, wrapping; texel centres at
# (i + 0.5) / PATTERN_PX, row 0 at v = 0 (glTF's picture order).
PATTERN_PX, PATTERN_UNITS = 512, 32      # 32 features across before it repeats
PATTERN_SD = 0.132
DRIFT_READS = ((54.4, (0.11, 0.23)), (169.6, (0.71, 0.47)))   # green, blue: repeat (m), offset; 1.7 and 5.3 m features


@functools.lru_cache(maxsize=1)
def ground_pattern():
    """The ground pattern: PATTERN_PX square, RGB, uint8 (always the same picture)."""
    n, u = PATTERN_PX, PATTERN_PX / PATTERN_UNITS
    rng = np.random.default_rng(20261008)
    out = np.zeros((n, n, 3), np.uint8)
    for c in range(3):
        f = np.zeros((n, n))
        for sigma_u, wgt in ((0.5, 0.5), (0.25, 0.3), (0.12, 0.2)):   # broad, finer, finest, as an fbm's octaves
            z = ndi.gaussian_filter(rng.standard_normal((n, n)), sigma_u * u, mode="wrap")
            f += wgt * z / z.std()
        f = (f - f.mean()) / f.std()
        out[..., c] = np.clip(np.round((0.5 + PATTERN_SD * f) * 255), 0, 255)
    return out


@functools.lru_cache(maxsize=1)
def ground_pattern_png():
    import io
    buf = io.BytesIO()
    Image.fromarray(ground_pattern(), "RGB").save(buf, "PNG", optimize=True)
    return buf.getvalue()


@functools.lru_cache(maxsize=1)
def _pattern_sorted():
    """The red field as bilinear reads see it (texels and the points between), sorted: for thresholds."""
    r = ground_pattern()[..., 0].astype(np.float64) / 255
    rx, ry = (r + np.roll(r, -1, 1)) / 2, (r + np.roll(r, -1, 0)) / 2
    rxy = (rx + np.roll(rx, -1, 0)) / 2
    return np.sort(np.concatenate([a.ravel() for a in (r, rx, ry, rxy)]))


def pattern_at(channel, x, z, scale, offset):
    """The pattern's channel (0 red .. 2 blue) at world points (x, z) metres, as a GPU reads it."""
    P = ground_pattern()[..., channel].astype(np.float32) / 255
    n = P.shape[0]
    u = (np.asarray(x, np.float64) * scale + offset[0]) * n - 0.5
    v = (np.asarray(z, np.float64) * scale + offset[1]) * n - 0.5
    return ndi.map_coordinates(P, [v, u], order=1, mode="grid-wrap")


def mix_params(amount, size_m, seed):
    """
    Where a slot's Mix lies: its patches are where the pattern's red is above
    thresh, which a share `amount` of the ground is; patches about size_m across;
    each slot at its own place in the pattern. {"thresh", "scale", "offset"}.
    """
    a = float(amount)
    s = _pattern_sorted()
    thresh = 2.0 if a <= 0 else float(s[min(len(s) - 1, max(0, int(round((1 - a) * len(s)))))])
    k = int(seed)
    return {"thresh": round(thresh, 5), "scale": 1.0 / (PATTERN_UNITS * max(0.5, float(size_m))),
            "offset": [round((0.618034 * k + 0.13) % 1.0, 5), round((0.381966 * k + 0.57) % 1.0, 5)]}


def mix_weight(n, thresh, hA, depth_a, hB, depth_b):
    """
    How much of the second material (0..1) where the pattern is n: across the
    patch edge, the higher stones of either show (heights 0..1 of their depths, m).
    The 3D tab's and the photo's mixWeight.
    """
    s = (n - thresh) / 0.035 + 2.5 * (hB * depth_b - hA * depth_a) / max(depth_a, depth_b, 1e-4)
    t = np.clip(0.5 + s, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def drift_at(x, z):
    """The drift's two fields at world points, about -0.5..0.5 (the 3D tab's gDriftA, gDriftB)."""
    return tuple(pattern_at(1 + i, x, z, 1.0 / rep, off) - 0.5 for i, (rep, off) in enumerate(DRIFT_READS))


def drift_colour(rgb_lin, amount, da, db):
    """Linear colours (N, 3) drifted as the 3D tab drifts them."""
    k = (1.0 + amount * (2.0 * da + 1.6 * db))[:, None]
    return rgb_lin * k * np.stack([1.0 + amount * db, np.ones_like(db), 1.0 - amount * db], -1)


# ------------------------------------------------------------------ the islands texture
def _main_direction(rings, W, H):
    """An island's main kerb direction (radians): its edges' directions averaged four-fold, the image border left out."""
    if not rings:
        return 0.0
    c = np.asarray(max(rings, key=len), np.float64)
    d = np.diff(c, axis=0)
    mid = (c[:-1] + c[1:]) / 2
    inner = ~((mid[:, 0] < 1) | (mid[:, 1] < 1) | (mid[:, 0] > W - 1) | (mid[:, 1] > H - 1))
    ln = np.hypot(d[:, 0], d[:, 1]) * inner
    a = np.arctan2(d[:, 1], d[:, 0])
    return float(np.arctan2((ln * np.sin(4 * a)).sum(), (ln * np.cos(4 * a)).sum()) / 4) if ln.sum() > 0 else 0.0


class _Tile:
    """
    A material tile (a file or a picture; grey: a height map) ready to sample at
    the texture's scale: blurred to its pixel footprint (wrapping), or its mean.
    """

    def __init__(self, src, tile_m, mpp_out, grey=False):
        from scipy.ndimage import fourier_gaussian
        im = src if isinstance(src, Image.Image) else Image.open(src)
        img = np.asarray(im.convert("L" if grey else "RGB"), np.float32)
        if grey:
            img = img[..., None]
        self.nc = img.shape[2]
        self.px = img.shape[1]
        self.k = self.px / tile_m                       # tile pixels per metre
        foot = mpp_out * self.k                         # tile pixels under one texture pixel
        self.mean = img.reshape(-1, self.nc).mean(axis=0)
        self.flat = foot > self.px / 3                  # finer than the picture can show: its mean colour
        if not self.flat and foot > 1.0:
            sigma = 0.5 * foot
            img = np.stack([np.fft.ifft2(fourier_gaussian(np.fft.fft2(img[..., c]), sigma)).real
                            for c in range(self.nc)], -1).astype(np.float32)
        self.img = img

    def sample(self, u_m, v_m):
        if self.flat:
            return np.broadcast_to(self.mean, (len(u_m), self.nc)).copy()
        at = [np.asarray(v_m) * self.k, np.asarray(u_m) * self.k]
        return np.stack([ndi.map_coordinates(self.img[..., c], at, order=1, mode="grid-wrap") for c in range(self.nc)], -1)


def _lin(c):
    """sRGB 0..255 to linear 0..1 (the GPU's reading of a colour picture)."""
    c = np.clip(c / 255.0, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


_edges = {}           # the last mask's road edge at the texture's size and nearest islands (slow on a big map)


def _edge_of(gray, lab, s, soft_edges):
    import hashlib
    from app.generation import prepare_mask
    key = (hashlib.sha1(np.ascontiguousarray(gray).tobytes()).hexdigest(), gray.shape,
           hashlib.sha1(np.ascontiguousarray(lab).tobytes()).hexdigest(), int(s), bool(soft_edges))
    if key not in _edges:
        prog.stage("mask", "the road edge at the texture's size")
        prepared, coverage = prepare_mask(gray, s)
        cov = coverage if soft_edges else (prepared > 0).astype(np.float32)
        # every pixel's nearest island, so the soft edge under the road's has a colour
        near = lab
        if (lab == 0).any() and lab.any():
            iy, ix = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
            near = lab[iy, ix]
        _edges.clear()
        _edges[key] = (cov, near)
    return _edges[key]


def texture(gray, lab, islands, mpp, out_scale, soft_edges, slot_of, looks, default, seed, out_path,
            result_path=None, full_path=None, background=(18, 18, 20), only=None):
    """
    The islands texture: the same size as the road texture (out_scale times the
    mask), each island laid with its material at real size along its main kerb
    direction, the roads transparent (their edge the exact counterpart of the
    road texture's, so the two meet with no gap or overlap). Ground (sand,
    gravel, earth) is mixed from its tile variants and given broad light and
    dark, so it shows no repeat; paving keeps its joints on one grid.

    Ground also has the 3D model's variety, read from the same ground pattern at
    the same world points: a slot's Mix (its second material in patches, the
    higher stones of either at their edges) and the colour's drift.

    slot_of: {island id: slot}; looks: {slot: {"tiles": [paths], "tile_m", "ground": bool,
    and for ground: "heights": [height pictures, as tiles], "depth_m", "drift",
    "mix": {"tile": path, "height": picture, "depth_m", "thresh", "scale", "offset"}}};
    default: the same for islands in no slot. With result_path and full_path,
    also writes the road texture with the islands under it.

    only: the islands to lay again (their material changed), on the pictures as
    they are (out_path, full_path): every other pixel stays as it was, and those
    come out as laying the whole texture gives them. None: all of them.
    Returns {"size", "islands", "by_slot": {slot: count}, "pixels": pixels laid}.
    """
    gray = np.asarray(gray)
    s = int(out_scale)
    mpp_out = mpp / s
    cov, near = _edge_of(gray, lab, s, soft_edges)
    Ho, Wo = cov.shape
    H, W = lab.shape
    prog.stage("islands", "the islands")
    n = len(islands)
    rng = np.random.default_rng([int(seed), 77])
    theta = np.zeros(n + 1)
    off = rng.uniform(0, 1000.0, (n + 1, 2))
    pick = rng.integers(0, 10 ** 6, n + 1)
    for isl in islands:
        theta[isl["id"]] = _main_direction(isl["rings"], W, H)
    look_of = np.full(n + 1, -1, np.int64)             # -1: the default look
    for k, sl in slot_of.items():
        if 0 < int(k) <= n and sl in looks:
            look_of[int(k)] = int(sl)
    tiles = {}

    def tiles_of(key):
        if key not in tiles:
            L = default if key == -1 else looks[key]
            tm = L["tile_m"]
            T = {"tiles": [_Tile(p, tm, mpp_out) for p in L["tiles"]], "ground": bool(L.get("ground")),
                 "drift": float(L.get("drift") or 0.0), "mix": None}
            mx = L.get("mix")
            if mx and L.get("heights"):
                T["heights"] = [_Tile(h, tm, mpp_out, grey=True) for h in L["heights"]]
                T["mix"] = dict(mx, tile=_Tile(mx["tile"], tm, mpp_out), height=_Tile(mx["height"], tm, mpp_out, grey=True),
                                depth_a=float(L.get("depth_m") or 0.0), shift=(0.37 * tm, 0.61 * tm))
            tiles[key] = T
        return tiles[key]

    field = ground_field(Wo * mpp_out, Ho * mpp_out, seed)
    half_w, half_h = Wo * mpp_out / 2, Ho * mpp_out / 2           # the 3D model's x = 0, z = 0
    mix = ground_field(Wo * mpp_out, Ho * mpp_out, int(seed) + 1)
    out = np.zeros((Ho, Wo, 4), np.uint8) if only is None else np.array(Image.open(out_path).convert("RGBA"))
    full = road = None
    if full_path is not None and result_path is not None:
        road = np.asarray(Image.open(result_path).convert("RGB"))
        full = road.copy() if only is None else np.array(Image.open(full_path).convert("RGB"))
    redo = np.zeros(n + 1, bool)
    if only is not None:
        redo[[int(k) for k in only if 0 < int(k) <= n]] = True
    laid = 0
    bg = np.asarray(background, np.float32)
    prog.stage("texture", "laying the materials")
    step = max(1, 1_500_000 // max(1, Wo))
    for r0 in range(0, Ho, step):
        prog.part(r0 / Ho)
        r1 = min(Ho, r0 + step)
        rows = np.arange(r0, r1)
        lab_rows = near[np.minimum(rows // s, H - 1)]                    # (rows, W) at mask size
        L = np.repeat(lab_rows, s, axis=1)[:, :Wo]
        a = 1.0 - cov[r0:r1]
        want = (a > 0.002) & (L > 0)
        if only is not None:
            want &= redo[L]
        if not want.any():
            continue
        laid += int(want.sum())
        yy, xx = np.nonzero(want)
        ids = L[yy, xx]
        X = (xx + 0.5) * mpp_out
        Y = (yy + r0 + 0.5) * mpp_out
        rgb = np.zeros((len(ids), 3), np.float32)
        keys = look_of[ids]
        for key in np.unique(keys):
            sel = keys == key
            T = tiles_of(int(key))
            tl, ground, mx = T["tiles"], T["ground"], T["mix"]
            i = ids[sel]
            c, sn = np.cos(theta[i]), np.sin(theta[i])
            u = c * X[sel] + sn * Y[sel] + off[i, 0]
            v = -sn * X[sel] + c * Y[sel] + off[i, 1]
            hA = None
            if ground and len(tl) > 1:
                # two of its variants, mixed by a smooth random field: no repeat to see
                va, vb = tl[0].sample(u, v), tl[1].sample(u + 37.0, v + 11.0)
                w = np.clip(0.5 + 1.2 * field_at(mix, X[sel], Y[sel]), 0, 1)[:, None]
                col = va * (1 - w) + vb * w
                if mx:
                    hs = T["heights"]
                    hA = (hs[0].sample(u, v) * (1 - w) + hs[1 % len(hs)].sample(u + 37.0, v + 11.0) * w)[:, 0] / 255
            else:
                # one variant per island, as the 3D model lays them
                var = pick[i] % len(tl)
                col = np.zeros((len(i), 3), np.float32)
                hA = np.zeros(len(i), np.float32) if mx else None
                for vk in np.unique(var):
                    m = var == vk
                    col[m] = tl[int(vk)].sample(u[m], v[m])
                    if mx:
                        hA[m] = T["heights"][int(vk) % len(T["heights"])].sample(u[m], v[m])[:, 0] / 255
            if mx or T["drift"] > 0:
                # the ground pattern at these points' world positions (the 3D model's x, z)
                xw, zw = X[sel] - half_w, Y[sel] - half_h
                lin = _lin(col)
                if mx:
                    pn = pattern_at(0, xw, zw, mx["scale"], mx["offset"])
                    inpatch = pn >= mx["thresh"] - 0.12
                    if inpatch.any():
                        ub, vb_ = u[inpatch] + mx["shift"][0], v[inpatch] + mx["shift"][1]
                        hB = mx["height"].sample(ub, vb_)[:, 0] / 255
                        wm = mix_weight(pn[inpatch], mx["thresh"], hA[inpatch], mx["depth_a"], hB, mx["depth_m"])[:, None]
                        lin[inpatch] = lin[inpatch] * (1 - wm) + _lin(mx["tile"].sample(ub, vb_)) * wm
                if T["drift"] > 0:
                    da, db = drift_at(xw, zw)
                    lin = drift_colour(lin, T["drift"], da, db)
                col = _srgb(lin).astype(np.float32)
            amp = GROUND_AMP if ground else PAVED_AMP
            col *= (1.0 + amp * field_at(field, X[sel], Y[sel]))[:, None]
            rgb[sel] = col
        alpha = a[yy, xx]
        out[yy + r0, xx, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        out[yy + r0, xx, 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
        if full is not None:
            f = road[yy + r0, xx].astype(np.float32)
            full[yy + r0, xx] = np.clip(f + (rgb - bg) * alpha[:, None] + 0.5, 0, 255).astype(np.uint8)
    prog.stage("saving", "saving")
    level = 4 if only is None else 1
    if only is None or laid:
        Image.fromarray(out, "RGBA").save(out_path, compress_level=level)
        if full is not None:
            Image.fromarray(full).save(full_path, compress_level=level)
    by = {}
    for k in range(1, n + 1):
        by[int(look_of[k])] = by.get(int(look_of[k]), 0) + 1
    return {"size": [int(Wo), int(Ho)], "islands": n, "by_slot": by, "pixels": laid}
