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
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app import progress as prog

MIN_M2 = 4.0            # smaller areas between roads get no number (the 3D model fills only larger)
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
    """A material tile ready to sample at the texture's scale: blurred to its pixel footprint (wrapping), or its mean."""

    def __init__(self, path, tile_m, mpp_out):
        from scipy.ndimage import fourier_gaussian
        img = np.asarray(Image.open(path).convert("RGB"), np.float32)
        self.px = img.shape[1]
        self.k = self.px / tile_m                       # tile pixels per metre
        foot = mpp_out * self.k                         # tile pixels under one texture pixel
        self.mean = img.reshape(-1, 3).mean(axis=0)
        self.flat = foot > self.px / 3                  # finer than the picture can show: its mean colour
        if not self.flat and foot > 1.0:
            sigma = 0.5 * foot
            img = np.stack([np.fft.ifft2(fourier_gaussian(np.fft.fft2(img[..., c]), sigma)).real
                            for c in range(3)], -1).astype(np.float32)
        self.img = img

    def sample(self, u_m, v_m):
        if self.flat:
            return np.broadcast_to(self.mean, (len(u_m), 3)).copy()
        at = [np.asarray(v_m) * self.k, np.asarray(u_m) * self.k]
        return np.stack([ndi.map_coordinates(self.img[..., c], at, order=1, mode="grid-wrap") for c in range(3)], -1)


def texture(gray, lab, islands, mpp, out_scale, soft_edges, slot_of, looks, default, seed, out_path,
            result_path=None, full_path=None, background=(18, 18, 20)):
    """
    The islands texture: the same size as the road texture (out_scale times the
    mask), each island laid with its material at real size along its main kerb
    direction, the roads transparent (their edge the exact counterpart of the
    road texture's, so the two meet with no gap or overlap). Ground (sand,
    gravel, earth) is mixed from its tile variants and given broad light and
    dark, so it shows no repeat; paving keeps its joints on one grid.

    slot_of: {island id: slot}; looks: {slot: {"tiles": [paths], "tile_m", "ground": bool}};
    default: the same for islands in no slot. With result_path and full_path,
    also writes the road texture with the islands under it.
    Returns {"size", "islands", "by_slot": {slot: count}}.
    """
    from app.generation import prepare_mask
    gray = np.asarray(gray)
    s = int(out_scale)
    mpp_out = mpp / s
    prog.stage("mask", "the road edge at the texture's size")
    prepared, coverage = prepare_mask(gray, s)
    cov = coverage if soft_edges else (prepared > 0).astype(np.float32)
    Ho, Wo = cov.shape
    H, W = lab.shape
    # every pixel's nearest island, so the soft edge under the road's has a colour
    prog.stage("islands", "the islands")
    near = lab
    if (lab == 0).any() and lab.any():
        iy, ix = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        near = lab[iy, ix]
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
            tiles[key] = ([_Tile(p, L["tile_m"], mpp_out) for p in L["tiles"]], bool(L.get("ground")))
        return tiles[key]

    field = ground_field(Wo * mpp_out, Ho * mpp_out, seed)
    mix = ground_field(Wo * mpp_out, Ho * mpp_out, int(seed) + 1)
    out = np.zeros((Ho, Wo, 4), np.uint8)
    full = None
    if full_path is not None and result_path is not None:
        full = np.asarray(Image.open(result_path).convert("RGB")).copy()
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
        if not want.any():
            continue
        yy, xx = np.nonzero(want)
        ids = L[yy, xx]
        X = (xx + 0.5) * mpp_out
        Y = (yy + r0 + 0.5) * mpp_out
        rgb = np.zeros((len(ids), 3), np.float32)
        keys = look_of[ids]
        for key in np.unique(keys):
            sel = keys == key
            tl, ground = tiles_of(int(key))
            i = ids[sel]
            c, sn = np.cos(theta[i]), np.sin(theta[i])
            u = c * X[sel] + sn * Y[sel] + off[i, 0]
            v = -sn * X[sel] + c * Y[sel] + off[i, 1]
            if ground and len(tl) > 1:
                # two of its variants, mixed by a smooth random field: no repeat to see
                va, vb = tl[0].sample(u, v), tl[1].sample(u + 37.0, v + 11.0)
                w = np.clip(0.5 + 1.2 * field_at(mix, X[sel], Y[sel]), 0, 1)[:, None]
                col = va * (1 - w) + vb * w
            else:
                # one variant per island, as the 3D model lays them
                var = pick[i] % len(tl)
                col = np.zeros((len(i), 3), np.float32)
                for vk in np.unique(var):
                    m = var == vk
                    col[m] = tl[int(vk)].sample(u[m], v[m])
            amp = GROUND_AMP if ground else PAVED_AMP
            col *= (1.0 + amp * field_at(field, X[sel], Y[sel]))[:, None]
            rgb[sel] = col
        alpha = a[yy, xx]
        out[yy + r0, xx, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        out[yy + r0, xx, 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
        if full is not None:
            f = full[yy + r0, xx].astype(np.float32)
            full[yy + r0, xx] = np.clip(f + (rgb - bg) * alpha[:, None] + 0.5, 0, 255).astype(np.uint8)
    prog.stage("saving", "saving")
    Image.fromarray(out, "RGBA").save(out_path, compress_level=4)
    if full is not None:
        Image.fromarray(full).save(full_path, compress_level=4)
    by = {}
    for k in range(1, n + 1):
        by[int(look_of[k])] = by.get(int(look_of[k]), 0) + 1
    return {"size": [int(Wo), int(Ho)], "islands": n, "by_slot": by}
