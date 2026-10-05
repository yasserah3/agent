"""
Seamless material tiles for the 3D road.

A tile is a small square of road surface that repeats along every street. It
has to be sharp up close, so its detail cannot come from the training pairs,
which are photographed from far above at around a metre per pixel. It comes
from the priming close-up instead, the "material with noise" photo, which shows
the surface itself. The training pairs give each part of the road its tone and
contrast, so junction, kerb and open road still differ.

Built to hide repetition from the start:
  - the tile wraps around seamlessly, so there is no line where copies meet;
  - large blotches are removed from it, since a memorable feature is what gives
    repetition away; large-scale variation comes from a separate layer later;
  - extreme dark and bright spots are softened for the same reason;
  - several variants are made, so neighbouring streets need not share one.

Nothing here depends on a city. Tiles are rebuilt only when what the agent has
learned changes.
"""

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app import analysis as A

PARTS = ("open", "edge", "junction")


def _seamless_synth(src, size, patch, rng, count=400, seam=3.0, quarter_turns=True, turned=None):
    """
    Build a tile on a torus: a patch that runs off one side continues on the
    opposite side, so the finished tile joins itself on every edge.

    src may have any number of channels (colour, and a material's bump and
    roughness with it, all placed together). quarter_turns: patches may turn
    by quarter turns; False keeps them to half turns (a surface with a
    direction, like brushed concrete). turned(patch, k, flipped): fixes a
    patch's channels that are directions (a bump map's slopes) after it has
    been turned k quarter turns and maybe flipped left to right.
    """
    h, w = src.shape[:2]
    patch = int(min(patch, h // 2, w // 2, size // 2))
    acc = np.zeros((size, size, src.shape[2]), np.float64)
    wsum = np.zeros((size, size, 1), np.float64)
    ramp = np.minimum(np.arange(patch), np.arange(patch)[::-1]).astype(np.float64) + 1.0
    window = np.outer(ramp, ramp)
    window = (window / window.max()) ** seam           # blend only in the middle of overlaps
    window = window[:, :, None]

    step = max(2, int(patch * 0.75))
    # the random nudge must stay under half the overlap, or two neighbours
    # nudged apart leave an uncovered gap that comes out as a dark line
    jit = max(0, (patch - step) // 2 - 1)
    for y in range(0, size, step):
        for x in range(0, size, step):
            yy = y + int(rng.integers(-jit, jit + 1))
            xx = x + int(rng.integers(-jit, jit + 1))
            sy = int(rng.integers(0, h - patch)); sx = int(rng.integers(0, w - patch))
            p = src[sy:sy + patch, sx:sx + patch].astype(np.float64)
            k = int(rng.integers(0, 4))
            if not quarter_turns:
                k = 2 * (k % 2)
            p = np.rot90(p, k)
            flipped = bool(rng.integers(0, 2))
            if flipped:
                p = np.fliplr(p)
            if turned is not None:
                p = turned(p, k, flipped)
            ys = (yy + np.arange(patch)) % size
            xs = (xx + np.arange(patch)) % size
            acc[np.ix_(ys, xs)] += p * window
            wsum[np.ix_(ys, xs)] += window
    empty = wsum[:, :, 0] < 1e-9
    wsum[wsum < 1e-9] = 1.0
    out = acc / wsum
    if empty.any():
        # any pixel still uncovered takes its nearest covered neighbour
        _, (iy, ix) = ndi.distance_transform_edt(empty, return_indices=True)
        out[empty] = out[iy[empty], ix[empty]]
    return out


def _calm(tile, size):
    """
    Remove what would make repetition visible: large blotches and extreme spots.
    Fine grain is kept untouched. Filters wrap around, so the tile stays seamless.
    """
    t = tile.astype(np.float64)
    broad = ndi.gaussian_filter(t, sigma=(size / 12, size / 12, 0), mode="wrap")
    mean = t.reshape(-1, 3).mean(axis=0)
    t = t - broad + mean                                  # flatten large-scale variation
    lum = t.mean(axis=2)
    mu, sd = lum.mean(), lum.std() + 1e-6
    z = (lum - mu) / sd
    soft = np.tanh(z / 2.5) * 2.5                         # soften anything beyond about 2.5 sd
    t = t + ((soft - z) * sd)[:, :, None]
    return t


def build_tile(src_rgb, src_width_m, tile_m, px, tone=None, contrast_ratio=1.0, seed=1):
    """
    src_rgb:        the priming close-up, as an array
    src_width_m:    how wide that photo is in real life
    tile_m, px:     the tile's real size and its pixel size
    tone:           target mean brightness for this part of the road, from training
    contrast_ratio: this part's contrast relative to open road, from training
    """
    rng = np.random.default_rng(seed)
    h, w = src_rgb.shape[:2]
    src_mpp = src_width_m / w
    tile_mpp = tile_m / px
    k = src_mpp / tile_mpp                                # resize the photo to the tile's scale
    if abs(k - 1) > 0.02:
        nw, nh = max(8, int(round(w * k))), max(8, int(round(h * k)))
        src = np.array(Image.fromarray(src_rgb).resize(
            (nw, nh), Image.LANCZOS if k > 1 else Image.BOX)).astype(np.float64)
    else:
        src = src_rgb.astype(np.float64)

    per = A.periodicity(src.mean(axis=2))
    patch = px // 8
    if per.get("period_px") and patch % per["period_px"] == 0:
        patch += max(1, per["period_px"] // 2)            # keep patch edges off the photo's own grid

    tile = _seamless_synth(src, px, patch, rng)
    tile = _calm(tile, px)

    # tone and contrast of this part of the road, keeping the grain's own detail
    mean = tile.reshape(-1, 3).mean(axis=0)
    detail = tile - mean
    detail *= float(np.clip(contrast_ratio, 0.4, 2.0))
    if tone is not None:
        mean = mean - mean.mean() + float(tone)
    tile = np.clip(mean + detail, 0, 255).astype(np.uint8)
    return tile, {"source_scale": round(k, 3), "patch_px": int(patch),
                  "source_repeat_px": per.get("period_px")}


def seam_score(tile):
    """
    How visible the joins are when the tile repeats: the difference across the
    wrap-around edge, divided by the typical difference between neighbours
    inside. About 1 means no visible seam.
    """
    t = tile.astype(np.float64).mean(axis=2)
    inside = (np.abs(np.diff(t, axis=1)).mean() + np.abs(np.diff(t, axis=0)).mean()) / 2
    across = (np.abs(t[:, 0] - t[:, -1]).mean() + np.abs(t[0, :] - t[-1, :]).mean()) / 2
    return round(float(across / max(inside, 1e-6)), 3)


def noticeability(tile):
    """
    How easy the repetition would be to spot.

    An exact tile always repeats perfectly, so the question is whether there is
    anything memorable to see repeating. That is medium-sized features, blotches
    and spots, compared with the fine grain. Low means an even surface where
    copies are hard to tell apart.
    """
    t = tile.astype(np.float64).mean(axis=2)
    size = t.shape[0]
    medium = ndi.gaussian_filter(t, size / 40, mode="wrap")
    fine = t - ndi.gaussian_filter(t, 1.5, mode="wrap")
    return round(float(medium.std() / max(fine.std(), 1e-6)), 3)


def build_paving_tile(src_rgb, src_width_m, tile_m, px, seed=1):
    """
    A sidewalk tile from a close-up of paving.

    Paving usually has a regular grid of slabs or bricks. Mixing random patches,
    as the asphalt tile does, would scramble that grid into jumbled slabs. So the
    photo is checked for repetition first (the far-pairs method). If it repeats,
    the tile is cut to a whole number of repeats in each direction, so the grid
    stays aligned and continues across the join. If it does not, the asphalt
    method is used.
    """
    lum = src_rgb.astype(np.float64).mean(axis=2)
    per = A.periodicity(lum)
    px_x = (per.get("along_x") or {}).get("period_px")
    px_y = (per.get("along_y") or {}).get("period_px")
    h, w = lum.shape
    src_mpp = src_width_m / w
    if px_x and px_y:
        # whole repeats that come closest to the tile's real size
        nx = max(1, int(round(tile_m / (px_x * src_mpp))))
        ny = max(1, int(round(tile_m / (px_y * src_mpp))))
        cw, ch = nx * px_x, ny * px_y
        if cw <= w and ch <= h:
            # start the cut on a joint line, so the tile's edge falls between
            # slabs, where the grid itself hides the join: nothing needs blending
            def joint_phase(profile, period):
                d = profile - ndi.uniform_filter1d(profile, period, mode="wrap")
                one = np.abs(d[:period])
                return int(np.argmax(one))
            phx = joint_phase(lum.mean(axis=0), px_x)
            phy = joint_phase(lum.mean(axis=1), px_y)
            rng = np.random.default_rng(seed)
            kx = int(rng.integers(0, max(1, (w - phx - cw) // px_x + 1)))
            ky = int(rng.integers(0, max(1, (h - phy - ch) // px_y + 1)))
            x0, y0 = phx + kx * px_x, phy + ky * px_y
            if x0 + cw > w or y0 + ch > h:
                x0, y0 = phx, phy
            crop = src_rgb[y0:y0 + ch, x0:x0 + cw]
            tile = np.array(Image.fromarray(crop).resize((px, px), Image.LANCZOS)).astype(np.float64)
            real = (cw * src_mpp, ch * src_mpp)
            return np.clip(tile, 0, 255).astype(np.uint8), {
                "method": "pattern", "repeats": [nx, ny], "period_px": [int(px_x), int(px_y)],
                "covers_m": [round(real[0], 2), round(real[1], 2)]}
    tile, info = build_tile(src_rgb, src_width_m, tile_m, px, seed=seed)
    return tile, {"method": "mixed patches", **info, "covers_m": [tile_m, tile_m]}


def concrete_placeholder(asphalt_rgb, src_width_m, tile_m, px, tone=165.0, seed=1):
    """Until a paving photo is given: light concrete made from the asphalt close-up."""
    grey = np.repeat(asphalt_rgb.astype(np.float64).mean(axis=2, keepdims=True), 3, axis=2)
    tile, info = build_tile(grey.astype(np.uint8), src_width_m, tile_m, px, tone=tone,
                            contrast_ratio=0.6, seed=seed)
    return tile, {"method": "placeholder concrete", **info, "covers_m": [tile_m, tile_m]}
