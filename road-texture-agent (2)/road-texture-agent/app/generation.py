"""
Generation.

A new mask comes in, a finished road texture goes out, in three layers:

  1. material  - the road surface, built from the patch libraries the trees
                 learned, using the right library for junctions, kerbs and open
                 road. Patches are placed with random orientation and blended
                 where they overlap, so the source texture never shows through
                 as a repeating grid.

  2. wear      - the isolated noise, laid over the surface at the strength you
                 choose. Drawn from the wear library, so it varies everywhere.

  3. markings  - lane lines along each street, as many as its width calls for
                 (app/lanes.py: none under 6 m, one centre line from 6 m, up to
                 four, and from 22 m a highway with a raised island). Position
                 and direction come from the skeleton, not from the agent:
                 dashes stop a set distance before every junction and each
                 street restarts its pattern, so no dash is ever cut in half at
                 a junction.

The agent decides what things look like. The geometry decides where they go.
"""

import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app import junctions as J
from app import lanes as LN
from app import progress as prog

EDGE_METRES = 1.5
ISLAND_RGB = (178, 175, 168)          # a highway island's concrete top, from above
ISLAND_RIM_RGB = (112, 110, 106)      # and its kerb's face and shadow round it


def load_patches(npz_path, with_masks=False):
    """
    Libraries trained before patch masks existed have none stored. For those,
    off-road pixels are recognised by tone: the black surround is far darker
    than any asphalt, so anything below that line is treated as not road.
    """
    with np.load(npz_path) as z:
        patches = z["patches"]
        masks = z["masks"] if "masks" in z.files else None
    if not with_masks:
        return patches
    if masks is None:
        lum = patches[..., :3].mean(axis=-1)
        masks = lum > 30
    return patches, masks.astype(bool)


def _oriented(patch, rng):
    out = np.rot90(patch, int(rng.integers(0, 4)))
    if rng.integers(0, 2):
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def synth_layer(shape, patches, rng, overlap=0.25, jitter=1.0, masks=None, seam=3.0, need=None):
    """
    Build a full canvas of texture from a patch library.

    Patches are laid on a jittered grid and feathered where they meet, so the
    joins do not show as hard edges. Each placement gets a random orientation.

    need: where the layer is used (a bool map), or None for everywhere. Patches
    that cannot reach it (with a margin of a patch, for filling any spot the
    patches leave empty) are skipped; their random draws are still made, so
    every pixel that is used comes out exactly as if all were laid. On a big
    map that is mostly background this skips most of the work.
    """
    H, W = shape
    s = int(patches.shape[1])
    step = max(2, int(s * (1 - overlap)))
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)

    ramp = np.minimum(np.arange(s), np.arange(s)[::-1]).astype(np.float32) + 1.0
    window = np.outer(ramp, ramp)[:, :, None]
    window /= window.max()
    # Raising the window to a power makes the patch whose centre is nearest win
    # almost outright; two patches only mix in a narrow band in the middle of
    # their overlap, where they are about equally near. Averaging patches
    # everywhere washes their grain out, so this keeps the surface detail while
    # still hiding the joins.
    if seam and seam > 1:
        window = window ** seam

    # each patch in each of its eight orientations, weighted by the window (and by
    # its road mask) once, not again at every placement
    oriented = {}

    def weighted(idx, k, flip):
        key = (idx, k, flip)
        if key not in oriented:
            p = np.rot90(patches[idx], k)
            pm = np.rot90(masks[idx], k) if masks is not None else None
            if flip:
                p = np.fliplr(p)
                pm = np.fliplr(pm) if pm is not None else None
            if p.ndim == 2:
                p = np.dstack([p] * 3)
            p = p[:, :, :3].astype(np.float32)
            # a pixel that was not road in the training photo adds nothing
            wnd = window * pm[:, :, None] if pm is not None else window
            oriented[key] = (p * wnd, wnd)
        return oriented[key]

    reach = None
    if need is not None:
        # the used area grown by a patch: a summed-area table answers, for any
        # placement, whether its square touches it
        grown = ndi.maximum_filter(np.asarray(need, bool), size=2 * (s + 2) + 1)
        reach = np.zeros((H + 1, W + 1), np.int64)
        reach[1:, 1:] = grown.cumsum(0).cumsum(1)

    rows = range(-s, H + s, step)
    for n_row, y in enumerate(rows):
        if n_row % 8 == 0:
            prog.part(n_row / len(rows))
        for x in range(-s, W + s, step):
            j = max(1, int(step * jitter / 3))
            yy = y + int(rng.integers(-j, j + 1))
            xx = x + int(rng.integers(-j, j + 1))
            idx = int(rng.integers(0, len(patches)))
            k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))

            y0, x0 = max(0, yy), max(0, xx)
            y1, x1 = min(H, yy + s), min(W, xx + s)
            if y1 <= y0 or x1 <= x0:
                continue
            if reach is not None and reach[y1, x1] - reach[y0, x1] - reach[y1, x0] + reach[y0, x0] == 0:
                continue
            pw, wnd = weighted(idx, k, flip)
            py0, px0 = y0 - yy, x0 - xx
            acc[y0:y1, x0:x1] += pw[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
            wsum[y0:y1, x0:x1] += wnd[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]

    empty = wsum[:, :, 0] == 0
    wsum[wsum == 0] = 1.0
    out = acc / wsum
    if empty.any() and (~empty).any():
        # a spot every nearby patch left empty takes its nearest filled neighbour
        _, (iy, ix) = ndi.distance_transform_edt(empty, return_indices=True)
        out[empty] = out[iy[empty], ix[empty]]
    return out


def _ordered_path(coords):
    """Put the pixels of one street segment in order from one end to the other."""
    pts = [tuple(c) for c in coords]
    if len(pts) < 2:
        return pts
    remaining = set(pts)
    # start from the pixel furthest from the centre of the segment
    arr = np.array(pts)
    centre = arr.mean(axis=0)
    start = tuple(arr[np.argmax(((arr - centre) ** 2).sum(axis=1))])
    path = [start]
    remaining.discard(start)
    cur = start
    while remaining:
        best, bestd = None, None
        cy, cx = cur
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand = (cy + dy, cx + dx)
                if cand in remaining:
                    d = dy * dy + dx * dx
                    if bestd is None or d < bestd:
                        best, bestd = cand, d
        if best is None:                      # segment is broken: stop here
            break
        path.append(best)
        remaining.discard(best)
        cur = best
    return path


def _smooth(path, window=7):
    """
    Average the centreline before placing dashes.

    The skeleton is one pixel wide and steps diagonally, so dashes laid straight
    on it wobble off the road axis. Averaging a few points together puts them
    back on the line the road actually follows.
    """
    if len(path) < window:
        return path
    arr = np.array(path, float)
    k = np.ones(window) / window
    ys = np.convolve(arr[:, 0], k, mode="same")
    xs = np.convolve(arr[:, 1], k, mode="same")
    half = window // 2
    ys[:half], ys[-half:] = arr[:half, 0], arr[-half:, 0]
    xs[:half], xs[-half:] = arr[:half, 1], arr[-half:, 1]
    return list(zip(ys, xs))


def _stroke(cover, line, half_w, strength=1.0):
    """
    Draw one dash as a smooth stroke.

    Every pixel near the dash gets the distance to the dash's centreline, and
    its coverage is how much of it falls inside the stroke: 1 well inside, 0
    outside, and a fraction across the one-pixel edge. The same idea as the
    road edges, so dashes on slanted roads no longer step. strength: for a line
    narrower than a pixel, the share of the pixel it covers.
    """
    H, W = cover.shape
    pad = half_w + 2
    y0 = int(max(0, np.floor(line[:, 0].min() - pad))); y1 = int(min(H, np.ceil(line[:, 0].max() + pad)))
    x0 = int(max(0, np.floor(line[:, 1].min() - pad))); x1 = int(min(W, np.ceil(line[:, 1].max() + pad)))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    best = np.full(yy.shape, np.inf, np.float32)
    if len(line) == 1:
        segs = [(line[0], line[0])]
    else:
        segs = zip(line[:-1], line[1:])
    for p0, p1 in segs:
        dy, dx = p1[0] - p0[0], p1[1] - p0[1]
        L2 = dy * dy + dx * dx
        if L2 < 1e-9:
            t = np.zeros_like(yy)
        else:
            t = np.clip(((yy - p0[0]) * dy + (xx - p0[1]) * dx) / L2, 0, 1)
        dist = np.hypot(yy - (p0[0] + t * dy), xx - (p0[1] + t * dx))
        np.minimum(best, dist, out=best)
    cov = np.clip(half_w + 0.5 - best, 0.0, 1.0)
    if strength < 1.0:
        cov *= strength
    np.maximum(cover[y0:y1, x0:x1], cov, out=cover[y0:y1, x0:x1])


def _strokes(cover, line, half_w, strength, piece_px):
    """A long line drawn as pieces of about piece_px, so each stroke's box stays small."""
    if len(line) < 2:
        return
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    total = float(d[-1])
    k = max(1, int(math.ceil(total / max(piece_px, 1.0))))
    for i in range(k):
        a, b = total * i / k, total * (i + 1) / k
        inner = line[(d > a) & (d < b)]
        ends = [np.array([np.interp(v, d, line[:, 0]), np.interp(v, d, line[:, 1])], np.float32) for v in (a, b)]
        _stroke(cover, np.vstack([ends[0][None], inner, ends[1][None]]) if len(inner) else np.vstack(ends),
                half_w, strength)


def place_dashes(det, metres_per_pixel, cycle_m, dash_share, width_m, setback_m, rng,
                 align=False, width_ratio=None, nomark=None):
    """
    Lay each street's lane lines (app/lanes.py: how many its width calls for,
    where across it, dashed or solid), stopping short of each junction, and on
    highways its raised island.

    The number of cycles per street is rounded to a whole number and the gaps
    stretched slightly to fit, so each street starts and ends on a full dash.
    Every lane's dashes are side by side, measured along the centreline. A
    street is read the canonical way (lanes.canonical), so its first side is
    the one the 3D model picks too.

    A line narrower than a pixel (15 cm paint on a map at 1 m a pixel) is drawn
    one pixel wide but only as strong as the share of the pixel it covers, the
    way a photo from that height shows it, not widened to a whole pixel.

    Returns the paint's coverage, a report, and the islands' (coverage, top's
    coverage): the top a little narrower, so its rim can be shaded as a kerb.
    """
    road = det["road"]
    H, W = road.shape
    cover = np.zeros((H, W), np.float32)
    isl = np.zeros((H, W), np.float32)
    isl_top = np.zeros((H, W), np.float32)
    seg_lab = det["segments"]
    n = int(seg_lab.max())
    px = max(metres_per_pixel, 1e-6)
    cycle_px = cycle_m / px
    setback_px = setback_m / px
    width_px = width_m / px
    placed = 0
    widths_used, street_m = [], []
    seg_px = det.get("segment_pixels") or J.label_coords(seg_lab, n)
    for s in range(1, n + 1):
        if s % 50 == 0:
            prog.part(s / n)
        coords = seg_px[s]
        if len(coords) < 4:
            continue
        if nomark is not None and nomark[coords[:, 0], coords[:, 1]].mean() > 0.5:
            continue                       # an inner street drawn without markings
        path = _ordered_path(coords)
        if len(path) < 4:
            continue
        # its width, measured along the middle of its centreline (its ends flare into
        # the junctions); the distance field reads half a pixel more than the half width
        pa = np.array(path)
        mid = pa[int(len(pa) * 0.15): max(int(len(pa) * 0.85), int(len(pa) * 0.15) + 1)]
        w_m = 2 * max(0.0, float(np.median(det["dt"][mid[:, 0], mid[:, 1]])) - 0.5) * px
        lay = LN.layout(w_m)
        if not lay["lines"]:
            continue                       # too narrow for a line
        street_m.append(w_m)
        line_w = LN.line_width(w_m, width_ratio, width_m) / px
        half_w, strength = max(0.5, line_w / 2), min(1.0, line_w)
        widths_used.append(line_w)
        # always smooth a little so dashes follow the road axis, not the
        # skeleton's pixel steps; "lines not aligned" smooths more
        path = _smooth(path, window=15 if align else 7)
        if not LN.canonical(path[0][::-1], path[-1][::-1]):
            path = path[::-1]
        # arc length along the street
        d = [0.0]
        for a, b in zip(path, path[1:]):
            d.append(d[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        total = d[-1]
        usable = total - 2 * setback_px
        if usable <= cycle_px * 0.6:
            continue                       # too short for even one dash
        cycles = max(1, int(round(usable / cycle_px)))
        step = usable / cycles             # gaps stretch a little to fit
        dash_len = step * dash_share

        pts = np.array(path, np.float32)                    # (y, x)
        dist = np.array(d, np.float32)
        # the sideways direction: (-ty, tx) in x, y, from a steadier copy of the line
        tg = np.gradient(np.array(_smooth(path, window=min(31, len(path) // 2 * 2 - 1)) if len(path) > 8 else path,
                                  np.float64), axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nrm = np.column_stack([tg[:, 1], -tg[:, 0]]).astype(np.float32)   # (y, x) of (-ty, tx)
        run = (dist >= setback_px) & (dist <= total - setback_px)

        def along(off_px, a, b):
            """The line off_px to the side, from a to b along the centreline."""
            q = pts + nrm * off_px
            inner = q[(dist > a) & (dist < b)]
            ends = [np.array([np.interp(v, dist, q[:, 0]), np.interp(v, dist, q[:, 1])], np.float32) for v in (a, b)]
            return np.vstack([ends[0][None], inner, ends[1][None]]) if len(inner) else np.vstack(ends)

        for x_m, kind in lay["lines"]:
            off = x_m / px
            if kind == "solid":
                _strokes(cover, along(off, setback_px, total - setback_px), half_w, strength, cycle_px)
                continue
            for c in range(cycles):
                a = setback_px + c * step
                _stroke(cover, along(off, a, a + dash_len), half_w, strength)
                placed += 1
        if lay["island"] and run.any():
            x0, x1 = lay["island"]
            spine = along((x0 + x1) / 2 / px, setback_px, total - setback_px)
            hw_isl = (x1 - x0) / 2 / px
            _strokes(isl, spine, hw_isl, min(1.0, 2 * hw_isl), cycle_px)
            _strokes(isl_top, spine, max(0.5, hw_isl - 0.12 / px), min(1.0, 2 * hw_isl), cycle_px)

    rc = road_cov if (road_cov := det.get("coverage")) is not None else road
    cover *= rc
    isl *= rc
    isl_top = np.minimum(isl_top, isl)
    width_shown = float(np.median(widths_used)) if widths_used else width_px
    warning = None
    if cycle_px < 6:
        # a dash and its gap need a few pixels each, or they merge into one line
        warning = (f"a dash cycle of {cycle_m:g} m is only {cycle_px:.1f} px at {metres_per_pixel:g} m a pixel: "
                   f"the dashes run together into a solid line (and there are thousands of them, which is slow). "
                   f"Real lane lines repeat every 9 to 12 m ({9 / px:.0f} to {12 / px:.0f} px here)")
    return cover, {"dashes": placed, "cycle_px": round(cycle_px, 1),
                   "width_px": round(width_shown, 2),
                   "width_mode": "learned" if width_ratio else "fixed",
                   "width_ratio": width_ratio, "faint": width_shown < 1.0, "warning": warning,
                   "lanes": LN.summary(street_m)}, (isl, isl_top)


def group_map(det, groups):
    """One array the review step can look up directly: 0 none, 1 junction, 2 kerb, 3 open."""
    import numpy as _np
    m = _np.zeros(det["road"].shape, _np.uint8)
    m[groups["open"]] = 3
    m[groups["edge"]] = 2
    m[groups["junction"]] = 1
    return m


def junction_id_map(det):
    import numpy as _np
    road = det["road"]
    ids = _np.zeros(road.shape, _np.int32)
    for j in det["junctions"]:
        win, inside = J.disc(road.shape, j["cy"], j["cx"], j["r"])
        ids[win][inside & road[win]] = j["id"]
    return ids


def groups_for(det, metres_per_pixel):
    """The same three groups training used, so generation asks the same questions."""
    road = det["road"]
    H, W = road.shape
    in_junction = np.zeros((H, W), bool)
    for j in det["junctions"]:
        if j["type"] == "interchange (flagged)":
            continue
        win, inside = J.disc((H, W), j["cy"], j["cx"], j["r"])
        in_junction[win] |= inside
    in_junction &= road

    widths = 2 * det["dt"][det["skeleton"].astype(bool)]
    road_width = float(np.median(widths)) if widths.size else 8.0
    band_m = min(EDGE_METRES, max(0.05, road_width * metres_per_pixel * 0.25))
    edge = road & ~in_junction & (det["dt"] * metres_per_pixel <= band_m)
    return {"junction": in_junction, "edge": edge,
            "open": road & ~in_junction & ~edge}, road_width, band_m


def prepare_mask(gray: np.ndarray, scale: int, smooth: float = 1.0):
    """
    Turn the pixel mask into a smooth road shape, at any output size.

    A pixel mask stores the road edge as steps. Blurring and thresholding it
    softens the steps but keeps them, and enlarging makes each step bigger.
    Instead, the mask becomes a signed distance field: every pixel holds its
    distance to the road edge, positive inside, negative outside. Distance
    changes smoothly even where the pixels step, so it can be smoothed and
    enlarged without stairs, and the edge is drawn exactly where it is zero.
    The same technique keeps text sharp when fonts are scaled.

    Cleaning happens first, at the mask's own size, because the divider and
    speck rules are written in pixels.

    Returns the road mask at output size and a coverage map from 0 to 1 that
    fades the edge into the background over one output pixel.
    """
    road, _ = J.clean_mask(gray)
    inside = ndi.distance_transform_edt(road)
    outside = ndi.distance_transform_edt(~road)
    # pixel centres sit half a pixel from the true edge on either side
    sdf = np.where(road, inside - 0.5, -(outside - 0.5)).astype(np.float32)
    if smooth > 0:
        sdf = ndi.gaussian_filter(sdf, smooth)            # rounds off the stairs
    if scale > 1:
        sdf = ndi.zoom(sdf, scale, order=1) * scale        # distances now in output pixels
    coverage = np.clip(sdf + 0.5, 0.0, 1.0).astype(np.float32)
    mask = (sdf >= 0).astype(np.uint8) * 255
    return mask, coverage


def scale_patches(patches: np.ndarray, scale: float, masks: np.ndarray = None):
    """
    Resize patches by any factor, so texture keeps its real-world size.

    A factor below 1 shrinks them: a photo taken much closer than the output
    has grain too fine to show, and shrinking averages it out the way distance
    would. A factor above 1 enlarges them.
    """
    if abs(scale - 1.0) < 1e-3:
        return (patches, masks) if masks is not None else patches
    s = max(3, int(round(patches.shape[1] * scale)))
    out = np.empty((len(patches), s, s, patches.shape[3]), patches.dtype)
    method = Image.LANCZOS if s >= patches.shape[1] else Image.BOX   # BOX averages when shrinking
    for i, p in enumerate(patches):
        out[i] = np.array(Image.fromarray(p.astype(np.uint8)).resize((s, s), method))
    if masks is None:
        return out
    mout = np.empty((len(masks), s, s), bool)
    for i, m in enumerate(masks):
        mout[i] = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((s, s), Image.NEAREST)) > 127
    return out, mout


def _at_size(m, scale, shape):
    """A mask-sized bool map at the output size (or None)."""
    if m is None:
        return None
    big = np.repeat(np.repeat(np.asarray(m, bool), scale, 0), scale, 1)
    out = np.zeros(shape, bool)
    h, w = min(shape[0], big.shape[0]), min(shape[1], big.shape[1])
    out[:h, :w] = big[:h, :w]
    return out


def lay_material(street, trained, road, mpp):
    """
    A scanned material (its colour picture, street["colour"], covering
    street["size_m"] metres) laid over the canvas at its real size, in place of
    the trained material. At the canvas's scale a pixel covers many of the
    material's stones, so each pixel takes the average of the ground it covers.
    The trained material's lighter and darker areas (over 2 m and more) are kept
    on it, so the streets keep their patches and stains rather than turning
    flat; with street["match"], its colour takes on the trained material's mean
    colour on the road.
    """
    H, W = road.shape
    img = Image.fromarray(np.asarray(street["colour"], np.uint8))
    sw, sh = [float(v) for v in street["size_m"]]
    # two texture pixels per canvas pixel, averaged down (box filter): the picture
    # stays one repeat of the material, so it still joins itself
    nx = int(np.clip(round(2 * sw / mpp), 4, img.width))
    ny = int(np.clip(round(2 * sh / mpp), 4, img.height))
    tex = np.asarray(img.resize((nx, ny), Image.BOX), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = ((xx + 0.5) * mpp / sw * nx - 0.5).ravel()
    cy = ((yy + 0.5) * mpp / sh * ny - 0.5).ravel()
    lay = np.stack([ndi.map_coordinates(tex[..., c], [cy, cx], order=1, mode="grid-wrap").reshape(H, W)
                    for c in range(3)], axis=-1)
    if street.get("match"):
        gain = trained[road].mean(axis=0) / np.maximum(lay[road].mean(axis=0), 1.0)
        lay = lay * np.clip(gain, 0.25, 4.0)
    lum = trained.mean(axis=2)
    broad = ndi.gaussian_filter(lum, max(1.0, 2.0 / mpp))
    rel = broad / max(float(broad[road].mean()), 1.0)
    return lay * np.clip(rel, 0.5, 1.6)[:, :, None]


def generate(mask_path, libraries, params, out_paths, map_path=None):
    """
    libraries: {"material": {group: npz path}, "wear": {group: npz path}}
    params:    metres_per_pixel, wear, seed, cycle_m, dash_share, width_m, setback_m
    out_paths: {"result","material","wear","markings"} -> file paths
    """
    rng = np.random.default_rng(int(params.get("seed", 7)))
    mpp = float(params.get("metres_per_pixel", 0.25))
    # quality 1 to 3: more overlap and less jitter means softer joins between
    # patches, at the cost of time. Nothing else about the result changes.
    quality = int(np.clip(int(params.get("quality", 1)), 1, 3))
    overlap = {1: 0.25, 2: 0.45, 3: 0.6}[quality]
    jitter = {1: 1.0, 2: 0.6, 3: 0.35}[quality]
    align_lines = bool(params.get("align_lines", False))
    scale = int(params.get("output_scale", 1))
    scale = scale if scale in (1, 2, 4) else 1
    soft_edges = bool(params.get("soft_edges", True))
    mpp_out = mpp / scale                     # each output pixel covers less ground

    gray = np.array(Image.open(mask_path).convert("L"))
    prog.stage("mask", "cleaning and smoothing the mask")
    prepared, coverage = prepare_mask(gray, scale)
    prog.stage("junctions", "finding the junctions and streets")
    det = J.detect(prepared)
    det["coverage"] = coverage
    road = det["road"]
    H, W = road.shape
    prog.stage("areas", "junction, kerb and open-road areas")
    groups, road_width, band_m = groups_for(det, mpp_out)

    # ---- 1. material ----
    # each part is painted over the whole canvas, then the parts are blended
    # with soft weights, so junction, kerb and open road fade into each other
    # over a few pixels instead of meeting at a hard line
    acc = np.zeros((H, W, 3), np.float32)
    wtot = np.zeros((H, W), np.float32)
    feather = 1.5 * scale
    used = {}
    size_ratio = {}
    for name, m in groups.items():
        prog.stage("material_" + name, {"open": "material patches: open road", "edge": "material patches: kerb band",
                                     "junction": "material patches: junctions"}.get(name, "material patches"))
        if not m.any():
            continue
        npz = libraries["material"].get(name) or libraries["material"].get("open")
        if not npz:
            continue
        patches, pmasks = load_patches(npz, with_masks=True)
        trained_mpp = (libraries.get("scales") or {}).get(name)
        ratio = (trained_mpp / mpp_out) if trained_mpp else float(scale)
        patches, pmasks = scale_patches(patches, ratio, pmasks)
        size_ratio[name] = round(ratio, 3)
        # the layer only counts where its blend weight is above zero: within the
        # weight's reach (the Gaussian's 4 sigma) of this part of the road
        w = ndi.gaussian_filter(m.astype(np.float32), feather)
        layer = synth_layer((H, W), patches, rng, overlap, jitter, pmasks, need=w > 0)
        acc += layer * w[:, :, None]
        wtot += w
        used[name] = {"patches": int(len(patches)), "size_px": int(patches.shape[1])}
    material = acc / np.maximum(wtot, 1e-6)[:, :, None]

    # ---- surface grain: turn the fine detail inside the material up or down.
    # The material is split into its broad tone (a blur) and its fine grain
    # (what the blur removed). Only the grain is scaled, so the road keeps its
    # colour and its larger light and dark areas while getting cleaner or dirtier.
    grain = float(params.get("grain", 1.0))
    if abs(grain - 1.0) > 1e-3 and road.any():
        broad = ndi.gaussian_filter(material, sigma=(max(1.0, 1.2 * scale), max(1.0, 1.2 * scale), 0))
        material = broad + (material - broad) * grain

    # ---- match the colour of the priming material, keeping the grain
    target = params.get("match_tone")
    if target is not None and road.any():
        lum = material.mean(axis=2)
        shift = float(target) - float(lum[road].mean())
        material = material + shift

    # ---- a scanned material for the streets (app/scans) instead of the trained one
    street = params.get("street_material")
    if street is not None and road.any():
        material = lay_material(street, material, road, mpp_out)

    # ---- 2. wear ----
    prog.stage("wear", "wear patches")
    wear_strength = float(params.get("wear", 0.5))
    wear_map = np.zeros((H, W), np.float32)
    wear_npz = libraries["wear"].get("open") or libraries["wear"].get("junction")
    if wear_npz and wear_strength > 0:
        wp, wm = load_patches(wear_npz, with_masks=True)
        wtrained = (libraries.get("scales") or {}).get("open")
        wp, wm = scale_patches(wp, (wtrained / mpp_out) if wtrained else float(scale), wm)
        wl = synth_layer((H, W), wp, rng, overlap, jitter, wm, need=road).mean(axis=2)
        lo, hi = np.percentile(wl[road], [5, 95]) if road.any() else (0, 1)
        wear_map = np.clip((wl - lo) / max(hi - lo, 1e-6), 0, 1)

    worn = material * (1.0 - 0.45 * wear_strength * wear_map[:, :, None])

    # ---- 3. markings ----
    prog.stage("markings", "markings")
    paint_rgb = np.array(params.get("paint_colour", [235, 232, 222]), np.float32)
    paint, dash_info, (island, island_top) = place_dashes(
        det, mpp_out,
        cycle_m=float(params.get("cycle_m", 9.0)),
        dash_share=float(params.get("dash_share", 0.6)),
        width_m=float(params.get("width_m", 0.15)),
        setback_m=float(params.get("setback_m", 2.0)),
        rng=rng, align=align_lines, width_ratio=params.get("width_ratio"),
        nomark=_at_size(params.get("nomark"), scale, road.shape))

    prog.stage("finishing", "blending and saving the pictures")
    result = worn.copy()
    if paint.any():
        # paint is worn too, so it is not a flat block of white; the coverage
        # blends it into the asphalt over the stroke's one-pixel edge
        a = paint[:, :, None]
        painted = paint_rgb[None, None, :] * (1.0 - 0.25 * wear_strength * wear_map)[:, :, None]
        result = result * (1 - a) + painted * a
    if island.any():
        # the highways' raised island: a concrete top with its grain, ringed by its
        # kerb's darker face and shadow, as it looks from above
        lum = worn.mean(axis=2, keepdims=True)
        grain = 0.9 + 0.1 * lum / max(float(lum[road].mean()) if road.any() else 1.0, 1e-6)
        top = np.array(ISLAND_RGB, np.float32)[None, None, :] * grain
        rim = np.array(ISLAND_RIM_RGB, np.float32)[None, None, :] * grain
        a_top, a_rim = island_top[:, :, None], np.clip(island - island_top, 0, 1)[:, :, None]
        result = result * (1 - a_top - a_rim) + top * a_top + rim * a_rim

    background = np.array([18, 18, 20], np.float32)
    # the edge ring just outside the road has partial coverage but no texture:
    # give each of those pixels the colour of the nearest road pixel
    if soft_edges and road.any():
        _, (iy, ix) = ndi.distance_transform_edt(~road, return_indices=True)
    cov = coverage if soft_edges else road.astype(np.float32)

    def save(arr, path, mask=None, soft=True):
        a = arr if arr.ndim == 3 else np.dstack([arr] * 3)
        a = a.astype(np.float32)
        if soft and soft_edges and mask is None:
            filled = a[iy, ix]                      # road colour carried to the edge ring
            alpha = cov[:, :, None]
            img = filled * alpha + background * (1 - alpha)
        else:
            m = road if mask is None else mask
            img = np.empty((H, W, 3), np.float32); img[...] = background
            img[m] = a[m]
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)

    save(result, out_paths["result"])
    save(material, out_paths["material"])
    save(np.dstack([wear_map * 255] * 3), out_paths["wear"], soft=False)
    mark = paint[:, :, None] * paint_rgb[None, None, :]
    if island.any():
        # the islands with the markings: neither is asphalt, so the 3D model's
        # tone (variation_factor) leaves both out
        mark = np.maximum(mark, island[:, :, None] * np.array(ISLAND_RGB, np.float32)[None, None, :])
    save(mark, out_paths["markings"], soft=False)

    if map_path is not None:
        np.savez_compressed(
            map_path,
            groups=group_map(det, groups),
            junction_ids=junction_id_map(det),
            edge_m=(det["dt"] * mpp_out).astype(np.float32),
            junction_types=np.array([f"{j['id']}:{j['type']}" for j in det["junctions"]]),
            metres_per_pixel=np.array([mpp_out], np.float32),
        )

    return {
        "ok": True,
        "size": [int(W), int(H)],
        "road_px": int(road.sum()),
        "road_width_px": round(road_width, 1),
        "road_width_m": round(road_width * mpp_out, 2),
        "output_scale": scale,
        "patch_size_ratio": size_ratio,
        "grain": round(float(params.get("grain", 1.0)), 2),
        "matched_tone": params.get("match_tone"),
        "soft_edges": soft_edges,
        "kerb_band_m": round(band_m, 2),
        "junctions": det["summary"]["junctions"],
        "by_type": det["summary"]["by_type"],
        "groups": {k: {"share": round(float(v.sum() / max(1, road.sum())), 4),
                       "library": used.get(k)} for k, v in groups.items()},
        "markings": dash_info,
        "wear_strength": wear_strength,
        "street_material": (street or {}).get("name"),
        "quality": quality,
        "align_lines": align_lines,
    }
