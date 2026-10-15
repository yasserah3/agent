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
YELLOW_RGB = (232, 172, 36)           # the guide's yellow (app/roadmarks.py)
# a dashed line shows as one line when its gaps are narrower than a pixel, or its cycle
# shorter than two (a pattern finer than two pixels cannot show in pixels): it is then drawn
# whole, not dash by dash (a 9 m cycle at 6 m a pixel: 1.5 px, thousands of dashes per km)
MERGE_GAP_PX = 1.0
MERGE_CYCLE_PX = 2.0
STROKE_PIECE_PX = 24.0                # a long line is drawn in pieces about this long


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


def street_paths(det, metres_per_pixel, nomark=None):
    """
    Every street's ordered centreline and width, {label: (path, width_m)}:
    path its skeleton pixels (y, x) from one end to the other, the width
    measured along the middle of it (its ends flare into the junctions). Streets
    under nomark (inner streets drawn without markings) are left out.
    """
    seg_lab = det["segments"]
    n = int(seg_lab.max())
    px = max(metres_per_pixel, 1e-6)
    seg_px = det.get("segment_pixels") or J.label_coords(seg_lab, n)
    found = {}
    for s in range(1, n + 1):
        coords = seg_px[s]
        if len(coords) < 4:
            continue
        if nomark is not None and nomark[coords[:, 0], coords[:, 1]].mean() > 0.5:
            continue                       # an inner street drawn without markings
        path = _ordered_path(coords)
        if len(path) < 4:
            continue
        # the distance field reads half a pixel more than the half width
        pa = np.array(path)
        mid = pa[int(len(pa) * 0.15): max(int(len(pa) * 0.85), int(len(pa) * 0.15) + 1)]
        found[s] = (path, 2 * max(0.0, float(np.median(det["dt"][mid[:, 0], mid[:, 1]])) - 0.5) * px)
    return found


def with_chains(found, ends, metres_per_pixel):
    """
    found ({label: (path, width_m)}) with the highways' whole centrelines (in
    the lines' ends: app/highways.py) as streets of their own, numbered from
    LN.CHAIN_ID (any kept from before replaced): lay_lanes lays a highway's
    lines along them, and the Lanes tool's choices on a highway are matched to
    them.
    """
    chains = LN.chains_of(ends)
    out = {k: v for k, v in found.items() if k < LN.CHAIN_ID}
    if not chains:
        return out
    # a highway's own segments are left out: its centreline lays its lines, takes its lane choices
    cidx = LN.ChainIndex(chains, max(metres_per_pixel, 1e-6))
    for k, (path, w) in list(out.items()):
        if cidx.covers_path(np.asarray(path, float)[:, ::-1]):
            del out[k]
    for k, c in enumerate(chains):
        if len(c["pts"]) >= 2:
            out[LN.CHAIN_ID + k] = ([(float(y), float(x)) for x, y in c["pts"]], float(np.median(c["w"])))
    return out


def lane_choices(found, lane_picks, metres_per_pixel):
    """The Lanes tool's choices ([{"pt": [x, y] in this picture's pixels, "sides"}]) by street: {label: (a, b)}."""
    if not lane_picks or not found:
        return {}
    px = max(metres_per_pixel, 1e-6)
    labs = np.concatenate([np.full(len(v[0]), k) for k, v in found.items()])
    pts = np.concatenate([np.array(v[0], float)[:, ::-1] for v in found.values()])
    return LN.match_picks(lane_picks, pts, labs, lambda k, d: d <= found[k][1] / 2 / px + 1.0 / px)


def _area_cover(area, shape, origin):
    """How much of each pixel of a canvas (shape, its corner at origin) the markings area covers: 0 to 1."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = LN.area_outside(area, xx + float(origin[1]), yy + float(origin[0]))
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def lay_lanes(shape, found, chosen, metres_per_pixel, cfg, align=False, origin=(0, 0), area=None, ends=None):
    """
    Draw the lane lines and islands of the streets in found ({label: (path,
    width_m)}), each with the lanes chosen for it (chosen: {label: (a, b)}, else
    even), on canvases of shape whose top left corner is origin (y, x) in the
    whole picture: a window, when only a few streets are laid again (the Lanes
    tool's quick update). cfg: the lines' settings (app/roadmarks.py's
    lines_config or legacy_config: each kind of line's dash, gap, width and
    colour). area: the markings area (app/lanes.py, in this picture's pixels):
    a dash is laid only when its middle lies inside, a solid line only as far as
    it is inside (the islands everywhere). ends: where junction markings ask the
    lines to stop (app/lanes.py's end_setbacks, this picture's pixels); elsewhere
    they stop cfg's setback short of each end. Returns the white paint's
    coverage, the islands' coverage and their tops', and {"placed", "widths",
    "layouts", "streets", "yellow": the yellow paint's coverage}.
    """
    from app import roadmarks as RM
    H, W = shape
    cover = np.zeros((H, W), np.float32)
    yellow = np.zeros((H, W), np.float32)
    isl = np.zeros((H, W), np.float32)
    isl_top = np.zeros((H, W), np.float32)
    px = max(metres_per_pixel, 1e-6)
    default_px = cfg["setback_m"] / px
    oy, ox = float(origin[0]), float(origin[1])
    # solid lines clipped to the area at the end
    solid = np.zeros((H, W), np.float32) if area else cover
    solid_y = np.zeros((H, W), np.float32) if area else yellow
    ends_here = ends or None                      # the whole picture's pixels, as the streets' paths are
    placed = 0
    widths_used, layouts, streets = [], [], []
    # the highways' whole centrelines (in found as streets of their own, numbered from
    # LN.CHAIN_ID, their own segments left out: with_chains): their lines are laid along them
    chains = LN.chains_of(ends_here)
    reservations = LN.ReservationIndex(ends_here)
    last = max([k for k in found if k < LN.CHAIN_ID], default=1)
    for s, (path, w_m) in found.items():
        if s % 50 == 0 and s < LN.CHAIN_ID:
            prog.part(s / last)
        chain = chains[s - LN.CHAIN_ID] if s >= LN.CHAIN_ID and s - LN.CHAIN_ID < len(chains) else None
        lay = LN.layout(w_m, chosen.get(s), rules=cfg["rules"], edge_inset=cfg.get("edge_inset"), hw=cfg.get("highway"),
                        group=LN.centre_group(np.asarray(path)[:, ::-1], px, cfg, w_m, on_chain=s >= LN.CHAIN_ID))
        layouts.append(lay)
        line = np.array(path[::-1] if not LN.canonical(path[0][::-1], path[-1][::-1]) else path, float)[:, ::-1]
        streets.append({"id": int(s), "line": line, "width_m": round(w_m, 2), "limit": lay["limit"],
                        "sides": list(lay["sides"]), "even": lay["even"], "cut": lay["cut"]})
        if not lay["lines"] or not LN.marking_eligible(w_m, cfg):
            continue                       # too narrow for a line
        main_cycle, _ = RM.main_cycle(cfg, lay)
        cycle_px = main_cycle / px
        piece_px = max(cycle_px, STROKE_PIECE_PX)
        styles = {k: dict(RM.kind_style(cfg, k, w_m), **({"yellow": lay["group"] == 2 and cfg.get("standard") != "saudi"} if lay.get("group") and k in LN.CENTRE_KINDS else {})) for k in {k for _, k in lay["lines"]}}
        line_w = styles[lay["lines"][0][1]]["width"] / px
        widths_used.append(line_w)
        # always smooth a little so dashes follow the road axis, not the
        # skeleton's pixel steps; "lines not aligned" smooths more
        path = _smooth(path, window=15 if align else 7)
        # a highway's width at each point of its centreline (the lines move with it)
        wid = np.asarray(chain["w"], np.float32) if chain is not None else None
        if not LN.canonical(path[0][::-1], path[-1][::-1]):
            path = path[::-1]
            wid = wid[::-1] if wid is not None else None
        # how far short of each end the lines stop: at a junction's markings, else the setback;
        # a stop line inside the junction's circle: the street runs on straight to it
        sb0 = sb1 = default_px
        if ends_here:
            sb0, sb1 = LN.end_setbacks(np.array(path, float)[:, ::-1], ends_here, default_px, px, inside=wid is not None)
            if sb0 < 0 or sb1 < 0:
                ext, bef, aft = LN.extend_ends(np.array(path, float), max(0.0, -sb0), max(0.0, -sb1))
                path = [tuple(q) for q in ext]
                sb0, sb1 = max(0.0, sb0), max(0.0, sb1)
                if wid is not None:
                    wid = np.concatenate([np.full(len(bef), wid[0]), wid, np.full(len(aft), wid[-1])])
        # arc length along the street
        d = [0.0]
        for a, b in zip(path, path[1:]):
            d.append(d[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        total = d[-1]
        usable = total - sb0 - sb1
        if usable <= 0 or (usable <= cycle_px * 0.6 and not any(k in LN.SOLID_KINDS for _, k in lay["lines"])):
            continue                       # too short for even one dash

        pts = np.array(path, np.float32) - np.array([oy, ox], np.float32)    # (y, x) on the canvas
        dist = np.array(d, np.float32)
        # the sideways direction: (-ty, tx) in x, y, from a steadier copy of the line
        tg = np.gradient(np.array(_smooth(path, window=min(31, len(path) // 2 * 2 - 1)) if len(path) > 8 else path,
                                  np.float64), axis=0)
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nrm = np.column_stack([tg[:, 1], -tg[:, 0]]).astype(np.float32)   # (y, x) of (-ty, tx)
        run = (dist >= sb0) & (dist <= total - sb1)

        # each line's offset: one for the street, or along a highway one at each point
        if wid is not None:
            xs_, isl_at = LN.chain_offsets(lay, wid, cfg["rules"], cfg.get("edge_inset"))
            x_at = [x.astype(np.float32) for x in xs_]
        else:
            x_at, isl_at = [x for x, _ in lay["lines"]], None
        kinds, outside = [k for _, k in lay["lines"]], [[] for _ in lay["lines"]]
        traces = getattr(chosen.get(s), "traces", None)
        if traces:
            # the satellite's lines (app/observed.py): each where it is painted, and only where it is
            kinds, x_at, t_isl, outside = LN.traced_offsets(pts[:, ::-1] + [ox, oy], nrm[:, ::-1], dist, px,
                                                            lay["lines"], lay["island"], traces)
            isl_at = t_isl if t_isl is not None else isl_at
            for k in set(kinds) - set(styles):
                styles[k] = dict(RM.kind_style(cfg, k, w_m))

        def along(off_px, a, b):
            """The line off_px to the side (one value, or one per point), from a to b along the centreline."""
            q = pts + nrm * (off_px[:, None] if np.ndim(off_px) else off_px)
            inner = q[(dist > a) & (dist < b)]
            ends_ = [np.array([np.interp(v, dist, q[:, 0]), np.interp(v, dist, q[:, 1])], np.float32) for v in (a, b)]
            return np.vstack([ends_[0][None], inner, ends_[1][None]]) if len(inner) else np.vstack(ends_)

        for li_, kind in enumerate(kinds):
            stl = styles[kind]
            w_px = stl["width"] / px
            half_w, strength = max(0.5, w_px / 2), min(1.0, w_px)
            off = x_at[li_] / px
            tgt, tgt_solid = (yellow, solid_y) if stl.get("yellow") else (cover, solid)
            def blocks_for(offset):
                q = pts + nrm * (offset[:, None] if np.ndim(offset) else offset)
                # Raster strokes have a one-pixel soft cap beyond their end.
                # Keep that cap outside the reserved pedestrian footprint too.
                return reservations.blocked(q[:, ::-1] + [ox, oy], dist, half_w + 1.0 + .2 / px)
            blocked = blocks_for(off)
            if outside[li_]:
                blocked = LN.merge_runs(blocked + outside[li_])
            if kind in LN.SOLID_KINDS:
                offs = [off]
                if kind == "cdouble":                      # two lines, a gap between them
                    sep = (stl["width"] + stl.get("double_gap", 0.1)) / 2 / px
                    offs = [off - sep, off + sep]
                for o_ in offs:
                    for a_, b_ in LN.clear_runs(sb0, total - sb1, LN.merge_runs(blocks_for(o_) + outside[li_])):
                        _strokes(tgt_solid, along(o_, a_, b_), half_w, strength, piece_px)
                continue
            k_cycle = stl["cycle"] / px
            cycles = max(1, int(round(usable / k_cycle)))
            step = usable / cycles             # gaps stretch a little to fit
            # the guide's dashes keep their length (only the gaps stretch); before, both did
            dash_len = min(stl["cycle"] * stl["share"] / px, step * 0.95) if cfg.get("exact") else step * stl["share"]
            starts = sb0 + step * np.arange(cycles)
            keep = np.ones(cycles, bool)
            if area:
                # a dash is laid whole, or not at all: by where its middle lies
                q = pts + nrm * (off[:, None] if np.ndim(off) else off)
                mid = starts + dash_len / 2
                keep = LN.in_area(area, np.interp(mid, dist, q[:, 1]) + ox, np.interp(mid, dist, q[:, 0]) + oy)
            gap_px = step - dash_len
            if gap_px < MERGE_GAP_PX or step < MERGE_CYCLE_PX:
                # the dashes run together: the line is drawn whole (as a solid line, stopping
                # at the area's edge), as strong as the dashes show on average along it. Each
                # dash's soft ends reach a pixel into the gap (g: the gap their half widths
                # past half a pixel do not already cover): two ends meeting fill all but g^2 / 4
                # of it, apart they fill one pixel of it
                g = max(0.0, gap_px - 2 * (half_w - 0.5))
                filled = step - g * g / 4.0 if g < 2.0 else step - g + 1.0
                for a_, b_ in LN.clear_runs(sb0, total - sb1, blocked):
                    _strokes(tgt_solid, along(off, a_, b_), half_w,
                             strength * filled / step, piece_px)
                placed += int(keep.sum())
                continue
            # a stroke's round ends reach half its width past its two ends: the guide's dashes are
            # drawn that much shorter at each end, so the paint is as long as the dash
            cap = min(half_w, dash_len * 0.45) if cfg.get("exact") else 0.0
            for a, k in zip(starts, keep):
                if k and LN.clear_runs(a, a + dash_len, blocked) == [(a, a + dash_len)]:
                    _stroke(tgt, along(off, a + cap, a + dash_len - cap), half_w, strength)
                    placed += 1
        if lay["island"] and run.any():
            x0, x1 = lay["island"]
            mid_isl = ((isl_at[0] + isl_at[1]) / 2).astype(np.float32) if isl_at is not None else (x0 + x1) / 2
            spine = along(mid_isl / px, sb0, total - sb1)
            hw_isl = (x1 - x0) / 2 / px
            _strokes(isl, spine, hw_isl, min(1.0, 2 * hw_isl), piece_px)
            _strokes(isl_top, spine, max(0.5, hw_isl - 0.12 / px), min(1.0, 2 * hw_isl), piece_px)
    if area:
        ac = None
        if solid.any():
            ac = _area_cover(area, (H, W), origin)
            np.maximum(cover, solid * ac, out=cover)       # solid lines stop at the edge
        if solid_y.any():
            ac = _area_cover(area, (H, W), origin) if ac is None else ac
            np.maximum(yellow, solid_y * ac, out=yellow)
    return cover, isl, isl_top, {"placed": placed, "widths": widths_used, "layouts": layouts, "streets": streets,
                                 "yellow": yellow}


def place_dashes(det, metres_per_pixel, cfg, rng, align=False, nomark=None, lane_picks=None, area=None, ends=None):
    """
    Lay each street's lane lines (app/lanes.py: its lanes on each side, even
    or as picked, where across it, which kind of line; app/roadmarks.py: each
    kind's dash, gap, width and colour, cfg), stopping short of each junction
    (at its markings, ends, else cfg's setback), and on highways its raised
    island. lane_picks: the Lanes tool's choices, [{"pt": [x, y] in this
    picture's pixels, "sides": [a, b]}], each for the street nearest its point.
    area: the markings area (this picture's pixels): lines only inside it
    (lay_lanes).

    The number of cycles per street is rounded to a whole number and the gaps
    stretched slightly to fit, so each street starts and ends on a full dash.
    Every lane's dashes are side by side, measured along the centreline. A
    street is read the canonical way (lanes.canonical), so its first side is
    the one the 3D model picks too.

    A line narrower than a pixel (15 cm paint on a map at 1 m a pixel) is drawn
    one pixel wide but only as strong as the share of the pixel it covers, the
    way a photo from that height shows it, not widened to a whole pixel.

    Returns the paint's coverage (white and yellow), a report, the islands'
    (coverage, top's coverage): the top a little narrower, so its rim can be
    shaded as a kerb, the streets for the Lanes tool: [{"id", "line":
    centreline (x, y) the canonical way, "width_m", "limit", "sides", "even",
    "cut"}], and every street's centreline and width (street_paths), kept so a
    street's lanes can be laid again on their own (repaint_lanes).
    """
    road = det["road"]
    px = max(metres_per_pixel, 1e-6)
    cycle_m = cfg["kinds"]["dash"]["cycle"]
    cycle_px = cycle_m / px
    # every street's ordered centreline and width first, so the Lanes tool's
    # choices can be matched to the street they were made on
    found = with_chains(street_paths(det, metres_per_pixel, nomark), ends, metres_per_pixel)
    chosen = lane_choices(found, lane_picks, metres_per_pixel)
    cover, isl, isl_top, laid = lay_lanes(road.shape, found, chosen, metres_per_pixel, cfg, align, area=area, ends=ends)
    rc = road_cov if (road_cov := det.get("coverage")) is not None else road
    cover *= rc
    yellow = laid["yellow"] * rc
    isl *= rc
    isl_top = np.minimum(isl_top, isl)
    widths_used, streets = laid["widths"], laid["streets"]
    width_shown = float(np.median(widths_used)) if widths_used else cfg.get("fixed_width_m", 0.15) / px
    warning = None
    if cycle_px < 6:
        # a dash and its gap need a few pixels each, or they merge into one line
        warning = (f"a dash and its gap of {cycle_m:g} m is only {cycle_px:.1f} px at {metres_per_pixel:g} m a pixel: "
                   f"the dashes run together into a solid line (and there are thousands of them, which is slow). "
                   f"Real lane lines repeat every 4.5 to 12 m ({4.5 / px:.0f} to {12 / px:.0f} px here)")
    return (cover, yellow), {"dashes": laid["placed"], "cycle_px": round(cycle_px, 1),
                             "width_px": round(width_shown, 2),
                             "width_mode": "learned" if cfg.get("width_ratio") else "fixed",
                             "width_ratio": cfg.get("width_ratio"), "faint": width_shown < 1.0, "warning": warning,
                             "lanes": LN.summary(laid["layouts"]), "picked": len(chosen),
                             "cut": sum(1 for st in streets if st["cut"]), "rules": cfg["rules"],
                             "yellow": bool(yellow.any())}, (isl, isl_top), streets, found


def with_lanes(worn, wear_map, paint, island, island_top, paint_rgb, wear_strength, lum_mean, yellow=None):
    """
    The texture with its lane lines and islands on the worn road. The paint is
    worn too, so it is not a flat block of white; the coverage blends it into
    the asphalt over the stroke's one-pixel edge. The highways' raised island: a
    concrete top with the road's grain (lum_mean: the road's mean brightness),
    ringed by its kerb's darker face and shadow, as it looks from above.
    """
    result = worn.astype(np.float32, copy=True)
    for cov_, rgb_ in ((paint, paint_rgb), (yellow, YELLOW_RGB)):
        if cov_ is not None and cov_.any():
            a = cov_[:, :, None]
            painted = np.asarray(rgb_, np.float32)[None, None, :] * (1.0 - 0.25 * wear_strength * wear_map)[:, :, None]
            result = result * (1 - a) + painted * a
    if island.any():
        lum = worn.mean(axis=2, keepdims=True)
        grain = 0.9 + 0.1 * lum / max(float(lum_mean), 1e-6)
        top = np.array(ISLAND_RGB, np.float32)[None, None, :] * grain
        rim = np.array(ISLAND_RIM_RGB, np.float32)[None, None, :] * grain
        a_top, a_rim = island_top[:, :, None], np.clip(island - island_top, 0, 1)[:, :, None]
        result = result * (1 - a_top - a_rim) + top * a_top + rim * a_rim
    return result


def markings_picture(paint, island, paint_rgb, yellow=None):
    """The markings layer: the paint, and the islands with it (neither is asphalt, so the 3D model's tone leaves both out)."""
    mark = paint[:, :, None] * np.asarray(paint_rgb, np.float32)[None, None, :]
    if yellow is not None and yellow.any():
        mark = mark * (1 - yellow[:, :, None]) + yellow[:, :, None] * np.asarray(YELLOW_RGB, np.float32)[None, None, :]
    if island.any():
        mark = np.maximum(mark, island[:, :, None] * np.array(ISLAND_RGB, np.float32)[None, None, :])
    return mark


BACKGROUND = (18, 18, 20)


def finish(arr, road, cov, soft_edges, nearest=None, mask=None, soft=True):
    """
    A layer as it is saved (uint8): with soft edges, the edge ring just outside
    the road takes the colour of the nearest road pixel (nearest: its (iy, ix),
    from the distance transform) faded into the background by the coverage;
    else, and for layers saved hard (mask, soft False), the road (or mask) as it
    is and the background elsewhere.
    """
    H, W = road.shape
    a = (arr if arr.ndim == 3 else np.dstack([arr] * 3)).astype(np.float32)
    background = np.array(BACKGROUND, np.float32)
    if soft and soft_edges and mask is None:
        if nearest is None:
            _, nearest = ndi.distance_transform_edt(~road, return_indices=True)
        iy, ix = nearest
        img = a[iy, ix] * cov[:, :, None] + background * (1 - cov[:, :, None])
    else:
        m = road if mask is None else mask
        img = np.empty((H, W, 3), np.float32); img[...] = background
        img[m] = a[m]
    return np.clip(img, 0, 255).astype(np.uint8)


def save_lane_state(path, worn, wear_map, road, cov, found, meta):
    """
    What laying a street's lanes again needs (repaint_lanes): the worn road
    without its markings, the wear, the road and its soft edge's coverage, every
    street's centreline and width, and the settings the lines were laid with.
    One zip of .npy files (np.load reads it), packed fast.
    """
    import io
    import json
    import zipfile
    labels = np.array(sorted(found), np.int32)
    paths = [np.asarray(found[int(k)][0], np.float64).reshape(-1, 2) for k in labels]
    arrays = {
        "worn": np.clip(worn + 0.5, 0, 255).astype(np.uint8),
        "wear": np.clip(wear_map * 255 + 0.5, 0, 255).astype(np.uint8),
        "road": np.packbits(road.astype(bool), axis=None),
        "cov": np.clip(cov * 65535 + 0.5, 0, 65535).astype(np.uint16),
        "labels": labels,
        "widths": np.array([found[int(k)][1] for k in labels], np.float64),
        "offsets": np.concatenate([[0], np.cumsum([len(p) for p in paths])]).astype(np.int64),
        "paths": np.concatenate(paths) if paths else np.zeros((0, 2), np.float64),
        "meta": np.array(json.dumps(meta)),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.save(buf, v, allow_pickle=False)
            z.writestr(k + ".npy", buf.getvalue())


def _same_lines(width_m, a, b, cfg=None):
    """Whether two lane choices lay the same lines and island on a street this wide."""
    rules, hw = (cfg or {}).get("rules", "simple"), (cfg or {}).get("highway")
    la, lb = LN.layout(width_m, a, rules=rules, hw=hw), LN.layout(width_m, b, rules=rules, hw=hw)
    return (la["lines"] == lb["lines"] and la["island"] == lb["island"]
            and getattr(a, "traces", None) == getattr(b, "traces", None))


def repaint_lanes(state_path, old_picks, new_picks, files, decals=None, compress_level=1,
                  old_area=None, new_area=None, old_ends=None, new_ends=None, new_cfg=None, new_align=None):
    """
    Lay again only the lane lines of the streets whose lanes changed (the Lanes
    tool's Done), on the texture as it is, and leave everything else as it was.

    old_picks, new_picks: the Lanes tool's choices the texture has, and the ones
    it should have (mask pixels). Each street whose lines change gets a window
    round it; in it, the lines of every street that reaches into the window are
    drawn as generation draws them, onto the worn road kept from generation
    (save_lane_state), and the pixels within the changed street's width (and
    half a metre) where a line or an island was or is now are written back. files: the pictures to update in place,
    {"result", "markings"} and optionally "base" (the texture before painted
    decals: then result is base with the decals, decals = (their picture's
    RGBA, painted) ) and "full" (roads + islands: it takes the same change).
    old_area, new_area: the markings area the texture has and the one it
    should have (mask pixels, app/lanes.py): a street whose lines it changes is
    laid again too, so markings outside a new area are taken away and those
    inside one come back, without generating anything else.
    old_ends, new_ends: where junction markings stopped the lines and where
    they should stop now (app/roadmarks.py, mask pixels; the presets changed):
    a street whose ends move is laid again too.
    Returns {"changed": [{"id", "sides", "even", "cut", "box": [x0, y0, x1, y1]
    in mask pixels}], "area_streets": how many were laid again for the area}.
    """
    z = np.load(state_path)
    import json
    meta = json.loads(str(z["meta"]))
    mpp, scale = float(meta["mpp_out"]), int(meta["scale"])
    px = max(mpp, 1e-6)
    labels, widths, offs, P = z["labels"], z["widths"], z["offsets"], z["paths"]
    found = {int(k): ([tuple(p) for p in P[offs[i]:offs[i + 1]]], float(widths[i])) for i, k in enumerate(labels)}

    def scaled(picks):
        return LN.picks_scaled(picks, scale)
    previous_cfg = lines_cfg(meta)
    cfg = new_cfg or previous_cfg
    e_old, e_new = ends_scaled(old_ends, scale), ends_scaled(new_ends, scale)
    # the highways' centrelines as streets of their own (as the texture was laid with them)
    found = with_chains(found, e_new, mpp)
    old, new = lane_choices(found, scaled(old_picks), mpp), lane_choices(found, scaled(new_picks), mpp)
    changed = [s for s, (_, w) in found.items() if not _same_lines(w, old.get(s), new.get(s), cfg)]
    if new_align is not None and bool(new_align) != bool(meta["align"]):
        meta["align"] = bool(new_align)
        changed = list(found)
    if cfg != previous_cfg:
        changed = list(found)
    if LN.chains_of(old_ends) != LN.chains_of(new_ends):
        changed = list(found)                  # other highways (other settings): every street's lines again
    by_ends = []
    if (old_ends or None) != (new_ends or None):
        # a street whose lines stop somewhere else now (the junction markings changed)
        d_px = cfg["setback_m"] / px
        for s_, (p_, w_) in found.items():
            if s_ in changed or not LN.layout(w_, new.get(s_), rules=cfg["rules"], hw=cfg.get("highway"))["lines"]:
                continue
            xy = np.asarray(p_, float)[:, ::-1]
            a_ = LN.end_setbacks(xy, e_old, d_px, px) if e_old else (d_px, d_px)
            b_ = LN.end_setbacks(xy, e_new, d_px, px) if e_new else (d_px, d_px)
            if max(abs(a_[0] - b_[0]), abs(a_[1] - b_[1])) > 0.01:
                by_ends.append(s_)
        changed += by_ends
    a_old, a_new = LN.area_scaled(LN.area_check(old_area), scale), LN.area_scaled(LN.area_check(new_area), scale)
    by_area = []
    if a_old != a_new:
        # a street the area's change reaches: across either edge, or in one and out of the other
        for s_, (p_, w_) in found.items():
            if s_ in changed or not LN.layout(w_, new.get(s_), rules=cfg["rules"], hw=cfg.get("highway"))["lines"]:
                continue
            pa_ = np.asarray(p_, float)
            r_ = (w_ / 2 + 0.5) / px + 3
            st_old = LN.area_status(a_old, pa_[:, 1], pa_[:, 0], r_)
            st_new = LN.area_status(a_new, pa_[:, 1], pa_[:, 0], r_)
            if st_old != st_new or st_old == "edge":
                by_area.append(s_)
        changed += by_area
    full_update = cfg != previous_cfg or LN.chains_of(old_ends) != LN.chains_of(new_ends) or [e for e in e_old or [] if e.get("reserve")] != [e for e in e_new or [] if e.get("reserve")]
    if full_update:
        changed = list(found)
    out = {"changed": [], "area_streets": len(by_area), "ends_streets": len(by_ends)}
    if not changed:
        return out

    H, W = int(meta["shape"][0]), int(meta["shape"][1])
    worn, wear, cov = z["worn"], z["wear"], z["cov"]
    road = np.unpackbits(z["road"], count=H * W).reshape(H, W).astype(bool)
    soft = bool(meta["soft_edges"])
    paint_rgb = np.array(meta["paint_rgb"], np.float32)
    overlays = {k: z[k].copy() for k in ("paint", "yellow", "island", "island_top") if k in z.files}
    pics = {k: np.array(Image.open(p).convert("RGB")) for k, p in files.items() if p}
    target = "base" if "base" in pics else "result"
    if full_update:
        paint, island, top, laid = lay_lanes((H,W),found,new,mpp,cfg,meta["align"],area=a_new,ends=e_new)
        yellow = laid["yellow"]
        coverage = cov.astype(np.float32)/65535
        paint *= coverage; yellow *= coverage; island *= coverage; top = np.minimum(top,island)
        nearest = ndi.distance_transform_edt(~road, return_indices=True)[1] if soft and road.any() else None
        fc = coverage if soft else road.astype(np.float32)
        dirty = (pics["markings"] != 0).any(axis=2) | (paint>0) | (yellow>0) | (island>0)
        if nearest is not None:
            dirty |= dirty[nearest[0],nearest[1]] & (coverage>0)
        result = with_lanes(worn,wear.astype(np.float32)/255,paint,island,top,paint_rgb,
                            float(meta["wear_strength"]),float(meta["lum_mean"]),yellow)
        previous = pics["result"].copy() if "full" in pics else None
        pics[target][dirty] = finish(result,road,fc,soft,nearest)[dirty]
        pics["markings"] = finish(markings_picture(paint,island,paint_rgb,yellow),road,fc,soft,soft=False)
        if target == "base":
            if decals is not None and decals[1]:
                from app import decals as DC
                rgba = decals[0].astype(np.float32)/255; rgba[...,:3] *= rgba[...,3:]
                pics["result"][dirty] = DC.composite(pics["base"],rgba)[dirty]
            else:
                pics["result"][dirty] = pics["base"][dirty]
        if previous is not None:
            pics["full"][dirty] = np.clip(pics["full"][dirty].astype(np.int16)+pics["result"][dirty].astype(np.int16)-previous[dirty],0,255)
        for k,p in files.items():
            if p:
                Image.fromarray(pics[k]).save(p,compress_level=compress_level)
        state={k:z[k] for k in z.files};z.close()
        state.update(paint=paint,yellow=yellow,island=island,island_top=top)
        np.savez_compressed(state_path,**state)
        for st in laid["streets"]:
            st["line"] = (np.asarray(st["line"])/scale).round(2).tolist()
        out["streets"] = laid["streets"]
        out["changed"] = [dict(id=int(s),sides=list(LN.layout(w,new.get(s),rules=cfg["rules"],hw=cfg.get("highway"))["sides"]),
                               even=new.get(s) is None,cut=False,box=[0,0,W/scale,H/scale]) for s,(_,w) in found.items()]
        return out

    before = pics["result"].copy() if "full" in pics else None
    # each street's centreline, run on past its ends as far as its lines run (to a stop line
    # inside the junction's circle, as laid before or now)
    d_px = cfg["setback_m"] / px
    lines_of = {}
    for s_, (p_, _) in found.items():
        pa_ = np.asarray(p_, float)
        x0_ = x1_ = 0.0
        for E in (e_old, e_new):
            if E:
                a_, b_ = LN.end_setbacks(pa_[:, ::-1], E, d_px, px)
                x0_, x1_ = max(x0_, -a_), max(x1_, -b_)
        lines_of[s_] = LN.extend_ends(pa_, x0_ + 1.0, x1_ + 1.0)[0] if (x0_ or x1_) else pa_
    # each street's reach: its centreline's box, out by half its width and a little
    reach = {s: (w / 2 + 0.5) / px + 3 for s, (_, w) in found.items()}
    lo = {s: np.min(lines_of[s], axis=0) - reach[s] for s in found}
    hi = {s: np.max(lines_of[s], axis=0) + reach[s] for s in found}
    for s in changed:
        pad = 6
        y0, x0 = np.maximum(np.floor(lo[s] - pad).astype(int), 0)
        y1, x1 = np.ceil(hi[s] + pad).astype(int) + 1
        y1, x1 = min(int(y1), H), min(int(x1), W)
        near = {t: found[t] for t in found
                if lo[t][0] <= y1 and hi[t][0] >= y0 and lo[t][1] <= x1 and hi[t][1] >= x0}
        cover, isl, isl_top, laid = lay_lanes((y1 - y0, x1 - x0), near, new, mpp, cfg, meta["align"],
                                              origin=(y0, x0), area=a_new, ends=e_new)
        yellow = laid["yellow"]
        rc = cov[y0:y1, x0:x1].astype(np.float32) / 65535.0
        rw = road[y0:y1, x0:x1]
        if not rw.any():
            continue
        cover *= rc
        yellow *= rc
        isl *= rc
        isl_top = np.minimum(isl_top, isl)
        res = with_lanes(worn[y0:y1, x0:x1], wear[y0:y1, x0:x1].astype(np.float32) / 255.0, cover, isl, isl_top,
                         paint_rgb, float(meta["wear_strength"]), float(meta["lum_mean"]), yellow)
        # the pixels this street's lines can reach: within its width (and half a metre) of its centreline
        line = np.zeros(rw.shape, bool)
        pa = np.round(lines_of[s]).astype(int) - np.array([y0, x0])
        pa = pa[(pa[:, 0] >= 0) & (pa[:, 0] < rw.shape[0]) & (pa[:, 1] >= 0) & (pa[:, 1] < rw.shape[1])]
        line[pa[:, 0], pa[:, 1]] = True
        R = ndi.distance_transform_edt(~line) <= reach[s]
        nearest = ndi.distance_transform_edt(~rw, return_indices=True)[1] if soft and rw.any() else None
        win = (slice(y0, y1), slice(x0, x1))
        # and of those only where a line or an island was or is now: the road's own surface
        # between them stays exactly as it was (a pixel of the edge ring follows the road
        # pixel it takes its colour from)
        lined = rw & ((pics["markings"][win] != 0).any(axis=2) | (cover > 0) | (yellow > 0) | (isl > 0))
        R &= lined[nearest[0], nearest[1]] if nearest is not None else lined
        for key_, value_ in (("paint", cover), ("yellow", yellow), ("island", isl), ("island_top", isl_top)):
            if key_ in overlays:
                overlays[key_][win][R] = value_[R]
        fc = rc if soft else rw.astype(np.float32)
        pics[target][win][R] = finish(res, rw, fc, soft, nearest)[R]
        pics["markings"][win][R] = finish(markings_picture(cover, isl, paint_rgb, yellow), rw, fc, soft, soft=False)[R]
        if target == "base":
            if decals is not None and decals[1]:
                from app import decals as DC
                rgba = decals[0][win].astype(np.float32) / 255.0
                rgba[..., :3] *= rgba[..., 3:]
                pics["result"][win][R] = DC.composite(pics["base"][win], rgba)[R]
            else:
                pics["result"][win][R] = pics["base"][win][R]
        if before is not None:
            d = pics["result"][win][R].astype(np.int16) - before[win][R].astype(np.int16)
            pics["full"][win][R] = np.clip(pics["full"][win][R].astype(np.int16) + d, 0, 255).astype(np.uint8)
        lay = LN.layout(found[s][1], new.get(s), rules=cfg["rules"], hw=cfg.get("highway"))
        out["changed"].append({"id": int(s), "sides": list(lay["sides"]), "even": lay["even"], "cut": lay["cut"],
                               "area": s in by_area, "ends": s in by_ends,
                               "box": [round(x0 / scale, 1), round(y0 / scale, 1), round(x1 / scale, 1),
                                       round(y1 / scale, 1)]})
    if overlays:
        state = {k: z[k] for k in z.files}
        z.close()
        state.update(overlays)
        np.savez_compressed(state_path, **state)
    import os
    import uuid
    for k, p in files.items():
        if not p:
            continue
        # written under another name, then put in place whole
        tmp = f"{p}.{uuid.uuid4().hex[:6]}.part.png"
        Image.fromarray(pics[k]).save(tmp, compress_level=compress_level)
        os.replace(tmp, p)
    return out


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


def lines_cfg(params):
    """
    The lines' settings a generation (or its kept state) was made with:
    app/roadmarks.py's lines_config as the server worked it out ("marks"), or
    from the settings before road markings had their own (every line dashed alike).
    """
    from app import roadmarks as RM
    if params.get("marks"):
        return params["marks"]
    return RM.legacy_config(float(params.get("cycle_m", 9.0)), float(params.get("dash_share", 0.6)),
                            float(params.get("width_m", 0.15)), params.get("width_ratio"),
                            float(params.get("setback_m", 2.0)))


def ends_scaled(ends, scale):
    """Junction markings' line ends (mask pixels) in the pixels of a picture scale times the mask's size."""
    if not ends:
        return None
    c = (scale - 1) / 2
    return [dict(e, reserve=dict(e["reserve"], cx=e["reserve"]["cx"] * scale + c,
                                 cy=e["reserve"]["cy"] * scale + c,
                                 w=e["reserve"]["w"] * scale, h=e["reserve"]["h"] * scale)) if e.get("reserve")
            else dict(e, pts=[[x * scale + c, y * scale + c] for x, y in e["pts"]]) if e.get("chain")
            else dict(e, x=e["x"] * scale + c, y=e["y"] * scale + c) for e in ends]


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
    import pickle
    from pathlib import Path
    geometry = params.get("_geometry_cache")
    if geometry and Path(geometry).exists():
        with open(geometry, "rb") as fh:
            det = pickle.load(fh)
        coverage = det["coverage"]
    else:
        prepared, coverage = prepare_mask(gray, scale)
        prog.stage("junctions", "finding the junctions and streets")
        det = J.detect(prepared)
        det["coverage"] = coverage
        if geometry:
            with open(geometry, "wb") as fh:
                pickle.dump(det, fh, protocol=4)
    road = det["road"]
    H, W = road.shape
    prog.stage("areas", "junction, kerb and open-road areas")
    groups, road_width, band_m = groups_for(det, mpp_out)

    # ---- 1. material ----
    # each part is painted over the whole canvas, then the parts are blended
    # with soft weights, so junction, kerb and open road fade into each other
    # over a few pixels instead of meeting at a hard line
    cached_surface = params.get("_surface_cache")
    reuse = params.get("_reuse_surface") and cached_surface and Path(cached_surface).exists()
    cached = np.load(cached_surface) if reuse else None
    if reuse:
        material = cached["material"].copy()
        used, size_ratio = {}, {}
    else:
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
    
    raw_material = material.copy()

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
    if reuse:
        wear_map = cached["wear"].copy()
        cached.close()
    else:
        if wear_npz:
            wp, wm = load_patches(wear_npz, with_masks=True)
            wtrained = (libraries.get("scales") or {}).get("open")
            wp, wm = scale_patches(wp, (wtrained / mpp_out) if wtrained else float(scale), wm)
            wl = synth_layer((H, W), wp, rng, overlap, jitter, wm, need=road).mean(axis=2)
            lo, hi = np.percentile(wl[road], [5, 95]) if road.any() else (0, 1)
            wear_map = np.clip((wl - lo) / max(hi - lo, 1e-6), 0, 1)
    
    if cached_surface:
        np.savez_compressed(cached_surface, material=raw_material, wear=wear_map)

    worn = material * (1.0 - 0.45 * wear_strength * wear_map[:, :, None])

    # ---- 3. markings ----
    prog.stage("markings", "markings")
    paint_rgb = np.array(params.get("paint_colour", [235, 232, 222]), np.float32)
    picks = LN.picks_scaled(params.get("lane_picks"), scale)
    marks = lines_cfg(params)
    line_ends = params.get("line_ends") or None
    det_lines = det
    if params.get("lines_mask"):
        # with a project area: the streets the lines follow traced from the mask without it, so
        # the streets round its islands keep their lines and junction markings as they were
        prog.stage("junctions", "finding the streets under the project areas")
        det_lines = J.detect(prepare_mask(np.array(Image.open(params["lines_mask"]).convert("L")), scale)[0])
    (paint, yellow), dash_info, (island, island_top), streets, found = place_dashes(
        det_lines, mpp_out, marks, rng=rng, align=align_lines,
        nomark=_at_size(params.get("nomark"), scale, road.shape), lane_picks=picks,
        area=LN.area_scaled(LN.area_check(params.get("mark_area")), scale), ends=ends_scaled(line_ends, scale))
    if params.get("_keep_markings") and out_paths.get("lanes") and Path(out_paths["lanes"]).exists():
        with np.load(out_paths["lanes"]) as old_lanes:
            if "paint" in old_lanes.files:
                paint, yellow, island, island_top = [old_lanes[k].copy() for k in ("paint", "yellow", "island", "island_top")]
    if out_paths.get("streets"):
        # the streets for the Lanes tool, in mask pixels, their lines simplified to about 0.5 m
        import json
        from shapely.geometry import LineString
        tol = max(0.5, 0.5 / mpp_out)
        for st in streets:
            ls = LineString(st["line"]) if len(st["line"]) > 1 else None
            q = np.asarray(ls.simplify(tol).coords) if ls is not None else st["line"]
            st["line"] = np.round(q / scale, 1).tolist()
        with open(out_paths["streets"], "w") as fh:
            json.dump({"mpp": mpp, "streets": streets}, fh, separators=(",", ":"))

    prog.stage("finishing", "blending and saving the pictures")
    lum_mean = float(worn.mean(axis=2)[road].mean()) if road.any() else 1.0
    result = with_lanes(worn, wear_map, paint, island, island_top, paint_rgb, wear_strength, lum_mean, yellow)

    # the edge ring just outside the road has partial coverage but no texture:
    # give each of those pixels the colour of the nearest road pixel
    nearest = ndi.distance_transform_edt(~road, return_indices=True)[1] if soft_edges and road.any() else None
    cov = coverage if soft_edges else road.astype(np.float32)

    def save(arr, path, mask=None, soft=True):
        Image.fromarray(finish(arr, road, cov, soft_edges, nearest, mask, soft)).save(path)

    save(result, out_paths["result"])
    save(material, out_paths["material"])
    save(np.dstack([wear_map * (255 if wear_strength > 0 else 0)] * 3), out_paths["wear"], soft=False)
    if not params.get("_keep_markings"):
        save(markings_picture(paint, island, paint_rgb, yellow), out_paths["markings"], soft=False)
    if out_paths.get("lanes"):
        # what laying one street's lanes again needs, without generating the rest (repaint_lanes)
        save_lane_state(out_paths["lanes"], worn, wear_map, road, coverage,
                        street_paths(det_lines, mpp_out, _at_size(params.get("nomark"), scale, road.shape)), {
            "mpp_out": mpp_out, "scale": scale, "shape": [int(H), int(W)], "soft_edges": soft_edges,
            "paint_rgb": [float(v) for v in paint_rgb], "wear_strength": wear_strength, "lum_mean": lum_mean,
            "cycle_m": float(params.get("cycle_m", 9.0)), "dash_share": float(params.get("dash_share", 0.6)),
            "width_m": float(params.get("width_m", 0.15)), "setback_m": float(params.get("setback_m", 2.0)),
            "align": align_lines, "width_ratio": params.get("width_ratio"), "marks": marks})

    if out_paths.get("lanes"):
        with np.load(out_paths["lanes"]) as state:
            arrays = {k: state[k] for k in state.files}
        arrays.update(paint=paint, yellow=yellow, island=island, island_top=island_top)
        np.savez_compressed(out_paths["lanes"], **arrays)
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
