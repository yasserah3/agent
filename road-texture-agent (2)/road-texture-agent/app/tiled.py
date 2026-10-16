"""
Texture in tiles, for areas made from coordinates (app/coords.py).

Such an area is textured in the squares its satellite analysis read (app/sat_analysis.py's
tile_grid: 512 pixels of the alpha a side, named r<row>c<col>), so each texture tile lies
on one satellite tile. The tiles are made side by side in worker processes
(python -m app.tiled --worker, as the analysis's and the 3D patches') and put together.

One set of materials for the whole area, never one per tile:
  - the libraries are chosen once and scaled once, so every tile lays the same patches;
  - where the patches go is one pattern for the whole area: each spot of the patch
    lattice takes its patch, turn and nudge from the seed and its own place (a hash),
    not from a run of random numbers a tile would start again. A tile, with a margin,
    lays every patch that reaches it, so along its edge it lays what its neighbour lays:
    the joins are as if the whole area were made at once (the tiles put together equal
    the area made as one tile, bit for bit);
  - what is worked out over the whole area (the road's width and kerb band, the tone to
    match, the wear's range, a scanned street material's colour) is worked out once, on
    the whole area, after the tiles (app/generation.py).

Adaptive to the analysis:
  - a tile with no road lays no patches (background only);
  - the asphalt's tone (optional): in the tiles the analysis rated at or above the
    threshold, each street takes the lighter or darker tone the satellite shows there
    (new black asphalt, old grey), relative to the area's usual tone; tiles under the
    threshold, and roads the satellite does not show, keep the area's tone (the rules).
    The tone is a smooth field, eased between seen and unseen tiles, so no tile edge
    shows.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

TILE = 512                     # the analysis's tile, in mask pixels (app/sat_analysis.py)
CODES = {"junction": 1, "edge": 2, "open": 3}     # app/generation.py's group_map
LAYERS = {"junction": 1, "edge": 2, "open": 3, "wear": 4}

TONE_CELL_M = 2.0              # the satellite's asphalt read in cells this big
TONE_VALUE_M = 5.0             # each street's tone: the cells this near
TONE_TRUST_M = 12.0            # eased from seen tiles to the area's tone over about this
TONE_MIN_SHARE = 0.3           # a cell counts when this much of it is clean asphalt
TONE_RANGE = (0.55, 1.6)       # how much lighter or darker than the area's tone, at most


# ------------------------------------------------------------------ the tiles

def tile_grid(w, h, scale=1, tile=TILE):
    """The analysis's tiles of a w x h mask, in output pixels (scale times the mask's): id, row, col, core."""
    out = []
    for r, y0 in enumerate(range(0, h, tile)):
        for c, x0 in enumerate(range(0, w, tile)):
            x1, y1 = min(w, x0 + tile), min(h, y0 + tile)
            out.append({"id": f"r{r}c{c}", "row": r, "col": c, "mask_core": [x0, y0, x1, y1],
                        "core": [x0 * scale, y0 * scale, x1 * scale, y1 * scale]})
    return out


def margin(scale):
    """A tile's margin (output pixels): the material parts' blend weight (a Gaussian, 1.5 px a mask pixel) reaches this far."""
    feather = 1.5 * scale
    return int(4.0 * feather + 0.5) + 2


# ------------------------------------------------------------------ one pattern for the area

_K0 = np.uint64(0x9E3779B97F4A7C15)
_K1 = np.uint64(0xBF58476D1CE4E5B9)
_K2 = np.uint64(0x94D049BB133111EB)
_K3 = np.uint64(0xD6E8FEB86659FD93)


def _mix(x):
    """SplitMix64's finaliser on uint64 arrays: every bit of the input stirs every bit of the output."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * _K1
        x = x ^ (x >> np.uint64(27))
        x = x * _K2
        return x ^ (x >> np.uint64(31))


def lattice_draws(seed, layer, ii, jj, n, jit):
    """
    The draws of lattice spots (ii, jj) (int arrays, any shape) of one layer: the nudge
    (dy, dx, each -jit..jit), the patch (0..n-1), its quarter turns (0..3) and whether it
    is mirrored, from the seed, the layer and the spot alone.
    """
    with np.errstate(over="ignore"):
        base = _mix(np.array([(int(seed) & 0xFFFFFFFF) * 0x10001 + int(layer) * 0x9E37], np.uint64))[0]
        key = _mix(_mix(np.asarray(ii, np.int64).astype(np.uint64) * _K0 + base)
                   ^ (np.asarray(jj, np.int64).astype(np.uint64) * _K3))
        h = [_mix(key + np.uint64(t + 1) * _K0) >> np.uint64(11) for t in range(5)]
    span = np.uint64(2 * jit + 1)
    dy = (h[0] % span).astype(np.int64) - jit
    dx = (h[1] % span).astype(np.int64) - jit
    idx = (h[2] % np.uint64(max(1, n))).astype(np.int64)
    k = (h[3] % np.uint64(4)).astype(np.int64)
    flip = (h[4] % np.uint64(2)).astype(bool)
    return dy, dx, idx, k, flip


def synth_window(win, shape, patches, rng_key, overlap=0.25, jitter=1.0, masks=None, seam=3.0, need=None):
    """
    app/generation.py's synth_layer over one window [x0, y0, x1, y1] of a canvas of shape
    (H, W): the patches of the area's one lattice (its spots every step from -s, as
    synth_layer lays them) that reach the window, each drawn by lattice_draws, laid in the
    lattice's order. rng_key: (seed, layer). need: where in the window the layer is used
    (a bool map the window's size), or None; a patch that cannot reach it (grown by a
    patch) is not laid. Returns the window's pixels and how many patches were laid.
    """
    H, W = shape
    x0, y0, x1, y1 = win
    h, w = y1 - y0, x1 - x0
    s = int(patches.shape[1])
    step = max(2, int(s * (1 - overlap)))
    jit = max(1, int(step * jitter / 3))
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)

    ramp = np.minimum(np.arange(s), np.arange(s)[::-1]).astype(np.float32) + 1.0
    window = np.outer(ramp, ramp)[:, :, None]
    window /= window.max()
    if seam and seam > 1:
        window = window ** seam
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
            wnd = window * pm[:, :, None] if pm is not None else window
            oriented[key] = (p * wnd, wnd)
        return oriented[key]

    reach = None
    if need is not None:
        need = np.asarray(need, bool)
        if not need.any():
            return acc, 0
        grown = ndi.maximum_filter(need, size=2 * (s + 2) + 1)
        reach = np.zeros((h + 1, w + 1), np.int64)
        reach[1:, 1:] = grown.cumsum(0).cumsum(1)

    n_rows, n_cols = len(range(-s, H + s, step)), len(range(-s, W + s, step))
    i0, i1 = max(0, (y0 - jit) // step - 1), min(n_rows, (y1 + jit + s) // step + 2)
    j0, j1 = max(0, (x0 - jit) // step - 1), min(n_cols, (x1 + jit + s) // step + 2)
    laid = 0
    if i1 > i0 and j1 > j0:
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        dy, dx, idx, kk, ff = lattice_draws(rng_key[0], rng_key[1], ii, jj, len(patches), jit)
        yy, xx = -s + ii * step + dy, -s + jj * step + dx
        # clipped to the canvas, then to the window
        cy0 = np.maximum(np.maximum(yy, 0), y0)
        cy1 = np.minimum(np.minimum(yy + s, H), y1)
        cx0 = np.maximum(np.maximum(xx, 0), x0)
        cx1 = np.minimum(np.minimum(xx + s, W), x1)
        ok = (cy1 > cy0) & (cx1 > cx0)
        for a, b in zip(*np.nonzero(ok)):
            ly0, ly1, lx0, lx1 = int(cy0[a, b] - y0), int(cy1[a, b] - y0), int(cx0[a, b] - x0), int(cx1[a, b] - x0)
            if reach is not None and reach[ly1, lx1] - reach[ly0, lx1] - reach[ly1, lx0] + reach[ly0, lx0] == 0:
                continue
            pw, wnd = weighted(int(idx[a, b]), int(kk[a, b]), bool(ff[a, b]))
            py0, px0 = int(cy0[a, b] - yy[a, b]), int(cx0[a, b] - xx[a, b])
            acc[ly0:ly1, lx0:lx1] += pw[py0:py0 + (ly1 - ly0), px0:px0 + (lx1 - lx0)]
            wsum[ly0:ly1, lx0:lx1] += wnd[py0:py0 + (ly1 - ly0), px0:px0 + (lx1 - lx0)]
            laid += 1

    empty = wsum[:, :, 0] == 0
    wsum[wsum == 0] = 1.0
    out = acc / wsum
    if empty.any() and (~empty).any():
        _, (iy, ix) = ndi.distance_transform_edt(empty, return_indices=True)
        out[empty] = out[iy[empty], ix[empty]]
    return out, laid


# ------------------------------------------------------------------ one tile (a worker's job)

_LIBS = {}


def _library(npz, ratio):
    """A material library scaled to the output (kept between a worker's tiles)."""
    from app import generation as G
    key = (npz, os.path.getmtime(npz), round(float(ratio), 6))
    if key not in _LIBS:
        if len(_LIBS) > 6:
            _LIBS.clear()
        patches, pmasks = G.load_patches(npz, with_masks=True)
        _LIBS[key] = G.scale_patches(patches, ratio, pmasks)
    return _LIBS[key]


def _arr(d, name, win, out=False):
    """A window of one of the run's shared arrays (memory-mapped), or None when it was not written."""
    f = Path(d) / f"{name}.npy"
    if not f.exists():
        return None
    a = np.load(f, mmap_mode="r+" if out else "r")
    if out:
        return a
    x0, y0, x1, y1 = win
    return np.asarray(a[y0:y1, x0:x1])


def make_tile(task):
    """
    One tile's material (the parts blended as app/generation.py blends them) and raw
    wear, written into the run's shared arrays (task["dir"]: material.npy, wear_raw.npy)
    at the tile's core.
    """
    t0 = time.time()
    d = task["dir"]
    H, W = task["shape"]
    cx0, cy0, cx1, cy1 = task["core"]
    x0, y0, x1, y1 = win = task["win"]
    labels = _arr(d, "labels", win)
    road = labels > 0
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    seed, overlap, jitter, feather = int(task["seed"]), float(task["overlap"]), float(task["jitter"]), float(task["feather"])
    info = {"id": task["id"], "road_px": int(road[sl].sum()), "laid": 0, "libs": {}}
    if not road.any():
        # no road within reach: nothing laid (the arrays start at 0, as an empty canvas)
        info["seconds"] = round(time.time() - t0, 3)
        return info
    acc = np.zeros(road.shape + (3,), np.float32)
    wtot = np.zeros(road.shape, np.float32)
    for g in task["groups"]:
        m = labels == CODES[g["name"]]
        if not m.any():
            continue
        patches, pmasks = _library(g["npz"], g["ratio"])
        info["libs"][g["name"]] = [int(len(patches)), int(patches.shape[1])]
        wgt = ndi.gaussian_filter(m.astype(np.float32), feather)
        layer, n = synth_window(win, (H, W), patches, (seed, LAYERS[g["name"]]), overlap, jitter, pmasks, need=wgt > 0)
        acc += layer * wgt[:, :, None]
        wtot += wgt
        info["laid"] += n
    material = acc / np.maximum(wtot, 1e-6)[:, :, None]
    out = _arr(d, "material", None, True)
    out[cy0:cy1, cx0:cx1] = material[sl]
    out.flush()
    del out
    if task.get("wear"):
        wp, wm = _library(task["wear"]["npz"], task["wear"]["ratio"])
        wl, n = synth_window(win, (H, W), wp, (seed, LAYERS["wear"]), overlap, jitter, wm, need=road)
        out = _arr(d, "wear_raw", None, True)
        out[cy0:cy1, cx0:cx1] = wl.mean(axis=2)[sl]
        out.flush()
        del out
        info["laid"] += n
    info["seconds"] = round(time.time() - t0, 3)
    return info


def finish_tile(task):
    """
    One tile's four pictures as app/generation.py saves them (the texture with its lines,
    the material, the wear, the markings), from the run's shared arrays, into its four
    shared pictures at the tile's core. The soft edge's ring takes the colour of the
    nearest road pixel, found within the tile's margin.
    """
    from app import generation as G
    t0 = time.time()
    d, o = task["dir"], task["opts"]
    cx0, cy0, cx1, cy1 = task["core"]
    x0, y0, x1, y1 = win = task["win"]
    sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    road = _arr(d, "road", win)
    soft_edges = bool(o["soft_edges"])
    names = ["result", "material", "wear"] + ([] if o["keep_markings"] else ["markings"])
    if not road.any():
        for name in names:
            out = _arr(d, "out_" + name, None, True)
            out[cy0:cy1, cx0:cx1] = np.asarray(G.BACKGROUND, np.uint8)
            out.flush()
            del out
        return {"id": task["id"], "seconds": round(time.time() - t0, 3)}
    material = _arr(d, "material", win)
    wear_map = _arr(d, "wear_map", win)
    cov = _arr(d, "coverage", win) if soft_edges else road.astype(np.float32)
    paint_rgb = np.asarray(o["paint_rgb"], np.float32)
    zero = np.zeros(road.shape, np.float32)
    paint = _arr(d, "paint", win)
    paint = zero if paint is None else paint
    yellow = _arr(d, "yellow", win)
    island = _arr(d, "island", win)
    island_top = _arr(d, "island_top", win)
    island = zero if island is None else island
    island_top = zero if island_top is None else island_top
    ws = float(o["wear_strength"])
    worn = material * (1.0 - 0.45 * ws * wear_map[:, :, None])
    nearest = ndi.distance_transform_edt(~road, return_indices=True)[1] if soft_edges else None
    pics = {"result": lambda: (G.with_lanes(worn, wear_map, paint, island, island_top, paint_rgb, ws,
                                            float(o["lum_mean"]), yellow), True),
            "material": lambda: (material, True),
            "wear": lambda: (np.dstack([wear_map * (255 if ws > 0 else 0)] * 3), False),
            "markings": lambda: (G.markings_picture(paint, island, paint_rgb, yellow), False)}
    for name in names:
        arr, soft = pics[name]()
        img = G.finish(arr, road, cov, soft_edges, nearest, None, soft)
        out = _arr(d, "out_" + name, None, True)
        out[cy0:cy1, cx0:cx1] = img[sl]
        out.flush()
        del out
    return {"id": task["id"], "seconds": round(time.time() - t0, 3)}


TASKS = {"texture_tile": make_tile, "finish_tile": finish_tile}


# ------------------------------------------------------------------ the whole area

_pool = None
_pool_lock = threading.Lock()


def workers():
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            from app import patches as PT
            _pool = PT.Pool(n=workers(), module="app.tiled")
        return _pool


class Run:
    """
    One texture made in tiles: a folder of the arrays its tiles share (memory-mapped),
    the tiles (the analysis's, at the output's size) and their windows (each tile with a
    margin). cfg: params["tiles"] ("mask_size": the mask's [w, h], "tmp": where the
    folder goes, "local": every tile in this process, one after another, for tests).
    """

    def __init__(self, cfg, shape, scale):
        self.cfg, self.shape, self.scale = cfg, tuple(shape), int(scale)
        H, W = self.shape
        mw, mh = cfg.get("mask_size") or (W // self.scale, H // self.scale)
        tmp = cfg.get("tmp")
        if tmp:
            Path(tmp).mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix="tiles_", dir=tmp))
        M = margin(self.scale)
        self.grid = []
        for t in tile_grid(int(mw), int(mh), self.scale, int(cfg.get("tile") or TILE)):
            x0, y0, x1, y1 = t["core"]
            x1, y1 = min(x1, W), min(y1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            self.grid.append(dict(t, core=[x0, y0, x1, y1],
                                  win=[max(0, x0 - M), max(0, y0 - M), min(W, x1 + M), min(H, y1 + M)]))
        self.reports = {t["id"]: {"id": t["id"], "row": t["row"], "col": t["col"], "core": t["mask_core"]}
                        for t in self.grid}

    def put(self, name, a):
        np.save(self.dir / f"{name}.npy", np.ascontiguousarray(a))

    def blank(self, name, shape, dtype):
        np.lib.format.open_memmap(self.dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape).flush()

    def take(self, name):
        return np.array(np.load(self.dir / f"{name}.npy", mmap_mode="r"))

    def _run(self, kind, extra, on_tile, order=None):
        tasks = [dict(extra, kind=kind, dir=str(self.dir), id=t["id"], shape=list(self.shape), core=t["core"],
                      win=t["win"]) for t in (order or self.grid)]
        got = {}
        if self.cfg.get("local"):
            for task in tasks:
                got[task["id"]] = TASKS[kind](task)
                if on_tile:
                    on_tile(kind, got[task["id"]], len(got), len(tasks))
            return got
        done, lock, failed = threading.Event(), threading.Lock(), []

        def answer(task):
            def fn(res):
                with lock:
                    if not res.get("ok"):
                        failed.append(f"{task['id']}: {res.get('error')}")
                        got[task["id"]] = {"id": task["id"], "failed": True}
                    else:
                        got[task["id"]] = res["info"]
                    if on_tile:
                        try:
                            on_tile(kind, got[task["id"]], len(got), len(tasks))
                        except Exception:
                            pass
                    if len(got) == len(tasks):
                        done.set()
            return fn
        p = pool()
        for task in tasks:
            p.submit({"key": f"{kind}:{self.dir.name}:{task['id']}", "fn": answer(task), "task": task})
        while not done.wait(0.25):
            pass
        if failed:
            raise RuntimeError(f"{len(failed)} texture tile(s) failed: " + "; ".join(failed[:3]))
        return got

    def synthesize(self, groups, libraries, ratios, seed, overlap, jitter, on_tile=None):
        """
        The material and the raw wear of the whole canvas, tile by tile. groups: {name: bool
        map} (app/generation.py's groups_for); ratios: {name: the patches' scale}, "wear"
        too. Returns material (H, W, 3), raw wear (H, W), {group: [patches, size]}.
        """
        try:
            return self._synthesize(groups, libraries, ratios, seed, overlap, jitter, on_tile)
        except BaseException:
            self.close()
            raise

    def _synthesize(self, groups, libraries, ratios, seed, overlap, jitter, on_tile):
        H, W = self.shape
        labels = np.zeros((H, W), np.uint8)
        for name, code in CODES.items():
            if name in groups:
                labels[groups[name]] = code
        self.put("labels", labels)
        self.blank("material", (H, W, 3), np.float32)
        self.blank("wear_raw", (H, W), np.float32)
        glist = []
        for name in groups:
            if not groups[name].any():
                continue
            npz = libraries["material"].get(name) or libraries["material"].get("open")
            if npz:
                glist.append({"name": name, "npz": npz, "ratio": ratios[name]})
        wear_npz = libraries["wear"].get("open") or libraries["wear"].get("junction")
        # the tiles with the most road first: the slowest start first, the empty ones fill the gaps
        order = sorted(self.grid, key=lambda t: -int((labels[t["win"][1]:t["win"][3]:8, t["win"][0]:t["win"][2]:8] > 0).sum()))
        del labels
        got = self._run("texture_tile", {"seed": int(seed), "overlap": overlap, "jitter": jitter,
                                         "feather": 1.5 * self.scale, "groups": glist,
                                         "wear": {"npz": wear_npz, "ratio": ratios["wear"]} if wear_npz else None},
                        on_tile, order)
        libs = {}
        for tid, info in got.items():
            self.reports[tid].update({k: info.get(k) for k in ("road_px", "laid", "seconds")})
            for k, v in (info.get("libs") or {}).items():
                libs.setdefault(k, v)
        material = self.take("material")
        wear_raw = self.take("wear_raw") if wear_npz else None
        for name in ("material", "wear_raw", "labels"):
            (self.dir / f"{name}.npy").unlink(missing_ok=True)
        return material, wear_raw, libs

    def finish(self, arrays, opts, on_tile=None):
        """
        The four pictures (uint8, H x W x 3) from the arrays (material, wear_map, road,
        coverage, paint, and yellow, island, island_top when there are any), tile by tile.
        opts: soft_edges, paint_rgb, wear_strength, lum_mean, keep_markings.
        """
        H, W = self.shape
        for name, a in arrays.items():
            if a is not None:
                self.put(name, a)
        names = ["result", "material", "wear"] + ([] if opts["keep_markings"] else ["markings"])
        for name in names:
            self.blank("out_" + name, (H, W, 3), np.uint8)
        got = self._run("finish_tile", {"opts": opts}, on_tile)
        for tid, info in got.items():
            self.reports[tid]["finish_seconds"] = info.get("seconds")
        return {name: self.take("out_" + name) for name in names}

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------ the asphalt's tone from the satellite

def _cell_median(lum, ok, f, step):
    """The median of each f x f cell's pixels where ok (every step-th pixel each way), NaN where it has under 8."""
    h, w = lum.shape
    nby, nbx = -(-h // f), -(-w // f)
    pad = np.full((nby * f, nbx * f), np.nan, np.float32)
    pad[:h, :w] = np.where(ok, lum, np.nan)
    blocks = pad.reshape(nby, f, nbx, f)[:, ::step, :, ::step].transpose(0, 2, 1, 3).reshape(nby, nbx, -1)
    n = np.isfinite(blocks).sum(axis=2)
    out = np.full((nby, nbx), np.nan, np.float32)
    has = n >= 8
    if has.any():
        out[has] = np.nanmedian(blocks[has], axis=1)
    return out, n


def tone_field(area_dir, analysis, threshold, mpp, strength, shape_out):
    """
    The asphalt's tone the satellite shows, as a gain on the material's brightness (H, W, 1
    at the output's size, 1 = the area's usual asphalt), and a report per tile; (None, {})
    when there is nothing to follow. Read in cells of TONE_CELL_M on the street map's road:
    each cell's median brightness, leaving out what is not asphalt (missing tiles, shadows,
    trees, colours, then paint and cars: far from the asphalt around them), only in tiles
    the analysis rated at or above threshold. Brightness only: a satellite picture's colour
    cast is the camera's, not the road's.
    """
    area_dir = Path(area_dir)
    if not analysis or strength <= 0:
        return None, {}
    sat = np.asarray(Image.open(area_dir / "satellite.png").convert("RGB"), np.float32)
    road = np.asarray(Image.open(area_dir / "alpha.png").convert("L")) > 127
    h, w = road.shape
    if sat.shape[:2] != (h, w):
        return None, {}
    try:
        area = json.loads((area_dir / "area.json").read_text())
    except (OSError, ValueError):
        area = {}
    missing = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in area.get("missing_tiles") or []:
        missing[y0:y1, x0:x1] = True
    lum = sat @ np.array([0.299, 0.587, 0.114], np.float32)
    on = road & ~missing
    if on.sum() < 500:
        return None, {}
    m_l = float(np.median(lum[on][:: max(1, int(on.sum() // 200_000))]))
    mx, mn = sat.max(axis=2), sat.min(axis=2)
    sat_s = (mx - mn) / np.maximum(mx, 1)
    green = (sat[..., 1] > sat[..., 0] + 10) & (sat[..., 1] > sat[..., 2] + 6) & (sat_s > 0.15)
    del sat, mx, mn
    clean = on & (lum > 0.55 * m_l) & ~green & (sat_s < 0.35)
    # the tiles the analysis trusts
    trusted = np.zeros((h, w), bool)
    tiles = analysis.get("tiles") or {}
    for t in tiles.values():
        c = t.get("confidence")
        if t.get("state") == "done" and c is not None and c >= threshold:
            x0, y0, x1, y1 = t["core"]
            trusted[y0:y1, x0:x1] = True
    f = max(2, int(round(TONE_CELL_M / mpp)))
    step = max(1, f // 8)
    nby, nbx = -(-h // f), -(-w // f)
    # each cell's asphalt: a first median, the asphalt around it (10 m), then again without
    # what lies far from that (paint, cars, a patch of sand)
    first, _ = _cell_median(lum, clean, f, step)
    ref = first.copy()
    if np.isfinite(ref).any():
        gaps = ~np.isfinite(ref)
        if gaps.any():
            _, (iy, ix) = ndi.distance_transform_edt(gaps, return_indices=True)
            ref = ref[iy, ix]
        ref = ndi.median_filter(ref, size=5, mode="nearest")
    near = np.repeat(np.repeat(np.nan_to_num(ref, nan=m_l), f, 0), f, 1)[:h, :w]
    keep = clean & (np.abs(lum - near) < 25)
    del near
    val, n = _cell_median(lum, keep, f, step)

    def cells(a):
        pad = np.zeros((nby * f, nbx * f), np.float32)
        pad[:h, :w] = a
        return pad.reshape(nby, f, nbx, f).mean(axis=(1, 3))
    road_cells = cells(road)
    tr = cells(trusted) > 0.5
    samples = ((f + step - 1) // step) ** 2
    good = np.isfinite(val) & (n >= TONE_MIN_SHARE * samples * np.maximum(road_cells, 0.2)) & tr
    if good.sum() < 4:
        return None, {"cells": int(good.sum())}
    usual = float(np.median(val[good]))                           # the area's usual asphalt
    wgt = good.astype(np.float32)
    vals = np.where(good, val, 0).astype(np.float32)
    sv, st = TONE_VALUE_M / (f * mpp), TONE_TRUST_M / (f * mpp)
    num, den = ndi.gaussian_filter(vals * wgt, sv), ndi.gaussian_filter(wgt, sv)
    local = np.where(den > 1e-4, num / np.maximum(den, 1e-6), usual)
    # how much each place trusts the satellite: the share of its road that is in seen cells
    present = (road_cells > 0.05).astype(np.float32)
    trust = np.clip(ndi.gaussian_filter(wgt, st) / np.maximum(ndi.gaussian_filter(present, st), 1e-4), 0, 1)
    ratio = np.clip(1.0 + trust * (local / max(usual, 1.0) - 1.0), *TONE_RANGE)
    gain_cells = (ratio ** float(strength)).astype(np.float32)
    # per tile, for the card: where its tone came from and how far from the usual
    report = {}
    for tid, t in tiles.items():
        x0, y0, x1, y1 = t["core"]
        cy0, cy1, cx0, cx1 = y0 // f, max(y0 // f + 1, -(-y1 // f)), x0 // f, max(x0 // f + 1, -(-x1 // f))
        rc = road_cells[cy0:cy1, cx0:cx1] > 0.05
        if not rc.any():
            report[tid] = {"tone": "no road"}
            continue
        c = t.get("confidence")
        report[tid] = {"tone": "satellite" if (t.get("state") == "done" and c is not None and c >= threshold) else "rules",
                       "gain": round(float(gain_cells[cy0:cy1, cx0:cx1][rc].mean()), 3)}
    return _upsample(gain_cells[..., None], f, (h, w), shape_out), {
        "tiles": report, "usual": round(usual, 1), "cells": int(good.sum()), "cell_px": f,
        "lighter_max": round(float(gain_cells.max()), 3), "darker_min": round(float(gain_cells.min()), 3)}


def _upsample(cells, f, mask_shape, shape_out):
    """A field of cells (each f mask pixels) at every output pixel, linear between cell centres."""
    H, W = shape_out
    h, w = mask_shape
    sy, sx = H / h, W / w
    ny, nx = cells.shape[:2]
    yc = np.clip(((np.arange(H) + 0.5) / sy) / f - 0.5, 0, ny - 1)
    xc = np.clip(((np.arange(W) + 0.5) / sx) / f - 0.5, 0, nx - 1)
    y0 = np.minimum(yc.astype(np.int64), ny - 1)
    x0 = np.minimum(xc.astype(np.int64), nx - 1)
    y1, x1 = np.minimum(y0 + 1, ny - 1), np.minimum(x0 + 1, nx - 1)
    wy, wx = (yc - y0).astype(np.float32)[:, None, None], (xc - x0).astype(np.float32)[None, :, None]
    rows = cells[y0] * (1 - wy) + cells[y1] * wy                     # (H, nx, 3)
    return (rows[:, x0] * (1 - wx) + rows[:, x1] * wx).astype(np.float32)


# ------------------------------------------------------------------ the worker

def _worker():
    """A worker's loop: one tile a line in, one answer a line out (as app/patches.py's)."""
    out = sys.stdout
    sys.stdout = sys.stderr
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            task = json.loads(line)
            ans = {"ok": True, "info": TASKS[task.get("kind", "texture_tile")](task)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            ans = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        ans["seconds"] = round(time.time() - t0, 2)
        out.write(json.dumps(ans, default=float) + "\n")
        out.flush()


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
