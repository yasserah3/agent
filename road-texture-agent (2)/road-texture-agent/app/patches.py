"""
Big maps in patches: the 3D model of a city built one square kilometre at a time.

A whole city in one model took 23 minutes and 10.6 GB of memory to build, and
the view then held all of it. Built in patches instead:

- Patches are squares of the map in metres (PATCH_M, 1 km), whatever its scale.
- What every patch shares is worked out once for the whole map (prepare):
  the cleaned mask and its coverage, the junctions and streets (numbered once,
  so a street keeps its width, lanes and lines in every patch it crosses), the
  variation layer, and the paved areas between the roads (each laid by its own
  number and whole outline). It is kept on disk; the workers read it without
  each holding a copy.
- Each patch is built from its square and MARGIN_M of its neighbours around it
  (a street or block cut at the patch's edge is the same on both sides), in the
  whole map's coordinates, and then cut exactly at its square (clip_scene): the
  triangles crossing an edge are cut there, so patches side by side meet with
  no gap and no overlap. Objects and street lamps belong to the patch their foot
  stands in.
- Patches are built in parallel, in worker processes (Pool), nearest the
  camera first, and kept by their place on the map (their key: what they were
  built from, the parts inside their own window included), so a change builds
  again only the patches it reaches.
"""

import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from app import lanes as LN

PATCH_M = 1000.0          # a patch's side (metres)
WORKER_GB = 1.5           # what a worker may take at most (1.4 GB measured on a 12 x 9 km city at 2.5 m a pixel)
MARGIN_M = 100.0          # built with this much of its neighbours around it, then cut to its square
BIG_KM2 = 4.0             # maps bigger than this are built in patches


# ------------------------------------------------------------------ the grid
def grid(W, H, mpp, patch_m=PATCH_M, road=None):
    """
    The patches of a map W x H mask pixels at mpp metres a pixel, row by row:
    [{"id": "i_j", "i", "j", "keep": (x0, y0, x1, y1) its square in mask
    pixels, "m": (x0, z0, x1, z1) the same in the model's metres, "road": how
    much of it is road (0 to 1; road: the mask, True on road)}].
    """
    side = patch_m / mpp
    nx, ny = max(1, math.ceil(W / side - 1e-9)), max(1, math.ceil(H / side - 1e-9))
    out = []
    for j in range(ny):
        for i in range(nx):
            keep = (i * side, j * side, min(float(W), (i + 1) * side), min(float(H), (j + 1) * side))
            m = (keep[0] * mpp - W * mpp / 2, keep[1] * mpp - H * mpp / 2,
                 keep[2] * mpp - W * mpp / 2, keep[3] * mpp - H * mpp / 2)
            share = 1.0
            if road is not None:
                sub = road[int(keep[1]):int(math.ceil(keep[3])), int(keep[0]):int(math.ceil(keep[2]))]
                share = float(sub.mean()) if sub.size else 0.0
            out.append({"id": f"{i}_{j}", "i": i, "j": j, "keep": tuple(float(v) for v in keep),
                        "m": tuple(round(float(v), 4) for v in m), "road": round(share, 4)})
    return out


def window_of(keep, W, H, mpp, margin_m=MARGIN_M):
    """A patch's square with its margin, inside the map (mask pixels)."""
    m = margin_m / mpp
    return (max(0.0, keep[0] - m), max(0.0, keep[1] - m), min(W - 1.0, keep[2] + m), min(H - 1.0, keep[3] + m))


def is_big(W, H, mpp):
    """Whether a map is big enough to be built in patches."""
    return W * H * mpp * mpp / 1e6 > BIG_KM2


# ------------------------------------------------------------- the cut
def _clip_poly(poly, axis, value, keep_below):
    """
    Sutherland-Hodgman against one plane: poly is [(t, a, b, c)] corners as
    (weights over the triangle's three corners, x, z); the part on one side of
    x (axis 0) or z (axis 1) = value.
    """
    out = []
    n = len(poly)
    for k in range(n):
        cur, nxt = poly[k], poly[(k + 1) % n]
        cv, nv = cur[1 + axis], nxt[1 + axis]
        c_in = cv <= value if keep_below else cv >= value
        n_in = nv <= value if keep_below else nv >= value
        if c_in:
            out.append(cur)
        if c_in != n_in:
            t = (value - cv) / (nv - cv)
            w = tuple(cur[0][q] + (nxt[0][q] - cur[0][q]) * t for q in range(3))
            x = cur[1] + (nxt[1] - cur[1]) * t
            z = cur[2] + (nxt[2] - cur[2]) * t
            if axis == 0:
                x = value
            else:
                z = value
            out.append((w, x, z))
    return out


def clip_prim(p, rect):
    """
    A primitive (app/model3d.py: positions, normals, uv0, colors, indices)
    cut to rect = (x0, z0, x1, z1) in the model's metres: the triangles inside
    kept as they are, those across an edge cut exactly at it (new corners with
    every value interpolated), the rest left out. None when nothing is left.
    """
    x0, z0, x1, z1 = rect
    P = np.asarray(p["positions"], float)
    T = np.asarray(p["indices"], np.int64).reshape(-1, 3)
    if not len(T):
        return None
    X, Z = P[:, 0], P[:, 2]
    vin = (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1)
    tin = vin[T].all(axis=1)
    tx, tz = X[T], Z[T]
    out_any = ((tx < x0).all(axis=1) | (tx > x1).all(axis=1) | (tz < z0).all(axis=1) | (tz > z1).all(axis=1))
    cross = ~tin & ~out_any
    attrs = [k for k in ("positions", "normals", "uv0", "colors") if p.get(k) is not None]
    A = {k: np.asarray(p[k], float) for k in attrs}
    new = {k: [] for k in attrs}
    new_tris = []
    base = len(P)
    for t in T[cross]:
        poly = [((1.0, 0.0, 0.0), X[t[0]], Z[t[0]]), ((0.0, 1.0, 0.0), X[t[1]], Z[t[1]]),
                ((0.0, 0.0, 1.0), X[t[2]], Z[t[2]])]
        for axis, value, below in ((0, x0, False), (0, x1, True), (1, z0, False), (1, z1, True)):
            poly = _clip_poly(poly, axis, value, below)
            if len(poly) < 3:
                break
        if len(poly) < 3:
            continue
        ids = []
        for w, x, z in poly:
            for k in attrs:
                v = w[0] * A[k][t[0]] + w[1] * A[k][t[1]] + w[2] * A[k][t[2]]
                if k == "positions":
                    v = np.array([x, v[1], z])
                elif k == "normals":
                    nv = np.linalg.norm(v)
                    v = v / nv if nv > 1e-12 else A[k][t[0]]
                new[k].append(v)
            ids.append(base + len(new["positions"]) - 1)
        for q in range(1, len(ids) - 1):
            a, b, c = ids[0], ids[q], ids[q + 1]
            new_tris.append((a, b, c))
    keep_t = [T[tin]]
    if new_tris:
        keep_t.append(np.array(new_tris, np.int64))
    T2 = np.concatenate(keep_t) if keep_t else np.zeros((0, 3), np.int64)
    full = {k: (np.vstack([A[k], np.array(new[k])]) if new[k] else A[k]) for k in attrs}
    if new_tris:
        # cut pieces with (next to) no area left out: a sliver along the edge
        Pp = full["positions"]
        e1, e2 = Pp[T2[:, 1]] - Pp[T2[:, 0]], Pp[T2[:, 2]] - Pp[T2[:, 0]]
        area = np.linalg.norm(np.cross(e1, e2), axis=1)
        T2 = T2[(area > 1e-10) | (np.arange(len(T2)) < len(keep_t[0]))]
    if not len(T2):
        return None
    used = np.unique(T2)
    remap = np.full(len(full["positions"]), -1, np.int64)
    remap[used] = np.arange(len(used))
    q = dict(p)
    for k in attrs:
        q[k] = full[k][used]
    q["indices"] = remap[T2]
    return q


def clip_scene(scene, rect):
    """
    A scene (meshes, materials, images) cut to rect (x0, z0, x1, z1), the
    model's metres: every mesh's primitives cut (clip_prim); a mesh placed
    many times (objects) keeps the copies standing inside, on its west and
    north edges included and its east and south edges not, so a copy on the
    line between two patches is in one of them.
    """
    meshes, materials, images = scene
    x0, z0, x1, z1 = rect
    out = []
    for m in meshes:
        if m.get("instances"):
            inst = [it for it in m["instances"] if x0 <= it[0][0] < x1 and z0 <= it[0][2] < z1]
            if inst:
                out.append(dict(m, instances=inst))
            continue
        prims = [q for q in (clip_prim(p, rect) for p in m["primitives"]) if q is not None]
        if prims:
            out.append(dict(m, primitives=prims))
    return out, materials, images


# ------------------------------------------------------------- what the patches share
_LOADED = {}                # store directory -> what load() gave (one copy per process)


def prepare(store, mask_path, base_mask_path, output_scale, mpp, result_path, markings_path, variation,
            divider_limit=None):
    """
    Work out what every patch of a map shares and keep it in the folder store
    (arrays as .npy files the workers map instead of copying; the rest as one
    pickle). base_mask_path: the mask without the inner streets (they are laid
    as exact shapes), else None. Returns the folder.
    """
    from PIL import Image
    from scipy import ndimage as ndi
    from app.generation import prepare_mask
    from app import quadmesh as QM
    from app import model3d as M3
    from app import junctions as J
    from app import progress as prog
    store = Path(store)
    tmp = store.with_name(store.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    prog.stage("prepare", "the whole map once: its mask, junctions and streets")
    gray = np.array(Image.open(mask_path).convert("L"))
    base = np.array(Image.open(base_mask_path).convert("L")) if base_mask_path else gray
    mask1, cov1 = prepare_mask(base, 1)
    coverage = cov1 if (output_scale == 1 and base_mask_path is None) else prepare_mask(gray, output_scale)[1]
    city = QM.city(mask1, mpp, cov1, divider_limit if divider_limit is not None else J.DIVIDER_LIMIT_PX)
    prog.stage("prepare_layers", "the whole map once: its variation layer and paved areas")
    mpp_out = mpp / output_scale
    fac = M3.variation_factor(result_path, markings_path, coverage, mpp_out, variation)
    fb = ndi.gaussian_filter(fac, sigma=max(1.0, 25.0 / mpp_out))
    lab, n = ndi.label(cov1 < 0.5)
    det = city["det"]
    # the widest street, by its usual width (painted markings: what their textures span
    # across); not the widest point of the centreline, which is in a junction or a square
    meds = [float(np.median(det["dt"][c[:, 0], c[:, 1]])) for c in city["seg_px"][1:] if len(c)]
    widest_m = min(150.0, 2 * max(meds) * mpp) if meds else 20.0
    arrays = {"gray": gray, "mask1": mask1, "cov1": cov1.astype(np.float32),
              "coverage": coverage.astype(np.float32), "dt": det["dt"].astype(np.float32),
              "seg_lab": city["seg_lab"].astype(np.int32), "fac": fac.astype(np.float32),
              "fb": fb.astype(np.float32), "blocks": lab.astype(np.int32)}
    for k, a in arrays.items():
        np.save(tmp / f"{k}.npy", a)
    rest = {"junctions": det["junctions"], "clusters": city["clusters"], "piece_of": city["piece_of"],
            "seg_px": city["seg_px"], "road_w_m": city["road_w_m"], "boxes": city["boxes"],
            "shape": tuple(mask1.shape), "slices": ndi.find_objects(lab), "widest_m": widest_m,
            "mpp": mpp, "output_scale": output_scale}
    with open(tmp / "shared.pkl", "wb") as f:
        pickle.dump(rest, f, protocol=pickle.HIGHEST_PROTOCOL)
    (tmp / "done").write_text("1")
    if store.exists():
        import shutil
        shutil.rmtree(store, ignore_errors=True)
    os.replace(tmp, store)
    return store


def load(store):
    """What prepare kept, as export_road_tiled_glb's shared (arrays mapped from disk, once per process)."""
    store = str(store)
    got = _LOADED.get(store)
    if got is not None:
        return got
    p = Path(store)
    a = {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in
         ("gray", "mask1", "cov1", "coverage", "dt", "seg_lab", "fac", "fb", "blocks")}
    with open(p / "shared.pkl", "rb") as f:
        r = pickle.load(f)
    city = {"det": {"dt": a["dt"], "junctions": r["junctions"]}, "clusters": r["clusters"],
            "seg_lab": a["seg_lab"], "piece_of": r["piece_of"], "seg_px": r["seg_px"],
            "road_w_m": r["road_w_m"], "boxes": r["boxes"], "shape": r["shape"]}
    got = {"gray": a["gray"], "mask1": a["mask1"], "cov1": a["cov1"], "coverage": a["coverage"],
           "fac": a["fac"], "fb": a["fb"], "city": city, "widest_m": r["widest_m"],
           "blocks": {"labels": a["blocks"], "slices": r["slices"], "scale": r["output_scale"], "theta": {}}}
    _LOADED.clear()                     # one map at a time
    _LOADED[store] = got
    return got


def build(store, args_path, patch, out_path):
    """
    One patch's model: export_road_tiled_glb with the map's shared store, the
    export's settings (args_path: a pickle of its keyword arguments) and the
    patch's square, to out_path. Returns its report (lamps included).
    """
    from app import model3d as M3
    shared = load(store)
    args = _args(args_path)
    H, W = shared["mask1"].shape
    keep = tuple(patch["keep"])
    window = window_of(keep, W, H, args["metres_per_pixel"])
    tmp = Path(out_path).with_suffix(".part.glb")
    info = M3.export_road_tiled_glb(**dict(args, out_path=tmp), shared=shared, window=window, keep=keep)
    os.replace(tmp, out_path)
    return {k: v for k, v in info.items() if not k.startswith("_") and k != "layouts"}


_ARGS = {}


def _args(path):
    path = str(path)
    if path not in _ARGS:
        with open(path, "rb") as f:
            _ARGS.clear()
            _ARGS[path] = pickle.load(f)
    return _ARGS[path]


def save_args(folder, args):
    """The export's settings for the workers, by their content (the same settings: the same file)."""
    blob = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
    h = hashlib.sha1(blob).hexdigest()[:16]
    p = Path(folder) / f"args_{h}.pkl"
    if not p.exists():
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, p)
    return p, h


# ------------------------------------------------------------- what reaches a patch
class Streets:
    """
    Which street a point is on (the nearest street's centreline pixel) and each
    street's box, from a map's store: a lane choice or a junction's line stop
    reaches every patch its street crosses.
    """

    def __init__(self, store):
        from scipy.spatial import cKDTree
        with open(Path(store) / "shared.pkl", "rb") as f:
            r = pickle.load(f)
        have = [k for k in range(1, len(r["seg_px"])) if len(r["seg_px"][k])]
        pts = np.concatenate([r["seg_px"][k][:, ::-1] for k in have]).astype(float) if have else np.zeros((1, 2))
        self.labels = np.concatenate([np.full(len(r["seg_px"][k]), k) for k in have]) if have else np.zeros(1, int)
        self.tree = cKDTree(pts)
        self.boxes = r["boxes"]

    def box_of(self, x, y):
        _, i = self.tree.query((float(x), float(y)))
        return self.boxes.get(int(self.labels[i]))


def _sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _reaches(box, window):
    return box is not None and not (box[2] < window[0] or box[0] > window[2] or box[3] < window[1] or box[1] > window[3])


def part_sigs(patches, W, H, mpp, base, lane_picks=None, ends=None, mark_area=None, decals=None, scatter=None,
              islands=None, streets=None, centre_groups=None):
    """
    Each patch's key: the settings every patch shares (base, a key) and the
    parts that reach into its window: the decals and objects near it, its
    islands' materials, the Lanes tool's choices and the junctions' line stops
    on the streets that cross it (streets: Streets; None: all of them), the
    markings area if it reaches it. A change makes a new key only for the
    patches it reaches, and only those are built again.
    """
    out = {}
    # a highway's centreline (app/highways.py) reaches every patch along it: so do the lane
    # choices made on it
    chains = LN.chains_of(ends)
    cboxes = []
    for c in chains:
        P = np.asarray(c["pts"], float)
        m = max(c["w"]) / mpp if c["w"] else 0.0
        cboxes.append((P[:, 0].min() - m, P[:, 1].min() - m, P[:, 0].max() + m, P[:, 1].max() + m))
    cidx = LN.ChainIndex(chains, mpp) if chains else None
    which = (np.concatenate([np.full(len(c["pts"]), k) for k, c in enumerate(chains) if len(c["pts"])])
             if chains else None)

    def pick_box(p):
        b = streets.box_of(*p["pt"]) if streets else None
        if cidx is not None and cidx.ok:
            d, i = cidx.tree.query(p["pt"])
            if d <= cidx.W[i] / 2 / mpp + 1.0:
                return cboxes[int(which[i])]
        return b
    picks = [(p, pick_box(p) if streets else None) for p in (lane_picks or [])]
    groups = [(p, pick_box(p) if streets else None) for p in (centre_groups or [])]
    def end_box(e):
        if e.get("reserve"):
            a = e["reserve"]; r = math.hypot(a["w"], a["h"]) / 2
            return (a["cx"]-r, a["cy"]-r, a["cx"]+r, a["cy"]+r)
        return (cboxes[chains.index(e)] if e.get("chain") else streets.box_of(e["x"], e["y"])) if streets else None
    ends_b = [(e, end_box(e)) for e in (ends or [])]
    area_box = None
    if mark_area:                                       # the turned rectangle's reach all round its middle
        r = math.hypot(mark_area["w"], mark_area["h"]) / 2
        area_box = (mark_area["cx"] - r, mark_area["cy"] - r, mark_area["cx"] + r, mark_area["cy"] + r)
    dpl = (decals or {}).get("placements") or []
    reach = max([math.hypot(p.get("w_m", 1.0), p.get("h_m", 1.0)) for p in dpl] + [1.0]) / mpp
    for p in patches:
        win = window_of(p["keep"], W, H, mpp)
        parts = {
            "picks": [q for q, b in picks if streets is None or _reaches(b, win)],
            "centre_groups": [q for q,b in groups if streets is None or _reaches(b,win)],
            "ends": [e for e, b in ends_b if streets is None or _reaches(b, win)],
            "area": None if not mark_area else (mark_area if area_box is None or _reaches(area_box, win) else "outside"),
            "decals": [[q.get("decal"), q["x"], q["y"], q.get("ux"), q.get("uy"), q.get("w_m"), q.get("h_m")]
                       for q in dpl if win[0] - reach <= q["x"] <= win[2] + reach
                       and win[1] - reach <= q["y"] <= win[3] + reach],
            "decal_mode": (decals or {}).get("mode"),
        }
        if scatter:
            r = 300.0 / mpp
            parts["scatter"] = [q for q in scatter["placements"]
                                if win[0] - r - q.get("length", 0) <= q["cx"] <= win[2] + r + q.get("length", 0)
                                and win[1] - r - q.get("length", 0) <= q["cy"] <= win[3] + r + q.get("length", 0)]
        if islands:
            lab = islands["labels"]
            sc = lab.shape[1] / W
            sub = lab[int(win[1] * sc):int(math.ceil(win[3] * sc)) + 1, int(win[0] * sc):int(math.ceil(win[2] * sc)) + 1]
            here = set(int(v) for v in np.unique(sub)) - {0}
            slots = {int(k): int(v) for k, v in islands["slot_of"].items() if int(k) in here}
            parts["islands"] = [slots, {s: islands["looks"].get(s) for s in sorted(set(slots.values()))}]
        out[p["id"]] = _sha([base, parts])
    return out


# ------------------------------------------------------------- the workers
def free_gb():
    """The memory free for new work, in GB (None if this system does not say)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1e6
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullAvailPhys / 1e9
    except Exception:
        pass
    return None


def workers():
    """How many patches to build at once: all cores but one, as many as the free memory holds (2 GB kept back)."""
    n = max(1, (os.cpu_count() or 2) - 1)
    gb = free_gb()
    if gb is None:
        return min(n, 4)
    return max(1, min(n, int((gb - 2.0) / WORKER_GB)))


class Pool:
    """
    Worker processes that build patches (python -m app.patches --worker): each
    reads one job a line on its input and answers one line when done. Jobs
    wait in a list the server may reorder (nearest the camera first); a worker
    takes the first when it is free. A worker that dies is started again and
    its job reported failed.
    """

    def __init__(self, n=None, module="app.patches"):
        self.n = n or workers()
        self.module = module                  # the workers' module (python -m <module> --worker)
        self.lock = threading.Condition()
        self.queue = []                       # [job]: {"key", "fn": callable(result) , "task": dict}
        self.busy = {}                        # worker index -> job
        self.procs = [None] * self.n
        self.threads = []
        for k in range(self.n):
            t = threading.Thread(target=self._run, args=(k,), daemon=True)
            t.start()
            self.threads.append(t)

    def _start(self):
        root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.Popen([sys.executable, "-m", self.module, "--worker"], cwd=root, env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)

    def submit(self, job):
        with self.lock:
            self.queue.append(job)
            self.lock.notify()

    def reorder(self, key_fn):
        """Sort the waiting jobs (key_fn: job -> sort key), e.g. by distance to the camera."""
        with self.lock:
            self.queue.sort(key=key_fn)

    def drop(self, pred):
        """Take the waiting jobs that match out of the list."""
        with self.lock:
            gone = [j for j in self.queue if pred(j)]
            self.queue = [j for j in self.queue if not pred(j)]
        return gone

    def waiting(self):
        with self.lock:
            return list(self.queue)

    def _run(self, k):
        while True:
            with self.lock:
                while not self.queue:
                    self.lock.wait()
                job = self.queue.pop(0)
                self.busy[k] = job
            res = None
            try:
                p = self.procs[k]
                if p is None or p.poll() is not None:
                    p = self.procs[k] = self._start()
                t0 = time.time()
                p.stdin.write(json.dumps(job["task"]) + "\n")
                p.stdin.flush()
                line = p.stdout.readline()
                if not line:
                    raise RuntimeError("the worker stopped (out of memory?)")
                res = json.loads(line)
                res.setdefault("seconds", round(time.time() - t0, 1))
            except Exception as e:
                res = {"ok": False, "error": str(e)}
                try:
                    if self.procs[k] is not None:
                        self.procs[k].kill()
                except Exception:
                    pass
                self.procs[k] = None
            finally:
                with self.lock:
                    self.busy.pop(k, None)
            try:
                job["fn"](res)
            except Exception as e:                      # the server's own bookkeeping
                print("patch result could not be kept:", repr(e))


def _worker():
    """A worker's loop: one job a line in, one answer a line out."""
    out = sys.stdout
    sys.stdout = sys.stderr                             # anything printed while building goes to the log
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            task = json.loads(line)
            if task.get("kind") == "prepare":
                prepare(task["store"], task["mask"], task.get("base"), task["output_scale"], task["mpp"],
                        task["result"], task.get("markings"), task["variation"])
                ans = {"ok": True, "info": {}}
            else:
                info = build(task["store"], task["args"], task["patch"], task["out"])
                ans = {"ok": True, "info": info}
        except Exception as e:
            import traceback
            traceback.print_exc()
            ans = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            import resource
            ans["peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        except Exception:
            pass
        ans["seconds"] = round(time.time() - t0, 1)
        out.write(json.dumps(ans, default=str) + "\n")
        out.flush()


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker()
