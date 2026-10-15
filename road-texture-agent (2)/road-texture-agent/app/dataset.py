"""
The dataset: what the satellite analysis of a Coordinates area found (app/sat_analysis.py),
kept as training examples when you ask (Save to the dataset on the satellite card), only
the kinds you tick. Saving runs in the background, one area after another, each kind
with its own progress.

workspace/dataset/
  dataset.json               every kind's total, and the areas saved (their bounds, zoom,
                             metres a pixel, when, and how many of each kind)
  <kind>/index.jsonl         one line for each example: its id (area_finding), its picture,
                             where it is (longitude, latitude, the area's pixels), its label
                             (kind, angle, size in metres, which way an arrow points and its
                             kind, a solid line's points), how sure (its score, its tile's
                             confidence) and the analysis version
  <kind>/<area>_<id>.png     the satellite round it, at the area's own resolution: its box
                             and 2 m all round (a solid line: its points' box)
  road_tiles/<area>_<tile>_sat.png and _alpha.png
                             each analysed tile with road: the satellite and the alpha
                             (white road) of the same square, its confidence and reasons
                             in index.jsonl: the pairs the layout is learned from

Saving an area again replaces its examples of the kinds saved (no example twice).
"""
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

KINDS = [("zebra", "Zebra crossings"), ("stop_line", "Stop lines"), ("giveway", "Give way lines"),
         ("cycle_crossing", "Cyclist crossings"), ("arrow", "Arrows"), ("parking_bay", "Parking bays"),
         ("lane_dash", "Lane lines (dashed)"), ("lane_solid", "Lane lines (solid)"),
         ("car", "Cars (what hides the road)"), ("unknown", "Unknown paint (to name later)")]
ROAD = "road_tiles"
LABELS = dict(KINDS, **{ROAD: "Road tiles (satellite with its alpha)"})
MARGIN_M = 2.0                       # the satellite kept round a mark
MIN_CROP_PX = 24
KEEP_JOBS = 20


def _read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _totals(root):
    """How many examples of each kind the dataset holds, and the areas in it."""
    d = _read(Path(root) / "dataset.json", {})
    return d.get("totals") or {}, d.get("areas") or {}


def categories(root, folder):
    """
    What an area's analysis can give the dataset: each kind it found (how many in the
    area, how many the dataset holds already, how many of them from this area), and its
    road tiles. [] until the analysis is done.
    """
    folder = Path(folder)
    fj, tj = folder / "analysis" / "findings.json", folder / "analysis" / "tiles.json"
    if not fj.exists() or not tj.exists():
        return []
    found = _read(fj, {}).get("findings") or []
    tiles = _read(tj, {}).get("tiles") or []
    totals, areas = _totals(root)
    mine = (areas.get(folder.name) or {}).get("counts") or {}
    counts = {}
    for f in found:
        counts[f["cls"]] = counts.get(f["cls"], 0) + 1
    out = [{"id": k, "label": label, "count": counts[k], "in_dataset": totals.get(k, 0), "from_area": mine.get(k, 0)}
           for k, label in KINDS if counts.get(k)]
    rated = sum(1 for t in tiles if t.get("confidence") is not None)
    if rated:
        out.append({"id": ROAD, "label": LABELS[ROAD], "count": rated, "in_dataset": totals.get(ROAD, 0),
                    "from_area": mine.get(ROAD, 0)})
    return out


# ------------------------------------------------------------------ saving

def _crop_box(f, mpp, w, h):
    """The satellite to keep round a finding: [x0, y0, x1, y1] in the area's pixels."""
    m = MARGIN_M / mpp
    if f.get("pts"):
        xs, ys = [p[0] for p in f["pts"]], [p[1] for p in f["pts"]]
        x0, x1, y0, y1 = min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m
    else:
        r = max(f.get("length_m") or 0, f.get("width_m") or 0, f.get("across_m") or 0) / 2 / mpp + m
        r = max(r, MIN_CROP_PX / 2)
        x0, x1, y0, y1 = f["x"] - r, f["x"] + r, f["y"] - r, f["y"] + r
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    return [x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)]


LABEL_KEYS = ("angle", "road_angle", "road_sure", "length_m", "width_m", "across_m", "stripe_m", "n", "toward", "kind",
              "kind_match", "yellow", "contrast", "score")


def _clear_area(kind_dir, key):
    """An area's earlier examples of a kind: their pictures gone, their lines out of the index."""
    for p in kind_dir.glob(f"{key}_*.png"):
        p.unlink(missing_ok=True)
    idx = kind_dir / "index.jsonl"
    if idx.exists():
        keep = [ln for ln in idx.read_text().splitlines() if ln.strip() and json.loads(ln).get("area") != key]
        idx.write_text("".join(ln + "\n" for ln in keep))


def save_kind(root, folder, area, kind, step=None, stop=None):
    """
    Keep an area's examples of one kind in the dataset (replacing the area's earlier
    ones). step(done, total) after each; stop() true: end early (what is kept so far
    stays). Returns how many were kept.
    """
    root, folder = Path(root), Path(folder)
    key, mpp = folder.name, float(area["mpp"])
    an = folder / "analysis"
    sat = np.load(an / "sat.npy", mmap_mode="r")
    h, w = sat.shape[:2]
    kind_dir = root / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    _clear_area(kind_dir, key)
    tiles = {t["id"]: t for t in (_read(an / "tiles.json", {}).get("tiles") or [])}
    version = _read(an / "findings.json", {}).get("version")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    base = {"area": key, "mpp": round(mpp, 6), "zoom": area.get("zoom"), "analysis_version": version, "saved": stamp}
    kept = 0
    with open(kind_dir / "index.jsonl", "a") as idx:
        if kind == ROAD:
            road = np.load(an / "road.npy", mmap_mode="r")
            items = [t for t in tiles.values() if t.get("confidence") is not None]
            for i, t in enumerate(items):
                if stop and stop():
                    break
                x0, y0, x1, y1 = t["core"]
                name = f"{key}_{t['id']}"
                Image.fromarray(np.ascontiguousarray(sat[y0:y1, x0:x1])).save(kind_dir / f"{name}_sat.png")
                Image.fromarray((np.asarray(road[y0:y1, x0:x1]) * 255).astype(np.uint8)).save(kind_dir / f"{name}_alpha.png")
                idx.write(json.dumps(dict(base, id=name, sat=f"{name}_sat.png", alpha=f"{name}_alpha.png", tile=t["id"],
                                          box=[x0, y0, x1, y1], confidence=t.get("confidence"),
                                          confidence_marks=t.get("confidence_marks"), reasons=t.get("reasons"),
                                          counts=t.get("counts")), separators=(",", ":")) + "\n")
                kept += 1
                if step:
                    step(i + 1, len(items))
            return kept
        found = [f for f in (_read(an / "findings.json", {}).get("findings") or []) if f.get("cls") == kind]
        for i, f in enumerate(found):
            if stop and stop():
                break
            x0, y0, x1, y1 = _crop_box(f, mpp, w, h)
            name = f"{key}_{f['id']}"
            Image.fromarray(np.ascontiguousarray(sat[y0:y1, x0:x1])).save(kind_dir / f"{name}.png")
            t = tiles.get(f.get("tile")) or {}
            label = {k: f[k] for k in LABEL_KEYS if k in f}
            if f.get("pts"):
                label["pts"] = [[round(p[0] - x0, 1), round(p[1] - y0, 1)] for p in f["pts"]]
            idx.write(json.dumps(dict(base, id=name, image=f"{name}.png", cls=kind, lon=f.get("lon"), lat=f.get("lat"),
                                      at=[f["x"], f["y"]], box=[x0, y0, x1, y1],
                                      centre=[round(f["x"] - x0, 1), round(f["y"] - y0, 1)], label=label,
                                      tile=f.get("tile"), tile_confidence=t.get("confidence"),
                                      tile_confidence_marks=t.get("confidence_marks")), separators=(",", ":")) + "\n")
            kept += 1
            if step:
                step(i + 1, len(found))
    return kept


def _recount(root, folder, area, saved):
    """dataset.json again: every kind's total from its index, and this area's entry."""
    root = Path(root)
    d = _read(root / "dataset.json", {})
    totals = {}
    for k in [k for k, _ in KINDS] + [ROAD]:
        idx = root / k / "index.jsonl"
        if idx.exists():
            totals[k] = sum(1 for ln in idx.read_text().splitlines() if ln.strip())
    areas = d.get("areas") or {}
    key = Path(folder).name
    entry = areas.get(key) or {}
    entry.update(bounds=area.get("bounds"), zoom=area.get("zoom"), mpp=area.get("mpp"),
                 saved=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 counts=dict(entry.get("counts") or {}, **saved))
    areas[key] = entry
    tmp = root / "dataset.json.tmp"
    tmp.write_text(json.dumps({"version": 1, "totals": totals, "areas": areas}, indent=1))
    tmp.replace(root / "dataset.json")
    return totals


# ------------------------------------------------------------------ in the background

_JOBS = {}                           # id -> job (what the page shows)
_QUEUE = []
_LOCK = threading.Lock()
_WAKE = threading.Event()
_WORKER = None


def _public(j):
    """A job as the page shows it: each kind's progress, the time so far and left."""
    now = time.time()
    total = sum(c["total"] for c in j["cats"]) or 0
    done = sum(c["done"] for c in j["cats"])
    t0 = j.get("started")
    elapsed = (j.get("finished") or now) - t0 if t0 else 0.0
    left = None
    if j["state"] == "running" and done > 0 and total > done:
        rate = done / max(elapsed, 1e-3)
        left = (total - done) / rate
    cats = []
    for c in j["cats"]:
        cl = None
        if j["state"] == "running" and done > 0 and c["total"] > c["done"]:
            cl = (c["total"] - c["done"]) / (done / max(elapsed, 1e-3))
        cats.append(dict(c, left=None if cl is None else round(cl, 1)))
    return {k: j[k] for k in ("id", "key", "title", "state", "error", "kept", "root")} | {
        "done": done, "total": total, "elapsed": round(elapsed, 1), "left": None if left is None else round(left, 1),
        "cats": cats, "queued": j.get("queued")}


def jobs():
    with _LOCK:
        return [_public(j) for j in sorted(_JOBS.values(), key=lambda j: j["queued"], reverse=True)]


def start(root, folder, area, kinds, title):
    """Save an area's chosen kinds in the background; returns the job (it waits its turn behind others)."""
    global _WORKER
    folder = Path(folder)
    found = _read(folder / "analysis" / "findings.json", {}).get("findings") or []
    tiles = _read(folder / "analysis" / "tiles.json", {}).get("tiles") or []
    cats = []
    for k in kinds:
        n = (sum(1 for t in tiles if t.get("confidence") is not None) if k == ROAD
             else sum(1 for f in found if f.get("cls") == k))
        cats.append({"id": k, "label": LABELS.get(k, k), "total": n, "done": 0, "state": "waiting", "kept": 0})
    j = {"id": uuid.uuid4().hex[:10], "key": folder.name, "title": title, "state": "waiting", "error": None,
         "kept": 0, "cats": cats, "queued": time.time(), "started": None, "finished": None, "stop": False,
         "root": str(root), "_args": (str(root), str(folder), area)}
    with _LOCK:
        _JOBS[j["id"]] = j
        _QUEUE.append(j["id"])
        for old in sorted(_JOBS.values(), key=lambda q: q["queued"])[:-KEEP_JOBS]:
            if old["state"] in ("done", "failed", "stopped"):
                _JOBS.pop(old["id"], None)
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(target=_work, name="dataset", daemon=True)
            _WORKER.start()
    _WAKE.set()
    return _public(j)


def cancel(jid):
    with _LOCK:
        j = _JOBS.get(jid)
        if not j:
            return None
        j["stop"] = True
        if j["state"] == "waiting":
            j["state"] = "stopped"
            j["finished"] = time.time()
            for c in j["cats"]:
                c["state"] = "stopped"
            if jid in _QUEUE:
                _QUEUE.remove(jid)
        return _public(j)


def dismiss(jid):
    """A finished job out of the list."""
    with _LOCK:
        j = _JOBS.get(jid)
        if j and j["state"] in ("done", "failed", "stopped"):
            _JOBS.pop(jid)
            return True
    return False


def _work():
    while True:
        with _LOCK:
            jid = _QUEUE.pop(0) if _QUEUE else None
            j = _JOBS.get(jid) if jid else None
        if j is None:
            _WAKE.wait(5)
            _WAKE.clear()
            continue
        if j["state"] == "stopped":
            continue
        root, folder, area = j["_args"]
        j.update(state="running", started=time.time())
        saved = {}
        try:
            for c in j["cats"]:
                if j["stop"]:
                    c["state"] = "stopped"
                    continue
                c["state"] = "running"

                def step(done, total, c=c):
                    c["done"], c["total"] = done, total
                n = save_kind(root, folder, area, c["id"], step, lambda: j["stop"])
                c["kept"], c["state"] = n, ("stopped" if j["stop"] and n < c["total"] else "done")
                c["done"] = n
                saved[c["id"]] = n
                j["kept"] += n
            _recount(root, folder, area, saved)
            j["state"] = "stopped" if j["stop"] else "done"
        except Exception as e:                     # kept in the job, for the page
            import traceback
            traceback.print_exc()
            j.update(state="failed", error=f"{type(e).__name__}: {e}")
            try:
                _recount(root, folder, area, saved)
            except Exception:
                pass
        j["finished"] = time.time()
