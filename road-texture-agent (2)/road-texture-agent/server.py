"""
Road Texture Agent - local server.

Run:  python server.py        then open http://127.0.0.1:8765

Everything stays on your machine. Nothing is sent anywhere.

What is real today:
  - uploads, stored and recorded in memory
  - noise isolation by subtraction
  - junction detection, classification and overlay
  - the memory database: step log, keyed situations, attempts, trees

What is not built yet (these endpoints say so honestly instead of pretending):
  - Stage A analysis, the three methods
  - Stage B training
  - generation
"""

import functools
import json
import os
import threading
import time
import mimetypes
import re
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

import numpy as np

# scripts as JavaScript whatever the system says (some Windows set-ups say text/plain,
# and browsers will not run a module script, like the 3D tab's, with that type)
mimetypes.add_type("text/javascript", ".js")

from app import analysis as A
from app import progress as prog
from app import images as I
from app import generation as G
from app import junctions as J
from app import model3d as M3
from app import library as LIB
from app import looks as LK
from app import autoplace as AP
from app import buildings as BLD
from app import placements as PL
from app import tiles as TL
from app import objects as OB
from app import quadmesh as QMB
from app import bridges as BRG
from app import streets as ST
from app import islands as ISL
from app import formats as FMT
from app import decals as DC
from app import lanes as LN
from app.generation import prepare_mask as _prep
from app import routing as R
from app import training as T
from app.memory import Memory

ROOT = Path(__file__).parent
VERSION = "2026.10.10-front1"   # must match UI_VERSION in ui/app.js


def _workspace_path():
    """
    Where everything learned is kept. By default the 'workspace' folder next
    to this file. To keep it somewhere else, for example so that updates can
    never touch it, write its full path into a file called workspace.txt next
    to this file, or set the RTA_WORKSPACE environment variable.
    """
    import os
    env = os.environ.get("RTA_WORKSPACE", "").strip()
    if env:
        return Path(env).expanduser()
    cfg = ROOT / "workspace.txt"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8").strip().strip('"')
        if text:
            return Path(text).expanduser()
    return ROOT / "workspace"


WORK = _workspace_path().resolve()
_FRESH = not (WORK / "memory.db").exists()
UPLOADS = WORK / "uploads"
ARTIFACTS = WORK / "artifacts"
OBJECTS = WORK / "objects"
PACKAGES = WORK / "packages"
BUILDINGS = WORK / "buildings"
for d in (UPLOADS, ARTIFACTS, OBJECTS, PACKAGES, BUILDINGS):
    d.mkdir(parents=True, exist_ok=True)

mem = Memory(WORK / "memory.db")
app = FastAPI(title="Road Texture Agent")
ERRORS = WORK / "errors"


@app.exception_handler(Exception)
async def _unexpected(request, exc):
    """
    An unexpected error: say what broke and where, instead of a bare 500, and
    keep the full details in workspace/errors so they can be sent on.
    """
    import traceback, time as _t
    from fastapi.responses import JSONResponse
    tb = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in tb if "road-texture-agent" in f.filename.replace("\\", "/") or "/app/" in f.filename.replace("\\", "/")]
    where = ours[-1] if ours else (tb[-1] if tb else None)
    at = ("%s line %d, in %s" % (Path(where.filename).name, where.lineno, where.name)) if where else "unknown place"
    ERRORS.mkdir(parents=True, exist_ok=True)
    report = ERRORS / ("error-%s.txt" % _t.strftime("%Y%m%d-%H%M%S"))
    report.write_text("Road Texture Agent %s\n%s %s\n\n%s" % (
        VERSION, request.method, request.url.path,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))), encoding="utf-8")
    return JSONResponse(status_code=500, content={
        "detail": "%s: %s (at %s). Full report saved to %s" % (type(exc).__name__, exc, at, report)})
_sm = mem.summary()
print("\n  Road Texture Agent version %s" % VERSION)
print("  Program folder: %s" % ROOT)
print("  Workspace: %s" % WORK)
print("  %s: %d trained pairs, %d steps recorded\n" % (
    "NEW, EMPTY workspace created here" if _FRESH else "Existing workspace", _sm.get("trained_pairs", 0), _sm.get("steps", 0)))
if _FRESH:
    print("  If you expected your earlier training: stop the server and copy your old 'workspace'")
    print("  folder here, or put its path into workspace.txt next to server.py.\n")
if mem.moved:
    print("  %d file references were full paths (from an earlier version, another folder or another" % mem.moved)
    print("  computer): each now points into this workspace, by its place in it, so the workspace")
    print("  can be moved or copied to another computer.\n")

ROLES = {"material", "line", "noisy", "pair_mask", "pair_photo", "gen_mask", "sidewalk"}


@app.get("/api/health")
def health():
    return {"ok": True, "engine": "partial", "memory": mem.summary(),
            "workspace": str(WORK), "fresh": _FRESH, "version": VERSION}


# ------------------------------------------------------------------ uploads
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), role: str = Form(...), pair: str = Form("")):
    if role not in ROLES:
        raise HTTPException(400, f"unknown role: {role}")
    ext = Path(file.filename or "image.png").suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "use PNG, JPEG or WebP")
    image_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{image_id}{ext}"
    path.write_bytes(await file.read())

    try:
        w, h = I.size_of(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "could not read that image")

    meta = {"pair": pair}
    if role in {"pair_mask", "gen_mask"}:
        meta["road_px"] = I.road_pixels(path)

    mem.add_image(image_id, role, file.filename or path.name, path, w, h,
                  I.sha256_file(path), meta)
    step = mem.record("upload", f"loaded {role}: {file.filename} ({w} x {h})",
                      {"image_id": image_id, "role": role, "pair": pair,
                       "width": w, "height": h, **meta})
    return {"id": image_id, "role": role, "name": file.filename, "width": w, "height": h,
            "url": f"/api/image/{image_id}", "step": step, **meta}


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    rec = mem.image(image_id)
    if not rec:
        raise HTTPException(404, "unknown image")
    return FileResponse(rec["path"])


@app.get("/api/artifact/{artifact_id}")
def get_artifact(artifact_id: str):
    rec = mem.artifact(artifact_id)
    if not rec:
        raise HTTPException(404, "unknown artifact")
    return FileResponse(rec["path"])


# ------------------------------------------------------------ noise isolate
@app.post("/api/noise")
def noise(payload: dict):
    clean, noisy = mem.image(payload.get("clean")), mem.image(payload.get("noisy"))
    if not clean or not noisy:
        raise HTTPException(400, "need both material images")
    art_id = uuid.uuid4().hex[:12]
    out = ARTIFACTS / f"noise_{art_id}.png"
    res = I.isolate_noise(clean["path"], noisy["path"], out)
    if not res.get("ok"):
        mem.record("noise_failed", f"could not isolate noise: {res['error']}", res)
        raise HTTPException(400, res["error"])
    mem.add_artifact(art_id, "noise", out, res)
    mem.record("noise", f"isolated noise, average difference {res['mean']}",
               {"artifact": art_id, "clean": clean["id"], "noisy": noisy["id"], **res})
    return {"artifact": art_id, "url": f"/api/artifact/{art_id}", **res}


# --------------------------------------------------------------- junctions
# ---- progress of the long jobs, for the console: the page sends an id with the job
# ("progress") and asks /api/progress/{id} about once a second while it waits. Each
# plan is the job's stages with their share of its time, measured on a city-sized map
# (seconds on the 3.7 x 2.5 km test map, 3328 x 2304 px, 1.1 m per px)
GEN_PLAN = [("preparing", 0.5), ("mask", 5.0), ("junctions", 5.0), ("areas", 1.0), ("material_open", 3.0),
            ("material_edge", 3.0), ("material_junction", 4.0), ("wear", 2.5), ("markings", 1.0),
            ("finishing", 8.0), ("saving", 0.5), ("islands", 5.0)]
ISLANDS_PLAN = [("materials", 3.0), ("mask", 5.0), ("islands", 2.0), ("texture", 20.0), ("saving", 3.0)]
EXPORT_PLAN = [("preparing", 0.5), ("materials", 6.0), ("junctions", 5.0), ("groups", 3.0), ("streets", 2.0),
               ("patches", 4.0), ("sidewalks", 24.0), ("apron", 24.0), ("road", 19.0), ("sidewalk_meshes", 15.0),
               ("objects", 0.5), ("blocks", 58.0), ("lamps", 1.0), ("writing", 1.5), ("check", 3.0)]
TRAIN_PLAN = [("pairs", 8.0), ("libraries", 2.0)]
PRIME_PLAN = [("material", 3.0), ("line", 2.0), ("noise", 2.0), ("trees", 1.0)]
TILES_PLAN = [("tiles", 1.0)]
JUNCTIONS_PLAN = [("junctions", 1.0)]
AUTOPLACE_PLAN = [("islands", 1.0)]


def _write_json(path, obj):
    """
    A settings file the page saves while it may be reading it (lane choices, island slots,
    placements...): written beside it and put in place whole, so a read never finds it half
    written (an empty file, read at that moment, failed to load).
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def tracked(name, plan):
    """Run an endpoint as a job the console can follow (when the page sent an id)."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(payload: dict):
            jid = payload.get("progress") if isinstance(payload, dict) else None
            with prog.job(str(jid)[:64] if jid else None, name, plan):
                return fn(payload)
        return inner
    return wrap


@app.get("/api/progress/{jid}")
def progress_of(jid: str):
    snap = prog.of(jid)
    return snap if snap else {"id": jid, "unknown": True}


@app.post("/api/junctions")
@tracked("Finding junctions", JUNCTIONS_PLAN)
def junctions(payload: dict):
    rec = mem.image(payload.get("image"))
    if not rec:
        raise HTTPException(400, "unknown image")
    limit = float(payload.get("divider_limit_px", J.DIVIDER_LIMIT_PX))
    prog.stage("junctions", "finding the junctions")
    result = J.detect(I.load_gray(rec["path"]), limit)

    art_id = uuid.uuid4().hex[:12]
    out = ARTIFACTS / f"junctions_{art_id}.png"
    J.overlay(result, out)

    clean = [{k: v for k, v in j.items() if k != "segment"} for j in result["junctions"]]
    for j in clean:
        j["arms"] = [{k: v for k, v in a.items() if k != "segment"} for a in j["arms"]]

    mem.add_artifact(art_id, "junction_overlay", out,
                     {"image": rec["id"], "summary": result["summary"]})
    (ARTIFACTS / f"junctions_{art_id}.json").write_text(
        json.dumps({"summary": result["summary"], "junctions": clean}, indent=2))
    mem.record("junctions",
               f"found {result['summary']['junctions']} junctions in {rec['name']}",
               {"image": rec["id"], "artifact": art_id, "summary": result["summary"],
                "junctions": clean})
    return {"artifact": art_id, "overlay_url": f"/api/artifact/{art_id}",
            "summary": result["summary"], "junctions": clean}


# ------------------------------------------------------------------ memory
@app.get("/api/memory")
def memory_summary():
    return {**mem.summary(), "workspace": str(WORK), "fresh": _FRESH}


@app.get("/api/memory/steps")
def memory_steps(limit: int = 100, kind: str = ""):
    return {"steps": mem.steps(limit, kind or None)}


@app.get("/api/memory/recall")
def memory_recall(key: str):
    return mem.recall(key)


@app.get("/api/memory/trees")
def memory_trees():
    return {name: mem.tree(name) for name in mem.tree_names()}


@app.post("/api/memory/attempt")
def memory_attempt(payload: dict):
    """Records a review decision: accepted, rejected or flagged, with its reason."""
    key = payload.get("key")
    route = payload.get("route")
    outcome = payload.get("outcome")
    if not key or not route or outcome not in {"accepted", "rejected", "flagged"}:
        raise HTTPException(400, "need key, route and outcome")
    mem.touch_situation(key, payload.get("tree", "material"), payload.get("features", {}))
    step = mem.record("review", f"{route} {outcome} at {key}", payload)
    mem.add_attempt(key, route, outcome, payload.get("reason"), step)
    return mem.recall(key)


# ------------------------------------------------------- not built yet
@app.post("/api/prime")
@tracked("Priming", PRIME_PLAN)
def prime(payload: dict):
    """Stage A: analyse the three reference images and create the three trees."""
    need = ["material", "line", "noisy"]
    have = {r: mem.image(payload.get(r)) for r in need}
    missing = [r for r in need if not have[r]]
    if missing:
        raise HTTPException(400, f"missing: {', '.join(missing)}")

    # the noise tree learns from the difference, not the raw noisy image
    noise_art = uuid.uuid4().hex[:12]
    noise_path = ARTIFACTS / f"noise_{noise_art}.png"
    diff = I.isolate_noise(have["material"]["path"], have["noisy"]["path"], noise_path)
    if not diff.get("ok"):
        raise HTTPException(400, diff["error"])

    prog.stage("material", "analysing the material photo")
    material = A.analyse_material(have["material"]["path"])
    prog.stage("line", "analysing the line image")
    line = A.analyse_line(have["line"]["path"])
    prog.stage("noise", "analysing the wear image")
    noise = A.analyse_noise(noise_path)
    prog.stage("trees", "building the trees")

    # keep the patch libraries: generation draws real fragments from these
    libs, previews = {}, {}
    for name, res in (("material", material), ("line", line), ("noise", noise)):
        patches = res["patches"].pop("_patches", None)
        res["patches"].pop("_masks", None)
        lib_id = uuid.uuid4().hex[:12]
        libs[name] = lib_id
        if patches is not None and len(patches):
            np.savez_compressed(ARTIFACTS / f"patches_{lib_id}.npz", patches=patches)
            sheet = ARTIFACTS / f"patches_{lib_id}.png"
            A.contact_sheet(patches, sheet)
            mem.add_artifact(lib_id, "patch_library", sheet,
                             {"kind": name, "count": int(len(patches)),
                              "size_px": int(patches.shape[1]),
                              "npz": str(ARTIFACTS / f"patches_{lib_id}.npz")})
            previews[name] = f"/api/artifact/{lib_id}"

    trees = A.build_trees(material, line, noise, libs)
    for name, tree in trees.items():
        mem.save_tree(name, tree, source="stage A priming")

    fingerprints = {"material": material, "line": line, "noise": noise}
    fp_id = uuid.uuid4().hex[:12]
    (ARTIFACTS / f"fingerprints_{fp_id}.json").write_text(json.dumps(fingerprints, indent=2))
    mem.add_artifact(fp_id, "fingerprints", ARTIFACTS / f"fingerprints_{fp_id}.json", {})
    mem.add_artifact(noise_art, "noise", noise_path, diff)

    dash = line["dash"]
    mem.record("prime",
               "priming finished: " + (dash["reading"] if dash.get("found")
                                       else "no dash pattern measured"),
               {"images": {r: payload.get(r) for r in need},
                "fingerprints": fp_id, "libraries": libs,
                "material": material["neighbours"], "noise_coverage": noise["coverage"],
                "dash": dash, "trees": list(trees)})

    return {"ok": True, "fingerprints": fingerprints, "libraries": libs,
            "library_previews": previews, "noise_url": f"/api/artifact/{noise_art}",
            "trees": list(trees)}


def _priming_tone():
    """The tone measured on the material-alone image during priming."""
    tree = mem.tree("material")
    if not tree:
        return None
    route = tree["children"]["no"]["children"]["1.5 m or more"]
    primed = route.get("primed_fallback") or route
    return (primed.get("params") or {}).get("tone")


def _pair_labels():
    return {p["id"]: f"{p['mask_name']} + {p['photo_name']}" for p in mem.trained_pairs()}


TILE_DEFAULTS = {"photo_width_m": 2.0, "tile_m": 4.0, "px": 1024, "variants": 3,
                 "sidewalk_width_m": 2.0}


def _priming_closeup():
    """The 'material with noise' photo from the latest priming run."""
    step = next(iter(mem.steps(1, "prime")), None)
    if not step:
        return None
    img_id = (step["payload"].get("images") or {}).get("noisy")
    rec = mem.image(img_id) if img_id else None
    return rec["path"] if rec else None


def _build_tiles(settings=None):
    """
    Seamless tiles for each part of the road, several variants each.

    Close-up detail from the priming photo; each part's tone and contrast from
    the pair at the top of its order. Rebuilt whenever the libraries change.
    """
    prev = mem.latest_artifact("tileset")
    # later wins: defaults, then the last saved settings, then what was asked for now
    cfg = {**TILE_DEFAULTS, **((prev or {}).get("meta", {}).get("settings", {})), **(settings or {})}
    src_path = _priming_closeup()
    if not src_path:
        return {"ok": False, "reason": "run priming first: tiles take their detail from the close-up photo"}
    src = np.array(Image.open(src_path).convert("RGB"))

    # each part's tone and contrast from the pair at the top of its order
    pairs = {p["id"]: p for p in mem.trained_pairs()}
    stats = {}
    for part in TL.PARTS:
        top = next(iter(mem.index(part)), None)
        g = (pairs.get(top["pair_id"], {}).get("groups", {}).get(part) or {}) if top else {}
        stats[part] = {"tone": g.get("mean"), "contrast": g.get("local_contrast")}
    base_c = stats["open"]["contrast"] or next((v["contrast"] for v in stats.values() if v["contrast"]), None)
    prime_tone = _priming_tone()

    set_id = uuid.uuid4().hex[:12]
    tiles = {}
    # every tile is built at the open road's tone; each part's own tone is kept
    # separately and applied through vertex colours at 3D export, where it can
    # fade smoothly from one part to the next instead of changing at a hard line
    part_tones = {p: (stats[p]["tone"] if stats[p]["tone"] is not None else prime_tone) for p in TL.PARTS}
    common = part_tones["open"] if part_tones["open"] is not None else prime_tone
    prog.stage("tiles", "building the material tiles")
    n_tiles, made = (len(TL.PARTS) + 2) * int(cfg["variants"]), 0
    for part in TL.PARTS:
        tone = common
        ratio = (stats[part]["contrast"] / base_c) if (stats[part]["contrast"] and base_c) else 1.0
        tiles[part] = []
        for v in range(int(cfg["variants"])):
            made += 1; prog.part(made / n_tiles); prog.note(f"building the material tiles: {part} {v + 1}")
            tile, info = TL.build_tile(src, float(cfg["photo_width_m"]), float(cfg["tile_m"]),
                                       int(cfg["px"]), tone=tone, contrast_ratio=ratio, seed=v + 1)
            aid = f"tile_{set_id}_{part}_{v + 1}"
            path = ARTIFACTS / f"{aid}.png"
            Image.fromarray(tile).save(path)
            meta = {"part": part, "variant": v + 1, "seam": TL.seam_score(tile),
                    "noticeability": TL.noticeability(tile), "tone": tone,
                    "contrast_ratio": round(ratio, 3), **info}
            mem.add_artifact(aid, "tile", path, meta)
            tiles[part].append({"id": aid, "url": f"/api/artifact/{aid}", **meta})

    # sidewalk: from the paving close-up if there is one, else light concrete.
    # kerb stone and kerb face: light concrete
    paving = next(iter(mem.images("sidewalk")), None)
    tiles["sidewalk"] = []
    for v in range(int(cfg["variants"])):
        if paving:
            psrc = np.array(Image.open(paving["path"]).convert("RGB"))
            tile, info = TL.build_paving_tile(psrc, float(cfg["sidewalk_width_m"]), float(cfg["tile_m"]),
                                              int(cfg["px"]), seed=v + 1)
        else:
            tile, info = TL.concrete_placeholder(src, float(cfg["photo_width_m"]), float(cfg["tile_m"]),
                                                 int(cfg["px"]), seed=v + 11)
        aid = f"tile_{set_id}_sidewalk_{v + 1}"
        path = ARTIFACTS / f"{aid}.png"
        Image.fromarray(tile).save(path)
        meta = {"part": "sidewalk", "variant": v + 1,
                "seam": None if info.get("method") == "pattern" else TL.seam_score(tile),
                "noticeability": None if info.get("method") == "pattern" else TL.noticeability(tile),
                **{k: v2 for k, v2 in info.items() if k in ("method", "repeats", "period_px", "covers_m")}}
        mem.add_artifact(aid, "tile", path, meta)
        tiles["sidewalk"].append({"id": aid, "url": f"/api/artifact/{aid}", **meta})
    ktile, kinfo = TL.concrete_placeholder(src, float(cfg["photo_width_m"]), float(cfg["tile_m"]),
                                           max(256, int(cfg["px"]) // 2), tone=185.0, seed=21)
    aid = f"tile_{set_id}_kerbstone_1"
    Image.fromarray(ktile).save(ARTIFACTS / f"{aid}.png")
    kmeta = {"part": "kerbstone", "variant": 1, "seam": TL.seam_score(ktile),
             "noticeability": TL.noticeability(ktile), "method": "light concrete"}
    mem.add_artifact(aid, "tile", ARTIFACTS / f"{aid}.png", kmeta)
    tiles["kerbstone"] = [{"id": aid, "url": f"/api/artifact/{aid}", **kmeta}]

    manifest = {"settings": cfg, "tiles": tiles, "part_tones": part_tones, "tile_tone": common,
                "sidewalk_source": "paving photo" if paving else "placeholder concrete",
                "mm_per_px": round(float(cfg["tile_m"]) / int(cfg["px"]) * 1000, 2)}
    mpath = ARTIFACTS / f"tileset_{set_id}.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    # older tile sets are no longer used: drop their files
    for row in mem.db.execute("SELECT id, path FROM artifacts WHERE kind IN ('tile','tileset')").fetchall():
        if set_id not in row["id"] and not row["path"].endswith(f"tileset_{set_id}.json"):
            Path(mem.resolve(row["path"])).unlink(missing_ok=True)
            mem.db.execute("DELETE FROM artifacts WHERE id=?", (row["id"],))
    mem.db.commit()
    mem.add_artifact(f"tileset_{set_id}", "tileset", mpath, manifest)
    mem.record("tiles", f"built {sum(len(v) for v in tiles.values())} tiles, "
               f"{manifest['mm_per_px']} mm per pixel", {"settings": cfg})
    return {"ok": True, "id": set_id, **manifest}


def _rebuild_libraries(note: str):
    """
    Rebuild the ordered index for each part of the road.

    Nothing is merged. Each pair keeps its own library, and the index lists the
    pairs in order of weight: how much road a pair supplied times how much its
    look has been trusted so far. The agent reads the list top to bottom.
    """
    pairs = mem.trained_pairs()
    trees = {name: mem.tree(name) for name in ("material", "noise", "lines")}
    if not all(trees.values()):
        raise HTTPException(400, "run priming first: the trees do not exist yet")
    labels = _pair_labels()

    report = {}
    for part in ("junction", "edge", "open"):
        rows = []
        for p in pairs:
            g = p["groups"].get(part) or {}
            f = p["patch_files"].get(part)
            if not f or not Path(f).exists() or not g.get("usable"):
                continue
            rows.append({"pair_id": p["id"], "pixels": int(g.get("pixels") or 0),
                         "patches": int(g.get("patches") or 0),
                         "size_px": int(g.get("patch_size_px") or 0), "npz": f})
        mem.set_index(part, rows)
        report[part] = [{"rank": r["rank"], "pair_id": r["pair_id"],
                         "label": labels.get(r["pair_id"], r["pair_id"]),
                         "pixels": r["pixels"], "patches": r["patches"],
                         "size_px": r["size_px"], "confidence": r["confidence"],
                         "weight": round(r["weight"])}
                        for r in mem.index(part)]

    # merged libraries from the previous design are no longer used
    for row in mem.db.execute(
            "SELECT id, path, meta FROM artifacts WHERE kind='patch_library'").fetchall():
        meta = json.loads(row["meta"])
        if str(meta.get("kind", "")).startswith("merged."):
            for f in (Path(mem.resolve(row["path"])), Path(mem.resolve(meta.get("npz", "")) or "/nonexistent")):
                if f.exists():
                    f.unlink()
            mem.db.execute("DELETE FROM artifacts WHERE id=?", (row["id"],))
    mem.db.commit()

    # routes now point at the index for their part; the primed route stays as last resort
    def point(route, part):
        primed = route.get("primed_fallback")
        if primed is None and str(route.get("source", "")).startswith("stage A"):
            primed = dict(route, id=route["id"] + ".primed",
                          lesson="Kept from priming, the last resort if every trained pair is rejected.")
        new = dict(route)
        n = len(report[part])
        new["params"] = dict(route.get("params", {}), library_index=part)
        new["source"] = f"stage B, ordered index of {n} pair{'s' if n != 1 else ''}"
        new["lesson"] = (f"Reads the {part} index top to bottom: {n} pair{'s' if n != 1 else ''}, "
                         f"each used whole so its photo's look is never mixed with another's.")
        if primed is not None:
            new["primed_fallback"] = primed
        return new

    # line width as a share of road width, learned from every pair that showed paint.
    # A weighted median, so one odd photo cannot drag it far.
    samples = sorted((p["groups"]["_paint"]["line_ratio"], p["groups"]["_paint"]["paint_px"])
                     for p in pairs
                     if (p["groups"].get("_paint") or {}).get("line_ratio"))
    line_ratio = None
    if samples:
        total = sum(w for _, w in samples)
        run = 0
        for r_, w in samples:
            run += w
            if run >= total / 2:
                line_ratio = r_
                break
    dash_route = trees["lines"]["children"]["yes"]["children"]["no"]
    dash_route["params"] = dict(dash_route.get("params", {}),
                                width_ratio=line_ratio, width_ratio_pairs=len(samples))
    report["lines"] = {"width_ratio": line_ratio, "pairs": len(samples)}

    m = trees["material"]
    m["children"]["yes"] = point(m["children"]["yes"], "junction")
    br = m["children"]["no"]["children"]
    br["under 1.5 m"] = point(br["under 1.5 m"], "edge")
    br["1.5 m or more"] = point(br["1.5 m or more"], "open")
    for name, tree in trees.items():
        mem.save_tree(name, tree, source=note)

    changes = [f"{part}: {len(rows)} pair{'s' if len(rows) != 1 else ''} in order"
               + (f", first is {rows[0]['label']}" if rows else ", none usable, priming is used")
               for part, rows in report.items() if part != "lines"]
    lr = report["lines"]
    changes.append(f"lines: {lr['width_ratio']*100:.1f}% of road width, learned from "
                   f"{lr['pairs']} pair{'s' if lr['pairs'] != 1 else ''}" if lr["width_ratio"]
                   else "lines: no painted lines found in the pairs, the fixed width is used")
    mem.record("rebuild", f"{note}: index rebuilt from {len(pairs)} pair(s)",
               {"index": report})
    tiles = _build_tiles()
    if tiles.get("ok"):
        changes.append(f"tiles: {sum(len(v) for v in tiles['tiles'].values())} rebuilt, "
                       f"{tiles['mm_per_px']} mm per pixel")
    return {"pairs": len(pairs), "index": report, "changes": changes}


@app.post("/api/train")
@tracked("Training", TRAIN_PLAN)
def train(payload: dict):
    """
    Stage B: measure any pair not measured before, then rebuild the libraries.

    Pairs already trained on are not measured again, so adding three pairs to
    ten costs three pairs of work, not thirteen.
    """
    mpp = float(payload.get("metres_per_pixel", 0.25))
    pairs = payload.get("pairs") or []
    if not pairs:
        raise HTTPException(400, "no pairs given")
    if not all(mem.tree(n) for n in ("material", "noise", "lines")):
        raise HTTPException(400, "run priming first: the trees do not exist yet")

    reports, added, skipped = [], 0, 0
    prog.stage("pairs", "measuring the pairs")
    for n, pair in enumerate(pairs, start=1):
        prog.part((n - 1) / len(pairs)); prog.note(f"measuring pair {n} of {len(pairs)}")
        mask, photo = mem.image(pair.get("mask")), mem.image(pair.get("photo"))
        if not mask or not photo:
            raise HTTPException(400, f"pair {n}: missing mask or photo")
        existing = mem.find_trained_pair(mask["id"], photo["id"], mpp)
        if existing:
            skipped += 1
            continue

        label = f"pair {n} ({mask['name']})"
        res = T.analyse_pair(mask["path"], photo["path"], mpp)
        if not res.get("ok"):
            mem.record("train_failed", f"{label}: {res['error']}", {"pair": pair})
            raise HTTPException(400, f"{label}: {res['error']}")

        pair_id = uuid.uuid4().hex[:12]
        files, groups = {}, {}
        for group, g in res["groups"].items():
            patches = g["patches"].pop("_patches", None)
            pmasks = g["patches"].pop("_masks", None)
            groups[group] = {**g["stats"], "usable": g["usable"],
                             "patch_size_px": g.get("patch_size_px"),
                             "share_of_road": g["share_of_road"],
                             "patches": g["patches"].get("count", 0)}
            if patches is not None and len(patches):
                f = ARTIFACTS / f"pair_{pair_id}_{group}.npz"
                if pmasks is not None:
                    np.savez_compressed(f, patches=patches, masks=pmasks)
                else:
                    np.savez_compressed(f, patches=patches)
                files[group] = str(f)
        groups["_paint"] = {k: res["paint"].get(k) for k in
                            ("line_width_px", "line_ratio", "paint_px", "share_of_road")}
        mem.add_trained_pair(pair_id, mask, photo, mpp, res["road_px"],
                             res["junctions"]["junctions"], groups, files)
        mem.record("train_pair", f"{label}: measured {res['road_px']:,} road pixels",
                   {"pair_id": pair_id, "mask": mask["id"], "photo": photo["id"],
                    "metres_per_pixel": mpp, "groups": groups})
        reports.append({"pair": label, "pair_id": pair_id,
                        "junctions": res["junctions"]["junctions"],
                        "road_px": res["road_px"], "paint": res["paint"],
                        "road_width_px": res.get("road_width_px"),
                        "road_width_m": res.get("road_width_m"),
                        "resolution": res.get("resolution"),
                        "kerb_band": res.get("kerb_band"),
                        "groups": {k: {"share_of_road": v["share_of_road"],
                                       "stats": v["stats"], "usable": v["usable"],
                                       "patches": v["patches"].get("count", 0),
                                       "patch_size_px": v.get("patch_size_px")}
                                   for k, v in res["groups"].items()}})
        added += 1

    prog.stage("libraries", "rebuilding the libraries and tiles")
    rebuilt = _rebuild_libraries(f"training: {added} new pair(s)")
    return {"ok": True, "added": added, "already_trained": skipped,
            "pairs": reports, "rebuilt": rebuilt, "memory": mem.summary()}


@app.get("/api/pairs")
def list_pairs():
    """Every pair the agent has been trained on."""
    out = []
    for p in mem.trained_pairs():
        out.append({
            "id": p["id"], "mask": p["mask_name"], "photo": p["photo_name"],
            "mask_id": p["mask_id"], "photo_id": p["photo_id"],
            "scale": p["scale"], "road_px": p["road_px"], "junctions": p["junctions"],
            "groups": {g: {"pixels": v.get("pixels"), "patches": v.get("patches"),
                           "usable": v.get("usable"), "tone": v.get("mean")}
                       for g, v in p["groups"].items()},
            "created": p["created"],
        })
    return {"pairs": out}


@app.delete("/api/pairs/{pair_id}")
def delete_pair(pair_id: str):
    """Delete one pair and rebuild the libraries without it."""
    rec = mem.delete_trained_pair(pair_id)
    if not rec:
        raise HTTPException(404, "unknown pair")
    for f in rec["patch_files"].values():
        p = Path(f)
        if p.exists():
            p.unlink()
    mem.drop_from_index(pair_id)
    mem.record("pair_deleted", f"deleted pair {rec['mask_name']} + {rec['photo_name']}",
               {"pair_id": pair_id})
    rebuilt = _rebuild_libraries("pair deleted")
    return {"ok": True, "deleted": pair_id, "rebuilt": rebuilt}


# the parts a library material can stand in for, and the tiles each replaces
# square: the blocks and islands between the roads (by default paved as the sidewalks)
MATERIAL_PARTS = {"street": ("open", "edge", "junction"), "sidewalk": ("sidewalk",), "kerb": ("kerbstone",),
                  "square": ("block",)}


def _apply_materials(tileset, choice, match_tone):
    """
    Put the chosen library materials (app/scans) in place of the trained tiles:
    for each part (street, sidewalk, kerb) its id, or "tiles" to keep yours.
    With match_tone, a material's colour takes on the mean colour of the tile it
    replaces. Returns {part: material name} for what was replaced.
    """
    used = {}
    tile_m = float(tileset["settings"]["tile_m"])
    px = int(tileset["settings"].get("px", 1024))
    for part, tile_parts in MATERIAL_PARTS.items():
        mid = choice.get(part)
        if not mid or mid in ("tiles", "same"):
            continue
        e = LIB.entry(mid)
        if not e or e["kind"] not in LIB.KINDS_FOR_PART[part]:
            raise ValueError(f"{mid} is not a {part} material in the library")
        def tone_of(t):
            own = (tileset["tiles"].get(t) or [None])[0]
            if own is None and part in ("kerb", "square"):
                own = (tileset["tiles"].get("sidewalk") or [None])[0]
            if not match_tone or own is None:
                return None
            return np.asarray(Image.open(own["path"]).convert("RGB")).reshape(-1, 3).mean(axis=0).tolist()
        if part == "street":
            # streets: fresh tiles made from the scan's patches, as many variants as your
            # own tiles and different ones for open road, the kerb band and junctions, so
            # no street shows the scan repeating; with match tone, each takes its part's tone
            n = max(1, int(tileset["settings"].get("variants", 3)))
            for i, t in enumerate(tile_parts):
                tone = tone_of(t)
                tileset["tiles"][t] = [LIB.tile_for(mid, tile_m, px, ARTIFACTS / "library", tone, variant=i * n + v + 1)
                                       for v in range(n)]
            used[part] = e["name"] + f" ({n * len(tile_parts)} random tiles" + (", toned to yours)" if tone is not None else ")")
        elif e["kind"] == "ground":
            # desert and ground: fresh tiles made from the scan's patches (no joints to
            # keep on a grid), several, in its own colour (it replaces nothing of yours)
            n = max(1, int(tileset["settings"].get("variants", 3)))
            for t in tile_parts:
                tileset["tiles"][t] = [LIB.tile_for(mid, tile_m, px, ARTIFACTS / "library", None, variant=v + 1)
                                       for v in range(n)]
            used[part] = e["name"]
        else:
            # paving and kerbs: the scan itself, so its joints stay on their grid
            tone = tone_of(tile_parts[0])
            rec = LIB.tile_for(mid, tile_m, px, ARTIFACTS / "library", tone)
            for t in tile_parts:
                tileset["tiles"][t] = [rec]
            used[part] = e["name"] + (" (toned to your tiles)" if tone is not None else "")
    return used


def _island_tiles(tileset, slots, match_tone):
    """
    The tiles of each island material slot: {slot: {"tiles", "name", "kind"}}.
    A slot on "same" is left out: its islands are laid as all the others (the
    Squares choice, or the sidewalks'). Desert and ground: fresh tiles made from
    the scan's patches, several, in its own colour; paving and concrete: the
    scan itself, toned as the squares are with match tone.
    """
    out = {}
    tile_m = float(tileset["settings"]["tile_m"])
    px = int(tileset["settings"].get("px", 1024))
    n = max(1, int(tileset["settings"].get("variants", 3)))
    for s, slot in enumerate(slots):
        mid = slot.get("material")
        if not mid or mid in ("same", "tiles"):
            continue
        e = LIB.entry(mid)
        if not e or e["kind"] not in LIB.KINDS_FOR_PART["island"]:
            raise ValueError(f"{mid} is not an island material in the library")
        if e["kind"] == "ground":
            recs = [LIB.tile_for(mid, tile_m, px, ARTIFACTS / "library", None, variant=v + 1) for v in range(n)]
        else:
            own = (tileset["tiles"].get("sidewalk") or [None])[0]
            tone = (np.asarray(Image.open(own["path"]).convert("RGB")).reshape(-1, 3).mean(axis=0).tolist()
                    if match_tone and own is not None else None)
            recs = [LIB.tile_for(mid, tile_m, px, ARTIFACTS / "library", tone)]
        out[s] = {"tiles": recs, "name": e["name"], "kind": e["kind"]}
        # ground: a second ground material in patches through it (the slot's mix, or the
        # library's partner), its tone halfway to this one's so patches differ in grain more than hue
        mix = slot.get("mix") or {}
        bid = e.get("mix_with") if mix.get("material", "auto") == "auto" else mix.get("material")
        be = LIB.entry(bid) if bid and bid != "none" and bid != mid else None
        if e["kind"] == "ground" and be and be["kind"] == "ground" and mix.get("amount", ISL.MIX_AMOUNT) > 0:
            a_mean = np.asarray(Image.open(recs[0]["path"]).convert("RGB")).reshape(-1, 3).mean(axis=0)
            b_mean = np.asarray(Image.open(LIB.DIR / be["colour"]).convert("RGB")).reshape(-1, 3).mean(axis=0)
            tone = np.sqrt(np.maximum(a_mean, 1.0) * np.maximum(b_mean, 1.0)).tolist()
            # one tile of it: its patches never repeat in step anyway, and every variant then shares
            # the same three pictures (the GLB and the photo render hold them once)
            out[s]["mix"] = {"tiles": [LIB.tile_for(bid, tile_m, px, ARTIFACTS / "library", tone, variant=1)],
                             "name": be["name"], "amount": float(mix.get("amount", ISL.MIX_AMOUNT)),
                             "size_m": float(mix.get("size_m", ISL.MIX_SIZE_M)), "seed": s}
    return out


def _tileset_with_paths():
    """Your latest tileset with each tile's file, or None when there are no tiles yet."""
    row = mem.latest_artifact("tileset")
    if not row or not row["meta"].get("tiles"):
        return None
    tileset = json.loads(json.dumps(row["meta"]))
    for part, lst in tileset["tiles"].items():
        for tile in lst:
            tile["path"] = mem.artifact(tile["id"])["path"]
    return tileset


def _islands_for(gid, tileset, payload):
    """
    The island materials of a generation for the 3D model and the islands
    texture: its numbered islands, which slot each picked island is in, and
    each slot's tiles. None when the generation has no numbered islands (made
    before they were numbered).
    """
    lab_art = mem.artifact(f"{gid}_islandmap")
    if not lab_art or not Path(lab_art["path"]).exists():
        return None
    lab = ISL.load_labels(lab_art["path"])
    slots = ISL.slot_list(payload.get("island_slots"))
    looks = _island_tiles(tileset, slots, bool(payload.get("match_tone", True)))
    slot_of = {k: s for k, s in ISL.resolve(slots, lab).items() if s in looks}
    return {"labels": lab, "slot_of": slot_of, "looks": looks, "seed": int(payload.get("seed", 7))}


def _your_tile(part):
    """Your trained tile that a part (street, sidewalk, kerb) uses, with its file and the tileset's id, or None."""
    row = mem.latest_artifact("tileset")
    if not row or not row["meta"].get("tiles"):
        return None
    tiles = row["meta"]["tiles"]
    names = list(MATERIAL_PARTS[part]) + (["sidewalk"] if part in ("kerb", "square") else [])
    tile = next((tiles[t][0] for t in names if tiles.get(t)), None)
    art = mem.artifact(tile["id"]) if tile else None
    if not art or not Path(art["path"]).exists():
        return None
    return {**tile, "path": art["path"], "tileset": row["id"], "tile_m": float(row["meta"]["settings"]["tile_m"])}


@app.get("/api/materials")
def list_materials():
    """
    The scanned material library: what each part (street, sidewalk, kerb) can
    use; and for each part whether you have trained tiles for it (their ball's
    version, so a new tileset shows a new picture).
    """
    yours = {}
    for part in MATERIAL_PARTS:
        t = _your_tile(part)
        yours[part] = t["tileset"] if t else None
    return {"parts": {p: list(k) for p, k in LIB.KINDS_FOR_PART.items()},
            "yours": yours,
            "materials": [{k: e.get(k) for k in ("id", "name", "kind", "size_m", "source", "title", "authors", "licence", "pattern", "mix_with")}
                          for e in LIB.materials()]}


@app.get("/api/materials/tiles/{part}/ball")
def your_tiles_ball(part: str):
    """Your trained tile for a part on a ball, with the bump and roughness the 3D model gives it."""
    if part not in MATERIAL_PARTS:
        raise HTTPException(404, "no such part")
    t = _your_tile(part)
    if t is None:
        raise HTTPException(404, "no material tiles yet")
    cache = ARTIFACTS / "library" / f"ball_tiles_{t['id']}_{part}.png"
    if not cache.exists():
        from app import surface as SF
        import io as _io
        img = Image.open(t["path"]).convert("RGB")
        kind = "asphalt" if part == "street" else ("concrete" if part == "kerb" else M3._sidewalk_kind(t))
        nrm, rgh = SF.maps(img, (t["tile_m"], t["tile_m"]), kind, source="scan")[:2]
        normal = np.asarray(Image.open(_io.BytesIO(nrm)).convert("RGB")).astype(np.float32) / 127.5 - 1.0
        rough = np.asarray(Image.open(_io.BytesIO(rgh)).convert("L")).astype(np.float32) / 255.0
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(LIB.ball(np.asarray(img), normal, rough, (t["tile_m"], t["tile_m"])))
    return Response(content=cache.read_bytes(), media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/materials/{mid}/ball")
def material_ball(mid: str):
    """A library material on a ball (PNG), at its real size, lit to show its bump and roughness."""
    data = LIB.material_ball(mid, ARTIFACTS / "library")
    if data is None:
        raise HTTPException(404, "no such material")
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


# ------------------------------------------------------------------ looks
# The 3D tab's Look panel: your saved looks and imported colour tables (LUTs),
# kept in workspace/looks; the ones that ship with the program in app/looks.
LOOKS = LK.Store(WORK / "looks")


@app.get("/api/looks")
def list_looks():
    return {"looks": LOOKS.look_list(), "luts": LOOKS.lut_list()}


@app.post("/api/looks")
def save_look(payload: dict):
    """Save a look: {"name", "settings", "id" (optional: overwrite your look with this id)}."""
    lk = LOOKS.save_look(payload.get("name") or "My look", payload.get("settings") or {}, payload.get("id"))
    mem.record("look_save", "saved look %s" % lk["name"], {"id": lk["id"]})
    return lk


@app.delete("/api/looks/{lid}")
def delete_look(lid: str):
    if not LOOKS.delete_look(lid):
        raise HTTPException(404, "no such look of yours")
    return {"ok": True}


@app.post("/api/looks/import")
async def import_look(file: UploadFile = File(...)):
    """A LUT (.cube, or a Hald CLUT .png) or a look file downloaded from the Look panel (.json)."""
    raw = await file.read()
    if len(raw) > 40 * 1024 * 1024:
        raise HTTPException(400, "%s is too large for a LUT or a look (over 40 MB)" % file.filename)
    try:
        out = LOOKS.import_file(file.filename or "lut.cube", raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    mem.record("look_import", "imported %s" % file.filename, {k: v["id"] for k, v in out.items() if v})
    return out


@app.get("/api/looks/lut/{lid}")
def lut_file(lid: str):
    text = LOOKS.lut_text(lid)
    if text is None:
        raise HTTPException(404, "no such LUT")
    return Response(content=text, media_type="text/plain", headers={"Cache-Control": "no-cache"})


@app.delete("/api/looks/lut/{lid}")
def delete_lut(lid: str):
    if not LOOKS.delete_lut(lid):
        raise HTTPException(404, "no such LUT of yours")
    return {"ok": True}


# ------------------------------------------------------------------ skies
# The 3D tab's HDRI skies: the ones that come with the program (ui/skies, CC0 from Poly
# Haven: a sharp picture of the sky and a small HDR of its light each) and your own
# (.hdr, .exr, or a .jpg or .png panorama), kept in workspace/skies with a note of what
# kind of light they are (day, sunset, overcast or night: it decides the street lamps
# and the exposure, since HDR pictures do not say how bright they really are).
SKIES_BUILTIN = ROOT / "ui" / "skies"
SKIES = WORK / "skies"
SKY_KINDS = ("day", "sunset", "overcast", "night")
SKY_EXT = {".hdr", ".exr", ".jpg", ".jpeg", ".png"}


def _sky_meta(sid):
    f = SKIES / f"{sid}.json"
    return json.loads(f.read_text()) if re.fullmatch(r"[0-9a-f]{12}", sid or "") and f.exists() else None


@app.get("/api/skies")
def list_skies():
    built = json.loads((SKIES_BUILTIN / "skies.json").read_text()).get("skies", []) if (SKIES_BUILTIN / "skies.json").exists() else []
    out = [{**e, "builtin": True, "background_url": f"skies/{e['background']}", "light_url": f"skies/{e['light']}",
            "thumb_url": f"skies/{e['thumb']}"} for e in built]
    if SKIES.exists():
        for f in sorted(SKIES.glob("*.json"), key=lambda p: p.stat().st_mtime):
            m = json.loads(f.read_text())
            out.append({**m, "builtin": False, "file_url": f"/api/skies/{m['id']}/file"})
    return {"skies": out, "kinds": list(SKY_KINDS)}


@app.post("/api/skies/import")
async def import_sky(file: UploadFile = File(...), kind: str = Form("day")):
    """Your own sky: an equirectangular (2:1) panorama, HDR (.hdr, .exr) or not (.jpg, .png)."""
    name = Path(file.filename or "sky.hdr").name
    ext = Path(name).suffix.lower()
    if ext not in SKY_EXT:
        raise HTTPException(400, "use an .hdr or .exr file (or a .jpg or .png panorama)")
    raw = await file.read()
    if len(raw) > 400 * 1024 * 1024:
        raise HTTPException(400, f"{name} is too large (over 400 MB)")
    SKIES.mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex[:12]
    (SKIES / f"{sid}{ext}").write_bytes(raw)
    meta = {"id": sid, "name": Path(name).stem.replace("_", " ")[:60], "file": f"{sid}{ext}", "format": ext[1:],
            "kind": kind if kind in SKY_KINDS else "day", "original": name}
    (SKIES / f"{sid}.json").write_text(json.dumps(meta, indent=2))
    mem.record("sky_import", f"imported sky {name}", {"id": sid})
    return {**meta, "builtin": False, "file_url": f"/api/skies/{sid}/file"}


@app.post("/api/skies/{sid}")
def update_sky(sid: str, payload: dict):
    """Your sky's name or kind of light."""
    m = _sky_meta(sid)
    if not m:
        raise HTTPException(404, "no such sky of yours")
    if payload.get("kind") in SKY_KINDS:
        m["kind"] = payload["kind"]
    if payload.get("name"):
        m["name"] = str(payload["name"])[:60]
    (SKIES / f"{sid}.json").write_text(json.dumps(m, indent=2))
    return {**m, "builtin": False, "file_url": f"/api/skies/{sid}/file"}


@app.get("/api/skies/{sid}/file")
def sky_file(sid: str):
    m = _sky_meta(sid)
    if not m or not (SKIES / m["file"]).exists():
        raise HTTPException(404, "no such sky")
    return FileResponse(SKIES / m["file"], headers={"Cache-Control": "max-age=86400"})


@app.delete("/api/skies/{sid}")
def delete_sky(sid: str):
    m = _sky_meta(sid)
    if not m:
        raise HTTPException(404, "no such sky of yours")
    (SKIES / m["file"]).unlink(missing_ok=True)
    (SKIES / f"{sid}.json").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/materials/{mid}/thumb")
def material_thumb(mid: str):
    data = LIB.thumbnail(mid)
    if data is None:
        raise HTTPException(404, "no such material")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _surface_choice(v):
    """Surface detail: "scan" (scanned maps where bundled, else the tile's grain), "grain", or off."""
    if v is True or v == "scan":
        return "scan"
    return "grain" if v == "grain" else False


def _bridge_list(raw):
    out = []
    for b in raw or []:
        try:
            out.append({"cx": float(b["cx"]), "cy": float(b["cy"]), "length": max(4.0, float(b["length"])),
                        "width": max(4.0, float(b["width"])), "angle": float(b.get("angle", 0.0))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ------------------------------------------------------------------ objects
@app.post("/api/objects/import")
async def import_object(file: UploadFile = File(...), package: str = Form(""), foliage: str = Form(""),
                        front: str = Form("+y")):
    """
    Import a GLB, OBJ or FBX object: normalised to metres, footprint centred,
    base at 0. With a package, the object becomes a new slot of that package
    instead of a layer of its own. Plants (foliage) are imported the same way,
    marked foliage, or into a foliage package. front: which way its front faces in
    the file ("+y", "+x", "-y", "-x", as Blender shows the axes): it is turned so
    its front is this app's, +Y.
    """
    front = str(front).strip().lower()
    if front not in OB.FRONT_TURN:
        raise HTTPException(400, "front must be +y, +x, -y or -x")
    name = Path(file.filename or "object").name
    ext = name.lower().rsplit(".", 1)[-1]
    if ext not in ("glb", "obj", "fbx"):
        raise HTTPException(400, "use a GLB, OBJ or FBX file")
    pk = None
    if package:
        package = _safe_id(package)
        pk = _package(package)
        if pk is None:
            raise HTTPException(404, "unknown package")
    oid = uuid.uuid4().hex[:12]
    folder = OBJECTS / oid
    folder.mkdir(parents=True, exist_ok=True)
    src = folder / ("source." + ext)
    src.write_bytes(await file.read())
    try:
        parts, info = OB.load(src)
    except Exception as e:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "could not read %s: %s" % (name, e))
    if front != "+y":
        info.update(OB.face_front(parts, front))                # its front made +Y, as every object here
    OB.save(parts, folder)
    meta = {"name": Path(name).stem, "file": name, "format": ext, "scale": 1.0, "turn": 0, "front": front, **info}
    if (pk.get("foliage") if pk is not None else foliage.strip().lower() in ("1", "true", "yes")):
        meta["foliage"] = True
    if pk is not None:
        meta["package"] = package
        pk["slots"].append({"object": oid, "weight": 1.0})
        _save_package(package, pk)
    (folder / "meta.json").write_text(json.dumps(meta))
    mem.record("object_import", "imported %s: %.2f x %.2f x %.2f m%s" % (
        name, info["width_m"], info["depth_m"], info["height_m"],
        " into package %s" % pk["name"] if pk is not None else ""), {"id": oid, **meta})
    return {"id": oid, **meta}


def _object_meta(oid):
    f = OBJECTS / oid / "meta.json"
    return json.loads(f.read_text()) if f.exists() else None


@app.get("/api/objects")
def list_objects():
    out = []
    for d in sorted(OBJECTS.glob("*"), key=lambda p: p.stat().st_mtime):
        m = _object_meta(d.name)
        if m:
            out.append({"id": d.name, **m})
    return {"objects": out}


@app.post("/api/objects/{oid}")
def update_object(oid: str, payload: dict):
    """Change an object's scale, for files whose units were written inconsistently."""
    oid = _safe_id(oid)
    m = _object_meta(oid)
    if not m:
        raise HTTPException(404, "unknown object")
    if "scale" in payload:
        m["scale"] = max(1e-4, float(payload["scale"]))
    if "turn" in payload:
        m["turn"] = int(payload["turn"]) % 4      # quarter turns, for objects modelled facing elsewhere
    (OBJECTS / oid / "meta.json").write_text(json.dumps(m))
    return {"id": oid, **m}


@app.delete("/api/objects/{oid}")
def delete_object(oid: str):
    import shutil
    oid = _safe_id(oid)
    if not (OBJECTS / oid).exists():
        raise HTTPException(404, "unknown object")
    m = _object_meta(oid) or {}
    shutil.rmtree(OBJECTS / oid, ignore_errors=True)
    # an object in a package is one of its slots: the slot goes with it
    if m.get("package") and _package(m["package"]) is not None:
        pk = _package(m["package"])
        pk["slots"] = [sl for sl in pk["slots"] if sl["object"] != oid]
        _save_package(m["package"], pk)
    return {"ok": True}


# ------------------------------------------------------------------ buildings
# Buildings made in the Dynamic creation tab (app/buildings.py), one buildings/<id>.json
# each. Use as object makes one an object layer, as an imported object, so it can be
# placed, put in a package or laid automatically on islands like any other.
def _building(bid):
    bid = _safe_id(bid)
    f = BUILDINGS / ("%s.json" % bid)
    if not f.exists():
        raise HTTPException(404, "unknown building")
    return bid, json.loads(f.read_text())


@app.get("/api/buildings")
def list_buildings():
    out = []
    for f in sorted(BUILDINGS.glob("*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            b = json.loads(f.read_text())
        except ValueError:
            continue
        out.append({"id": f.stem, "name": b.get("name", "Building"), "levels": len(b.get("levels", [])),
                    "object": b.get("object") if b.get("object") and _object_meta(b["object"]) else None,
                    "updated": round(f.stat().st_mtime)})
    return {"buildings": out}


@app.get("/api/buildings/{bid}")
def get_building(bid: str):
    bid, b = _building(bid)
    if b.get("object") and not _object_meta(b["object"]):
        b.pop("object")                                  # its object was deleted since
    return {"id": bid, **b}


@app.post("/api/buildings")
def save_building(payload: dict):
    """Save a building (a new one without an id): its name, preset and levels."""
    b = BLD.check(payload)
    bid = payload.get("id")
    if bid:
        bid, old = _building(bid)
        if old.get("object") and "object" not in b:
            b["object"] = old["object"]
    else:
        bid = uuid.uuid4().hex[:12]
    _write_json(BUILDINGS / ("%s.json" % bid), b)
    return {"id": bid, **b}


@app.delete("/api/buildings/{bid}")
def delete_building(bid: str):
    """Delete a building; an object made from it stays (delete it in the Objects panel)."""
    bid, _ = _building(bid)
    (BUILDINGS / ("%s.json" % bid)).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/buildings/{bid}/object")
def building_object(bid: str):
    """
    The building as an object layer: a box for each drawn level, footprint centred,
    base at 0, its front (-Z) the object's front. Sent again, the same layer is
    updated (its scale and quarter turn kept), so its placements follow.
    """
    bid, b = _building(bid)
    parts = BLD.parts(b)
    if not parts:
        raise HTTPException(400, "draw at least one level first")
    allp = np.vstack([p["pos"] for p in parts])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    shift = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    for p in parts:
        p["pos"] = p["pos"] - shift
    size = hi - lo
    oid = b.get("object") if b.get("object") and _object_meta(b["object"]) else None
    old = _object_meta(oid) if oid else None
    if oid is None:
        oid = uuid.uuid4().hex[:12]
    folder = OBJECTS / oid
    OB.save(parts, folder)
    meta = {"name": b["name"], "file": "building", "format": "building", "building": bid,
            "scale": (old or {}).get("scale", 1.0), "turn": (old or {}).get("turn", 0),
            "frame": "the building's own axes, its front towards the top of the map",
            "width_m": round(float(size[0]), 3), "depth_m": round(float(size[2]), 3), "height_m": round(float(size[1]), 3),
            "triangles": int(sum(len(p["faces"]) for p in parts)), "parts": len(parts)}
    if old and old.get("package"):
        meta["package"] = old["package"]                  # a slot of a package stays one
    _write_json(folder / "meta.json", meta)
    if b.get("object") != oid:
        b["object"] = oid
        _write_json(BUILDINGS / ("%s.json" % bid), b)
    mem.record("building_object", "%s %s as an object: %.1f x %.1f x %.1f m, %d levels" % (
        b["name"], "updated" if old else "made", meta["width_m"], meta["depth_m"], meta["height_m"], len(parts)), {"id": oid, "building": bid})
    return {"id": oid, "updated": bool(old), **meta}


# ------------------------------------------------------------------ packages
# A package mixes several objects in one placement. Each is stored as
# packages/<id>.json: a name and its slots, each slot an imported object (its
# meta says which package it belongs to) with a weight for how often it is picked.
def _safe_id(x):
    """Ids are 12 hex characters: anything else never reaches the file system."""
    x = str(x)
    if not re.fullmatch(r"[0-9a-f]{12}", x):
        raise HTTPException(404, "unknown id")
    return x


def _package(pid):
    if not re.fullmatch(r"[0-9a-f]{12}", str(pid)):
        return None
    f = PACKAGES / ("%s.json" % pid)
    if not f.exists():
        return None
    pk = json.loads(f.read_text())
    pk["slots"] = [sl for sl in pk.get("slots", []) if _object_meta(sl["object"])]   # slots whose object is gone drop out
    return pk


def _save_package(pid, pk):
    keep = {"name": pk["name"], "slots": pk["slots"]}
    if pk.get("foliage"):
        keep["foliage"] = True                    # a package of plants
    (PACKAGES / ("%s.json" % pid)).write_text(json.dumps(keep))


@app.get("/api/packages")
def list_packages():
    out = []
    for f in sorted(PACKAGES.glob("*.json"), key=lambda p: p.stat().st_mtime):
        pk = _package(f.stem)
        if pk is not None:
            out.append({"id": f.stem, **pk})
    return {"packages": out}


@app.post("/api/packages")
def create_package(payload: dict):
    pid = uuid.uuid4().hex[:12]
    pk = {"name": str(payload.get("name") or "Package %d" % (len(list(PACKAGES.glob("*.json"))) + 1))[:80], "slots": []}
    if payload.get("foliage"):
        pk["foliage"] = True                      # a mix of plants, in the Foliage panel
    _save_package(pid, pk)
    mem.record("package_create", "created package %s" % pk["name"], {"id": pid})
    return {"id": pid, **pk}


@app.post("/api/packages/{pid}")
def update_package(pid: str, payload: dict):
    """Rename a package, or set its slots' weights: {"weights": {object id: weight}}."""
    pid = _safe_id(pid)
    pk = _package(pid)
    if pk is None:
        raise HTTPException(404, "unknown package")
    if str(payload.get("name") or "").strip():
        pk["name"] = str(payload["name"]).strip()[:80]
    for sl in pk["slots"]:
        if sl["object"] in (payload.get("weights") or {}):
            sl["weight"] = max(0.0, float(payload["weights"][sl["object"]]))
    _save_package(pid, pk)
    return {"id": pid, **pk}


@app.delete("/api/packages/{pid}")
def delete_package(pid: str):
    """A package and the objects in its slots."""
    import shutil
    pid = _safe_id(pid)
    pk = _package(pid)
    if pk is None:
        raise HTTPException(404, "unknown package")
    for sl in pk["slots"]:
        if (_object_meta(sl["object"]) or {}).get("package") == pid:
            shutil.rmtree(OBJECTS / _safe_id(sl["object"]), ignore_errors=True)
    (PACKAGES / ("%s.json" % pid)).unlink(missing_ok=True)
    mem.record("package_delete", "deleted package %s" % pk["name"], {"id": pid})
    return {"ok": True}


def _scatter_for_export(raw, parts=True):
    pl = _placement_list(raw)
    if not pl:
        return None
    objs, pkgs, keep = {}, {}, []

    def have(oid):
        if oid not in objs:
            m = _object_meta(oid)
            if m:
                objs[oid] = {"meta": m}
                if parts:
                    objs[oid]["parts"] = OB.load_saved(OBJECTS / oid)
        return oid in objs
    for p in pl:
        if "package" in p:
            pk = pkgs.get(p["package"]) or _package(p["package"])
            if pk is not None and [have(sl["object"]) for sl in pk["slots"]].count(True):
                pkgs[p["package"]] = pk
                keep.append(p)
        elif have(p["object"]):
            keep.append(p)
    return {"placements": keep, "objects": objs, "packages": pkgs}


def _placement_list(raw):
    out = []
    for p in raw or []:
        try:
            # one object, or a package: a mix of objects, filling a length (mask
            # pixels) in rows, its random mix fixed by the seed
            q = ({"package": str(p["package"]), "seed": int(p.get("seed", 1)) & 0xFFFFFFFF,
                  "length": max(0.0, float(p.get("length", 0.0)))}
                 if p.get("package") else {"object": str(p["object"])})
            q.update({"cx": float(p["cx"]), "cy": float(p["cy"]),
                      "angle": float(p.get("angle", 0.0)), "nx": max(1, int(p.get("nx", 1))),
                      "ny": max(1, int(p.get("ny", 1))), "gap_x": max(0.0, float(p.get("gap_x", 0.0))),
                      "gap_y": max(0.0, float(p.get("gap_y", 0.0)))})
            # random spaces: each gap a random distance in metres between min and max,
            # along the rows (x) and between them (y); kept when off, so the values stay
            sp = p.get("spaces")
            if isinstance(sp, dict):
                (x0, x1), (y0, y1) = sp.get("x", (0, 0)), sp.get("y", (0, 0))
                q["spaces"] = {"on": bool(sp.get("on", False)),
                               "x": [max(0.0, float(x0)), max(0.0, float(x1))],
                               "y": [max(0.0, float(y0)), max(0.0, float(y1))],
                               "seed": int(sp.get("seed", 1)) & 0xFFFFFFFF}
            # single objects turned on their own, by id ("row-column"): degrees, clockwise
            turns = {str(k): float(v) % 360.0 for k, v in (p.get("turns") or {}).items()
                     if re.fullmatch(r"\d{1,6}-\d{1,4}", str(k))}
            if turns:
                q["turns"] = {k: v for k, v in turns.items() if v}
            # Draw spaced: single objects made empty spots, by id ("row-column"); the
            # rest of the grid stays where it is
            empty = sorted({str(k) for k in (p.get("empty") or []) if re.fullmatch(r"\d{1,6}-\d{1,4}", str(k))})
            if empty:
                q["empty"] = empty[:100000]
            # foliage: a placement of plants, each with its own random scale, turn (degrees)
            # and offset (metres) between a min and a max, and whether plants may overlap
            if p.get("foliage"):
                q["foliage"] = True
            jt = p.get("jitter")
            if isinstance(jt, dict):
                def pair(key, lo, hi, default):
                    v = [min(hi, max(lo, float(x))) for x in (jt.get(key) or default)][:2]
                    return (v * 2)[:2] if v else list(default)
                q["jitter"] = {"seed": int(jt.get("seed", 1)) & 0xFFFFFFFF, "scale": pair("scale", 0.01, 100.0, (1.0, 1.0)),
                               "rotate": pair("rotate", -3600.0, 3600.0, (0.0, 0.0)),
                               "offset": pair("offset", 0.0, 1000.0, (0.0, 0.0)), "overlap": bool(jt.get("overlap", True))}
            # objects alignment (app/curves.py): every second row turned round, the last
            # row turned round, the ends of rows facing out, the first and last rows left out
            al = p.get("align")
            if isinstance(al, dict):
                al = {k: bool(al.get(k, False)) for k in ("rows", "last", "columns", "exclude")}
                if any(al.values()):
                    q["align"] = al
            # inner streets: the space cells drawn as streets (app/curves.py cells),
            # with the road and sidewalk widths, corner radius, how far a street
            # reaching the edge goes on to meet a street, and whether they get markings
            st = p.get("streets")
            if isinstance(st, dict):
                q["streets"] = {"cells": [c for c in map(str, st.get("cells") or [])
                                          if re.fullmatch(r"[xyj]\d{1,4}-\d{1,4}", c)][:5000],
                                "sidewalk_m": min(10.0, max(0.5, float(st.get("sidewalk_m", 2.0)))),
                                "road_m": min(30.0, max(1.0, float(st.get("road_m", 6.0)))),
                                "corner_m": min(30.0, max(0.0, float(st.get("corner_m", 4.0)))),
                                "reach_m": min(2000.0, max(0.0, float(st.get("reach_m", 200.0)))),
                                "markings": bool(st.get("markings", True))}
            # a curved placement: copies along the smooth line through these points;
            # more curve lines for the rows behind, each with as many points
            path = [[float(x), float(y)] for x, y in (p.get("path") or [])]
            if len(path) >= 2:
                q["path"] = path
                q["flip"] = bool(p.get("flip", False))
                lines = [[[float(x), float(y)] for x, y in ln] for ln in (p.get("lines") or [])]
                lines = [ln for ln in lines if len(ln) == len(path)][:500]
                if lines:
                    q["lines"] = lines
            elif p.get("mirror"):
                q["mirror"] = True              # a rectangle laid out from its other end: its mirror image
            # an automatic placement (app/autoplace.py): objects laid by rule on the islands,
            # its settings and the objects it laid, kept as laid
            if isinstance(p.get("auto"), dict):
                q["auto"] = AP.settings(p["auto"])
                q["items"] = AP.item_list(p.get("items"))
            out.append(q)
        except (KeyError, TypeError, ValueError):
            continue
    return out


@app.post("/api/scatter")
def save_scatter(payload: dict):
    """Object placements are kept with their mask, like bridges."""
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    pl = _placement_list(payload.get("placements"))
    path = ARTIFACTS / f"scatter_{rec['id']}.json"
    _write_json(path, pl)
    mem.add_artifact(f"scatter_{rec['id']}", "scatter", path, {"mask": rec["id"], "count": len(pl)})
    return {"ok": True, "count": len(pl)}


def _own_size(meta):
    # an object's width and depth (scaled) as it faces its front, its +Y (the arrow on the
    # map), whatever its quarter turn: laid on an island its width runs along the side
    k = float(meta.get("scale", 1.0))
    return meta["width_m"] * k, meta["depth_m"] * k


# automatic placement: each island's spots for the last few sizes and settings, so islands
# picked or taken out, a slot's islands changed, or a new mix are laid at once (a city's
# 1,300 islands take a few seconds the first time); and the latest request of each
# placement, so one asked again before it is done stops
_auto_cache, _auto_latest, _auto_lock = OrderedDict(), {}, threading.Lock()
AUTO_CACHE_KEEP = 8


@app.post("/api/autoplace")
@tracked("Laying objects on the islands", AUTOPLACE_PLAN)
def autoplace(payload: dict):
    """
    An automatic placement's objects (app/autoplace.py): the selected object, or a
    package's mix, laid by rule on the islands of a generation: every island, or the
    ones given (the page knows which islands were picked, or are in a material slot).
    Returns the objects laid ("items", mask pixels) and how many islands got some.
    payload "key": the placement's own key; a newer request with it stops this one (409).
    """
    t0 = time.time()
    gid = str(payload.get("generation") or "")
    art = mem.artifact(f"{gid}_islands")
    if not art or not Path(art["path"]).exists():
        raise HTTPException(400, "no islands for this texture: run Generate texture first")
    doc = json.loads(Path(art["path"]).read_text())
    kinds = []
    if payload.get("package"):
        pk = _package(payload["package"])
        if pk is None:
            raise HTTPException(400, "unknown package")
        for sl in pk["slots"]:
            meta = _object_meta(sl["object"])
            if meta:
                kinds.append((sl["object"],) + _own_size(meta) + (float(sl.get("weight", 1.0)),))
    else:
        meta = _object_meta(str(payload.get("object") or ""))
        if meta:
            kinds.append((str(payload["object"]),) + _own_size(meta) + (1.0,))
    if not kinds:
        raise HTTPException(400, "no object to lay: import one, or put one in the package")
    au = AP.settings(payload.get("auto"))
    ids = None if au["islands"] == "all" else [int(i) for i in (payload.get("island_ids") or [])]
    w, d = max(k[1] for k in kinds), max(k[2] for k in kinds)
    ckey = (gid, round(w, 4), round(d, 4), au["weight"], au["setback"], au["gap"])
    pkey = str(payload.get("key") or "")[:64]
    with _auto_lock:
        cache = _auto_cache.pop(ckey, None)
        cache = {} if cache is None else cache
        _auto_cache[ckey] = cache                            # the latest used last
        while len(_auto_cache) > AUTO_CACHE_KEEP:
            _auto_cache.popitem(last=False)
        mine = _auto_latest[pkey] = _auto_latest.get(pkey, 0) + 1
    prog.stage("islands", "laying objects on the islands")
    try:
        items, note = AP.lay(doc, ids, kinds, au["weight"], au["setback"], au["gap"], int(payload.get("seed", 1)),
                             cache=cache, progress=prog.part,
                             cancel=(lambda: _auto_latest.get(pkey) != mine) if pkey else None)
    except AP.Superseded:
        raise HTTPException(409, "asked again with other settings")
    return {"items": items, "note": note, "generation": gid, "seconds": round(time.time() - t0, 2)}


@app.get("/api/scatter")
def load_scatter(mask: str):
    rec = mem.artifact(f"scatter_{mask}")
    return {"placements": json.loads(Path(rec["path"]).read_text()) if rec else []}


@app.post("/api/bridges/preview")
def bridges_preview(payload: dict):
    """
    Which road each bridge rectangle lifts, and where its ramps run, before any
    export: the deck and ramps as lines along the road, in mask pixels.
    """
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    bridges = _bridge_list(payload.get("bridges"))
    if not bridges:
        return {"bridges": []}
    mpp = float(payload.get("scale", 1.1))
    gray = np.array(Image.open(rec["path"]).convert("L"))
    mask1, cov1 = _prep(gray, 1)
    mesh, det, rep = QMB.build(mask1, 1, mpp, float(payload.get("straightness", 70)) / 100.0,
                               float(payload.get("spacing_m", 2.0)), coverage=cov1,
                               kerb_band=True, bridges=bridges)
    plans = BRG.plan(mesh, bridges, 1, mpp, height_m=float(payload.get("height_m", 5.0)),
                     ramp_m=float(payload.get("ramp_m", 120.0)))
    out = []
    for p in plans:
        if not p.get("ok"):
            out.append({"index": p["index"], "ok": False, "warnings": p["warnings"]})
            continue
        L, arc = p["line"], p["arc"]
        pick = lambda a, b: L[(arc >= a) & (arc <= b)].round(1).tolist()
        out.append({"index": p["index"], "ok": True,
                    "deck": pick(p["s_in"], p["s_out"]),
                    "ramps": [pick(p["s_in"] - p["r_before"], p["s_in"]),
                              pick(p["s_out"], p["s_out"] + p["r_after"])],
                    "deck_m": round(p["s_out"] - p["s_in"], 1),
                    "ramp_m": [round(p["r_before"], 1), round(p["r_after"], 1)],
                    "steepest_pct": p["steepest_pct"], "warnings": p["warnings"],
                    "crossings": rep.get("crossings", 0)})
    return {"bridges": out}


@app.post("/api/bridges")
def save_bridges(payload: dict):
    """Bridges are kept with their mask, so they are there next time it is loaded."""
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    bridges = _bridge_list(payload.get("bridges"))
    path = ARTIFACTS / f"bridges_{rec['id']}.json"
    _write_json(path, bridges)
    mem.add_artifact(f"bridges_{rec['id']}", "bridges", path, {"mask": rec["id"], "count": len(bridges)})
    return {"ok": True, "count": len(bridges)}


@app.get("/api/bridges")
def load_bridges(mask: str):
    rec = mem.artifact(f"bridges_{mask}")
    return {"bridges": json.loads(Path(rec["path"]).read_text()) if rec else []}


@app.get("/api/tiles")
def get_tiles():
    row = mem.latest_artifact("tileset")
    return row["meta"] if row else {"tiles": {}, "settings": TILE_DEFAULTS}


@app.post("/api/tiles")
@tracked("Building the material tiles", TILES_PLAN)
def rebuild_tiles(payload: dict):
    settings = {k: payload[k] for k in TILE_DEFAULTS if k in payload}
    res = _build_tiles(settings)
    if not res.get("ok"):
        raise HTTPException(400, res.get("reason", "could not build tiles"))
    return res


@app.get("/api/lines")
def get_lines():
    """The learned line width, as a share of road width."""
    tree = mem.tree("lines")
    params = (tree["children"]["yes"]["children"]["no"].get("params") or {}) if tree else {}
    return {"width_ratio": params.get("width_ratio"), "pairs": params.get("width_ratio_pairs", 0)}


@app.get("/api/index")
def get_index():
    """The ordered list for each part of the road, as the agent will read it."""
    labels = _pair_labels()
    return {part: [{"rank": r["rank"], "pair_id": r["pair_id"],
                    "label": labels.get(r["pair_id"], r["pair_id"]),
                    "pixels": r["pixels"], "patches": r["patches"], "size_px": r["size_px"],
                    "confidence": r["confidence"], "weight": round(r["weight"]),
                    "rejections": r.get("rejections", 0), "acceptances": r.get("acceptances", 0),
                    "sitting_out": r.get("cooldown", 0) > 0,
                    "usable": r["patches"] >= R.MIN_USABLE}
                   for r in mem.index(part)]
            for part in ("junction", "edge", "open")}


@app.post("/api/recalculate")
def recalculate():
    """Rebuild the libraries from the pairs that are stored now."""
    return {"ok": True, "rebuilt": _rebuild_libraries("recalculated")}


@app.post("/api/generate")
@tracked("Generating the texture", GEN_PLAN)
def generate(payload: dict):
    """Fill a new mask with material, wear and markings, using the learned libraries."""
    prog.stage("preparing", "choosing the libraries")
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    trees = {name: mem.tree(name) for name in ("material", "noise", "lines")}
    if not all(trees.values()):
        raise HTTPException(400, "run priming first: the trees do not exist yet")

    # read each index top to bottom. Open road picks first; the kerb and the
    # junctions then try the same pair, so one road's parts share one photo.
    labels = _pair_labels()
    open_pick = R.choose_pair(mem, trees, "open", labels=labels)
    prefer = open_pick["chosen"]["pair_id"] if open_pick["chosen"] else None
    picks = {
        "open": open_pick,
        "edge": R.choose_pair(mem, trees, "edge", prefer_pair=prefer, labels=labels),
        "junction": R.choose_pair(mem, trees, "junction", prefer_pair=prefer, labels=labels),
    }
    libraries = {"material": {}, "wear": {}}
    decisions = {}
    for part, pick in picks.items():
        c = pick["chosen"]
        libraries["material"][part] = c["npz"] if c else None
        decisions[part] = {"key": pick["key"], "route": c["id"] if c else None,
                           "label": c["label"] if c else None,
                           "rank": c["rank"] if c else None,
                           "exhausted": pick["exhausted"],
                           "candidates": [{k: v for k, v in x.items() if k != "npz"}
                                          for x in pick["candidates"]]}
    # each library was measured at its own scale; generation resizes it to the output scale
    pair_scale = {pp["id"]: pp["scale"] for pp in mem.trained_pairs()}
    libraries["scales"] = {part: (pair_scale.get(pick["chosen"]["pair_id"])
                                  if pick["chosen"] and pick["chosen"].get("pair_id") else None)
                           for part, pick in picks.items()}
    # wear comes from the same photo as the road it lies on
    libraries["wear"]["open"] = libraries["material"]["open"]
    libraries["wear"]["junction"] = libraries["material"]["junction"]
    if not any(libraries["material"].values()):
        raise HTTPException(400, "no usable library: run priming, then training")
    routes = {"material": {part: {"id": d["route"], "confidence": None, "lesson": d["label"]}
                           for part, d in decisions.items()}, "wear": {}}

    dash = trees["lines"]["children"]["yes"]["children"]["no"].get("params", {})
    dash_px, gap_px = dash.get("dash_px"), dash.get("gap_px")
    dash_share = (dash_px / (dash_px + gap_px)) if dash_px and gap_px else 0.6
    # a dash length set in metres wins over the measured share of the cycle
    cycle_m = float(payload.get("cycle_m", 9.0))
    if payload.get("dash_m") and cycle_m > 0:
        dash_share = float(np.clip(float(payload["dash_m"]) / cycle_m, 0.05, 0.95))

    gid = uuid.uuid4().hex[:12]
    # inner streets drawn between objects become streets in a copy of the mask,
    # which everything from here on uses: texture, junctions, the 3D model
    base = rec
    streets = _inner_streets(rec, float(payload.get("scale", 0.25)))
    if streets["mask"]:
        rec = streets["mask"]
    outs = {k: ARTIFACTS / f"gen_{gid}_{k}.png"
            for k in ("result", "material", "wear", "markings")}
    map_path = ARTIFACTS / f"gen_{gid}_map.npz"
    streets_path = ARTIFACTS / f"gen_{gid}_streets.json"
    lanes_path = ARTIFACTS / f"gen_{gid}_lanes.npz"
    # the Lanes tool's choices: sent with the request, else the ones kept for this mask
    lane_picks = LN.picks_list(payload["lane_picks"] if "lane_picks" in payload else _load_lanes(base))
    # the markings area: lines (and decals) only inside it; sent with the request, else the one kept
    mark_area = LN.area_check(payload["mark_area"] if "mark_area" in payload else _load_area(base))
    res = G.generate(rec["path"], libraries, {
        "lane_picks": lane_picks,
        "mark_area": mark_area,
        "nomark": streets["nomark"],
        "metres_per_pixel": float(payload.get("scale", 0.25)),
        "wear": float(payload.get("wear", 50)) / 100.0,
        "seed": int(payload.get("seed", 7)),
        "cycle_m": float(payload.get("cycle_m", 9.0)),
        "dash_share": dash_share,
        "width_m": float(payload.get("marking_width_m", 0.15)),
        "width_ratio": (dash.get("width_ratio")
                        if payload.get("line_width_mode", "learned") == "learned" else None),
        "setback_m": float(payload.get("setback_m", 2.0)),
        "quality": int(payload.get("quality", 1)),
        "align_lines": bool(payload.get("align_lines", False)),
        "output_scale": int(payload.get("output_scale", 1)),
        "grain": float(payload.get("grain", 50)) / 50.0,
        "match_tone": _priming_tone() if payload.get("match_material") else None,
        "soft_edges": bool(payload.get("soft_edges", True)),
        "street_material": _street_material(payload),
    }, dict(outs, streets=streets_path, lanes=lanes_path), map_path)
    mem.add_artifact(f"{gid}_streets", "generated_streets", streets_path, {"mask": rec["id"]})
    # what a street's lanes need to be laid again on their own (the Lanes tool's Done), and the
    # choices the texture has now
    mem.add_artifact(f"{gid}_lanes", "generation_lanes", lanes_path, {"mask": rec["id"], "picks": lane_picks,
                                                                      "area": mark_area})

    prog.stage("saving", "saving")
    urls = {}
    for kind, path in outs.items():
        aid = f"{gid}_{kind}"
        mem.add_artifact(aid, f"generated_{kind}", path, {"mask": rec["id"]})
        urls[kind] = f"/api/artifact/{aid}"

    # the islands (the areas between the roads) with their numbers, for the island materials
    prog.stage("islands", "numbering the islands")
    lab, islands = ISL.find(np.array(Image.open(rec["path"]).convert("L")), float(payload.get("scale", 0.25)))
    ISL.save(lab, islands, float(payload.get("scale", 0.25)), ARTIFACTS / f"gen_{gid}_islands.json",
             ARTIFACTS / f"gen_{gid}_islands.npz")
    mem.add_artifact(f"{gid}_islands", "generated_islands", ARTIFACTS / f"gen_{gid}_islands.json",
                     {"mask": rec["id"], "count": len(islands)})
    mem.add_artifact(f"{gid}_islandmap", "island_labels", ARTIFACTS / f"gen_{gid}_islands.npz", {"mask": rec["id"]})

    for part in ("open", "edge", "junction"):
        mem.tick_cooldowns(part)
    mem.add_artifact(f"{gid}_map", "generation_map", map_path,
                     {"mask": rec["id"], "base_mask": base["id"], "streets_sig": streets["sig"],
                      "soft_edges": bool(payload.get("soft_edges", True)),
                      "nomark": streets["nomark_path"], "scale": float(payload.get("scale", 0.25)),
                      "output_scale": int(res.get("output_scale", 1)),
                      "dashes": {"cycle_m": float(payload.get("cycle_m", 9.0)),
                                 "dash_share": dash_share,
                                 "width_ratio": res["markings"].get("width_ratio"),
                                 "width_m": float(payload.get("marking_width_m", 0.15)),
                                 "setback_m": float(payload.get("setback_m", 2.0))}})
    mem.record("generate",
               f"generated {res['size'][0]} x {res['size'][1]} from {rec['name']}: "
               f"{res['junctions']} junctions, {res['markings']['dashes']} dashes",
               {"mask": rec["id"], "result": res, "libraries": libraries,
                "decisions": decisions})

    return {"ok": True, "id": gid, "urls": urls, "summary": res, "decisions": decisions,
            "routes": routes, "inner_streets": streets["report"],
            "dash_share": round(dash_share, 3),
            "islands": {"count": len(islands), "url": f"/api/artifact/{gid}_islands"},
            "streets": {"url": f"/api/artifact/{gid}_streets"}, "lane_picks": lane_picks, "mark_area": mark_area}


def _street_material(payload):
    """The scanned material chosen for the streets, for the texture (its colour and size), or None for yours."""
    mid = (payload.get("materials") or {}).get("street")
    if not mid or mid == "tiles":
        return None
    e = LIB.entry(mid)
    if not e or e["kind"] not in LIB.KINDS_FOR_PART["street"]:
        raise HTTPException(400, f"{mid} is not a street material in the library")
    tone = bool(payload.get("match_tone", True))
    return {"colour": np.asarray(Image.open(LIB.DIR / e["colour"]).convert("RGB")), "size_m": e["size_m"],
            "match": tone, "name": e["name"] + (" (toned to your material)" if tone else "")}


def _inner_streets(rec, mpp):
    """
    The mask with the inner streets of its saved placements added (app/streets.py):
    {"mask": the new mask's image record or None, "nomark": where they carry no
    markings, its file, a signature of what was added, and a report}.
    """
    out = {"mask": None, "nomark": None, "nomark_path": None, "sig": None, "report": [], "shapes": []}
    f = ARTIFACTS / f"scatter_{rec['id']}.json"
    if not f.exists():
        return out
    sc = _scatter_for_export(json.loads(f.read_text()), parts=False)
    if not sc or not any((p.get("streets") or {}).get("cells") for p in sc["placements"]):
        return out
    import hashlib
    gray = np.array(Image.open(rec["path"]).convert("L"))
    add, nomark, out["report"], out["shapes"] = ST.inner_streets(gray > 127, mpp, sc["placements"], sc["objects"],
                                                                 sc["packages"])
    if not add.any():
        return out
    out["sig"] = hashlib.sha1(np.packbits(add).tobytes()).hexdigest()
    sid = "st" + out["sig"][:10]
    path = UPLOADS / f"{sid}.png"
    if not path.exists():
        g2 = gray.copy()
        g2[add] = 255
        Image.fromarray(g2).save(path)
    if not mem.image(sid):
        mem.add_image(sid, "gen_mask_streets", f"{rec['name']} + inner streets", path, gray.shape[1], gray.shape[0],
                      I.sha256_file(path), {"base": rec["id"]})
    out["mask"] = mem.image(sid)
    if nomark.any():
        out["nomark"] = nomark
        out["nomark_path"] = str(ARTIFACTS / f"nomark_{sid}.png")
        Image.fromarray((nomark * 255).astype(np.uint8)).save(out["nomark_path"])
    return out


# What each answer to "what is wrong?" does. These change settings rather than
# routes: the same libraries are used, so nothing else about the result moves.
CORRECTIONS = {
    "material_too_noisy": {"label": "Material is too noisy", "adjust": {"grain": -15},
                           "explain": "less grain inside the material"},
    "material_too_flat": {"label": "Material looks too flat", "adjust": {"grain": +15},
                          "explain": "more grain inside the material"},
    "lines_not_aligned": {"label": "Lines are not aligned with the road",
                          "adjust": {"align_lines": True},
                          "explain": "the centreline is averaged before the dashes are placed, "
                                     "so they follow the road axis"},
    "bad_quality": {"label": "Image has bad quality", "adjust": {"quality": +1},
                    "explain": "patches overlap more and are placed with less jitter, "
                               "which softens the joins"},
    "wrong_library": {"label": "Material looks wrong, try the next library", "adjust": {},
                      "explain": "this library sits out and the next one in the order is used",
                      "rejects_library": True},
}


@app.get("/api/corrections")
def corrections():
    return {"corrections": [{"id": k, **v} for k, v in CORRECTIONS.items()]}


@app.post("/api/inspect")
def inspect(payload: dict):
    """What painted this spot, and what memory already knows about places like it."""
    art = mem.artifact(f"{payload.get('generation')}_map")
    if not art or not Path(art["path"]).exists():
        raise HTTPException(400, "unknown generation: run Generate first")
    x, y = int(payload.get("x", 0)), int(payload.get("y", 0))

    with np.load(art["path"]) as z:
        groups = z["groups"]
        if not (0 <= y < groups.shape[0] and 0 <= x < groups.shape[1]):
            raise HTTPException(400, "point is outside the image")
        code = int(groups[y, x])
        if code == 0:
            return {"on_road": False,
                    "reading": "that spot is off the road, so nothing was painted there"}
        group = {1: "junction", 2: "edge", 3: "open"}[code]
        edge_m = float(z["edge_m"][y, x])
        jid = int(z["junction_ids"][y, x])
        types = {int(t.split(":")[0]): t.split(":", 1)[1] for t in z["junction_types"].tolist()}
    jtype = types.get(jid) if group == "junction" else None

    trees = {name: mem.tree(name) for name in ("material", "noise")}
    labels = _pair_labels()
    open_pick = R.choose_pair(mem, trees, "open", labels=labels)
    prefer = open_pick["chosen"]["pair_id"] if open_pick["chosen"] else None
    pick = open_pick if group == "open" else R.choose_pair(mem, trees, group, prefer_pair=prefer,
                                                           labels=labels)
    c = pick["chosen"]
    return {"on_road": True, "point": [x, y], "part": group, "edge_m": round(edge_m, 2),
            "junction": {"id": jid, "type": jtype} if jtype else None,
            "layers": {"material": {
                "key": pick["key"], "features": pick["features"], "part": group,
                "route": ({"id": c["id"], "label": c["label"], "rank": c["rank"],
                           "confidence": c["confidence"], "patches": c["patches"],
                           "size_px": c["size_px"]} if c else None),
                "candidates": [{k: v for k, v in x.items() if k != "npz"}
                               for x in pick["candidates"]],
                "exhausted": pick["exhausted"],
                "rejected": pick["recall"]["rejected"],
            }}}


@app.post("/api/review")
def review(payload: dict):
    """Record a correction; a rejected pair drops down the order for this part of the road."""
    correction = CORRECTIONS.get(payload.get("correction"))
    outcome = payload.get("outcome")
    # a settings correction changes wear, quality or line alignment, and must
    # not also swap the library: that would change two things at once
    if correction and not correction.get("rejects_library"):
        outcome = "adjusted"
    if outcome not in {"accepted", "rejected", "flagged", "adjusted"}:
        raise HTTPException(400, "outcome must be accepted, rejected, flagged or adjusted")
    part = payload.get("part") or payload.get("features", {}).get("part", "open")
    reason = payload.get("reason") or (correction["label"] if correction else None)
    res = R.record_pair_outcome(mem, part, payload["key"], payload.get("features", {}),
                                payload["route"], outcome, reason)
    if correction:
        mem.record("correction", f"{correction['label']}: {correction['explain']}",
                   {"key": payload["key"], "route": payload["route"],
                    "correction": payload.get("correction"), "adjust": correction["adjust"]})

    trees = {name: mem.tree(name) for name in ("material", "noise")}
    pick = R.choose_pair(mem, trees, part, labels=_pair_labels())
    nxt = pick["chosen"]
    return {"ok": True, "recall": res["recall"], "moved": res["moved"],
            "next_route": nxt["id"] if nxt else None,
            "next_label": nxt["label"] if nxt else None,
            "exhausted": pick["exhausted"],
            "candidates": [{k: v for k, v in x.items() if k != "npz"} for x in pick["candidates"]],
            "correction": ({"id": payload.get("correction"), **correction} if correction else None)}


# One build per model: the top view, Generate 3D scene and Export 3D model ask for the
# same model (same texture and settings). A request for one already being built waits
# for it, its console line following that build, instead of building it a second time
# alongside (twice the time, and both writing the same file); one built already, and
# still the file on disk, is answered at once.
_builds = {}                       # key -> {"done": Event, "result", "error", "job", "mtime", "at"}
_builds_lock = threading.Lock()
BUILD_KEEP_S = 1800                # a finished build is reused for half an hour


def _build_key(payload):
    gid = payload.get("generation")
    art = mem.artifact(f"{gid}_map")
    if not art:
        return None
    import hashlib
    base = art["meta"].get("base_mask") or art["meta"].get("mask")
    sc = ARTIFACTS / f"scatter_{base}.json"                  # objects and inner streets
    dc = ARTIFACTS / f"gen_{gid}_decals.json"                 # the decals' places
    tiles = mem.latest_artifact("tileset") if payload.get("mesh", "tiled") == "tiled" else None
    return json.dumps([gid, {k: v for k, v in payload.items() if k != "progress"},
                       hashlib.sha1(sc.read_bytes()).hexdigest() if sc.exists() else None,
                       hashlib.sha1(dc.read_bytes()).hexdigest() if dc.exists() else None,
                       tiles["id"] if tiles else None], sort_keys=True, default=str)


# ------------------------------------------------------------------ islands
# The areas between the roads, numbered when the texture is generated
# (app/islands.py). Island material slots are kept with their mask's picture
# (by its contents, so loading the same mask again brings them back); each
# remembers its islands by a point inside each.
def _slots_key(rec):
    return f"islandslots_{(rec.get('sha256') or rec['id'])[:20]}"


# ------------------------------------------------------------------ lanes
# The Lanes tool's choices (app/lanes.py): streets given their own lanes on each
# side, by a point on each, kept with the mask as the island slots are.
def _lanes_key(rec):
    return f"lanepicks_{(rec.get('sha256') or rec['id'])[:20]}"


def _load_lanes(rec):
    art = mem.artifact(_lanes_key(rec)) if rec else None
    return json.loads(Path(art["path"]).read_text()) if art and Path(art["path"]).exists() else []


@app.get("/api/lanes")
def load_lanes(mask: str):
    return {"picks": _load_lanes(mem.image(mask))}


@app.post("/api/lanes")
def save_lanes(payload: dict):
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    picks = LN.picks_list(payload.get("picks"))
    key = _lanes_key(rec)
    path = ARTIFACTS / f"{key}.json"
    _write_json(path, picks)
    mem.add_artifact(key, "lane_picks", path, {"mask": rec["id"], "picks": len(picks)})
    return {"ok": True, "picks": len(picks)}


# ------------------------------------------------------------------ markings area
# One rectangle on the map outside which no lane lines and no decals are laid
# (app/lanes.py), kept with the mask as the lane choices are.
def _area_key(rec):
    return f"markarea_{(rec.get('sha256') or rec['id'])[:20]}"


def _load_area(rec):
    art = mem.artifact(_area_key(rec)) if rec else None
    if not art or not Path(art["path"]).exists():
        return None
    return LN.area_check(json.loads(Path(art["path"]).read_text()).get("area"))


@app.get("/api/markarea")
def get_markarea(mask: str):
    return {"area": _load_area(mem.image(mask))}


@app.post("/api/markarea")
def save_markarea(payload: dict):
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    area = LN.area_check(payload.get("area"))
    path = ARTIFACTS / f"{_area_key(rec)}.json"
    _write_json(path, {"area": area})
    mem.add_artifact(_area_key(rec), "markings_area", path, {"mask": rec["id"], "area": area})
    return {"ok": True, "area": area}


@app.post("/api/markarea/apply")
def apply_markarea(payload: dict):
    """
    A markings area set (or changed, or taken away) after the texture was
    generated: kept with the mask, then laid on the texture by taking away what
    now lies outside it and laying what now lies inside it, nothing else: the
    lane lines of the streets the change reaches (lanes_repaint) and the decals
    (_lay_decals, as last laid: only those that go or come are painted).
    """
    gid = payload.get("generation")
    art = mem.artifact(f"{gid}_map")
    if not art:
        raise HTTPException(400, "unknown generation: run Generate texture first")
    area = LN.area_check(payload.get("area"))
    base_rec = mem.image(art["meta"].get("base_mask") or art["meta"]["mask"])
    if base_rec:
        save_markarea({"mask": base_rec["id"], "area": area})
    t0 = time.time()
    lanes = mem.artifact(f"{gid}_lanes")
    lane_res = lanes_repaint({"generation": gid, "lane_picks": (lanes["meta"] or {}).get("picks") or [] if lanes else [],
                              "mark_area": area})
    dec_res = None
    dart = mem.artifact(f"{gid}_decals")
    if dart and Path(dart["path"]).exists():
        m = dart["meta"] or {}
        dec_res = _lay_decals(gid, m.get("layers") if m.get("layers") is not None else _decal_layers(),
                              m.get("mode", "separate"), int(m.get("seed", 7)), area)
    v = uuid.uuid4().hex[:6]
    return {"ok": True, "area": area, "lanes": lane_res, "decals": dec_res, "seconds": round(time.time() - t0, 2),
            "urls": {"result": f"/api/artifact/{gid}_result?v={v}", "markings": f"/api/artifact/{gid}_markings?v={v}",
                     "decals": f"/api/artifact/{gid}_decalimg?v={v}", "full": f"/api/artifact/{gid}_full?v={v}"}}


_repaint_lock = threading.Lock()


@app.post("/api/lanes/repaint")
def lanes_repaint(payload: dict):
    """
    The Lanes tool's Done: lay again only the lane lines of the streets whose
    lanes changed, on the generated texture as it is (app/generation.py's
    repaint_lanes), instead of generating the whole texture again. The road
    texture, its markings layer, the texture under painted decals and the
    roads + islands picture all take the change; nothing else is touched.
    """
    gid = payload.get("generation")
    art, result, markings = mem.artifact(f"{gid}_lanes"), mem.artifact(f"{gid}_result"), mem.artifact(f"{gid}_markings")
    if not result or not markings:
        raise HTTPException(400, "unknown generation: run Generate texture first")
    if not art or not Path(art["path"]).exists():
        raise HTTPException(409, "this texture was generated before lanes could be laid on their own: "
                                 "press Generate texture once, then lane changes are quick")
    picks = LN.picks_list(payload.get("lane_picks"))
    t0 = time.time()
    with _repaint_lock:
        art = mem.artifact(f"{gid}_lanes")
        old_area = (art["meta"] or {}).get("area")
        area = LN.area_check(payload["mark_area"]) if "mark_area" in payload else LN.area_check(old_area)
        base = ARTIFACTS / f"gen_{gid}_result_base.png"
        full = mem.artifact(f"{gid}_full")
        files = {"result": result["path"], "markings": markings["path"],
                 "base": str(base) if base.exists() else None,
                 "full": full["path"] if full and Path(full["path"]).exists() else None}
        decals = None
        dec = mem.artifact(f"{gid}_decalimg")
        if files["base"] and dec and Path(dec["path"]).exists():
            places = mem.artifact(f"{gid}_decals")
            decals = (np.asarray(Image.open(dec["path"]).convert("RGBA")),
                      bool(places and (places["meta"] or {}).get("mode") == "painted"))
        out = G.repaint_lanes(art["path"], (art["meta"] or {}).get("picks") or [], picks, files, decals,
                              old_area=old_area, new_area=area)
        meta = dict(art["meta"] or {}, picks=picks, area=area)
        mem.add_artifact(f"{gid}_lanes", "generation_lanes", Path(art["path"]), meta)
        st = mem.artifact(f"{gid}_streets")
        if out["changed"] and st and Path(st["path"]).exists():
            # the Lanes tool's streets: each with the lanes it has now
            data = json.loads(Path(st["path"]).read_text())
            now = {c["id"]: c for c in out["changed"]}
            for s_ in data.get("streets", []):
                c = now.get(s_["id"])
                if c:
                    s_.update(sides=c["sides"], even=c["even"], cut=c["cut"])
            Path(st["path"]).write_text(json.dumps(data, separators=(",", ":")))
    v = uuid.uuid4().hex[:6]
    secs = round(time.time() - t0, 2)
    if out["changed"]:
        mem.record("lanes_repaint", f"lanes laid again on {len(out['changed'])} street(s) in {secs} s",
                   {"generation": gid, "changed": out["changed"]})
    return {"ok": True, "changed": out["changed"], "area_streets": out.get("area_streets", 0), "seconds": secs,
            "urls": {"result": f"/api/artifact/{gid}_result?v={v}", "markings": f"/api/artifact/{gid}_markings?v={v}",
                     **({"full": f"/api/artifact/{gid}_full?v={v}"} if files["full"] else {})}}


@app.get("/api/islands")
def load_island_slots(mask: str):
    rec = mem.image(mask)
    art = mem.artifact(_slots_key(rec)) if rec else None
    return {"slots": json.loads(Path(art["path"]).read_text()) if art and Path(art["path"]).exists() else []}


@app.post("/api/islands")
def save_island_slots(payload: dict):
    rec = mem.image(payload.get("mask"))
    if not rec:
        raise HTTPException(400, "unknown mask")
    slots = ISL.slot_list(payload.get("slots"))
    key = _slots_key(rec)
    path = ARTIFACTS / f"{key}.json"
    _write_json(path, slots)
    mem.add_artifact(key, "island_slots", path, {"mask": rec["id"], "slots": len(slots)})
    return {"ok": True, "slots": len(slots)}


@app.post("/api/islands/texture")
@tracked("Generating the islands texture", ISLANDS_PLAN)
def islands_texture(payload: dict):
    """
    The islands texture of a generated texture: only the islands, each laid
    with its slot's material (or the Squares choice, or the sidewalks'), at real
    size, the roads transparent. Also the road texture with the islands under it.
    """
    gid = payload.get("generation")
    art, result = mem.artifact(f"{gid}_map"), mem.artifact(f"{gid}_result")
    isl_art = mem.artifact(f"{gid}_islands")
    if not art or not result:
        raise HTTPException(400, "unknown generation: run Generate texture first")
    if not isl_art or not mem.artifact(f"{gid}_islandmap"):
        raise HTTPException(400, "this texture was generated before the islands were numbered: press Generate texture again")
    rec = mem.image(art["meta"]["mask"])
    if not rec:
        raise HTTPException(400, "the mask for this generation is missing")
    prog.stage("materials", "the island materials")
    tileset = _tileset_with_paths() or {"settings": {"tile_m": 4.0, "px": 1024, "variants": 3}, "tiles": {}}
    try:
        used = _apply_materials(tileset, {"square": (payload.get("materials") or {}).get("square")},
                                bool(payload.get("match_tone", True)))
        islands = _islands_for(gid, tileset, payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    tile_m = float(tileset["settings"]["tile_m"])

    def look(recs, kind, mix=None):
        # ground: the 3D model's variety too (its drift, and the slot's Mix with both height maps,
        # placed by the same ground pattern), so the islands texture matches the model
        out = {"tiles": [t["path"] for t in recs], "tile_m": tile_m, "ground": kind == "ground"}
        if kind == "ground":
            out["drift"] = M3.GROUND_DRIFT
            if mix and mix.get("tiles"):
                hs = [M3.tile_relief(t, tile_m, kind) for t in recs[:2]]
                bh, bd = M3.tile_relief(mix["tiles"][0], tile_m, "ground")
                out.update(heights=[h for h, _ in hs], depth_m=hs[0][1],
                           mix={"tile": mix["tiles"][0]["path"], "height": bh, "depth_m": bd,
                                **ISL.mix_params(mix["amount"], mix["size_m"], mix.get("seed", 0))})
        return out
    looks = {s: look(v["tiles"], v["kind"], v.get("mix")) for s, v in islands["looks"].items()}
    base = tileset["tiles"].get("block") or tileset["tiles"].get("sidewalk")
    if not base:
        if len(islands["slot_of"]) < len(json.loads(Path(isl_art["path"]).read_text())["islands"]):
            raise HTTPException(400, "no sidewalk tiles yet for the islands without a material: build them in the "
                                     "Memory tab, choose a Squares material, or give every island a material")
        base = next(iter(islands["looks"].values()))["tiles"]
    default = look(base, M3._sidewalk_kind(base[0]))
    data = json.loads(Path(isl_art["path"]).read_text())
    out, full = ARTIFACTS / f"gen_{gid}_islandtex.png", ARTIFACTS / f"gen_{gid}_full.png"
    # what each island is laid from, as a key: made before with these, only the islands
    # whose key changed are laid again (the rest of the picture stays as it was)
    keys_path = ARTIFACTS / f"gen_{gid}_islandtex.keys.json"
    import hashlib
    sig = lambda v: hashlib.sha1(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()[:16]
    look_key = {s_: sig([v["tiles"], v.get("kind"), v.get("mix")]) for s_, v in islands["looks"].items()}
    look_key[-1] = sig([base, M3._sidewalk_kind(base[0])])
    now = {"whole": sig([rec["path"], Path(rec["path"]).stat().st_mtime, art["meta"].get("scale"),
                         art["meta"].get("output_scale", 1), art["meta"].get("soft_edges", True), int(payload.get("seed", 7)),
                         tile_m, M3.GROUND_DRIFT, ISL.TEXTURE_VERSION]),
           "islands": {str(i["id"]): look_key[islands["slot_of"].get(i["id"], -1)] for i in data["islands"]}}
    only = None
    if keys_path.exists() and out.exists() and full.exists() and not payload.get("whole"):
        was = json.loads(keys_path.read_text())
        if was.get("whole") == now["whole"]:
            only = {int(k) for k, v in now["islands"].items() if was["islands"].get(k) != v}
    t0 = time.time()
    info = ISL.texture(np.array(Image.open(rec["path"]).convert("L")), islands["labels"], data["islands"],
                       float(art["meta"]["scale"]), int(art["meta"].get("output_scale", 1)),
                       bool(art["meta"].get("soft_edges", True)), islands["slot_of"], looks, default,
                       int(payload.get("seed", 7)), out, result_path=result["path"], full_path=full, only=only)
    keys_path.write_text(json.dumps(now))
    mem.add_artifact(f"{gid}_islandtex", "generated_islands_texture", out, {"generation": gid, **info})
    mem.add_artifact(f"{gid}_full", "generated_full_texture", full, {"generation": gid})
    stem = Path((mem.image(art["meta"].get("base_mask")) or rec)["name"]).stem
    names = {s: v["name"] for s, v in islands["looks"].items()}
    by = [{"slot": s, "name": names.get(s) if s >= 0 else (used.get("square") or "Same as sidewalks"),
           "islands": c} for s, c in sorted(info["by_slot"].items())]
    mem.record("islands_texture", f"islands texture {info['size'][0]} x {info['size'][1]}: {info['islands']} islands",
               {"generation": gid, "by_slot": by})
    v = uuid.uuid4().hex[:6]
    return {"ok": True, "size": info["size"], "islands": info["islands"], "by_slot": by,
            "partial": None if only is None else {"islands": sorted(only), "pixels": info["pixels"]},
            "seconds": round(time.time() - t0, 2),
            "urls": {"islands": f"/api/artifact/{gid}_islandtex?v={v}", "full": f"/api/artifact/{gid}_full?v={v}"},
            "downloads": {"islands": f"/api/download/{gid}_islandtex?name={stem}_islands.png",
                          "full": f"/api/download/{gid}_full?name={stem}_roads_and_islands.png"}}


# ------------------------------------------------------------------ decals
# Pictures laid on the streets (app/decals.py), kept in workspace/decals: each
# a PNG with its name and size in index.json, and the decal layers (which
# picture, where it goes) in layers.json, for every map.
DECALS = WORK / "decals"
DECAL_EXT = {".png", ".webp", ".jpg", ".jpeg"}


def _decal_index():
    f = DECALS / "index.json"
    return json.loads(f.read_text()) if f.exists() else []


def _decal_layers():
    f = DECALS / "layers.json"
    return json.loads(f.read_text()) if f.exists() else []


@app.get("/api/decals")
def list_decals():
    idx = _decal_index()
    # layers saved before the four kinds of place come back in them (Place randomly: intersections or streets)
    return {"decals": [{**d, "url": f"/api/decals/{d['id']}/image"} for d in idx],
            "layers": DC.layer_list(_decal_layers(), {d["id"] for d in idx})}


@app.post("/api/decals/import")
async def import_decal(file: UploadFile = File(...)):
    """A decal picture (PNG with a transparent background, or WebP or JPEG): kept as a PNG."""
    name = Path(file.filename or "decal.png").name
    if Path(name).suffix.lower() not in DECAL_EXT:
        raise HTTPException(400, "use a PNG (with a transparent background), WebP or JPEG picture")
    raw = await file.read()
    try:
        im = Image.open(__import__("io").BytesIO(raw)).convert("RGBA")
    except Exception:
        raise HTTPException(400, f"could not read {name}")
    if max(im.size) > 4096:
        im.thumbnail((4096, 4096), Image.LANCZOS)
    DECALS.mkdir(parents=True, exist_ok=True)
    did = uuid.uuid4().hex[:12]
    im.save(DECALS / f"{did}.png", optimize=True)
    entry = {"id": did, "name": Path(name).stem.replace("_", " ")[:60], "width": im.width, "height": im.height,
             "see_through": bool(np.asarray(im)[..., 3].min() < 250)}
    (DECALS / "index.json").write_text(json.dumps(_decal_index() + [entry], indent=1))
    mem.record("decal_import", f"imported decal {name}", {"id": did})
    return {**entry, "url": f"/api/decals/{did}/image"}


@app.get("/api/decals/{did}/image")
def decal_image(did: str):
    if not re.fullmatch(r"[0-9a-f]{12}", did) or not (DECALS / f"{did}.png").exists():
        raise HTTPException(404, "no such decal")
    return FileResponse(DECALS / f"{did}.png", media_type="image/png")


@app.delete("/api/decals/{did}")
def delete_decal(did: str):
    idx = _decal_index()
    if not any(d["id"] == did for d in idx):
        raise HTTPException(404, "no such decal")
    (DECALS / "index.json").write_text(json.dumps([d for d in idx if d["id"] != did], indent=1))
    (DECALS / f"{did}.png").unlink(missing_ok=True)
    (DECALS / "layers.json").write_text(json.dumps([l for l in _decal_layers() if l.get("decal") != did]))
    return {"ok": True}


@app.post("/api/decals/layers")
def save_decal_layers(payload: dict):
    DECALS.mkdir(parents=True, exist_ok=True)
    layers = DC.layer_list(payload.get("layers"), {d["id"] for d in _decal_index()})
    _write_json(DECALS / "layers.json", layers)
    return {"ok": True, "layers": len(layers)}


_prep_cache = {}                   # the last mask's road edge at its texture's size (slow on a big map)


def _prep_cached(path, gray, s):
    key = (str(path), Path(path).stat().st_mtime, int(s))
    if key not in _prep_cache:
        _prep_cache.clear()
        _prep_cache[key] = _prep(gray, s)
    return _prep_cache[key]


@app.post("/api/decals/apply")
def apply_decals(payload: dict):
    """
    Lay a generated texture's decals: where each layer's decals go (from the
    generation's mask), the decals as a picture of their own (the Decals layer),
    and, painted into the road texture or not (mode "painted" or "separate").
    The texture as generated is kept, so laying them again starts from it.

    Laid before on this texture, only what changed is painted again: the decals
    are placed (that is quick), compared with the ones laid last time, and only
    the footprints of those gone, moved or new (all of them when the mode
    changes) are worked out again in the Decals layer and the road texture
    (app/decals.py's repaint). payload "whole": lay everything again. Only
    decals whose middle lies inside the markings area (payload "mark_area",
    else the one kept for the mask) are laid.
    """
    gid = payload.get("generation")
    art = mem.artifact(f"{gid}_map")
    if not art:
        raise HTTPException(400, "unknown generation: run Generate texture first")
    area = LN.area_check(payload["mark_area"] if "mark_area" in payload
                         else _load_area(mem.image(art["meta"].get("base_mask") or art["meta"]["mask"])))
    return _lay_decals(gid, payload.get("layers"), "painted" if payload.get("mode") == "painted" else "separate",
                       int(payload.get("seed", 7)), area, bool(payload.get("whole")))


def _lay_decals(gid, raw_layers, mode, seed, area, whole=False):
    """Lay a generation's decals (apply_decals): only what changed since last time, unless whole."""
    t0 = time.time()
    art, result = mem.artifact(f"{gid}_map"), mem.artifact(f"{gid}_result")
    if not art or not result:
        raise HTTPException(400, "unknown generation: run Generate texture first")
    rec = mem.image(art["meta"]["mask"])
    idx = _decal_index()
    layers = DC.layer_list(raw_layers, {d["id"] for d in idx})
    base = ARTIFACTS / f"gen_{gid}_result_base.png"
    if not base.exists():
        import shutil
        shutil.copyfile(result["path"], base)                 # the texture as generated, without decals
    mpp, s = float(art["meta"]["scale"]), int(art["meta"].get("output_scale", 1))
    gray = np.array(Image.open(rec["path"]).convert("L"))
    sizes = {d["id"]: (d["width"], d["height"]) for d in idx}
    bridges = None
    base_mask = mem.artifact(f"bridges_{art['meta'].get('base_mask') or rec['id']}")
    if base_mask and Path(base_mask["path"]).exists():
        bridges = _bridge_list(json.loads(Path(base_mask["path"]).read_text()))
    placements = DC.place(gray, mpp, layers, sizes, seed, bridges)
    if area:
        # the markings area: a decal is laid when its middle lies inside
        keep = LN.in_area(LN.area_scaled(area, 1), [p["x"] for p in placements], [p["y"] for p in placements])
        placements = [p for p, k in zip(placements, np.atleast_1d(keep)) if k]
    pics = DC.load_pictures(idx, DECALS)
    base_rgb = np.asarray(Image.open(base).convert("RGB"))
    _, coverage = _prep_cached(rec["path"], gray, s) if s else (None, None)
    if coverage.shape != base_rgb.shape[:2]:
        coverage = None
    layer_png = ARTIFACTS / f"gen_{gid}_decals.png"
    before = mem.artifact(f"{gid}_decals")
    old = None
    if before and Path(before["path"]).exists() and layer_png.exists() and not whole:
        old = json.loads(Path(before["path"]).read_text()).get("placements")
    if old is not None:
        # laid before: only where something changed
        old_mode = (before["meta"] or {}).get("mode", "separate")
        changes = DC.changed(old, placements) if old_mode == mode else old + placements
        Hh, Ww = base_rgb.shape[:2]
        area = sum(max(0, min(Ww, x1) - max(0, x0)) * max(0, min(Hh, y1) - max(0, y0))
                   for x0, y0, x1, y1 in (DC.footprint(p, mpp / s, s) for p in changes))
        layer = np.array(Image.open(layer_png).convert("RGBA"))
        out = np.array(Image.open(result["path"]).convert("RGB"))
        if layer.shape[:2] != base_rgb.shape[:2] or out.shape != base_rgb.shape or area > Hh * Ww / 3:
            old = None              # sizes do not agree, or most of the texture changes: lay everything (the same result)
    if old is not None:
        pixels = DC.repaint(changes, placements, pics, mpp / s, s, coverage, base_rgb, layer, out, mode == "painted")
        if changes:
            Image.fromarray(layer, "RGBA").save(layer_png, compress_level=1)
        now = {DC._key(p) for p in placements}
        partial = {"changed": len(changes), "gone": sum(1 for p in changes if DC._key(p) not in now),
                   "pixels": int(pixels), "share": round(pixels / max(1, out.shape[0] * out.shape[1]), 5)}
    else:
        rgba = DC.paint(base_rgb.shape[:2], placements, pics, mpp / s, s, coverage)
        Image.fromarray(DC.layer_picture(rgba), "RGBA").save(layer_png, compress_level=4)
        out = DC.composite(base_rgb, rgba) if mode == "painted" and placements else base_rgb
        partial = None
    if partial is None or partial["changed"]:
        # the roads + islands picture (Generate islands texture) takes the same change as the road texture
        full = mem.artifact(f"{gid}_full")
        if full and Path(full["path"]).exists():
            was = np.asarray(Image.open(result["path"]).convert("RGB")).astype(np.int16)
            fp = np.asarray(Image.open(full["path"]).convert("RGB")).astype(np.int16)
            if fp.shape == was.shape == out.shape:
                d = out.astype(np.int16) - was
                if d.any():
                    Image.fromarray(np.clip(fp + d, 0, 255).astype(np.uint8)).save(full["path"], compress_level=1)
        tmp = Path(result["path"]).with_name(f"gen_{gid}_result.{uuid.uuid4().hex[:6]}.part.png")
        Image.fromarray(out).save(tmp, compress_level=1 if partial else 6)
        os.replace(tmp, result["path"])
    DC.save(placements, ARTIFACTS / f"gen_{gid}_decals.json")
    mem.add_artifact(f"{gid}_decalimg", "generated_decals", layer_png, {"generation": gid})
    mem.add_artifact(f"{gid}_decals", "decal_places", ARTIFACTS / f"gen_{gid}_decals.json",
                     {"generation": gid, "count": len(placements), "mode": mode, "seed": seed, "area": area,
                      "layers": layers})
    names = {d["id"]: d["name"] for d in idx}
    v = uuid.uuid4().hex[:6]
    return {"ok": True, "count": len(placements), "mode": mode, "layers": DC.summary(placements, layers, names),
            "partial": partial, "seconds": round(time.time() - t0, 2),
            "urls": {"decals": f"/api/artifact/{gid}_decalimg?v={v}", "result": f"/api/artifact/{gid}_result?v={v}",
                     "full": f"/api/artifact/{gid}_full?v={v}"}}


def _decals_for(gid, payload):
    """A generation's decals for the 3D model: their places and pictures, and how they lie ("separate" or "painted")."""
    art = mem.artifact(f"{gid}_decals")
    if not art or not Path(art["path"]).exists():
        return None
    placements = json.loads(Path(art["path"]).read_text()).get("placements") or []
    if not placements:
        return None
    return {"placements": placements, "images": DC.load_pictures(_decal_index(), DECALS),
            "names": {d["id"]: d["name"] for d in _decal_index()},
            "mode": "painted" if payload.get("decals") == "painted" else "separate"}


@app.post("/api/export3d")
@tracked("Building the 3D model", EXPORT_PLAN)
def export3d(payload: dict):
    """Build a GLB road model from a generated result, in real metres (once per model)."""
    key = _build_key(payload)
    if key is None:
        return _export3d(payload)
    final = ARTIFACTS / f"gen_{payload.get('generation')}.glb"
    with _builds_lock:
        now = time.time()
        for k in [k for k, b in _builds.items() if b["done"].is_set() and now - b["at"] > BUILD_KEEP_S]:
            _builds.pop(k)
        b = _builds.get(key)
        if b is not None and b["done"].is_set() and not (
                b["result"] and final.exists() and final.stat().st_mtime == b["mtime"]):
            b = None                                         # the file has been rebuilt since, other settings
        mine = b is None
        if mine:
            b = _builds[key] = {"done": threading.Event(), "result": None, "error": None,
                                "job": payload.get("progress"), "mtime": None, "at": now}
    if not mine:
        if not b["done"].is_set():
            prog.follow(b["job"], "already being built: shared")
            b["done"].wait()
        if b["error"] is not None:
            raise b["error"]
        return dict(b["result"])
    try:
        res = _export3d(payload)
        b["result"], b["mtime"] = res, final.stat().st_mtime
        return dict(res)
    except BaseException as e:
        b["error"] = e
        with _builds_lock:
            if _builds.get(key) is b:
                _builds.pop(key)
        raise
    finally:
        b["at"] = time.time()
        b["done"].set()


# The last 3D models, by everything they were built from but the parts a change can
# build again on its own (app/model3d.py: the paving with the island material slots,
# the objects, the decals, the streets' lanes): a model whose only changes are in
# those parts builds just them again and the view swaps just them, not the whole model
PART_FIELDS = {"paving": ("island_slots",), "objects": ("scatter",), "decals": ("decals", "decal_version"),
               "lanes": ("lane_picks", "mark_area")}
_models = {}                       # base key -> {"rebuild", "parts", "info", "tileset", "lock", "at"}
_models_lock = threading.Lock()
MODELS_KEEP = 2


def _model_keys(gid, payload, streets_sig, picks, area=None):
    """A model's base key (all it is built from but its parts) and each part's key."""
    import hashlib
    skip = {"progress", "view"} | {f for fs in PART_FIELDS.values() for f in fs}
    tiles = mem.latest_artifact("tileset")
    base = json.dumps([gid, {k: v for k, v in payload.items() if k not in skip}, streets_sig,
                       tiles["id"] if tiles else None], sort_keys=True, default=str)
    dc = ARTIFACTS / f"gen_{gid}_decals.json"
    parts = {"paving": json.dumps(payload.get("island_slots"), sort_keys=True, default=str),
             "objects": json.dumps(payload.get("scatter"), sort_keys=True, default=str),
             "decals": json.dumps([payload.get("decals"), hashlib.sha1(dc.read_bytes()).hexdigest() if dc.exists() else None]),
             "lanes": json.dumps([picks, area], sort_keys=True)}
    return base, parts


def _sig(text):
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _rebuild_parts(gid, key, part_keys, payload, picks, out, part_out, area=None):
    """
    The model with only its changed parts built again, from a model built before
    with everything else the same, or None if there is none (or it cannot).
    payload "view": what the asking view holds ({"base", "parts": {part: its
    signature}}): the parts it needs are those that differ from it, which may be
    more than changed now (another view, the Top view, took some changes first).
    """
    with _models_lock:
        e = _models.get(key)
    if e is None:
        return None
    changed = [n for n, k in part_keys.items() if e["parts"].get(n) != k]
    inputs = {}
    for n in changed:
        inputs[n] = ({"picks": picks, "area": area} if n == "lanes" else _islands_for(gid, e["tileset"], payload) if n == "paving"
                     else _scatter_for_export(payload.get("scatter")) if n == "objects" else _decals_for(gid, payload))
    view = payload.get("view") if isinstance(payload.get("view"), dict) else {}
    send = None
    if view.get("base") == _sig(key) and isinstance(view.get("parts"), dict):
        send = [n for n in part_keys if view["parts"].get(n) != _sig(part_keys[n])]
    prog.stage("parts", "building again only what changed: " + (", ".join(changed) or "nothing"))
    with e["lock"]:
        try:
            r = e["rebuild"](inputs, out, part_out, send)
        except Exception as err:                         # its state is no longer sure: build it all
            print("the changed parts could not be built on their own, building the whole model:", repr(err))
            r = None
        if r is None:                                    # (or a street gained or lost all its lines: build it all)
            with _models_lock:
                _models.pop(key, None)
            return None
        e["parts"].update({n: part_keys[n] for n in changed})
        e["at"] = time.time()
    now, names = r
    return dict(e["info"], **{k: v for k, v in now.items() if not k.startswith("_") and k != "layouts"}, partial=names)


def _keep_model(key, part_keys, rebuild, info, tileset):
    gid = json.loads(key)[0]
    with _models_lock:
        # a model holds its whole build in memory: only the newest few, of this texture only
        for k in [k for k in _models if json.loads(k)[0] != gid]:
            _models.pop(k)
        _models[key] = {"rebuild": rebuild, "parts": dict(part_keys), "info": dict(info), "tileset": tileset,
                        "lock": threading.Lock(), "at": time.time()}
        for k in sorted(_models, key=lambda k: _models[k]["at"])[:-MODELS_KEEP]:
            _models.pop(k)


def _export3d(payload):
    prog.stage("preparing", "preparing")
    gid = payload.get("generation")
    art = mem.artifact(f"{gid}_map")
    result = mem.artifact(f"{gid}_result")
    if not art or not result:
        raise HTTPException(400, "unknown generation: run Generate first")
    rec = mem.image(art["meta"]["mask"])
    if not rec:
        raise HTTPException(400, "the mask for this generation is missing")
    # written under its own name, then put in place whole: a page loading the model
    # never reads a file half written
    final = ARTIFACTS / f"gen_{gid}.glb"
    out = final.with_name(f"gen_{gid}.{uuid.uuid4().hex[:8]}.part.glb")
    # the changed parts alone, when only parts changed (for a view that has the rest)
    lanes_final = ARTIFACTS / f"gen_{gid}.parts.glb"
    lanes_out = lanes_final.with_name(f"gen_{gid}.parts.{uuid.uuid4().hex[:8]}.part.glb")
    mode = payload.get("mesh", "tiled")
    dash_cfg = dict(art["meta"].get("dashes") or {})
    base_rec = mem.image(art["meta"].get("base_mask") or art["meta"]["mask"])
    dash_cfg["lane_picks"] = LN.picks_list(payload["lane_picks"] if "lane_picks" in payload else _load_lanes(base_rec))
    dash_cfg["mark_area"] = LN.area_check(payload["mark_area"] if "mark_area" in payload else _load_area(base_rec))
    if art["meta"].get("nomark") and Path(art["meta"]["nomark"]).exists():
        dash_cfg["nomark"] = np.array(Image.open(art["meta"]["nomark"]).convert("L")) > 127
    stale, inner, streets_now = None, None, {"sig": None}
    base = mem.image(art["meta"].get("base_mask") or art["meta"]["mask"])
    if base:
        streets_now = _inner_streets(base, float(art["meta"]["scale"]))
        if streets_now["sig"] != art["meta"].get("streets_sig"):
            stale = ("the inner streets have changed since this was generated: press Generate again "
                     "to build them into the streets")
        elif streets_now["shapes"] and base["id"] != rec["id"]:
            # the inner streets as they were drawn, laid exactly; the other roads from the mask without them
            inner = {"base_mask_path": base["path"], "shapes": streets_now["shapes"]}
    rkey, part_keys = (_model_keys(gid, payload, [streets_now["sig"], stale is None], dash_cfg["lane_picks"],
                                   dash_cfg["mark_area"]) if mode == "tiled" else (None, None))
    try:
        info = (_rebuild_parts(gid, rkey, part_keys, payload, dash_cfg["lane_picks"], out, lanes_out, dash_cfg["mark_area"])
                if rkey else None)
        if info is not None:
            pass                                          # only some parts changed: built again on their own
        elif mode == "tiled":
            tileset = _tileset_with_paths()
            if tileset is None:
                raise ValueError("no material tiles yet: build them in the Memory tab, or train, which builds them")
            prog.stage("materials", "street materials")
            used_materials = _apply_materials(tileset, payload.get("materials") or {},
                                              bool(payload.get("match_tone", True)))
            islands = _islands_for(gid, tileset, payload)
            decals = _decals_for(gid, payload)
            base_tex = ARTIFACTS / f"gen_{gid}_result_base.png"     # without painted decals (the variation layer)
            markings = mem.artifact(f"{gid}_markings")
            info = M3.export_road_tiled_glb(
                rec["path"], str(base_tex) if base_tex.exists() else result["path"], markings["path"] if markings else None, tileset,
                float(art["meta"]["scale"]), int(art["meta"].get("output_scale", 1)), out,
                straightness=float(payload.get("straightness", 70)) / 100.0,
                spacing_m=float(payload.get("spacing_m", 2.0)),
                variation=float(payload.get("variation", 100)) / 100.0,
                seed=int(payload.get("seed", 7)),
                dash_cfg=dash_cfg,
                sidewalk=({"share": float(payload.get("sw_share", 30)) / 100.0,
                           "min_m": float(payload.get("sw_min_m", 1.5)),
                           "max_m": float(payload.get("sw_max_m", 2.5)),
                           "height_m": float(payload.get("sw_height_cm", 10)) / 100.0,
                           "max_road_m": float(payload.get("sw_max_road_m", 20)),
                           "skip_interchanges": bool(payload.get("sw_skip_interchanges", False))}
                          if payload.get("sidewalks", True) else None),
                bridges=_bridge_list(payload.get("bridges")),
                markings="painted" if payload.get("markings") == "painted" else "strips",
                optimise=payload.get("mesh_detail") == "optimised",
                blocks=({"height_m": float(payload.get("sw_height_cm", 10)) / 100.0}
                        if payload.get("blocks", True) else None),
                scatter=_scatter_for_export(payload.get("scatter")),
                bridge_cfg={"height_m": float(payload.get("bridge_height_m", 5.0)),
                            "ramp_m": float(payload.get("bridge_ramp_m", 120.0)),
                            "deck_m": float(payload.get("bridge_deck_m", 1.0))},
                inner=inner, surface=_surface_choice(payload.get("surface_detail", "scan")), islands=islands,
                decals=decals)
            prog.stage("check", "checking the tile repeat")
            check = M3.repeat_check(out, info["_mesh"], info["_world"], info["tile_m"])
            rebuild = info["_rebuild"]
            info = {k: v for k, v in info.items() if not k.startswith("_") and k != "layouts"}
            info["materials_used"] = used_materials
            if islands is None and any(s.get("picks") for s in ISL.slot_list(payload.get("island_slots"))):
                info["islands_warning"] = ("this texture was generated before the islands were numbered: "
                                           "press Generate texture again to use the island materials")
            info["repeat_check"] = {k: float(v) for k, v in check.items()} if check else None
            if rkey:
                _keep_model(rkey, part_keys, rebuild, info, tileset)
        elif mode == "traced":
            info = M3.export_road_glb(rec["path"], result["path"], float(art["meta"]["scale"]),
                                      int(art["meta"].get("output_scale", 1)), out)
            info["mesh"] = "traced"
        else:
            info = M3.export_road_quads_glb(
                rec["path"], result["path"], float(art["meta"]["scale"]),
                int(art["meta"].get("output_scale", 1)), out,
                straightness=float(payload.get("straightness", 70)) / 100.0,
                spacing_m=float(payload.get("spacing_m", 2.0)),
                optimise=payload.get("mesh_detail") == "optimised", inner=inner)
    except ValueError as e:
        out.unlink(missing_ok=True)
        lanes_out.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except BaseException:
        out.unlink(missing_ok=True)
        lanes_out.unlink(missing_ok=True)
        raise
    os.replace(out, final)
    out = final
    aid = f"{gid}_glb"
    partial = None
    if info.get("partial") is not None:
        url = None
        if lanes_out.exists():
            os.replace(lanes_out, lanes_final)
            mem.add_artifact(f"{gid}_glbparts", "model_glb_parts", lanes_final, {"generation": gid})
            url = f"/api/artifact/{gid}_glbparts?v={uuid.uuid4().hex[:6]}"
        # replace: the parts whose objects the view takes out (url: the new ones, None when they come to nothing)
        partial = {"url": url, "replace": info.pop("partial"), "bytes": info.get("part_bytes")}
    mem.add_artifact(aid, "model_glb", out, {"generation": gid, **info})
    count = (f"{info['quads']:,} quads" if info.get("mesh") == "quads"
             else f"{info.get('triangles', 0):,} triangles")
    mem.record("export3d", f"exported road model: {count}, {info['size_m'][0]} x {info['size_m'][1]} m",
               {"generation": gid, **info})
    stem = Path(rec["name"]).stem
    # what the model was built from but its parts: a view holding a model of the same base can
    # take the changed parts alone (partial)
    base_sig = _sig(rkey) if rkey else None
    return {"ok": True, "url": f"/api/download/{aid}?name={stem}_roads.glb", **info, "streets_warning": stale,
            "partial": partial, "model_base": base_sig,
            "model_parts": {n: _sig(k) for n, k in part_keys.items()} if part_keys else None}


@app.post("/api/export3d/convert")
def export3d_convert(payload: dict):
    """
    The 3D model built by /api/export3d in the format asked (GLB, FBX or OBJ;
    FBX and OBJ as a zip with their textures), with the streets, sidewalks and
    kerbs as one object (combine_ground) and the placed objects as one object
    (combine_objects) if asked (app/formats.py).
    """
    gid = payload.get("generation")
    fmt = str(payload.get("format", "glb")).lower()
    if fmt not in ("glb", "fbx", "obj"):
        raise HTTPException(400, "format must be GLB, FBX or OBJ")
    glb = mem.artifact(f"{gid}_glb")
    if not glb or not Path(glb["path"]).exists():
        raise HTTPException(400, "build the 3D model first")
    art = mem.artifact(f"{gid}_map")
    rec = mem.image(art["meta"].get("base_mask") or art["meta"]["mask"]) if art else None
    stem = Path(rec["name"]).stem + "_roads" if rec else "roads"
    cg, co = bool(payload.get("combine_ground")), bool(payload.get("combine_objects"))
    front = str(payload.get("front") or "+y").lower()
    if front not in FMT.FRONT_TURN:
        raise HTTPException(400, "front must be +y, +x, -y or -x")
    if fmt == "glb" and not cg and not co and front == "+y":
        return {"ok": True, "url": f"/api/download/{gid}_glb?name={stem}.glb", "format": "glb", "objects": None, "front": front}
    tag = f"{fmt}_{int(cg)}{int(co)}" + ("" if front == "+y" else "_" + {"+x": "px", "-y": "ny", "-x": "nx"}[front])
    out = ARTIFACTS / f"gen_{gid}_{tag}.{'glb' if fmt == 'glb' else 'zip'}"
    part = out.with_name(out.stem + f".{uuid.uuid4().hex[:8]}.part{out.suffix}")
    try:
        info = FMT.convert(glb["path"], part, fmt, stem, cg, co, front)
        os.replace(part, out)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    mem.add_artifact(f"{gid}_{tag}", f"model_{fmt}", out, {"generation": gid, **info})
    name = f"{stem}.glb" if fmt == "glb" else f"{stem}_{fmt}.zip"
    return {"ok": True, "url": f"/api/download/{gid}_{tag}?name={name}", "format": fmt, "file_bytes": out.stat().st_size,
            "objects": info["count"], "names": info["objects"], "copies": info["copies"], "front": front}


@app.get("/api/download/{artifact_id}")
def download(artifact_id: str, name: str = ""):
    rec = mem.artifact(artifact_id)
    if not rec:
        raise HTTPException(404, "unknown file")
    return FileResponse(rec["path"], filename=name or Path(rec["path"]).name,
                        media_type="model/gltf-binary" if str(rec["path"]).endswith(".glb")
                        else None)


@app.get("/api/memory/tree-history")
def tree_history(name: str = ""):
    """Every saved version of every tree. Stage A stays here after Stage B runs."""
    return {"versions": mem.tree_history(name or None)}


@app.post("/api/memory/restore-tree")
def restore_tree(payload: dict):
    name, version = payload.get("name"), int(payload.get("version", 0))
    data = mem.restore_tree(name, version)
    if data is None:
        raise HTTPException(404, "unknown version")
    mem.record("tree_restored", f"restored {name} to version {version}", payload)
    return {"ok": True, "name": name, "version": version}


# --------------------------------------------------------------------- UI
@app.middleware("http")
async def no_cache_for_interface(request, call_next):
    """
    The interface files change with every update. Without this, browsers keep
    showing the previous version of the page until a hard refresh.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


app.mount("/", StaticFiles(directory=ROOT / "ui", html=True), name="ui")


def _port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sck:
        return sck.connect_ex(("127.0.0.1", port)) == 0


def _running_version(port):
    import json as _json, urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/health" % port, timeout=2) as r:
            return _json.loads(r.read().decode()).get("version") or "an older version (no version number)"
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    import uvicorn
    if _port_in_use(8765):
        other = _running_version(8765)
        print("")
        print("  !! Port 8765 is already in use, so this server cannot start.")
        if other:
            print("  !! Another Road Texture Agent is already running there: version %s." % other)
            print("  !! Your browser is talking to THAT one, which is why you may see an old page.")
        else:
            print("  !! Another program is using it.")
        print("  !! Close every other server window (or restart the computer), then start this one again.")
        print("  !! This one is version %s, in %s" % (VERSION, ROOT))
        print("")
        input("  Press Enter to close this window...")
        sys.exit(1)
    print("Road Texture Agent %s: http://127.0.0.1:8765" % VERSION)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
