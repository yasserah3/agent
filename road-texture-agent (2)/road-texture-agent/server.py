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

import json
import mimetypes
import re
import uuid
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
from app import images as I
from app import generation as G
from app import junctions as J
from app import model3d as M3
from app import library as LIB
from app import looks as LK
from app import tiles as TL
from app import objects as OB
from app import quadmesh as QMB
from app import bridges as BRG
from app import streets as ST
from app.generation import prepare_mask as _prep
from app import routing as R
from app import training as T
from app.memory import Memory

ROOT = Path(__file__).parent
VERSION = "2026.10.05-bake1"   # must match UI_VERSION in ui/app.js


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
for d in (UPLOADS, ARTIFACTS, OBJECTS, PACKAGES):
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
@app.post("/api/junctions")
def junctions(payload: dict):
    rec = mem.image(payload.get("image"))
    if not rec:
        raise HTTPException(400, "unknown image")
    limit = float(payload.get("divider_limit_px", J.DIVIDER_LIMIT_PX))
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

    material = A.analyse_material(have["material"]["path"])
    line = A.analyse_line(have["line"]["path"])
    noise = A.analyse_noise(noise_path)

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
    for part in TL.PARTS:
        tone = common
        ratio = (stats[part]["contrast"] / base_c) if (stats[part]["contrast"] and base_c) else 1.0
        tiles[part] = []
        for v in range(int(cfg["variants"])):
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
            Path(row["path"]).unlink(missing_ok=True)
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
            for f in (Path(row["path"]), Path(meta.get("npz", "") or "/nonexistent")):
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
    for n, pair in enumerate(pairs, start=1):
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
MATERIAL_PARTS = {"street": ("open", "edge", "junction"), "sidewalk": ("sidewalk",), "kerb": ("kerbstone",)}


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
        if not mid or mid == "tiles":
            continue
        e = LIB.entry(mid)
        if not e or e["kind"] not in LIB.KINDS_FOR_PART[part]:
            raise ValueError(f"{mid} is not a {part} material in the library")
        def tone_of(t):
            own = (tileset["tiles"].get(t) or [None])[0]
            if own is None and part == "kerb":
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
        else:
            # paving and kerbs: the scan itself, so its joints stay on their grid
            tone = tone_of(tile_parts[0])
            rec = LIB.tile_for(mid, tile_m, px, ARTIFACTS / "library", tone)
            for t in tile_parts:
                tileset["tiles"][t] = [rec]
            used[part] = e["name"] + (" (toned to your tiles)" if tone is not None else "")
    return used


def _your_tile(part):
    """Your trained tile that a part (street, sidewalk, kerb) uses, with its file and the tileset's id, or None."""
    row = mem.latest_artifact("tileset")
    if not row or not row["meta"].get("tiles"):
        return None
    tiles = row["meta"]["tiles"]
    names = list(MATERIAL_PARTS[part]) + (["sidewalk"] if part == "kerb" else [])
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
            "materials": [{k: e.get(k) for k in ("id", "name", "kind", "size_m", "source", "title", "authors", "licence")}
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
        nrm, rgh, _ = SF.maps(img, (t["tile_m"], t["tile_m"]), kind, source="scan")
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
async def import_object(file: UploadFile = File(...), package: str = Form(""), foliage: str = Form("")):
    """
    Import a GLB, OBJ or FBX object: normalised to metres, footprint centred,
    base at 0. With a package, the object becomes a new slot of that package
    instead of a layer of its own. Plants (foliage) are imported the same way,
    marked foliage, or into a foliage package.
    """
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
    OB.save(parts, folder)
    meta = {"name": Path(name).stem, "file": name, "format": ext, "scale": 1.0, "turn": 0, **info}
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
                     if re.fullmatch(r"\d{1,4}-\d{1,4}", str(k))}
            if turns:
                q["turns"] = {k: v for k, v in turns.items() if v}
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
    path.write_text(json.dumps(pl))
    mem.add_artifact(f"scatter_{rec['id']}", "scatter", path, {"mask": rec["id"], "count": len(pl)})
    return {"ok": True, "count": len(pl)}


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
    path.write_text(json.dumps(bridges))
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
def generate(payload: dict):
    """Fill a new mask with material, wear and markings, using the learned libraries."""
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
    res = G.generate(rec["path"], libraries, {
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
    }, outs, map_path)

    urls = {}
    for kind, path in outs.items():
        aid = f"{gid}_{kind}"
        mem.add_artifact(aid, f"generated_{kind}", path, {"mask": rec["id"]})
        urls[kind] = f"/api/artifact/{aid}"

    for part in ("open", "edge", "junction"):
        mem.tick_cooldowns(part)
    mem.add_artifact(f"{gid}_map", "generation_map", map_path,
                     {"mask": rec["id"], "base_mask": base["id"], "streets_sig": streets["sig"],
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
            "dash_share": round(dash_share, 3)}


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


@app.post("/api/export3d")
def export3d(payload: dict):
    """Build a GLB road model from a generated result, in real metres."""
    gid = payload.get("generation")
    art = mem.artifact(f"{gid}_map")
    result = mem.artifact(f"{gid}_result")
    if not art or not result:
        raise HTTPException(400, "unknown generation: run Generate first")
    rec = mem.image(art["meta"]["mask"])
    if not rec:
        raise HTTPException(400, "the mask for this generation is missing")
    out = ARTIFACTS / f"gen_{gid}.glb"
    mode = payload.get("mesh", "tiled")
    dash_cfg = dict(art["meta"].get("dashes") or {})
    if art["meta"].get("nomark") and Path(art["meta"]["nomark"]).exists():
        dash_cfg["nomark"] = np.array(Image.open(art["meta"]["nomark"]).convert("L")) > 127
    stale, inner = None, None
    base = mem.image(art["meta"].get("base_mask") or art["meta"]["mask"])
    if base:
        streets_now = _inner_streets(base, float(art["meta"]["scale"]))
        if streets_now["sig"] != art["meta"].get("streets_sig"):
            stale = ("the inner streets have changed since this was generated: press Generate again "
                     "to build them into the streets")
        elif streets_now["shapes"] and base["id"] != rec["id"]:
            # the inner streets as they were drawn, laid exactly; the other roads from the mask without them
            inner = {"base_mask_path": base["path"], "shapes": streets_now["shapes"]}
    try:
        if mode == "tiled":
            row = mem.latest_artifact("tileset")
            if not row or not row["meta"].get("tiles"):
                raise ValueError("no material tiles yet: build them in the Memory tab, or train, which builds them")
            tileset = json.loads(json.dumps(row["meta"]))
            for part, lst in tileset["tiles"].items():
                for tile in lst:
                    tile["path"] = mem.artifact(tile["id"])["path"]
            used_materials = _apply_materials(tileset, payload.get("materials") or {},
                                              bool(payload.get("match_tone", True)))
            markings = mem.artifact(f"{gid}_markings")
            info = M3.export_road_tiled_glb(
                rec["path"], result["path"], markings["path"] if markings else None, tileset,
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
                inner=inner, surface=_surface_choice(payload.get("surface_detail", "scan")))
            check = M3.repeat_check(out, info["_mesh"], info["_world"], info["tile_m"])
            info = {k: v for k, v in info.items() if not k.startswith("_") and k != "layouts"}
            info["materials_used"] = used_materials
            info["repeat_check"] = {k: float(v) for k, v in check.items()} if check else None
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
        raise HTTPException(400, str(e))
    aid = f"{gid}_glb"
    mem.add_artifact(aid, "model_glb", out, {"generation": gid, **info})
    count = (f"{info['quads']:,} quads" if info.get("mesh") == "quads"
             else f"{info.get('triangles', 0):,} triangles")
    mem.record("export3d", f"exported road model: {count}, {info['size_m'][0]} x {info['size_m'][1]} m",
               {"generation": gid, **info})
    stem = Path(rec["name"]).stem
    return {"ok": True, "url": f"/api/download/{aid}?name={stem}_roads.glb", **info, "streets_warning": stale}


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
