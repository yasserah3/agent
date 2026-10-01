"""
Stage B: training on a real street.

An aligned pair goes in: a black and white mask, and a photo of the same street
at the same size. The mask says where the road is, and where its junctions and
kerbs are. The photo says what those places look like.

Road pixels are sorted into three groups, using the junction geometry that is
already computed by rules:

    junction   inside a junction region
    edge       within 1.5 m of the road edge
    open       everything else

Each group is measured on its own with the same three methods used in priming,
and gets its own patch library. That is how the trees learn that a kerb looks
different from the middle of the road, and that junctions are more worn.

Stage A is never discarded. Its routes stay in the tree as fallbacks, so if a
Stage B route is rejected during review the agent can fall back to what priming
learned.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from app import analysis as A
from app import junctions as J

EDGE_METRES = 1.5
MIN_PATCHES = 6          # below this a group is too small to trust


def group_stats(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Tone, contrast and grain measured over one group of pixels only."""
    g = gray.astype(np.float32)
    vals = g[mask]
    if vals.size < 50:
        return {"pixels": int(vals.size), "reading": "too few pixels to measure"}

    diffs = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        diffs.append(np.abs(g - shifted)[mask])
    local_contrast = float(np.mean(diffs))
    fine = (g - ndi.gaussian_filter(g, 2.0))[mask]

    return {
        "pixels": int(vals.size),
        "mean": round(float(vals.mean()), 2),
        "std": round(float(vals.std()), 2),
        "p5": round(float(np.percentile(vals, 5)), 1),
        "p95": round(float(np.percentile(vals, 95)), 1),
        "local_contrast": round(local_contrast, 3),
        "grain": round(float(fine.std()), 3),
        "reading": ("flat, almost no grain" if fine.std() < 1.0 else
                    "fine grain" if fine.std() < 6 else "coarse, heavily worn"),
    }


def line_width_ratio(paint: np.ndarray, road_width: float) -> dict:
    """
    How wide the painted lines are, as a share of the road's width.

    The line's width is measured along its own centreline: twice the distance
    from the middle of the stroke to its edge. Measuring it as a proportion
    rather than in pixels or metres is what lets it stay right at any output
    size and on any road, without depending on the scale being exactly right.
    """
    from skimage.morphology import skeletonize
    if paint.sum() < 50 or road_width <= 0:
        return {"line_width_px": None, "line_ratio": None, "paint_px": int(paint.sum())}
    # ignore specks that are not strokes
    lab, n = ndi.label(paint)
    sizes = ndi.sum(paint, lab, index=np.arange(1, n + 1)) if n else []
    strokes = np.isin(lab, [i + 1 for i, a in enumerate(sizes) if a >= 12])
    if strokes.sum() < 50:
        return {"line_width_px": None, "line_ratio": None, "paint_px": int(paint.sum())}
    sk = skeletonize(strokes)
    inside = ndi.distance_transform_edt(strokes)
    widths = 2 * inside[sk] - 1          # a one-pixel line measures 1, not 2
    width = float(np.median(widths)) if widths.size else None
    if not width or width <= 0:
        return {"line_width_px": None, "line_ratio": None, "paint_px": int(paint.sum())}
    return {"line_width_px": round(width, 2),
            "line_ratio": round(width / road_width, 4),
            "paint_px": int(strokes.sum())}


def analyse_pair(mask_path, photo_path, metres_per_pixel: float,
                 divider_limit: float = J.DIVIDER_LIMIT_PX) -> dict:
    """Measure one aligned pair, grouped by junction, edge and open road."""
    mask_gray = np.array(Image.open(mask_path).convert("L"))
    photo_rgb = np.array(Image.open(photo_path).convert("RGB"))
    photo_gray = np.array(Image.open(photo_path).convert("L"))
    if mask_gray.shape != photo_gray.shape:
        return {"ok": False, "error": (f"not aligned: mask is "
                                       f"{mask_gray.shape[1]} x {mask_gray.shape[0]}, photo is "
                                       f"{photo_gray.shape[1]} x {photo_gray.shape[0]}")}

    det = J.detect(mask_gray, divider_limit)
    road = det["road"]
    H, W = road.shape

    # inside a junction, from the junction regions the rules already found
    yy, xx = np.mgrid[0:H, 0:W]
    in_junction = np.zeros((H, W), bool)
    for j in det["junctions"]:
        if j["type"] == "interchange (flagged)":
            continue                      # flagged, not learned from
        in_junction |= ((yy - j["cy"]) ** 2 + (xx - j["cx"]) ** 2) <= j["r"] ** 2
    in_junction &= road

    edge_m = det["dt"] * metres_per_pixel

    # The kerb band is 1.5 m, but it is taken from both sides, so on a narrow
    # road it can swallow the whole surface and leave no open road to learn
    # from. Cap it at a quarter of the road width: on a wide road the 1.5 m
    # applies as intended, on a narrow one the band shrinks to fit.
    widths = 2 * det["dt"][det["skeleton"].astype(bool)]
    road_width = float(np.median(widths)) if widths.size else 8.0
    cap_m = max(0.05, road_width * metres_per_pixel * 0.25)
    band_m = min(EDGE_METRES, cap_m)
    band_capped = band_m < EDGE_METRES - 1e-6

    # Paint is not material. If the training photo has markings on it, their
    # pixels leak into the material and wear libraries and then get scattered
    # at random over the whole road, because those layers have no idea where a
    # line belongs. Only the lines tree paints lines, so paint is cut out here.
    road_vals = photo_gray[road].astype(np.float32)
    med = float(np.median(road_vals))
    mad = float(np.median(np.abs(road_vals - med))) * 1.4826
    paint_threshold = max(med + 3.0 * max(mad, 1.0), med + 25.0)
    paint_core = road & (photo_gray > paint_threshold)
    paint = ndi.binary_dilation(paint_core, iterations=1) & road   # catch soft edges
    surface = road & ~paint

    in_junction &= surface
    near_edge = surface & ~in_junction & (edge_m <= band_m)
    open_road = surface & ~in_junction & ~near_edge

    groups = {"junction": in_junction, "edge": near_edge, "open": open_road}
    per = A.periodicity(photo_gray)

    # patches must fit inside a road. A fixed size fails on narrow roads: a
    # 48 px patch cannot sit inside an 8 px wide street, and every group comes
    # back empty. So the size follows the measured road width.
    junction_r = (float(np.median([j["r"] for j in det["junctions"]]))
                  if det["junctions"] else road_width)
    edge_band_px = band_m / max(metres_per_pixel, 1e-6)
    sizes = {
        "open": int(np.clip(round(road_width * 0.9), 6, 64)),
        # the kerb band is only as wide as EDGE_METRES, so its patches must fit
        # inside that band, not inside the whole road
        "edge": int(np.clip(round(min(road_width * 0.5, edge_band_px * 0.9)), 4, 48)),
        "junction": int(np.clip(round(junction_r * 0.9), 6, 64)),
    }

    def build_library(m, size, name):
        """
        Take clean patches, shrinking them until enough fit.

        On a narrow road with wide markings, a patch of the ideal size may never
        fit between the paint and the kerb. Rather than return nothing, the
        patch gets smaller until it does.
        """
        tried = []
        for attempt in range(4):
            lib = A.patch_library(photo_rgb, mask=m, count=48, size=size,
                                  avoid_period=per.get("period_px"),
                                  min_coverage=0.5 if name == "edge" else 0.6,
                                  exclude=paint)
            tried.append((size, lib.get("count", 0)))
            if lib.get("count", 0) >= MIN_PATCHES or size <= 4:
                lib["size_attempts"] = tried
                return lib
            size = max(4, int(size * 0.6))
        lib["size_attempts"] = tried
        return lib

    out = {}
    for name, m in groups.items():
        stats = group_stats(photo_gray, m)
        lib = build_library(m, sizes[name], name)
        out[name] = {"stats": stats, "patches": lib,
                     "patch_size_px": lib.get("size_px", sizes[name]),
                     "share_of_road": round(float(m.sum() / max(1, road.sum())), 4),
                     "usable": lib.get("count", 0) >= MIN_PATCHES}

    paint_share = float(paint.sum() / max(1, road.sum()))

    return {
        "ok": True,
        "junctions": det["summary"],
        "metres_per_pixel": metres_per_pixel,
        "road_px": int(road.sum()),
        "road_width_px": round(road_width, 1),
        "road_width_m": round(road_width * metres_per_pixel, 2),
        "kerb_band": {
            "metres": round(band_m, 2),
            "pixels": round(band_m / max(metres_per_pixel, 1e-6), 1),
            "capped": band_capped,
            "reading": (f"kerb band reduced to {band_m:.2f} m so open road still exists "
                        f"on a {road_width * metres_per_pixel:.1f} m road"
                        if band_capped else f"kerb band {band_m:.1f} m as intended"),
        },
        "groups": out,
        "periodicity": per,
        "resolution": {
            "road_width_px": round(road_width, 1),
            "ok": road_width >= 20,
            "reading": ("road is {:.0f} px wide, enough detail to learn from".format(road_width)
                        if road_width >= 20 else
                        "road is only {:.0f} px wide: too coarse to learn surface detail. "
                        "Use a pair where the road is at least 20 px across, "
                        "ideally 40 px or more.".format(road_width)),
        },
        "paint": {
            **line_width_ratio(paint_core, road_width),
            "share_of_road": round(paint_share, 4),
            "threshold": round(paint_threshold, 1),
            "excluded": True,
            "reading": (f"markings cover {paint_share*100:.1f}% of the road and were cut out, "
                        "so they cannot leak into the material or wear"
                        if paint_share > 0.002 else "no clear markings in this photo"),
        },
    }


def accumulate(agg: dict, result: dict, libs: dict, pair_label: str) -> dict:
    """
    Add one pair's measurements to the running totals.

    With several pairs this matters: without it, each pair overwrites the last
    and the trees end up knowing only the final pair. Here every pair
    contributes, weighted by how many pixels it actually supplied, and every
    pair's patch library is kept so generation can draw from all of them.
    """
    for group, g in result["groups"].items():
        if not g.get("usable"):
            agg.setdefault(group, _empty_group())["skipped"].append(pair_label)
            continue
        st = g["stats"]
        a = agg.setdefault(group, _empty_group())
        n = st["pixels"]
        a["pixels"] += n
        a["tone"] += st["mean"] * n
        a["grain"] += st["grain"] * n
        a["contrast"] += st["local_contrast"] * n
        a["libraries"].append(libs[group])
        a["patches"] += g["patches"].get("count", 0)
        a["pairs"].append(pair_label)
    return agg


def _empty_group():
    return {"pixels": 0, "tone": 0.0, "grain": 0.0, "contrast": 0.0,
            "libraries": [], "patches": 0, "pairs": [], "skipped": []}


def finalise(agg: dict) -> dict:
    """Turn the running totals into one measurement per group."""
    out = {}
    for group, a in agg.items():
        if a["pixels"] == 0:
            out[group] = {"usable": False, "skipped": a["skipped"]}
            continue
        n = a["pixels"]
        out[group] = {
            "usable": True, "pixels": n, "patches": a["patches"],
            "tone": round(a["tone"] / n, 2), "grain": round(a["grain"] / n, 3),
            "contrast": round(a["contrast"] / n, 3),
            "libraries": a["libraries"], "pairs": a["pairs"], "skipped": a["skipped"],
        }
    return out


def merge_into_trees(trees: dict, measured: dict) -> dict:
    """
    Update the trees with everything the pairs taught, keeping the primed routes.

    A route is trained only where the group had enough pixels across the pairs.
    The route created by priming is kept as the fallback, and only ever captured
    once: training again later does not quietly turn a previously trained route
    into the "primed" one.
    """
    changes = []

    def upgrade(route: dict, group: str, description: str) -> dict:
        g = measured.get(group, {})
        if not g.get("usable"):
            where = f" ({', '.join(g.get('skipped', [])[:3])})" if g.get("skipped") else ""
            changes.append(f"{route['id']}: left as primed, too little {group} in the pairs{where}")
            return route

        # capture the primed route once, and never overwrite it later
        primed = route.get("primed_fallback")
        if primed is None and str(route.get("source", "")).startswith("stage A"):
            primed = dict(route, id=route["id"] + ".primed",
                          lesson="Kept from priming, as a fallback if the trained route is rejected.")

        new = dict(route)
        new["params"] = dict(route.get("params", {}),
                             library=g["libraries"][0], libraries=g["libraries"],
                             tone=g["tone"], grain=g["grain"], local_contrast=g["contrast"])
        pairs = len(g["pairs"])
        new["lesson"] = (f"{description} Measured across {pairs} pair{'s' if pairs > 1 else ''}: "
                         f"tone {g['tone']}, grain {g['grain']}, contrast {g['contrast']}, "
                         f"from {g['pixels']:,} pixels and {g['patches']} patches.")
        new["confidence"] = round(min(0.85, 0.6 + 0.05 * pairs), 2)
        new["source"] = f"stage B, {pairs} pair{'s' if pairs > 1 else ''}"
        if primed is not None:
            new["primed_fallback"] = primed
            new["fallbacks"] = [primed["id"]] + [f for f in route.get("fallbacks", [])
                                                 if f != primed["id"]]
        changes.append(f"{route['id']}: trained on {g['pixels']:,} {group} pixels "
                       f"from {pairs} pair{'s' if pairs > 1 else ''}, {g['patches']} patches")
        return new

    m = trees["material"]
    m["children"]["yes"] = upgrade(m["children"]["yes"], "junction",
                                   "Junction surface, usually more polished by turning traffic.")
    edge_branch = m["children"]["no"]["children"]
    edge_branch["under 1.5 m"] = upgrade(edge_branch["under 1.5 m"], "edge",
                                         "Near the kerb, where dust and darker wear collect.")
    edge_branch["1.5 m or more"] = upgrade(edge_branch["1.5 m or more"], "open",
                                           "Open road, away from kerbs and junctions.")

    n = trees["noise"]
    n["children"]["yes"] = upgrade(n["children"]["yes"], "junction",
                                   "Junction wear, heavier from turning traffic.")
    n["children"]["no"] = upgrade(n["children"]["no"], "open", "Open road wear.")

    return {"trees": trees, "changes": changes}


# ---------------------------------------------------------------------------
# Merging every trained pair into one library per part of the road.
#
# The routes point at one library at a time. Whenever pairs are added or
# deleted, the library is rebuilt from all of them and the old one is dropped,
# so there is never a pile of libraries to choose between.
#
# Each pair's own patches are kept as the raw material. Deleting a pair is then
# just a re-merge, a second or two, instead of re-analysing every other pair.
# ---------------------------------------------------------------------------

MERGED_CAP = 300          # patches in a merged library; more adds variety you cannot see


def _resize_patches(arr, size):
    if arr.shape[1] == size:
        return arr
    out = np.empty((len(arr), size, size, arr.shape[3]), arr.dtype)
    for i, p in enumerate(arr):
        out[i] = np.array(Image.fromarray(p.astype(np.uint8)).resize((size, size), Image.LANCZOS))
    return out


def merge_group(pairs, group, cap=MERGED_CAP):
    """
    One library for this part of the road, built from every pair.

    Patches come in at whatever size suited each road, so they are brought to
    one common size first: mixing sizes in a single library would leave the
    agent choosing between fragments that do not describe the same thing.

    Pairs are represented in proportion to how much road they supplied, so a
    pair with ten times the pixels contributes about ten times the patches.
    """
    parts = []
    for p in pairs:
        f = p["patch_files"].get(group)
        stats = p["groups"].get(group) or {}
        if not f or not Path(f).exists() or not stats.get("usable"):
            continue
        with np.load(f) as z:
            arr = z["patches"]
        if len(arr):
            parts.append({"patches": arr, "pixels": stats.get("pixels", len(arr)),
                          "size": int(arr.shape[1]), "pair": p["id"],
                          "stats": stats})
    if not parts:
        return None

    sizes = [p["size"] for p in parts]
    common = int(np.median(sizes))
    total_px = sum(p["pixels"] for p in parts) or 1

    chosen, used = [], []
    for part in parts:
        share = part["pixels"] / total_px
        want = max(1, int(round(cap * share)))
        arr = part["patches"]
        if want < len(arr):
            idx = np.random.default_rng(7).choice(len(arr), want, replace=False)
            arr = arr[idx]
        chosen.append(_resize_patches(arr, common))
        used.append({"pair": part["pair"], "patches": int(len(arr)),
                     "pixels": part["pixels"], "original_size": part["size"]})

    merged = np.concatenate(chosen)[:cap]
    weighted = lambda key: sum(p["stats"].get(key, 0) * p["pixels"] for p in parts) / total_px
    return {
        "patches": merged, "size_px": common, "count": int(len(merged)),
        "pairs": used, "pixels": int(total_px),
        "tone": round(weighted("mean"), 2), "grain": round(weighted("grain"), 3),
        "contrast": round(weighted("local_contrast"), 3),
        "sizes_seen": sorted(set(sizes)),
    }


def apply_merged(trees, merged, libs):
    """Point every route at the freshly merged library for its part of the road."""
    changes = []

    def upgrade(route, group, description):
        m = merged.get(group)
        if not m:
            changes.append(f"{route['id']}: no pairs supply {group}, left as primed")
            return route
        primed = route.get("primed_fallback")
        if primed is None and str(route.get("source", "")).startswith("stage A"):
            primed = dict(route, id=route["id"] + ".primed",
                          lesson="Kept from priming, as a fallback if the trained route is rejected.")
        new = dict(route)
        new["params"] = dict(route.get("params", {}), library=libs[group], libraries=[libs[group]],
                             tone=m["tone"], grain=m["grain"], local_contrast=m["contrast"])
        n = len(m["pairs"])
        new["lesson"] = (f"{description} Built from {n} pair{'s' if n > 1 else ''}: "
                         f"tone {m['tone']}, grain {m['grain']}, contrast {m['contrast']}, "
                         f"from {m['pixels']:,} pixels and {m['count']} patches of {m['size_px']} px.")
        new["confidence"] = round(min(0.85, 0.6 + 0.05 * n), 2)
        new["source"] = f"stage B, {n} pair{'s' if n > 1 else ''}"
        if primed is not None:
            new["primed_fallback"] = primed
            new["fallbacks"] = [primed["id"]] + [f for f in route.get("fallbacks", [])
                                                 if f != primed["id"]]
        changes.append(f"{route['id']}: {m['count']} patches from {n} pair{'s' if n > 1 else ''}, "
                       f"{m['pixels']:,} pixels")
        return new

    m = trees["material"]
    m["children"]["yes"] = upgrade(m["children"]["yes"], "junction",
                                   "Junction surface, usually more polished by turning traffic.")
    branch = m["children"]["no"]["children"]
    branch["under 1.5 m"] = upgrade(branch["under 1.5 m"], "edge",
                                    "Near the kerb, where dust and darker wear collect.")
    branch["1.5 m or more"] = upgrade(branch["1.5 m or more"], "open",
                                      "Open road, away from kerbs and junctions.")
    n = trees["noise"]
    n["children"]["yes"] = upgrade(n["children"]["yes"], "junction",
                                   "Junction wear, heavier from turning traffic.")
    n["children"]["no"] = upgrade(n["children"]["no"], "open", "Open road wear.")
    return {"trees": trees, "changes": changes}
