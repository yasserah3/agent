"""
Stage A: priming.

Three isolated reference images go in, three fingerprints come out, and the
three decision trees are created from them.

The three methods, as agreed:

  Method 1  neighbour pixels   - how a pixel relates to the eight around it.
                                 Gives tone, local contrast, grain size and
                                 whether the surface has a direction.
                                 (Markov random field style statistics.)

  Method 2  far pixel pairs    - sample the profile between distant points and
                                 look for repetition. This is what measures the
                                 dash and gap length of road paint instead of
                                 guessing it.

  Method 3  random rectangles  - sample patches at random positions to build a
                                 library. Generation later draws from this
                                 library, which is why generated wear varies
                                 instead of repeating one stamped shape.

Nothing here is a neural network. Every number is computed directly from the
pixels, so the same image always gives the same fingerprint.
"""

import json
import math
import uuid

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


# ----------------------------------------------------------------- method 1
def neighbour_stats(gray: np.ndarray) -> dict:
    """How each pixel relates to the eight around it."""
    g = gray.astype(np.float32)

    # mean absolute difference to the 8 neighbours = local contrast
    diffs = []
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        diffs.append(np.abs(g - shifted))
    local_contrast = float(np.mean(diffs))

    # grain: what is left after a blur, i.e. the fine detail
    fine = g - ndi.gaussian_filter(g, 2.0)
    grain = float(fine.std())

    # grain size: how far the fine detail stays correlated
    f = fine - fine.mean()
    denom = float((f * f).sum()) or 1.0
    grain_size = 1
    for lag in range(1, 12):
        c = float((f[:, :-lag] * f[:, lag:]).sum()) / denom
        if c < 0.2:
            grain_size = lag
            break
        grain_size = lag

    # direction: structure tensor tells us if the surface is streaky
    gy, gx = np.gradient(ndi.gaussian_filter(g, 1.0))
    jxx = ndi.gaussian_filter(gx * gx, 3.0).sum()
    jyy = ndi.gaussian_filter(gy * gy, 3.0).sum()
    jxy = ndi.gaussian_filter(gx * gy, 3.0).sum()
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    anisotropy = float((l1 - l2) / l1) if l1 > 1e-9 else 0.0
    direction = float((math.degrees(0.5 * math.atan2(2 * jxy, jxx - jyy))) % 180)

    return {
        "mean": round(float(g.mean()), 2),
        "std": round(float(g.std()), 2),
        "p5": round(float(np.percentile(g, 5)), 1),
        "p95": round(float(np.percentile(g, 95)), 1),
        "local_contrast": round(local_contrast, 3),
        "grain": round(grain, 3),
        "grain_size_px": int(grain_size),
        "anisotropy": round(anisotropy, 3),
        "direction_deg": round(direction, 1),
        "reading": ("flat colour, no grain" if grain < 1.0 else
                    "fine even grain" if grain_size <= 3 else "coarse grain"),
    }


# ----------------------------------------------------------------- method 2
def _autocorr_1d(profile: np.ndarray) -> np.ndarray:
    p = profile - profile.mean()
    n = len(p)
    f = np.fft.rfft(p, n * 2)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / (ac[0] if ac[0] else 1.0)


def periodicity(gray: np.ndarray, max_period: int = None) -> dict:
    """
    Sample along rows and columns and look for repetition.

    A strong peak at lag L means the surface repeats every L pixels, which is
    exactly what a dash pattern does. A flat curve means no repetition, which is
    what plain asphalt should give.
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    max_period = max_period or min(h, w) // 3
    out = {}
    for axis, name in ((1, "along_x"), (0, "along_y")):
        prof = g.mean(axis=0 if axis == 1 else 1)
        variation = float(prof.std())
        # a nearly constant profile has no pattern to find. Without this check a
        # flat colour reports a strong "repeat", because even tiny smooth
        # variation correlates with itself at short lags.
        if variation < 0.5:
            out[name] = {"period_px": None, "strength": 0.0, "variation": round(variation, 3)}
            continue
        ac = _autocorr_1d(prof)
        # a real repeat is a peak, not just a point on a curve that decays
        best_k, best_v = None, 0.0
        for k in range(3, min(max_period, len(ac) - 1)):
            if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]:
                trough = float(ac[3:k].min()) if k > 3 else float(ac[3])
                prominence = float(ac[k]) - trough
                if float(ac[k]) > 0.35 and prominence > 0.08 and float(ac[k]) > best_v:
                    best_k, best_v = k, float(ac[k])
        out[name] = {"period_px": best_k, "strength": round(best_v, 3),
                     "variation": round(variation, 3)}
    best = max((v for v in out.values() if isinstance(v, dict)),
               key=lambda v: v.get("strength", 0.0))
    out["repeats"] = bool(best.get("period_px"))
    out["period_px"] = best.get("period_px")
    out["reading"] = (f"repeats every {out['period_px']} px "
                      f"(strength {best['strength']})" if out["repeats"]
                      else "no repeating pattern, as a natural surface should be")
    return out


def dash_pattern(gray: np.ndarray) -> dict:
    """
    Measure a dash and gap pattern directly, for the line image.

    This is the number that used to be guessed. Now it is read off the image.
    """
    paint = gray > 128
    if paint.sum() < 20:
        return {"found": False, "reading": "no clear paint found in this image"}

    ys, xs = np.nonzero(paint)
    vertical = (ys.max() - ys.min()) >= (xs.max() - xs.min())
    line = paint.any(axis=1) if vertical else paint.any(axis=0)
    width = int(np.median(paint.sum(axis=1 if vertical else 0)[line]))

    runs, cur, n = [], line[0], 1
    for v in line[1:]:
        if v == cur:
            n += 1
        else:
            runs.append((bool(cur), n)); cur, n = v, 1
    runs.append((bool(cur), n))
    dashes = [n for v, n in runs if v]
    gaps = [n for v, n in runs[1:-1] if not v]
    if not dashes or not gaps:
        return {"found": False, "reading": "found paint but no repeating dashes"}

    dash = float(np.median(dashes)); gap = float(np.median(gaps))
    return {
        "found": True,
        "orientation": "vertical" if vertical else "horizontal",
        "dash_px": round(dash, 1), "gap_px": round(gap, 1),
        "cycle_px": round(dash + gap, 1),
        "width_px": width,
        "ratio": round(dash / gap, 2) if gap else None,
        "dashes_measured": len(dashes),
        "reading": f"dash {dash:.0f} px, gap {gap:.0f} px, repeating every {dash+gap:.0f} px",
    }


# ----------------------------------------------------------------- method 3
def _variants(p: np.ndarray) -> list:
    """The eight ways a square patch can be flipped and rotated."""
    out = []
    for k in range(4):
        r = np.rot90(p, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def patch_library(rgb: np.ndarray, mask: np.ndarray = None, count: int = 64,
                  size: int = 64, seed: int = 7, avoid_period: int = None,
                  min_coverage: float = 0.6, exclude: np.ndarray = None,
                  max_exclude: float = 0.0) -> dict:
    """
    Sample random rectangles to build the library generation will draw from.

    Patches are kept, not averaged. That is the point: a library of many real
    fragments produces variety, where a single average would produce repetition.

    Two extra defences against a source texture that tiles:

      - every patch is stored with all eight flips and rotations, so the same
        fragment never lands twice looking the same;
      - if the source repeats every N pixels, the patch size is nudged off a
        multiple of N, so patch edges never line up with the source grid.
    """
    h, w = rgb.shape[:2]
    size = int(min(size, h // 2, w // 2)) or 8
    if avoid_period and avoid_period > 1 and size % avoid_period == 0:
        size = max(8, size + max(1, avoid_period // 2))   # step off the grid
    rng = np.random.default_rng(seed)
    patches, means, stds, masks = [], [], [], []
    tries = rejected_paint = 0
    while len(patches) < count and tries < count * 40:
        tries += 1
        y = int(rng.integers(0, h - size)); x = int(rng.integers(0, w - size))
        if mask is not None and mask[y:y+size, x:x+size].mean() < min_coverage:
            continue
        # a patch that clips a painted line would scatter bits of paint across
        # the road, so reject it outright rather than accept it partly
        if exclude is not None and exclude[y:y+size, x:x+size].mean() > max_exclude:
            rejected_paint += 1
            continue
        p = rgb[y:y+size, x:x+size]
        patches.append(p)
        # which pixels of this patch are really road; the rest must not be pasted
        masks.append(mask[y:y+size, x:x+size].copy() if mask is not None
                     else np.ones((size, size), bool))
        g = p.mean(axis=2) if p.ndim == 3 else p
        means.append(float(g.mean())); stds.append(float(g.std()))
    if not patches:
        return {"count": 0, "reading": "no usable patches found"}
    arr = np.stack(patches)
    return {
        "count": len(patches), "size_px": size,
        "rejected_for_paint": int(rejected_paint),
        "orientations_per_patch": 8,
        "placements_available": len(patches) * 8,
        "size_offset_applied": bool(avoid_period),
        "patch_mean": round(float(np.mean(means)), 2),
        "patch_std": round(float(np.mean(stds)), 2),
        "variation_between_patches": round(float(np.std(means)), 2),
        "_patches": arr,
        "_masks": np.stack(masks),
        "reading": (f"{len(patches)} patches of {size} px, each usable in 8 orientations, "
                    f"so {len(patches) * 8} different placements; "
                    f"they differ from each other by about {np.std(means):.1f} levels"),
    }


def draw_patch(patches: np.ndarray, rng) -> np.ndarray:
    """Pick a patch and one of its eight orientations at random."""
    p = patches[int(rng.integers(0, len(patches)))]
    k = int(rng.integers(0, 4))
    out = np.rot90(p, k)
    if rng.integers(0, 2):
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def contact_sheet(patches: np.ndarray, out_path, cols: int = 8) -> str:
    n, s = patches.shape[0], patches.shape[1]
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * s, cols * s, 3), np.uint8)
    for i, p in enumerate(patches):
        if p.ndim == 2:
            p = np.dstack([p] * 3)
        r, c = divmod(i, cols)
        sheet[r*s:(r+1)*s, c*s:(c+1)*s] = p[:, :, :3]
    Image.fromarray(sheet).save(out_path)
    return str(out_path)


# ------------------------------------------------------------- stage A runs
def analyse_material(path) -> dict:
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))
    per = periodicity(gray)
    # if the source texture tiles, keep the patch size off that grid
    lib = patch_library(rgb, count=64, size=64, avoid_period=per.get("period_px"))
    return {"neighbours": neighbour_stats(gray), "periodicity": per,
            "patches": lib, "colour_mean": [round(float(v), 1) for v in rgb.reshape(-1, 3).mean(0)]}


def analyse_line(path) -> dict:
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))
    paint = gray > 128
    dash = dash_pattern(gray)
    # the paint pixels are a scattered set, not an image, so only tone applies
    vals = gray[paint].astype(np.float32) if paint.any() else np.array([0.0])
    paint_stats = {
        "mean": round(float(vals.mean()), 2),
        "std": round(float(vals.std()), 2),
        "p5": round(float(np.percentile(vals, 5)), 1),
        "p95": round(float(np.percentile(vals, 95)), 1),
        "reading": ("flat paint with no wear or texture" if vals.std() < 6
                    else "paint with some wear and variation"),
    }
    lib = patch_library(rgb, mask=paint, count=24, size=min(24, int(dash.get("width_px", 16) or 16)))
    return {"dash": dash, "paint": paint_stats, "patches": lib,
            "paint_colour_mean": [round(float(v), 1) for v in rgb[paint].mean(0)] if paint.any() else None,
            "paint_coverage": round(float(paint.mean()), 4)}


def analyse_noise(path) -> dict:
    gray = np.array(Image.open(path).convert("L"))
    strong = gray > 20
    lab, n = ndi.label(strong)
    sizes = ndi.sum(strong, lab, index=np.arange(1, n + 1)) if n else np.array([])
    thickness = ndi.distance_transform_edt(strong)
    rgb = np.dstack([gray] * 3)
    per = periodicity(gray)
    lib = patch_library(rgb, count=48, size=48, avoid_period=per.get("period_px"))
    return {
        "neighbours": neighbour_stats(gray),
        "periodicity": per,
        "patches": lib,
        "coverage": round(float(strong.mean()), 4),
        "features": int(n),
        "feature_size_median_px": round(float(np.median(sizes)), 1) if len(sizes) else 0.0,
        "thickness_px": round(float(thickness[strong].mean() * 2), 2) if strong.any() else 0.0,
        "reading": (f"{n} separate marks covering {strong.mean()*100:.1f}% of the surface"
                    if n else "no clear marks: the two material images may be too similar"),
    }


# ------------------------------------------------------------- tree building
def _route(rid, action, lesson, fallbacks=None, params=None):
    return {"id": rid, "action": action, "lesson": lesson,
            "params": params or {}, "fallbacks": fallbacks or [],
            "confidence": 0.5, "visits": 0, "successes": 0, "failures": 0,
            "source": "stage A priming, not yet proven on a real street"}


def build_trees(material: dict, line: dict, noise: dict, libs: dict) -> dict:
    """
    Creates the three trees.

    Every tree asks about junctions first, because a junction surface, its wear
    and its markings all behave differently from open road. Stage B fills these
    branches with what real streets actually look like; for now each branch has
    one route from priming, with a low confidence to show it is unproven.
    """
    m, nb = material["neighbours"], noise["neighbours"]
    dash = line["dash"]

    material_tree = {
        "question": "inside a junction?",
        "children": {
            "yes": _route("material.junction", f"patch library {libs['material']}",
                          "Junction surface. Primed from the plain material, because priming "
                          "has no junction example yet. Stage B replaces this with real "
                          "junction pixels, which are usually more polished.",
                          ["material.open"], {"library": libs["material"]}),
            "no": {
                "question": "distance to road edge",
                "children": {
                    "under 1.5 m": _route("material.edge", f"patch library {libs['material']}",
                                          "Near the kerb. Same library for now; Stage B learns "
                                          "how the edge differs from the middle.",
                                          ["material.open"], {"library": libs["material"]}),
                    "1.5 m or more": _route("material.open", f"patch library {libs['material']}",
                                            f"Open road. Measured: {m['reading']}, tone {m['mean']}, "
                                            f"local contrast {m['local_contrast']}.",
                                            [], {"library": libs["material"],
                                                 "tone": m["mean"], "grain": m["grain"]}),
                },
            },
        },
    }

    noise_tree = {
        "question": "inside a junction?",
        "children": {
            "yes": _route("noise.junction", f"patch library {libs['noise']}",
                          "Junction wear. Primed from the general wear; Stage B learns the "
                          "heavier turning wear that junctions really have.",
                          ["noise.open"], {"library": libs["noise"],
                                           "coverage": noise["coverage"]}),
            "no": _route("noise.open", f"patch library {libs['noise']}",
                         f"Open road wear. Measured: {noise['reading']}, "
                         f"average mark thickness {noise['thickness_px']} px.",
                         ["noise.junction"], {"library": libs["noise"],
                                              "coverage": noise["coverage"],
                                              "thickness_px": noise["thickness_px"]}),
        },
    }

    lines_tree = {
        "question": "on a centreline?",
        "children": {
            "yes": {
                "question": "near a junction?",
                "children": {
                    "yes": _route("lines.junction_clear", "no paint",
                                  "Dashes stop before a junction and each arm restarts its "
                                  "pattern, so no dash is ever cut in half.",
                                  [], {"setback_m": 2.0}),
                    "no": _route("lines.dash", f"patch library {libs['line']}",
                                 (f"Dash geometry measured from the line image: {dash['reading']}."
                                  if dash.get("found") else
                                  "No dash pattern could be measured from the line image."),
                                 [], {"library": libs["line"],
                                      "dash_px": dash.get("dash_px"),
                                      "gap_px": dash.get("gap_px"),
                                      "cycle_px": dash.get("cycle_px"),
                                      "width_px": dash.get("width_px")}),
                },
            },
            "no": _route("lines.none", "no paint",
                         "Road away from a centreline gets no marking.", []),
        },
    }
    return {"material": material_tree, "noise": noise_tree, "lines": lines_tree}
