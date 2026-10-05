"""
Junction handling, all fixed geometry computed from the mask alone.

Nothing here is learned. The agent later learns how junction surfaces *look*,
but finding them, classifying them and deciding where dashes stop are rules.

Pipeline, in order:
  1. clean     - fill specks, merge narrow dividers, drop tiny white specks
  2. widths    - distance transform gives every road its real width
  3. skeleton  - one pixel centreline, with noise spurs pruned
  4. junctions - centreline points where 3+ roads meet, merged into one junction
  5. arms      - each road leaving a junction, with its angle and width
  6. classify  - T, Y/skewed, crossroads, skewed crossroads, complex, interchange
  7. regions   - mouth lines, and which road pixels are inside a junction
"""

import math
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---- big maps: work on each street's or junction's own pixels, never the whole
# picture once per street or junction (thousands of each on a city-sized map
# made that billions of steps, minutes of waiting). Same results, pixel for pixel.

def label_coords(lab, n=None):
    """
    Every label's pixels, found in one pass: coords[i] is what
    np.argwhere(lab == i) gives (row by row), for i from 1 to n.
    """
    flat = lab.ravel()
    if n is None:
        n = int(flat.max()) if flat.size else 0
    idx = np.flatnonzero(flat)
    order = idx[np.argsort(flat[idx], kind="stable")]
    counts = np.bincount(flat[idx], minlength=n + 1)[1:n + 1]
    W = lab.shape[1]
    out = [np.zeros((0, 2), np.int64)]
    for g in np.split(order, np.cumsum(counts)[:-1]):
        out.append(np.column_stack((g // W, g % W)).astype(np.int64))
    return out


def disc(shape, cy, cx, r):
    """
    The pixels within r of (cy, cx): a window of the picture (two slices) and,
    inside it, the disc (the same test as over the whole picture).
    """
    H, W = shape
    y0, y1 = max(0, int(math.floor(cy - r))), min(H, int(math.ceil(cy + r)) + 1)
    x0, x1 = max(0, int(math.floor(cx - r))), min(W, int(math.ceil(cx + r)) + 1)
    yy, xx = np.ogrid[y0:max(y0, y1), x0:max(x0, x1)]
    return (slice(y0, max(y0, y1)), slice(x0, max(x0, x1))), (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

# A black strip inside a road narrower than this is a divider: the two sides are
# one road. Anything wider stays two separate roads, so a wide central reserve
# is not silently turned into one huge road.
DIVIDER_LIMIT_PX = 6.0

TYPE_COLOURS = {
    "T-junction": (92, 175, 127),
    "Y / skewed": (217, 164, 65),
    "crossroads": (61, 123, 224),
    "skewed crossroads": (143, 182, 242),
    "complex": (200, 110, 200),
    "interchange (flagged)": (216, 96, 76),
}


def clean_mask(gray: np.ndarray, divider_limit: float = DIVIDER_LIMIT_PX):
    road = gray > 128
    lab, n = ndi.label(~road)
    if n:
        dtb = ndi.distance_transform_edt(~road)
        idx = np.arange(1, n + 1)
        thick = ndi.maximum(dtb, lab, index=idx)
        area = ndi.sum(np.ones_like(road), lab, index=idx)
    else:
        thick = area = []
    specks = merged = left_split = 0
    fill = np.zeros(n + 1, bool)
    for i, (t, a) in enumerate(zip(thick, area), start=1):
        gap = 2 * float(t)
        if gap <= 3.0 and a < 40:
            fill[i] = True; specks += 1
        elif gap <= divider_limit and a >= 40:
            fill[i] = True; merged += 1
        elif a >= 40 and t <= 12:
            left_split += 1
    if fill.any():
        road |= fill[lab]

    wl, nw = ndi.label(road)
    if nw:
        wa = ndi.sum(road, wl, index=np.arange(1, nw + 1))
        drop = np.zeros(nw + 1, bool)
        drop[1:] = np.asarray(wa) < 60
        if drop.any():
            road &= ~drop[wl]
    return road, {"specks_filled": specks, "dividers_merged": merged,
                  "gaps_left_split": left_split, "divider_limit_px": divider_limit}


def centreline(road: np.ndarray, dt: np.ndarray):
    H, W = road.shape
    sk = skeletonize(road).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8); kernel[1, 1] = 0

    def degree():
        return ndi.convolve(sk, kernel, mode="constant") * sk

    def nbrs(y, x):
        for dy, dx in N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and sk[yy, xx]:
                yield yy, xx

    pruned = 0
    for _ in range(4):
        deg = degree()
        removed = False
        for (y, x) in list(zip(*np.nonzero(deg == 1))):
            if not sk[y, x]:
                continue
            # a branch that ends near the image edge is a road leaving the
            # picture, not noise: never prune it
            margin = max(6, int(2 * dt[y, x]))
            if y < margin or x < margin or y >= H - margin or x >= W - margin:
                continue
            path, prev, cur = [(y, x)], None, (y, x)
            while True:
                nxt = [p for p in nbrs(*cur) if p != prev and p not in path]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                if deg[cur] >= 3:
                    break
                path.append(cur)
                if len(path) > 80:
                    break
            if deg[cur] >= 3 and len(path) < max(4.0, 2.5 * dt[cur]):
                for p in path:
                    sk[p] = 0
                pruned += 1
                removed = True
        if not removed:
            break
    sk = skeletonize(sk.astype(bool)).astype(np.uint8)
    return sk, pruned


def detect(gray: np.ndarray, divider_limit: float = DIVIDER_LIMIT_PX):
    road, clean_stats = clean_mask(gray, divider_limit)
    H, W = road.shape
    dt = ndi.distance_transform_edt(road)
    sk, pruned = centreline(road, dt)

    kernel = np.ones((3, 3), np.uint8); kernel[1, 1] = 0
    deg = ndi.convolve(sk, kernel, mode="constant") * sk

    # candidates, then merge the cluster that one real junction produces
    lab, n = ndi.label(ndi.binary_dilation(deg >= 3, iterations=1) & sk.astype(bool))
    cands = []
    cand_px = label_coords(lab, n)
    for i in range(1, n + 1):
        ys, xs = cand_px[i][:, 0], cand_px[i][:, 1]
        if not len(ys):
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        y0, x0 = max(0, int(cy) - 6), max(0, int(cx) - 6)
        cands.append({"y": cy, "x": cx, "hw": float(dt[y0:int(cy) + 7, x0:int(cx) + 7].max())})

    parent = list(range(len(cands)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # only candidates close enough to possibly merge are compared (a k-d tree finds
    # them), not every pair: the same groups, without millions of comparisons
    if len(cands) > 1:
        from scipy.spatial import cKDTree
        reach = 1.6 * 2 * max(c["hw"] for c in cands)
        tree = cKDTree(np.array([[c["y"], c["x"]] for c in cands]))
        for i, j in sorted(tree.query_pairs(reach)):
            a, b = cands[i], cands[j]
            if math.hypot(a["y"] - b["y"], a["x"] - b["x"]) < 1.6 * (a["hw"] + b["hw"]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(cands)):
        groups.setdefault(find(i), []).append(cands[i])

    junctions = []
    for members in groups.values():
        cy = float(np.mean([m["y"] for m in members]))
        cx = float(np.mean([m["x"] for m in members]))
        spread = max(math.hypot(m["y"] - cy, m["x"] - cx) for m in members)
        hw = max(m["hw"] for m in members)
        junctions.append({"cy": cy, "cx": cx, "hw": hw, "n_cand": len(members),
                          "r": max(6.0, 2.0 * hw + spread)})

    seg = sk.astype(bool).copy()
    for j in junctions:
        win, inside = disc((H, W), j["cy"], j["cx"], j["r"])
        seg[win][inside] = False
    seg_lab, nseg = ndi.label(seg, structure=np.ones((3, 3)))
    seg_px = label_coords(seg_lab, nseg)

    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    for jid, j in enumerate(junctions, start=1):
        j["id"] = jid
        win, inside = disc((H, W), j["cy"], j["cx"], j["r"] + 2.5)
        ring = inside & seg[win]
        arms = []
        for s in set(np.unique(seg_lab[win][ring])) - {0}:
            ys, xs = seg_px[s][:, 0], seg_px[s][:, 1]
            d = np.hypot(ys - j["cy"], xs - j["cx"])
            sel = (d >= j["r"]) & (d <= j["r"] + max(12, 4 * j["hw"]))
            if sel.sum() < 3:
                sel = d <= j["r"] + 12
            my, mx = ys[sel].mean(), xs[sel].mean()
            arms.append({
                "angle": round(math.degrees(math.atan2(-(my - j["cy"]), mx - j["cx"])) % 360, 1),
                "width_px": round(2 * float(np.median(dt[ys[sel], xs[sel]])), 1),
                "off_image": False,
                "segment": int(s),
            })
        # an arm that leaves the image inside the junction disc still counts
        win, inside = disc((H, W), j["cy"], j["cx"], j["r"] + 8)
        near = border[win] & road[win] & inside
        if near.any():
            blab, nb = ndi.label(near, structure=np.ones((3, 3)))
            for b in range(1, nb + 1):
                ys, xs = np.nonzero(blab == b)
                ys, xs = ys + win[0].start, xs + win[1].start
                ang = round(math.degrees(
                    math.atan2(-(ys.mean() - j["cy"]), xs.mean() - j["cx"])) % 360, 1)
                if all(abs((ang - a["angle"] + 180) % 360 - 180) > 25 for a in arms):
                    arms.append({"angle": ang, "width_px": float(len(ys)),
                                 "off_image": True, "segment": None})
        arms.sort(key=lambda a: a["angle"])
        j["arms"] = arms

        k = len(arms)
        gaps = [round(((arms[(i + 1) % k]["angle"] - arms[i]["angle"]) % 360), 1)
                for i in range(k)] if k else []
        j["gaps"] = gaps
        if j["n_cand"] >= 5 or k >= 7:
            j["type"] = "interchange (flagged)"
        elif k <= 2:
            j["type"] = "not a junction"
        elif k == 3:
            straight = any(abs(((arms[a]["angle"] - arms[b]["angle"]) % 360) - 180) <= 20
                           for a in range(3) for b in range(a + 1, 3))
            j["type"] = "T-junction" if straight else "Y / skewed"
        elif k == 4:
            j["type"] = "crossroads" if all(abs(g - 90) <= 20 for g in gaps) else "skewed crossroads"
        else:
            j["type"] = "complex"

        # mouth line for each arm: where that arm enters the junction
        for a in arms:
            t = math.radians(a["angle"])
            ex, ey = j["cx"] + math.cos(t) * j["r"], j["cy"] - math.sin(t) * j["r"]
            px, py = -math.sin(t), -math.cos(t)
            half = max(3.0, a["width_px"] * 0.7)
            a["mouth"] = [[round(ex - px * half, 1), round(ey - py * half, 1)],
                          [round(ex + px * half, 1), round(ey + py * half, 1)]]

    junctions = [j for j in junctions if j["type"] != "not a junction"]

    off_edge = 0
    for sl in ndi.find_objects(seg_lab):
        if sl is not None and (sl[0].start < 3 or sl[1].start < 3 or sl[0].stop - 1 >= H - 3 or sl[1].stop - 1 >= W - 3):
            off_edge += 1

    widths = 2 * dt[sk.astype(bool)]
    summary = {
        "image": [int(W), int(H)],
        **clean_stats,
        "spurs_pruned": pruned,
        "junction_candidates": len(cands),
        "junctions": len(junctions),
        "by_type": dict(Counter(j["type"] for j in junctions)),
        "street_segments": int(nseg),
        "segments_off_image": off_edge,
        "road_px": int(road.sum()),
        "road_width_px_median": round(float(np.median(widths)), 1) if widths.size else 0.0,
        "road_width_px_max": round(float(widths.max()), 1) if widths.size else 0.0,
    }
    return {"summary": summary, "junctions": junctions,
            "road": road, "skeleton": sk, "dt": dt, "segments": seg_lab, "segment_pixels": seg_px}


def overlay(result, out_path):
    road, sk = result["road"], result["skeleton"]
    H, W = road.shape
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = (30, 34, 38)
    base[road] = (88, 94, 100)
    img = Image.fromarray(base)
    dr = ImageDraw.Draw(img, "RGBA")
    for j in result["junctions"]:
        c = TYPE_COLOURS.get(j["type"], (200, 200, 200))
        r = j["r"]
        dr.ellipse([j["cx"] - r, j["cy"] - r, j["cx"] + r, j["cy"] + r],
                   fill=c + (70,), outline=c + (255,), width=2)
    ys, xs = np.nonzero(sk)
    px = img.load()
    for y, x in zip(ys, xs):
        px[int(x), int(y)] = (235, 231, 220)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = None
    for j in result["junctions"]:
        for a in j["arms"]:
            (x1, y1), (x2, y2) = a["mouth"]
            dr.line([x1, y1, x2, y2], fill=(216, 96, 76, 255), width=2)
        dr.text((j["cx"] + j["r"] + 2, j["cy"] - j["r"] - 12), str(j["id"]),
                fill=(255, 255, 255, 255), font=font)
    img.save(out_path)
    return out_path
