"""
A road mesh made of quads, with straight edges.

Streets are rebuilt as a modeller would: from each street's smoothed centreline,
with its width measured and smoothed along it, the two kerbs are offsets of the
centreline. That removes the pixel drift a traced outline inherits from the
mask, and makes each street a clean strip of quads.

Junctions are quad patches: a ring of quads along the junction's edge and a fan
of quads to its centre, joined to the street strips at shared vertices so the
mesh has no cracks. Corners between two streets are smooth curves that follow
both kerbs.

Straightness, 0 to 1:
  0  follow the mask closely: little smoothing of the centreline or width
  1  one constant width per street, and a strongly smoothed centreline

Everything is computed on the mask at its own size, which is fast, and scaled
to the texture's size at the end.
"""

import math

import numpy as np
from scipy import ndimage as ndi

from app import junctions as J
from app.generation import _ordered_path


def _lerp(a, b, t):
    return a + (b - a) * t


class Mesh:
    def __init__(self):
        self.v = []          # (x, y) in texture pixels
        self.st = []         # (along, across) in metres for street vertices, else None
        self.role = []       # per vertex: "kerb", "open" or "junction", which sets its tone
        self.quads = []
        self.part = []       # per quad: "open", "edge" (kerb band) or "junction"
        self.owner = []      # per quad: street id, or minus the junction id
        self.tris = []
        self.dashes = []     # per dash: list of (x, y) quad corners, in texture pixels
        self.h = []          # per vertex: height in metres (road 0, sidewalk top raised)
        self.sw_quads, self.sw_part, self.sw_owner = [], [], []   # sidewalk top: "paving" or "kerbstone"
        self.kerb_quads, self.kerb_facing = [], []                # vertical step, and the way it faces
        self.sw_rows = {}    # road edge vertex -> the sidewalk row beside it, for corners
        self.sw_cfg = None
        self.dash_owner = []   # per dash: the street or connector it lies on
        self.kerb_owner = []   # per kerb face: its street, junction or connector
        self.streets = {}      # street id -> centreline, rows and sidewalk rows, for bridges
        self.jinfo = {}        # junction id -> centre and the street ends that meet there
        self.connectors = {}   # owner id -> a straight piece joining two mouths across a crossing
        self.patch_rings = []  # outer ring of each junction patch, for the road surface
        self.sw_tris, self.sw_tri_owner = [], []   # small islands paved completely
        self.kerb_surface = None
        self.sw_kind = {}
        self.fill_polys = None # the knot and fallback fills, as polygons in texture pixels
        self.sw_mode = None

    def add(self, x, y, st=None, role="open", h=0.0):
        self.v.append((float(x), float(y)))
        self.st.append(st)
        self.role.append(role)
        self.h.append(float(h))
        return len(self.v) - 1

    def quad(self, q, part, owner):
        self.quads.append(q)
        self.part.append(part)
        self.owner.append(owner)


def _smooth_line(arr, window):
    """Average a path with a moving window, keeping its two ends fixed."""
    n = len(arr)
    window = int(window)
    if window < 3 or n < 5:
        return arr.copy()
    window = min(window, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if window < 3:
        return arr.copy()
    if window % 2 == 0:
        window -= 1
    half = window // 2
    k = np.ones(window) / window
    out = arr.copy()
    for c in range(2):
        padded = np.concatenate([np.full(half, arr[0, c]), arr[:, c], np.full(half, arr[-1, c])])
        out[:, c] = np.convolve(padded, k, mode="valid")
    # keep the ends exactly where the street meets its junction
    out[0], out[-1] = arr[0], arr[-1]
    return out


def _resample(arr, spacing):
    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(1, int(math.ceil(L / max(spacing, 1e-6))))
    t = np.linspace(0.0, L, n + 1)
    pts = np.column_stack([np.interp(t, s, arr[:, 0]), np.interp(t, s, arr[:, 1])])
    return pts, t, L, s


def _kerbs(coverage, pts, side, guess):
    """
    Distance to the kerb on each side of every centre point, to a quarter pixel.

    The skeleton is one whole pixel wide, so on a road with an even width it
    sits half a pixel off centre. Measuring both kerbs on the smooth coverage
    map and taking the middle puts the centre, and the width, where they really
    are. Where a street opens into a junction the kerb runs away, and the
    measured width is ignored in favour of the estimate.
    """
    from scipy.ndimage import map_coordinates
    step = 0.25
    maxd = float(np.max(guess)) * 3 + 2
    offs = np.arange(0.0, maxd, step)
    out = np.zeros((len(pts), 2))
    for k, sign in enumerate((1.0, -1.0)):
        xs = pts[:, 0:1] + sign * side[:, 0:1] * offs[None, :]
        ys = pts[:, 1:2] + sign * side[:, 1:2] * offs[None, :]
        vals = map_coordinates(coverage, [ys.ravel(), xs.ravel()], order=1, mode="constant",
                               cval=0.0).reshape(xs.shape)
        below = vals < 0.5
        first = np.where(below.any(axis=1), below.argmax(axis=1), len(offs) - 1)
        # interpolate between the last road sample and the first background one
        i1 = np.maximum(first, 1); i0 = i1 - 1
        v0 = vals[np.arange(len(pts)), i0]; v1 = vals[np.arange(len(pts)), i1]
        frac = np.clip((v0 - 0.5) / np.maximum(v0 - v1, 1e-6), 0, 1)
        out[:, k] = (i0 + frac) * step
    return out[:, 0], out[:, 1]


def build(gray_mask, scale, metres_per_pixel, straightness=0.7, spacing_m=2.0, across=2,
          divider_limit=J.DIVIDER_LIMIT_PX, coverage=None, kerb_band=False, dashes=None,
          sidewalk=None, bridges=None, optimise=False, exact=None):
    """
    gray_mask: the road mask at its own size (already smooth-thresholded is fine)
    scale:     texture size / mask size (the output scale)
    exact:     roads given as exact shapes, laid as they are instead of traced:
               the inner streets (app/streets.py), each {"road": shapely shape,
               "lines": [(centreline points, half width)], "markings": bool} in
               mask pixels. The mask then holds only the other roads.
    Returns the mesh in texture pixels, and a short report.
    """
    det = J.detect(gray_mask, divider_limit)
    dt = det["dt"]
    seg_lab = det["segments"]      # replaced below when there are groups
    H, W = gray_mask.shape
    mpp = metres_per_pixel                         # at mask size
    s_ = float(np.clip(straightness, 0.0, 1.0))
    spacing_px = max(1.0, spacing_m / mpp)         # in mask pixels

    # how much to smooth, in mask pixels
    centre_window = max(3.0, _lerp(3.0, 40.0, s_) / mpp)
    width_sigma = max(0.5, _lerp(1.0, 30.0, s_) / mpp) / spacing_px   # in samples

    mesh = Mesh()
    clusters = find_clusters(det, mpp, coverage if coverage is not None else (gray_mask > 127).astype(float))
    mesh.clusters = clusters
    in_cluster = {jid for c in clusters for jid in c["members"]}
    sk_all = det["skeleton"].astype(bool)
    road_w_m = float(np.median(2 * dt[sk_all])) * mpp if sk_all.any() else 8.0
    piece_of = {}
    if clusters:
        seg_lab, piece_of = _pieces(det, clusters, gray_mask.shape)
    if sidewalk and coverage is not None:
        mesh.sw_cfg = dict({"share": 0.3, "min_m": 1.5, "max_m": 2.5, "height_m": 0.10,
                            "kerbstone_m": 0.15, "max_road_m": 20.0}, **sidewalk)
        mesh.sw_cov, mesh.sw_mpp, mesh.sw_scale = coverage, mpp, scale
        # interchanges get no sidewalks
        # sidewalks go on every road. Leaving interchanges and very wide roads
        # without them is an option, off unless asked for
        skip = bool(mesh.sw_cfg.get("skip_interchanges", False))
        mesh.sw_exclude = [(j["cx"], j["cy"], j["r"] * 1.3) for j in det["junctions"]
                           if skip and j["type"] == "interchange (flagged)"]
        mesh.sw_exclude_polys = [c["region"] for c in clusters if skip and c["kind"] == "interchange"]
        # sidewalks follow the kerb line of the whole road surface; bridges need
        # sidewalks at two heights where roads cross, so they keep the per-street way
        mesh.sw_mode = "per_street" if bridges else "kerb_line"
    ends = []            # street ends, to be joined by junction patches
    rows_full = rows_kept = 0
    mesh.optimise = optimise
    streets = 0

    for sid in range(1, int(seg_lab.max()) + 1):
        coords = np.argwhere(seg_lab == sid)
        if len(coords) < 2:
            continue
        path = _ordered_path(coords)
        if len(path) < 2:
            continue
        if len(coords) < 3:
            continue                       # a few stray pixels: the knot fill covers them
        raw = np.array([(p[1], p[0]) for p in path], float)         # (x, y)
        hw_raw = np.array([max(0.5, dt[int(p[0]), int(p[1])] - 0.5) for p in path])

        # sample the street evenly, then find its true centre and width
        pts, t, L, s_raw_line = _resample(raw, spacing_px)
        s_raw = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(raw[:, 0]), np.diff(raw[:, 1])))])
        hw_guess = np.interp(t / max(L, 1e-6) * s_raw[-1], s_raw, hw_raw)
        if coverage is not None and len(pts) >= 2:
            pre = _smooth_line(pts, 5)
            tg = np.gradient(pre, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            sd = np.column_stack([-tg[:, 1], tg[:, 0]])
            d_plus, d_minus = _kerbs(coverage, pts, sd, hw_guess)
            good = (d_plus < 2.2 * hw_guess + 1) & (d_minus < 2.2 * hw_guess + 1)
            pts = pts + sd * np.where(good, (d_plus - d_minus) / 2, 0.0)[:, None]
            hw = np.where(good, (d_plus + d_minus) / 2, hw_guess)
        else:
            hw = hw_guess

        # straighten: smooth the centreline, in samples along the street
        # pieces inside a group are ramps and slip roads: they curve tightly, so
        # they are smoothed over a few metres only, or they would cut their corners
        cw = min(centre_window, 8.0 / mpp) if sid in piece_of else centre_window
        win = int(round(cw / spacing_px))
        pts = _smooth_line(pts, win if win % 2 == 1 else win + 1) if win >= 3 else pts

        # the width flares where a street opens into a junction; that belongs to
        # the junction's corners, not to the street, so it is measured away from the ends
        n = len(hw)
        inner = hw[int(n * 0.15): max(int(n * 0.85), int(n * 0.15) + 1)]
        hw_const = float(np.median(inner if len(inner) else hw))
        hw_med = ndi.median_filter(hw, size=min(5, n)) if n >= 3 else hw
        hw_smooth = ndi.gaussian_filter1d(hw_med, width_sigma, mode="nearest") if n >= 3 else hw_med
        hw_smooth = np.minimum(hw_smooth, 1.25 * hw_const)
        hw_final = _lerp(hw_smooth, np.full_like(hw_smooth, hw_const), s_)

        # direction and the sideways normal at each row
        tang = np.gradient(pts, axis=0)
        norm = np.linalg.norm(tang, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        tang = tang / norm
        side = np.column_stack([-tang[:, 1], tang[:, 0]])

        # offsets across the road, from one kerb to the other. With a kerb band,
        # the outer quads on each side are the kerb and the inner ones open road,
        # using the same band rule as the texture: 1.5 m, or a quarter of the width
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])

        # dashes first: in the optimised mesh their layout decides where rows must stay
        layout = None
        grp = piece_of.get(sid)
        # inside a group, narrow pieces are single-lane slip roads and ramps,
        # which have no centre line; and a piece with room for only one dash
        # would carry a lonely dash, so it needs room for at least two
        narrow_ramp = (grp is not None and 2 * float(np.median(hw_final)) * mpp < 0.8 * road_w_m)
        # an inner street drawn without markings (app/streets.py) gets none
        nomark = dashes.get("nomark") if dashes else None
        unmarked = False
        if nomark is not None and len(pts):
            iy = np.clip(pts[:, 1].astype(int), 0, nomark.shape[0] - 1)
            ix = np.clip(pts[:, 0].astype(int), 0, nomark.shape[1] - 1)
            unmarked = nomark[iy, ix].mean() > 0.5
        if dashes and not narrow_ramp and not unmarked:
            n0 = len(mesh.dashes)
            cfg_d = dict(dashes, min_cycles=2) if grp is not None else dashes
            layout = _dash_strips(mesh, pts, side, arc, hw_final, mpp, scale, cfg_d)
            mesh.dash_owner.extend([sid] * (len(mesh.dashes) - n0))

        # which rows to build: all of them, or in the optimised mesh only where
        # the road needs them
        if optimise:
            near_bridge = any(np.min(np.hypot(pts[:, 0] - b["cx"], pts[:, 1] - b["cy"]))
                              < b["length"] / 2 + 150.0 / mpp for b in (bridges or []))
            keep = _row_keep(pts, tang, arc * mpp, hw_final, layout,
                             max_gap_m=4.0 if near_bridge else 20.0)
        else:
            keep = np.arange(len(pts))
        P_r, S_r, A_r, HW_r = pts[keep], side[keep], arc[keep], hw_final[keep]

        # offsets across the road, from one kerb to the other. With a kerb band,
        # the outer quads on each side are the kerb and the inner ones open road,
        # using the same band rule as the texture: 1.5 m, or a quarter of the width.
        # The optimised mesh leaves out the centre line: 3 quads across, not 4
        rows = []
        for i in range(len(P_r)):
            hwi = HW_r[i]
            if kerb_band:
                kb = min(1.5 / mpp, 0.5 * hwi)
                offs = ([hwi, hwi - kb, -(hwi - kb), -hwi] if optimise
                        else [hwi, hwi - kb, 0.0, -(hwi - kb), -hwi])
            else:
                offs = [_lerp(hwi, -hwi, a / across) for a in range(across + 1)]
            row = []
            for k, off in enumerate(offs):
                x, y = P_r[i] + S_r[i] * off
                role = "kerb" if kerb_band and k in (0, len(offs) - 1) else "open"
                row.append(mesh.add(x * scale, y * scale, (A_r[i] * mpp, -off * mpp), role))
            rows.append(row)
        ncols = len(rows[0]) - 1
        for i in range(len(rows) - 1):
            for a in range(ncols):
                part = "edge" if kerb_band and a in (0, ncols - 1) else "open"
                mesh.quad((rows[i][a], rows[i][a + 1], rows[i + 1][a + 1], rows[i + 1][a]), part, sid)
        streets += 1
        rows_full += len(pts); rows_kept += len(P_r)

        mesh.streets[sid] = {"pts": P_r.copy(), "arc": A_r.copy(), "rows": rows,
                             "hw": HW_r.copy(), "side": S_r.copy(), "sw": {},
                             "length_m": float(arc[-1] * mpp), "dash": layout,
                             "half_width_m": float(np.max(hw_final) * mpp)}
        if mesh.sw_cfg:
            if mesh.sw_mode == "per_street":
                _street_sidewalks(mesh, rows, P_r, S_r, A_r, HW_r, sid)

        for end_row, centre, inward in ((rows[0], pts[0], tang[0]), (rows[-1], pts[-1], -tang[-1])):
            ends.append({"row": end_row, "centre": centre * scale,
                         "outward": -inward, "street": sid,
                         "which": 0 if end_row is rows[0] else 1})

    # ---- junction patches
    patched, skipped, flagged, crossings = 0, 0, [], 0
    for j in det["junctions"]:
        cx, cy, r = j["cx"] * scale, j["cy"] * scale, j["r"] * scale
        if int(j["id"]) in in_cluster:
            continue                       # part of a tight group: covered by its fill
        if j["type"] == "interchange (flagged)":
            flagged.append(j)
            continue
        mouths = [e for e in ends
                  if math.hypot(e["centre"][0] - cx, e["centre"][1] - cy) <= r + 4 * scale]
        if len(mouths) < 2:
            skipped += 1
            _fallback_fill(mesh, j)
            continue
        mesh.jinfo[int(j["id"])] = {"C": (cx, cy), "r": r,
                                    "mouths": [(m["street"], m["which"]) for m in mouths],
                                    "kind": "patch"}
        crossing = None
        for bi, b in enumerate(bridges or []):
            if inside_rect(np.array([[j["cx"], j["cy"]]]), b)[0]:
                crossing = (bi, b)
                break
        if crossing and _crossing(mesh, mouths, (cx, cy), spacing_px * scale, crossing, int(j["id"]),
                                  dashes, mpp, scale):
            mesh.jinfo[int(j["id"])]["kind"] = "crossing"
            crossings += 1
            continue
        if _patch(mesh, mouths, (cx, cy), spacing_px * scale, owner=-int(j["id"])):
            patched += 1
        else:
            skipped += 1
            _fallback_fill(mesh, j)

    holes = _cover_bare(mesh, coverage if coverage is not None else (gray_mask > 127).astype(float), scale)
    cov_mask = coverage if coverage is not None else (gray_mask > 127).astype(float)
    mesh.fill_polys = [_scale_poly(q, scale) for q in cluster_fill_polys(mesh.clusters, cov_mask, 1)]
    # roads given exactly (the inner streets): laid as they are, so their straight
    # edges stay straight in the kerb line and sidewalks, with their markings
    # along their centrelines
    exact_n = 0
    if exact:
        # their edges stay exactly where they are when the kerb line is smoothed
        from shapely.ops import unary_union
        mesh.exact_edges = unary_union([ex["road"].boundary for ex in exact if not ex["road"].is_empty])
    for ex in exact or []:
        road = ex["road"]
        for g in (road.geoms if hasattr(road, "geoms") else [road]):
            if g.geom_type == "Polygon" and not g.is_empty:
                mesh.fill_polys.append(_scale_poly(g, scale))
                exact_n += 1
        if not (dashes and ex.get("markings")):
            continue
        for pts, hw in ex["lines"]:
            pts = np.asarray(pts, float)
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
            tg = np.gradient(pts, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            n0 = len(mesh.dashes)
            _dash_strips(mesh, pts, np.column_stack([-tg[:, 1], tg[:, 0]]), arc, np.full(len(pts), float(hw)),
                         mpp, scale, dashes)
            mesh.dash_owner.extend([None] * (len(mesh.dashes) - n0))
    if mesh.sw_cfg and mesh.sw_mode == "kerb_line":
        _kerb_line_sidewalks(mesh, gray_mask.shape, scale, s_)

    return mesh, det, {
        "streets": streets, "junctions_patched": patched, "junctions_skipped": skipped,
        "bare_filled": holes,
        "rows_full": rows_full, "rows_kept": rows_kept,
        "interchanges_flagged": len(flagged), "flagged": flagged, "crossings": crossings,
        "clusters": [{"centre": [round(c["centre"][0]), round(c["centre"][1])], "junctions": len(c["members"]),
                      "islands": c["islands"]} for c in clusters],
        "straightness": round(s_, 2), "spacing_m": spacing_m, "exact_shapes": exact_n,
    }


def _ang(v):
    """Angle with y pointing up, so increasing angle runs anticlockwise on screen."""
    return math.atan2(-v[1], v[0])


def _patch(mesh, mouths, centre, spacing, owner=0):
    C = np.array(centre, float)
    V = mesh.v

    for m in mouths:
        d = m["centre"] - C
        m["dir"] = d / (np.linalg.norm(d) or 1.0)
        base = _ang(m["dir"])
        # this mouth's vertices, from its clockwise side to its anticlockwise side
        m["loop"] = sorted(m["row"], key=lambda idx: (
            (_ang(np.array(V[idx]) - C) - base + math.pi) % (2 * math.pi) - math.pi))
    mouths.sort(key=lambda m: _ang(m["dir"]))

    outer = []
    k = len(mouths)
    for i in range(k):
        a, b = mouths[i], mouths[(i + 1) % k]
        outer.extend(a["loop"])
        P0, P2 = np.array(V[a["loop"][-1]]), np.array(V[b["loop"][0]])
        # corner: leave along this street's kerb inward, arrive along the next street's kerb
        d0, d2 = -a["dir"], -b["dir"]
        M = np.array([[d0[0], -d2[0]], [d0[1], -d2[1]]])
        ctrl = (P0 + P2) / 2
        det_ = np.linalg.det(M)
        if abs(det_) > 1e-6:
            t, u = np.linalg.solve(M, P2 - P0)
            reach = np.linalg.norm(P2 - P0) * 3 + 1
            if 0 < t < reach and 0 < u < reach:
                ctrl = P0 + d0 * t
        # a few points per corner are enough to read as a curve; many more only
        # crowd the junction centre with edges
        length = np.linalg.norm(ctrl - P0) + np.linalg.norm(P2 - ctrl)
        steps = int(np.clip(round(length / max(spacing, 1e-6)), 4, 6))
        corner_ids = [a["loop"][-1]]
        for s in range(1, steps):
            q = s / steps
            p = (1 - q) ** 2 * P0 + 2 * (1 - q) * q * ctrl + q ** 2 * P2
            outer.append(mesh.add(p[0], p[1], None, "kerb"))
            corner_ids.append(outer[-1])
        corner_ids.append(b["loop"][0])
        if mesh.sw_cfg and mesh.sw_mode == "per_street":
            _corner_sidewalk(mesh, corner_ids, C, owner)

    mesh.patch_rings.append(list(outer))
    m = len(outer)
    if m < 3:
        return False
    # two rings of quads step in from the edge before the centre fan, so the
    # patch is mostly regular quads and fewer edges meet at the centre
    c_idx = mesh.add(C[0], C[1], None, "junction")
    rings = [outer]
    for f in (0.62, 0.3):
        ring = []
        for idx in outer:
            p = C + (np.array(V[idx]) - C) * f
            ring.append(mesh.add(p[0], p[1], None, "junction"))
        rings.append(ring)
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(m):
            j = (i + 1) % m
            mesh.quad((a[i], a[j], b[j], b[i]), "junction", owner)
    inner = rings[-1]
    for i in range(0, m - 1, 2):
        mesh.quad((c_idx, inner[i], inner[i + 1], inner[(i + 2) % m]), "junction", owner)
    if m % 2 == 1:
        mesh.tris.append((c_idx, inner[m - 1], inner[0]))
    return True


def _dash_strips(mesh, pts, side, arc, hw, mpp, scale, cfg):
    """
    Dashes as their own thin quads, following the street's smoothed centreline.

    Same rules as the texture: a setback from each end so no dash is cut at a
    junction, a whole number of cycles with the gaps stretched to fit, and a
    width that is either the learned share of the street's width or fixed.
    """
    L = arc[-1] * mpp
    setback, cycle = cfg.get("setback_m", 2.0), cfg.get("cycle_m", 9.0)
    share = cfg.get("dash_share", 0.6)
    usable = L - 2 * setback
    if usable < cycle * 0.6:
        return None
    n = max(1, int(round(usable / cycle)))
    if n < cfg.get("min_cycles", 1):
        return None
    step = usable / n
    ratio, fixed = cfg.get("width_ratio"), cfg.get("width_m", 0.15)
    width_m = (float(np.median(hw)) * 2 * mpp * ratio) if ratio else fixed
    layout = {"setback_m": setback, "step_m": step, "n": n, "share": share, "width_m": width_m}
    for c in range(n):
        a = (setback + c * step) / mpp
        b = a + step * share / mpp
        inside = (arc > a) & (arc < b)
        ts = np.concatenate([[a], arc[inside], [b]])
        cx = np.interp(ts, arc, pts[:, 0]); cy = np.interp(ts, arc, pts[:, 1])
        sx = np.interp(ts, arc, side[:, 0]); sy = np.interp(ts, arc, side[:, 1])
        nrm = np.hypot(sx, sy); nrm[nrm == 0] = 1; sx, sy = sx / nrm, sy / nrm
        halfw = np.interp(ts, arc, hw) * ratio if ratio else np.full_like(ts, fixed / 2 / mpp)
        strip = []
        for k in range(len(ts)):
            strip.append(((cx[k] + sx[k] * halfw[k]) * scale, (cy[k] + sy[k] * halfw[k]) * scale,
                          (cx[k] - sx[k] * halfw[k]) * scale, (cy[k] - sy[k] * halfw[k]) * scale))
        mesh.dashes.append(strip)
    return layout



# ---------------------------------------------------------------------------
# Sidewalks
#
# A strip beside each street, raised by the sidewalk height, with a vertical
# kerb face down to the road and a narrow kerb stone along its top edge.
#
# Width: a share of the road's width, kept between a minimum and a maximum.
# Before each row the free space beyond the kerb is measured on the distance
# field, out to the next road, and a sidewalk never takes more than half of it,
# so two sidewalks facing each other across a thin block meet in the middle
# instead of overlapping.
# ---------------------------------------------------------------------------

def _free_space(mesh, origins, normals, maxd):
    """Distance from each kerb point, outward, to the next road (mask pixels)."""
    from scipy.ndimage import map_coordinates
    step = 0.25
    offs = np.arange(0.75, maxd, step)
    xs = origins[:, 0:1] + normals[:, 0:1] * offs[None, :]
    ys = origins[:, 1:2] + normals[:, 1:2] * offs[None, :]
    vals = map_coordinates(mesh.sw_cov, [ys.ravel(), xs.ravel()], order=1, mode="constant",
                           cval=0.0).reshape(xs.shape)
    # near a junction the straightened kerb sits inside the painted road's rounded
    # flare, so the ray starts on road. That road is this street's own, not the
    # next one: first leave it, then look for the next road
    road = vals >= 0.5
    left = ~road
    exit_i = np.where(left.any(axis=1), left.argmax(axis=1), road.shape[1])
    after = road & (np.arange(road.shape[1])[None, :] > exit_i[:, None])
    free = np.where(after.any(axis=1), offs[after.argmax(axis=1)], np.inf)
    return free


def _excluded(mesh, pts_mask):
    out = np.zeros(len(pts_mask), bool)
    for cx, cy, r in mesh.sw_exclude:
        out |= np.hypot(pts_mask[:, 0] - cx, pts_mask[:, 1] - cy) <= r
    if getattr(mesh, "sw_exclude_polys", None):
        out |= _inside_clusters([{"region": r} for r in mesh.sw_exclude_polys], pts_mask)
    return out


def _sw_row(mesh, edge, n, w_px, along_m, H, ks_px):
    """One row across a sidewalk: kerb foot, kerb top, kerb stone edge, outer edge."""
    sc = mesh.sw_scale
    mpp = mesh.sw_mpp
    ks = min(ks_px, w_px * 0.5)
    st = lambda d: (along_m, d * mpp) if along_m is not None else None
    bottom = mesh.add(edge[0] * sc, edge[1] * sc, None, "kerbface", 0.0)
    top = mesh.add(edge[0] * sc, edge[1] * sc, st(0.0), "kerbstone", H)
    stone = mesh.add((edge[0] + n[0] * ks) * sc, (edge[1] + n[1] * ks) * sc, st(ks), "kerbstone", H)
    outer = mesh.add((edge[0] + n[0] * w_px) * sc, (edge[1] + n[1] * w_px) * sc, st(w_px), "paving", H)
    return (bottom, top, stone, outer)


def _sw_connect(mesh, r0, r1, owner, facing):
    b0, t0, s0, o0 = r0
    b1, t1, s1, o1 = r1
    mesh.sw_quads.append((t0, t1, s1, s0)); mesh.sw_part.append("kerbstone"); mesh.sw_owner.append(owner)
    mesh.sw_quads.append((s0, s1, o1, o0)); mesh.sw_part.append("paving"); mesh.sw_owner.append(owner)
    mesh.kerb_quads.append((b0, b1, t1, t0)); mesh.kerb_facing.append(facing)
    mesh.kerb_owner.append(owner)


def _street_sidewalks(mesh, rows, pts, side, arc, hw, sid):
    cfg, mpp = mesh.sw_cfg, mesh.sw_mpp
    road_m = 2 * float(np.median(hw)) * mpp
    if cfg.get("skip_interchanges", False) and road_m > cfg["max_road_m"]:
        return                                          # wide roads (highways) have no sidewalks
    H = cfg["height_m"]
    ks_px = cfg["kerbstone_m"] / mpp
    target = np.clip(cfg["share"] * 2 * hw * mpp, cfg["min_m"], cfg["max_m"]) / mpp
    maxd = 2 * cfg["max_m"] / mpp + 2
    for sign, col in ((1.0, 0), (-1.0, -1)):
        n = side * sign
        edge = pts + n * hw[:, None]
        free = _free_space(mesh, edge, n, maxd)
        w = np.minimum(target, free / 2)
        if len(w) >= 3:
            w = np.minimum(ndi.gaussian_filter1d(w, 1.0, mode="nearest"), free / 2)
        ok = (w > 0.3 / mpp) & ~_excluded(mesh, edge)
        # short scraps of sidewalk, between knots or on a short piece, look like
        # debris rather than pavement: runs under the minimum length are dropped
        ok = _drop_short_runs(ok, arc * mpp, mesh.sw_cfg.get("min_run_m", 8.0))
        # distance along is measured on this kerb, not the centreline: on the
        # inside of a bend the kerb travels much less than the centre does, and
        # the paving would be squeezed if it followed the centre
        edge_arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(edge[:, 0]), np.diff(edge[:, 1])))])
        made = {}
        for i in range(len(pts)):
            if ok[i]:
                made[i] = _sw_row(mesh, edge[i], n[i], w[i], edge_arc[i] * mpp, H, ks_px)
        for i in range(len(pts) - 1):
            if i in made and i + 1 in made:
                _sw_connect(mesh, made[i], made[i + 1], sid, (-n[i, 0], -n[i, 1]))
        if sid in mesh.streets:
            mesh.streets[sid]["sw"][sign] = made
        # remember the end rows, so corners can continue from them
        for i in (0, len(pts) - 1):
            if i in made:
                mesh.sw_rows[rows[i][col]] = {"row": made[i], "w": float(w[i]),
                                              "n": (float(n[i, 0]), float(n[i, 1]))}


def _corner_sidewalk(mesh, corner_ids, C, owner):
    """Wrap a sidewalk around a block corner, joining two streets' sidewalks."""
    a, b = mesh.sw_rows.get(corner_ids[0]), mesh.sw_rows.get(corner_ids[-1])
    if not a or not b or len(corner_ids) < 3:
        return
    from shapely.geometry import LineString
    cfg, sc, mpp = mesh.sw_cfg, mesh.sw_scale, mesh.sw_mpp
    H, ks_px = cfg["height_m"], cfg["kerbstone_m"] / mpp
    P = np.array([mesh.v[i] for i in corner_ids], float) / sc          # mask pixels
    Cm = np.array(C, float) / sc
    tg = np.gradient(P, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
    n = np.column_stack([-tg[:, 1], tg[:, 0]])
    flip = ((P - Cm) * n).sum(axis=1) < 0                                # point away from the road
    n[flip] *= -1
    n[0], n[-1] = a["n"], b["n"]
    w = np.linspace(a["w"], b["w"], len(P))
    maxd = 2 * cfg["max_m"] / mpp + 2
    w = np.minimum(w, _free_space(mesh, P, n, maxd) / 2)
    # the corner is concave on the block side; too wide a sidewalk folds over
    # itself there, so narrow it until its outer edge is a clean line
    for _ in range(6):
        outer = P + n * w[:, None]
        if LineString(outer).is_simple:
            break
        w[1:-1] *= 0.7
    rows = [a["row"]]
    for i in range(1, len(P) - 1):
        if w[i] <= 0.3 / mpp or _excluded(mesh, P[i:i + 1])[0]:
            return
        rows.append(_sw_row(mesh, P[i], n[i], w[i], None, H, ks_px))
    rows.append(b["row"])
    for i in range(len(rows) - 1):
        _sw_connect(mesh, rows[i], rows[i + 1], owner, (-n[i, 0], -n[i, 1]))



# ---------------------------------------------------------------------------
# Bridges: over-and-under crossings
#
# In a 2D mask a bridge crossing looks exactly like a crossroads. Inside a
# bridge rectangle the crossing is not patched as one junction: the two mouths
# running along the rectangle are joined by a straight piece (the bridge), and
# the two mouths across it by another (the road passing underneath). Heights
# are applied later; here the two pieces only need to be separate.
# ---------------------------------------------------------------------------

BRIDGE_OWNER = 1_000_000       # owner ids for the piece carried over a crossing
UNDER_OWNER = 2_000_000        # and for the piece passing underneath


def inside_rect(pts, b):
    """Whether points (mask pixels) fall inside a rotated rectangle."""
    a = math.radians(b["angle"])
    u = np.array([math.cos(a), math.sin(a)]); v = np.array([-u[1], u[0]])
    d = np.asarray(pts, float) - np.array([b["cx"], b["cy"]])
    return (np.abs(d @ u) <= b["length"] / 2) & (np.abs(d @ v) <= b["width"] / 2)


def _crossing(mesh, mouths, C, spacing, crossing, jid, dashes, mpp, scale):
    bi, b = crossing
    a = math.radians(b["angle"])
    u = np.array([math.cos(a), math.sin(a)])
    Cv = np.array(C, float)
    for m in mouths:
        d = m["centre"] - Cv
        m["dir"] = d / (np.linalg.norm(d) or 1.0)
    # the bridge: the two mouths most in line with the rectangle, on opposite sides
    best = None
    for i, mi in enumerate(mouths):
        for k, mk in enumerate(mouths):
            if k <= i:
                continue
            ai, ak = float(mi["dir"] @ u), float(mk["dir"] @ u)
            if ai * ak < 0 and min(abs(ai), abs(ak)) > 0.7 and float(mi["dir"] @ mk["dir"]) < -0.7:
                score = abs(ai) + abs(ak)
                if best is None or score > best[0]:
                    best = (score, i, k)
    if best is None:
        return False
    _, i, k = best
    _connector(mesh, mouths[i], mouths[k], spacing, BRIDGE_OWNER + bi * 1000 + (jid % 1000),
               True, bi, dashes, mpp, scale)
    rest = [m for n, m in enumerate(mouths) if n not in (i, k)]
    if len(rest) == 2 and float(rest[0]["dir"] @ rest[1]["dir"]) < -0.7:
        _connector(mesh, rest[0], rest[1], spacing, UNDER_OWNER + bi * 1000 + (jid % 1000),
                   False, bi, dashes, mpp, scale)
    elif rest:
        mesh.connectors.setdefault("_warnings", []).append(
            f"junction {jid}: {len(rest)} road(s) meet the bridge without passing straight under it")
    return True


def _connector(mesh, A, B, spacing, owner, is_bridge, bi, dashes, mpp, scale):
    """Join two facing mouths with straight rows of quads, and their sidewalks."""
    V = mesh.v
    ca, cb = np.array(A["centre"], float), np.array(B["centre"], float)
    d = cb - ca
    L = float(np.linalg.norm(d)) or 1.0
    d /= L
    lat = np.array([-d[1], d[0]])
    ra = sorted(A["row"], key=lambda v: float((np.array(V[v]) - ca) @ lat))
    rb = sorted(B["row"], key=lambda v: float((np.array(V[v]) - cb) @ lat))
    if len(ra) != len(rb):
        return
    n = max(1, int(round(L / max(spacing, 1e-6))))
    rows = [ra]
    for k in range(1, n):
        t = k / n
        row = []
        for c in range(len(ra)):
            p = np.array(V[ra[c]]) * (1 - t) + np.array(V[rb[c]]) * t
            row.append(mesh.add(p[0], p[1], None, "kerb" if c in (0, len(ra) - 1) else "open"))
        rows.append(row)
    rows.append(rb)
    last = len(ra) - 2
    for i in range(len(rows) - 1):
        for c in range(len(ra) - 1):
            part = "edge" if len(ra) >= 4 and c in (0, last) else "open"
            mesh.quad((rows[i][c], rows[i][c + 1], rows[i + 1][c + 1], rows[i + 1][c]), part, owner)

    info = {"bridge": is_bridge, "index": bi, "rows": rows, "sw": {},
            "ends": [(A["street"], A["which"]), (B["street"], B["which"])],
            "line": np.array([ca, cb])}
    # dashes along the piece
    if dashes and is_bridge:
        steps = max(2, int(round(L / scale / 1.0)))
        pts = np.array([ca + (cb - ca) * (t / steps) for t in range(steps + 1)]) / scale
        side = np.tile(lat, (len(pts), 1))
        arc = np.linspace(0.0, L / scale, len(pts))
        hw = np.full(len(pts), float(np.linalg.norm(np.array(V[ra[-1]]) - np.array(V[ra[0]]))) / 2 / scale)
        n0 = len(mesh.dashes)
        _dash_strips(mesh, pts, side, arc, hw, mpp, scale, dashes)
        mesh.dash_owner.extend([owner] * (len(mesh.dashes) - n0))
    # sidewalks carried across, joining the two mouths' sidewalks
    if mesh.sw_cfg:
        for col, facing in ((0, lat), (len(ra) - 1, -lat)):
            sa, sb = mesh.sw_rows.get(ra[col]), mesh.sw_rows.get(rb[col])
            if not sa or not sb:
                continue
            srows = [sa["row"]]
            for k in range(1, n):
                t = k / n
                row = []
                for c, role, hh in ((0, "kerbface", 0.0), (1, "kerbstone", mesh.sw_cfg["height_m"]),
                                    (2, "kerbstone", mesh.sw_cfg["height_m"]),
                                    (3, "paving", mesh.sw_cfg["height_m"])):
                    p = np.array(V[sa["row"][c]]) * (1 - t) + np.array(V[sb["row"][c]]) * t
                    row.append(mesh.add(p[0], p[1], None, role, hh))
                srows.append(tuple(row))
            srows.append(sb["row"])
            for i in range(len(srows) - 1):
                _sw_connect(mesh, srows[i], srows[i + 1], owner, (float(facing[0]), float(facing[1])))
            info["sw"][col] = srows
    mesh.connectors[owner] = info



# ---------------------------------------------------------------------------
# Tight groups of junctions
#
# Some places have several junctions a few metres apart: slip roads cutting the
# corners of a crossroads, a Y with a triangular island, a roundabout. There the
# centreline breaks into short pieces and junctions that overlap, and patching
# each junction on its own leaves holes and patches that join the wrong roads.
#
# Such a group is handled as one area instead. Its junctions get no patches and
# short links inside it get no strips; the road inside the area is covered by
# the traced outline, laid a few millimetres below the road surface. Streets
# leaving the group keep their quad strips on top, so where a strip reaches, the
# strip shows, and where it cannot, the fill shows: no holes.
# ---------------------------------------------------------------------------

def find_clusters(det, mpp, coverage, island_max_m2=2500.0, gap_m=10.0):
    """Group junctions that share a small island or sit nearly on top of each other."""
    from shapely.geometry import Point, MultiPoint
    from shapely.ops import unary_union
    js = [j for j in det["junctions"]]
    if not js:
        return []
    ids = [int(j["id"]) for j in js]
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        parent[find(a)] = find(b)

    # junctions almost touching
    for a in range(len(js)):
        for b in range(a + 1, len(js)):
            ja, jb = js[a], js[b]
            d = math.hypot(ja["cx"] - jb["cx"], ja["cy"] - jb["cy"])
            if d < ja["r"] + jb["r"] + gap_m / mpp:
                join(ids[a], ids[b])

    # junctions around the same small island
    road = coverage >= 0.5
    lab, n = ndi.label(~road)
    H, W = road.shape
    edge_labels = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) if n else []
    islands = {}
    widths = 2 * det["dt"][det["skeleton"].astype(bool)]
    road_w = float(np.median(widths)) if widths.size else 8.0
    for i, a in enumerate(sizes, start=1):
        if i in edge_labels or a * mpp * mpp > island_max_m2:
            continue
        ys, xs = np.nonzero(lab == i)
        near = []
        for j in js:
            dmin = np.min(np.hypot(xs - j["cx"], ys - j["cy"]))
            if dmin <= j["r"] + road_w * 1.5:
                near.append(int(j["id"]))
        if near:
            for k in near[1:]:
                join(near[0], k)
            islands[i] = (near[0], xs, ys)

    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    byid = {int(j["id"]): j for j in js}
    # a junction cut by the image border cannot be patched properly: its arms
    # leave the picture. It is filled instead
    def at_border(j):
        return min(j["cx"], j["cy"], W - 1 - j["cx"], H - 1 - j["cy"]) < j["r"] + road_w
    out = []
    for root, members in groups.items():
        mine = [k for k, (r0, _, _) in islands.items() if find(r0) == root]
        flagged = any(byid[m]["type"] == "interchange (flagged)" or at_border(byid[m]) for m in members)
        if len(members) < 2 and not mine and not flagged:
            continue
        shapes = [Point(byid[m]["cx"], byid[m]["cy"]).buffer(byid[m]["r"] + road_w * 0.5) for m in members]
        for k in mine:
            _, xs, ys = islands[k]
            shapes.append(MultiPoint(list(zip(xs.tolist(), ys.tolist()))).convex_hull.buffer(road_w))
        # the whole group, including the crossing between its junctions
        region = unary_union(shapes).convex_hull.buffer(road_w * 0.5)
        c = region.centroid
        kind = ("interchange" if any(byid[m]["type"] == "interchange (flagged)" for m in members)
                else "border" if any(at_border(byid[m]) for m in members) else "urban")
        out.append({"members": members, "region": region, "centre": (c.x, c.y),
                    "islands": len(mine), "kind": kind})
    # groups whose areas nearly touch become one, or the road between them
    # would fall outside both fills while counting as inside a group
    merged = True
    while merged and len(out) > 1:
        merged = False
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                if out[a]["region"].distance(out[b]["region"]) < road_w * 3:
                    ra, rb = out[a], out[b]
                    region = unary_union([ra["region"], rb["region"]]).convex_hull
                    c = region.centroid
                    kinds = {ra["kind"], rb["kind"]}
                    out[a] = {"members": ra["members"] + rb["members"], "region": region,
                              "centre": (c.x, c.y), "islands": ra["islands"] + rb["islands"],
                              "kind": "interchange" if "interchange" in kinds else
                                      "urban" if "urban" in kinds else "border"}
                    del out[b]
                    merged = True
                    break
            if merged:
                break
    return out


def _inside_clusters(clusters, pts):
    from shapely import contains_xy
    pts = np.asarray(pts, float)
    out = np.zeros(len(pts), bool)
    for c in clusters:
        out |= contains_xy(c["region"], pts[:, 0], pts[:, 1])
    return out


def cluster_fill(clusters, coverage_out, scale, polys=None):
    """Triangles for the fill, from the build's polygons if given."""
    from app.model3d import triangulate
    if polys is not None:
        return triangulate(polys) if polys else (np.zeros((0, 2)), np.zeros((0, 3), np.uint32))
    return triangulate(cluster_fill_polys(clusters, coverage_out, scale))


def _scale_poly(poly, scale):
    from shapely.affinity import scale as sscale
    return sscale(poly, xfact=scale, yfact=scale, origin=(0, 0)) if scale != 1 else poly


def cluster_fill_polys(clusters, coverage_out, scale):
    """
    The traced road inside each group's area, as triangles in texture pixels.
    coverage_out is the smooth road coverage at the texture's size.
    """
    from shapely.affinity import scale as sscale
    from app.model3d import outline_polygons
    if not clusters:
        return []
    # the road outline, with its pixel wobble smoothed away: a small opening and
    # closing rounds off steps without moving straight edges
    r = 1.2 * scale
    polys = []
    for p0 in outline_polygons(coverage_out):
        q = p0.buffer(r).buffer(-2 * r).buffer(r)
        if not q.is_empty:
            polys.extend(q.geoms if hasattr(q, "geoms") else [q])
    pieces = []
    for c in clusters:
        area = c.get("fill_region", c["region"])
        if area is None or area.is_empty:
            continue
        region = sscale(area, xfact=scale, yfact=scale, origin=(0, 0))
        for p in polys:
            inter = p.intersection(region)
            if inter.is_empty:
                continue
            geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
            pieces.extend(g for g in geoms if g.geom_type == "Polygon" and g.area > 1)
    return pieces



def _fallback_fill(mesh, j):
    """A junction that could not be patched is covered by the traced outline instead."""
    from shapely.geometry import Point
    region = Point(j["cx"], j["cy"]).buffer(j["r"] * 1.15)
    mesh.clusters.append({"members": [int(j["id"])], "region": region,
                          "centre": (j["cx"], j["cy"]), "islands": 0, "fallback": True,
                          "kind": "fallback"})



def _cover_bare(mesh, coverage, scale, min_px=12):
    """
    The final guarantee: any road in the mask that nothing covers gets filled.

    The mesh is drawn into an image the size of the mask and compared with the
    road. Each uncovered piece bigger than a few pixels becomes a fill area,
    widened a little so it tucks under the strips around it.
    """
    from PIL import Image, ImageDraw
    from shapely.geometry import MultiPoint
    H, W = coverage.shape
    k = 2
    img = Image.new("L", (W * k, H * k), 0)
    d = ImageDraw.Draw(img)
    V = np.array(mesh.v, float) / scale * k
    for q in mesh.quads:
        d.polygon([tuple(V[i]) for i in q], fill=255)
    for t in mesh.tris:
        d.polygon([tuple(V[i]) for i in t], fill=255)
    for c in mesh.clusters:
        area = c.get("fill_region", c["region"])
        if area is None or area.is_empty:
            continue
        geoms = area.geoms if hasattr(area, "geoms") else [area]
        for g in geoms:
            d.polygon([(x * k, y * k) for x, y in g.exterior.coords], fill=255)
    covered = np.array(img.resize((W, H), Image.BOX)) > 60
    bare = (coverage >= 0.5) & ~covered
    lab, n = ndi.label(bare)
    if not n:
        return 0
    sizes = ndi.sum(bare, lab, index=np.arange(1, n + 1))
    added = 0
    for i, a in enumerate(sizes, start=1):
        if a < min_px:
            continue
        ys, xs = np.nonzero(lab == i)
        region = MultiPoint(list(zip(xs.tolist(), ys.tolist()))).convex_hull.buffer(4.0)
        mesh.clusters.append({"members": [], "region": region, "centre": (float(xs.mean()), float(ys.mean())),
                              "islands": 0, "fallback": True, "kind": "fallback"})
        added += 1
    return added



def _pieces(det, clusters, shape):
    """
    Rebuild the centreline inside group areas at full detail.

    Knots are where pieces really meet (branch points of the centreline) and
    where roads enter a group area. Around each, a small disc is cut out; what
    is left of the centreline, inside the areas, becomes separate pieces that
    are built as ordinary road strips. Outside the areas the street segments are
    kept, clipped at the area edge. Each group's fill becomes just its knots.
    """
    from PIL import Image, ImageDraw
    from shapely.geometry import Point
    from shapely.ops import unary_union
    H, W = shape
    sk = det["skeleton"].astype(bool)
    segs = det["segments"]
    dt = det["dt"]
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    owner = np.full((H, W), -1, np.int32)
    for gi, c in enumerate(clusters):
        geoms = c["region"].geoms if hasattr(c["region"], "geoms") else [c["region"]]
        for g in geoms:
            gi_img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(gi_img).polygon(list(g.exterior.coords), fill=255)
            owner[np.array(gi_img) > 0] = gi
            d.polygon(list(g.exterior.coords), fill=255)
    inside = np.array(img) > 0

    # knots: branch points of the centreline inside the areas. Roads entering
    # an area are not cut at its edge; a street simply continues as one piece
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    node = sk & inside & (nb >= 3)
    base = ((segs > 0) & ~inside) | (sk & inside)
    lab_n, nn = ndi.label(node, structure=np.ones((3, 3)))
    nodes = []
    cut = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]

    def disc(cx, cy, r):
        y0, y1 = max(0, int(cy - r) - 1), min(H, int(cy + r) + 2)
        x0, x1 = max(0, int(cx - r) - 1), min(W, int(cx + r) + 2)
        cut[y0:y1, x0:x1] |= (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2 <= r * r

    for i in range(1, nn + 1):
        ys, xs = np.nonzero(lab_n == i)
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(dt[int(round(cy)), int(round(cx))]) + 1.0
        nodes.append((cx, cy, r))
        disc(cx, cy, r)

    # each piece is then trimmed back from a knot to where it has separated from
    # the others and returned to its own normal width: a right-angle crossing
    # needs only a small knot, a shallow merge a long one
    kept = base & ~cut
    lab, n = ndi.label(kept, structure=np.ones((3, 3)))
    knot_px = []                                    # (x, y, r) discs making up the knots
    for cx, cy, r in nodes:
        knot_px.append((cx, cy, r * 1.3))
    node_arr = np.array([(cx, cy, r) for cx, cy, r in nodes]) if nodes else np.zeros((0, 3))
    for i in range(1, n + 1):
        coords = np.argwhere(lab == i)
        if len(coords) < 3:
            continue
        path = _ordered_path(coords)
        if len(path) < 3:
            continue
        pw = np.array([dt[int(p_[0]), int(p_[1])] for p_ in path])
        normal = float(np.median(pw))
        for forward in (True, False):
            seq = path if forward else path[::-1]
            y0, x0 = seq[0]
            if not len(node_arr):
                break
            dn = np.hypot(node_arr[:, 0] - x0, node_arr[:, 1] - y0) - node_arr[:, 2]
            if dn.min() > 3:
                continue                            # this end does not touch a knot
            for y, x in seq[: len(seq) // 2]:
                if dt[int(y), int(x)] <= 1.15 * normal:
                    break
                kept[int(y), int(x)] = False
                knot_px.append((float(x), float(y), float(dt[int(y), int(x)]) * 1.15))
    lab, n = ndi.label(kept, structure=np.ones((3, 3)))
    knots = [(x, y, r, int(owner[min(H - 1, max(0, int(round(y)))), min(W - 1, max(0, int(round(x))))]))
             for x, y, r in knot_px]

    # which labels are pieces inside a group, and which group
    piece_of = {}
    if n:
        own = owner[kept]
        ids = lab[kept]
        for i in range(1, n + 1):
            o = own[ids == i]
            if len(o) and (o >= 0).mean() > 0.5:
                vals, counts = np.unique(o[o >= 0], return_counts=True)
                piece_of[i] = int(vals[counts.argmax()])

    # each group's fill is now just its knots, widened to tuck under the strips
    for gi, c in enumerate(clusters):
        discs = [Point(cx, cy).buffer(r * 1.15) for cx, cy, r, g in knots if g == gi]
        c["fill_region"] = unary_union(discs).intersection(c["region"].buffer(2)) if discs else None
        c["knots"] = len(discs)
    return lab, piece_of



def _drop_short_runs(ok, along_m, min_m):
    """Keep only runs of consecutive True rows covering at least min_m metres."""
    ok = ok.copy()
    i, n = 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        if along_m[j] - along_m[i] < min_m:
            ok[i:j + 1] = False
        i = j + 1
    return ok



# ---------------------------------------------------------------------------
# Sidewalks along the kerb line
#
# Instead of a sidewalk per street joined at corners, the road surface is
# assembled from the exact shapes of every strip, junction patch and fill, and
# its edge, the kerb line, is walked around each block and island. Each block
# gets one continuous sidewalk: where two roads merge, their kerbs are simply
# one line and the sidewalk follows it. Where the kerb line meets the image
# border, the sidewalk runs to the border and is cut exactly there.
# ---------------------------------------------------------------------------

def _road_surface(mesh):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    V = mesh.v
    polys = []

    def add(ring):
        if len(ring) >= 3:
            q = Polygon([V[i] for i in ring])
            if not q.is_valid:
                q = q.buffer(0)
            if not q.is_empty:
                polys.append(q)

    for st in mesh.streets.values():
        rows = st["rows"]
        add([r[0] for r in rows] + [r[-1] for r in rows[::-1]])
    for oid, c in mesh.connectors.items():
        if oid != "_warnings":
            add([r[0] for r in c["rows"]] + [r[-1] for r in c["rows"][::-1]])
    for ring in mesh.patch_rings:
        add(ring)
    polys.extend(mesh.fill_polys or [])
    # a hair of growth merges pieces that touch along an edge into one surface;
    # shrinking by the same hair afterwards puts the edge back exactly where the
    # road's edge is, or the kerb would sit that far away from it
    return unary_union([q.buffer(0.02) for q in polys]).buffer(-0.02)


def _smooth_ring(P, frame_w, frame_h, iterations, keep=None):
    """
    Straighten a closed kerb line the way the roads were straightened.

    Taubin smoothing: a step that pulls each point towards its neighbours and a
    slightly larger step back out. Wobble and pixel steps disappear, but a curve
    keeps its size, where plain smoothing would shrink islands a little every
    pass. Points on the image border stay put, so the border stays straight, and
    so do points on an exact edge (keep: an inner street's edge, already exact).
    """
    Q = P.copy()
    fixed = (Q[:, 0] < 0.6) | (Q[:, 1] < 0.6) | (Q[:, 0] > frame_w - 0.6) | (Q[:, 1] > frame_h - 0.6)
    if keep is not None and not keep.is_empty:
        import shapely
        fixed |= shapely.dwithin(keep, shapely.points(Q), 0.05)
    for _ in range(iterations):
        for lam in (0.5, -0.53):
            avg = (np.roll(Q, 1, axis=0) + np.roll(Q, -1, axis=0)) / 2
            step = lam * (avg - Q)
            step[fixed] = 0
            Q = Q + step
    return Q


def _resample_ring(P, spacing):
    closed = np.vstack([P, P[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    L = float(seg.sum())
    n = max(8, int(L / spacing))
    t = np.linspace(0, L, n + 1)[:-1]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.column_stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])])


def _adaptive(K, min_step, max_step, max_turn_deg):
    """
    Keep more points where the kerb bends and fewer where it runs straight: a
    new row whenever the kerb has turned by max_turn_deg or run max_step.
    """
    keep = [0]
    run, turn = 0.0, 0.0
    n = len(K)
    for i in range(1, n):
        a = K[i] - K[i - 1]
        b = K[(i + 1) % n] - K[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        run += la
        here = 0.0
        if la > 1e-9 and lb > 1e-9:
            here = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1)))
            turn += here
        # a sharp corner is always kept exactly, or the row before and after it
        # would cut across the tip and leave it uncovered
        if here >= 12.0 or (run >= max_step) or (turn >= max_turn_deg and run >= min_step):
            keep.append(i)
            run, turn = 0.0, 0.0
    return np.array(keep)


def _kerb_line_sidewalks(mesh, shape, scale, straightness=0.7):
    from shapely import contains_xy
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    cfg, mpp = mesh.sw_cfg, mesh.sw_mpp
    H, W = shape
    raw = _road_surface(mesh)
    if raw.is_empty:
        return

    # the kerb line, straightened: every ring of the road surface is smoothed,
    # in mask pixels, with more passes the higher the straightness setting
    iters = int(round(10 + 90 * straightness))
    fine = max(0.35, 0.5 / mpp)                            # 0.5 m spacing for smoothing
    rings_out = []
    geoms = raw.geoms if hasattr(raw, "geoms") else [raw]
    keep = getattr(mesh, "exact_edges", None)
    for g in geoms:
        ext = _smooth_ring(_resample_ring(np.array(g.exterior.coords)[:-1] / scale, fine), W, H, iters, keep)
        holes = [_smooth_ring(_resample_ring(np.array(r.coords)[:-1] / scale, fine), W, H, iters, keep)
                 for r in g.interiors if r.length / scale > 4 * fine]
        q = Polygon(ext, [h for h in holes if len(h) >= 4])
        if not q.is_valid:
            q = q.buffer(0)
        rings_out.append(q)
    surface = unary_union(rings_out).intersection(box(0, 0, W, H))
    mesh.kerb_surface = _scale_poly(surface, scale)

    # the knot fills now follow the straightened kerb, so the road's edge and
    # the sidewalk's edge are the same line
    # everything inside the kerb line that the street strips and junction
    # patches do not cover is filled: the knots, and any sliver where the
    # smoothed kerb runs just outside a strip. The road then always reaches the
    # kerb. The fill starts a little under the strips so no seam can open
    V = mesh.v
    built = []
    for q in mesh.quads:
        pg = Polygon([(V[i][0] / scale, V[i][1] / scale) for i in q])
        built.append(pg if pg.is_valid else pg.buffer(0))
    built = unary_union([b for b in built if not b.is_empty])
    # only what is actually uncovered, then widened a little so it tucks under
    # the strips beside it: no band under every road edge where nothing is missing
    gaps = surface.difference(built)
    gaps = [g for g in (gaps.geoms if hasattr(gaps, "geoms") else [gaps])
            if g.geom_type == "Polygon" and g.area > 0.02]
    fills = unary_union([g.buffer(0.3) for g in gaps]).intersection(surface) if gaps else Polygon()
    mesh.fill_polys = [_scale_poly(g, scale) for g in (fills.geoms if hasattr(fills, "geoms") else [fills])
                       if g.geom_type == "Polygon" and g.area > 0.05]

    # optimised: sidewalk rows only where the kerb bends, up to 20 m apart
    sw_max, sw_turn = (20.0, 4.0) if getattr(mesh, "optimise", False) else (2.0, 6.0)
    # the road as it will actually be exported: strips, patches and fills
    road_all = unary_union([built] + [_scale_poly(f, 1.0 / scale) for f in (mesh.fill_polys or [])])
    snap_px = 0.2 / mpp
    H_m, ks_px = cfg["height_m"], cfg["kerbstone_m"] / mpp
    maxd = 2 * cfg["max_m"] / mpp + 2
    big = box(0, 0, W, H).buffer(4 * cfg["max_m"] / mpp, join_style=2)
    blocks = big.difference(surface)
    blocks = list(blocks.geoms) if hasattr(blocks, "geoms") else [blocks]
    cen, cw = [], []
    for st in mesh.streets.values():
        cen.append(st["pts"]); cw.append(2 * st["hw"])
    cen = np.vstack(cen) if cen else np.zeros((1, 2))
    cw = np.concatenate(cw) if cw else np.array([8.0])

    geoms = surface.geoms if hasattr(surface, "geoms") else [surface]
    loop_id = 3_000_000
    for g in geoms:
        if g.geom_type != "Polygon":
            continue
        for ring_i, ring in enumerate([g.exterior] + list(g.interiors)):
            loop_id += 1
            K = _resample_ring(np.array(ring.coords)[:-1], fine)
            n_ = len(K)
            if n_ < 8:
                continue
            K = _snap_to(K, road_all, snap_px)
            tg = np.roll(K, -1, axis=0) - np.roll(K, 1, axis=0)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            nrm = np.column_stack([-tg[:, 1], tg[:, 0]])
            probe = K + nrm * 0.4
            nrm[contains_xy(surface, probe[:, 0], probe[:, 1])] *= -1       # away from the road
            on_frame = (K[:, 0] < 0.6) | (K[:, 1] < 0.6) | (K[:, 0] > W - 1.6) | (K[:, 1] > H - 1.6)
            _, idx = _nearest(cen, K)
            target = np.clip(cfg["share"] * cw[idx] * mpp, cfg["min_m"], cfg["max_m"]) / mpp
            wide = (cw[idx] * mpp > cfg["max_road_m"]) & bool(cfg.get("skip_interchanges", False))
            ok = ~on_frame & ~_excluded(mesh, K) & ~wide
            if not ok.any():
                continue
            is_hole = ring_i > 0                                  # a block or island enclosed by road

            if is_hole and ok.all():
                # a closed block or island: the sidewalk's inner edge is the
                # island itself shrunk by the sidewalk width, which cannot fold
                # or overlap; too small to shrink, and it is paved completely
                island = Polygon(K)
                if not island.is_valid:
                    island = island.buffer(0)
                w = float(np.median(target))
                inner = island.buffer(-w, join_style=1)
                keep = _adaptive(K, 0.4 / mpp, sw_max / mpp, sw_turn)
                Kk = K[keep]
                if inner.is_empty or inner.area < 0.5 or inner.geom_type != "Polygon":
                    mesh.sw_kind[loop_id] = "paved"
                    _paved_island(mesh, Kk, loop_id, H_m, scale)
                    continue
                mesh.sw_kind[loop_id] = "closed"
                O = _project_monotonic(Kk, inner.exterior)
                along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Kk, axis=0).T))]) * mpp
                _rows_from(mesh, Kk, O, along, loop_id, H_m, ks_px, closed=True)
                continue

            # open runs: blocks touching the image border, or interrupted by an
            # interchange. Width by the rule, never more than half the space to
            # the next road, narrowed on tight inside curves
            free = _free_space(mesh, K, nrm, maxd)
            w = np.minimum(target, free / 2)
            for _ in range(12):
                O = K + nrm * w[:, None]
                dk = np.roll(K, -1, axis=0) - K
                do = np.roll(O, -1, axis=0) - O
                fold = (dk * do).sum(axis=1) < 0.2 * (dk * dk).sum(axis=1)
                if not fold.any():
                    break
                bad = fold | np.roll(fold, 1)
                w[bad] *= 0.7
            w = ndi.minimum_filter1d(w, 5, mode="wrap")
            w = ndi.uniform_filter1d(w, 5, mode="wrap")
            ok &= w > 0.3 / mpp
            for run in _runs_circular(ok):
                if len(run) < 3:
                    continue
                idxs = list(run)
                # carry a run that stops at the image border right up to it
                for end in (0, -1):
                    j = idxs[end]
                    nb = (j - 1) % n_ if end == 0 else (j + 1) % n_
                    if on_frame[nb]:
                        idxs = [nb] + idxs if end == 0 else idxs + [nb]
                sub = K[idxs]
                keep = _adaptive(sub, 0.4 / mpp, sw_max / mpp, sw_turn)
                if keep[-1] != len(sub) - 1:
                    keep = np.append(keep, len(sub) - 1)
                pts = sub[keep]
                ws = np.array([w[i] if not on_frame[i] else w[idxs[1] if k == 0 else idxs[-2]]
                               for k, i in enumerate(np.array(idxs)[keep])])
                ns = nrm[np.array(idxs)[keep]]
                run_len = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) * mpp
                if run_len < cfg.get("min_run_m", 8.0):
                    continue
                O = _open_inner(pts, ns, ws, blocks)
                if O is None:
                    O = pts + ns * ws[:, None]
                O[:, 0] = np.clip(O[:, 0], 0, W - 1.0)
                O[:, 1] = np.clip(O[:, 1], 0, H - 1.0)
                along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]) * mpp
                mesh.sw_kind[loop_id] = "open"
                _rows_from(mesh, pts, O, along, loop_id, H_m, ks_px, closed=False)


def _project_monotonic(K, ring):
    """
    Pair each kerb point with a point on the inner edge, moving only forwards
    around it, so rows never cross each other.
    """
    from shapely.geometry import Point
    from shapely.geometry.polygon import orient, LinearRing
    L = ring.length
    s = np.array([ring.project(Point(x, y)) for x, y in K])
    # both loops must run the same way round
    kr = LinearRing(K)
    if kr.is_ccw != LinearRing(ring.coords).is_ccw:
        s = (L - s) % L
        ring = LinearRing(list(ring.coords)[::-1])
    s0 = s[0]
    rel = (s - s0) % L
    rel = np.maximum.accumulate(np.unwrap(rel * 2 * np.pi / L) * L / (2 * np.pi))
    # positions were measured from the first point; add that start back
    return np.array([ring.interpolate((s0 + v) % L).coords[0] for v in rel])


def _rows_from(mesh, K, O, along, owner, H_m, ks_px, closed):
    """Sidewalk rows from kerb points to outer points, and the quads between them."""
    sc = mesh.sw_scale
    rows = []
    for k in range(len(K)):
        d = O[k] - K[k]
        w = float(np.hypot(*d))
        if w < 1e-6:
            d = np.array([0.0, 0.0]); w = 0.0
        n = d / max(w, 1e-9)
        rows.append(_sw_row(mesh, K[k], n, max(w, 1e-3), float(along[k]), H_m, ks_px))
    if closed:
        # repeat the first row at the end with its full distance along, so the
        # paving does not jump back to the start in the last quad
        L = float(along[-1] + np.hypot(*(K[0] - K[-1])) * mesh.sw_mpp)
        d = O[0] - K[0]
        w = float(np.hypot(*d))
        rows.append(_sw_row(mesh, K[0], d / max(w, 1e-9), max(w, 1e-3), L, H_m, ks_px))
    for k in range(len(rows) - 1):
        d = K[(k + 1) % len(K)] - K[k]
        facing = (-(O[k] - K[k]) / max(np.hypot(*(O[k] - K[k])), 1e-9))
        _sw_connect(mesh, rows[k], rows[k + 1], owner, (float(facing[0]), float(facing[1])))


def _paved_island(mesh, K, owner, H_m, scale):
    """An island too small for a ring of sidewalk: paved over completely."""
    from shapely.geometry import Polygon
    from app.model3d import triangulate
    poly = Polygon(K)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return
    v2, t2 = triangulate([poly])
    base = len(mesh.v)
    for x, y in v2:
        mesh.add(x * scale, y * scale, None, "paving", H_m)
    for t in t2:
        mesh.sw_tris.append(tuple(int(i) + base for i in t))
        mesh.sw_tri_owner.append(owner)
    # the kerb face all round it
    ring = [mesh.add(x * scale, y * scale, None, "kerbface", 0.0) for x, y in K]
    top = [mesh.add(x * scale, y * scale, None, "kerbstone", H_m) for x, y in K]
    c = np.array(K).mean(axis=0)
    for i in range(len(K)):
        j = (i + 1) % len(K)
        out = np.array(K[i]) - c
        out /= max(np.hypot(*out), 1e-9)
        mesh.kerb_quads.append((ring[i], ring[j], top[j], top[i]))
        mesh.kerb_facing.append((float(out[0]), float(out[1])))
        mesh.kerb_owner.append(owner)


def _nearest(A, B):
    from scipy.spatial import cKDTree
    d, i = cKDTree(A).query(B)
    return d, i


def _runs_circular(ok):
    """Runs of consecutive True indices around a closed loop."""
    n = len(ok)
    if not ok.any():
        return []
    start = int(np.argmin(ok))                   # begin just after a gap
    runs, cur = [], []
    for k in range(1, n + 1):
        i = (start + k) % n
        if ok[i]:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return runs



def _open_inner(pts, ns, ws, blocks):
    """
    The inner edge for an open run of sidewalk: the block beside it shrunk by
    the sidewalk width, paired with the kerb points moving only forwards, so
    rows cannot cross or leave gaps at corners. None if that is not possible.
    """
    from shapely.geometry import Point
    mid = len(pts) // 2
    probe = Point(*(pts[mid] + ns[mid] * 0.6))
    block = next((b for b in blocks if b.contains(probe)), None)
    if block is None:
        return None
    w = float(np.median(ws))
    inner = block.buffer(-w, join_style=1)
    if inner.is_empty:
        return None
    parts = inner.geoms if hasattr(inner, "geoms") else [inner]
    target = Point(*(pts[mid] + ns[mid] * w))
    part = min(parts, key=lambda g: g.distance(target))
    ring = part.exterior
    L = ring.length
    s = np.array([ring.project(Point(x, y)) for x, y in pts])
    u = np.unwrap(s * 2 * np.pi / L) * L / (2 * np.pi)
    if u[-1] < u[0]:
        u = -np.maximum.accumulate(-u)          # the inner edge runs the other way round
    else:
        u = np.maximum.accumulate(u)
    O = np.array([ring.interpolate(v % L).coords[0] for v in u])
    # a pairing that jumps far from the kerb means the shape was not simple here
    if np.any(np.hypot(*(O - pts).T) > 3 * w + 2):
        return None
    return O



def _snap_to(K, road, max_px):
    """
    Move kerb points that sit just outside the road exactly onto its edge, so
    the kerb and the road meet with no gap. Points further out are left alone.
    """
    from shapely import distance as sdist, points as spoints
    from shapely.ops import nearest_points
    from shapely.geometry import Point
    d = sdist(road, spoints(K))
    out = (d > 1e-6) & (d < max_px)
    if out.any():
        K = K.copy()
        for i in np.nonzero(out)[0]:
            q = nearest_points(road, Point(*K[i]))[0]
            K[i] = (q.x, q.y)
    return K



def _row_keep(pts, tang, arc_m, hw, layout, max_gap_m=20.0, turn_deg=2.0, width_change=0.05):
    """
    Rows the optimised mesh keeps along a street.

    A new row where the road has turned turn_deg since the last one, where its
    width has changed by width_change, and at least every max_gap_m: the broad
    lighter and darker areas that hide the tile repeating are stored per vertex,
    so rows too far apart would lose them. Painted dashes need a row just
    before the first dash starts and just after the last ends, so the marked
    texture lines up; bridge ramps pass a smaller max_gap_m.
    """
    n = len(pts)
    must = {0, n - 1}
    if layout:
        start = layout["setback_m"] - (1 - layout["share"]) * layout["step_m"] + 0.05
        stop = layout["setback_m"] + layout["n"] * layout["step_m"] - 0.05
        i0 = int(np.searchsorted(arc_m, start))
        i1 = int(np.searchsorted(arc_m, stop, side="right")) - 1
        for i in (i0, i1):
            if 0 <= i < n:
                must.add(i)
    keep = [0]
    last = 0
    for i in range(1, n):
        turned = np.degrees(np.arccos(np.clip(float(tang[i] @ tang[last]), -1, 1)))
        widened = abs(hw[i] - hw[last]) > width_change * max(hw[last], 1e-6)
        far = arc_m[i] - arc_m[last] >= max_gap_m
        if i in must or turned >= turn_deg or widened or far:
            # keep the row before a turn too, so the straight part ends exactly there
            if (turned >= turn_deg or widened) and i - 1 > last:
                keep.append(i - 1)
            keep.append(i)
            last = i
    return np.array(sorted(set(keep)))



def block_polygons(mesh, shape, scale, mpp, min_area_m2=4.0):
    """
    The blocks and islands between roads, in texture pixels: everything inside
    the image that is not road. The road surface is the straightened kerb line
    when sidewalks were built along it, otherwise the road pieces themselves.
    """
    from shapely.geometry import box
    H, W = shape
    surface = getattr(mesh, "kerb_surface", None)
    if surface is None or surface.is_empty:
        surface = _road_surface(mesh)
    frame = box(0, 0, W * scale, H * scale)
    rest = frame.difference(surface)
    geoms = rest.geoms if hasattr(rest, "geoms") else [rest]
    px_area = (mpp / scale) ** 2                      # square metres per texture pixel
    out = []
    for g in geoms:
        if g.geom_type != "Polygon" or g.is_empty:
            continue
        if g.area * px_area < min_area_m2:
            continue
        # a light simplification: the outline is already smooth, and a large
        # flat plane gains nothing from thousands of edge points
        g2 = g.simplify(0.15 * scale / mpp, preserve_topology=True)
        out.append(g2 if g2.is_valid and not g2.is_empty else g)
    return out
