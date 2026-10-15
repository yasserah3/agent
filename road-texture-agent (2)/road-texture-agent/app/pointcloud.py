"""
The map as dots, for the 3D tab while its 3D model is being built (the look
of a laser scan or Gaussian splatting): the whole map at once, then each part
of it giving way to the real model as it comes in.

Dots on a jittered grid over the whole map: on the road at its height (0), on
the sidewalks along it and on the blocks beyond at theirs (the 3D model
settings), coloured from the texture (the road) and the paving's own colour
(the sidewalks and blocks); each placed object a cloud of dots on the ellipsoid
that fills its box, so the trees and buildings stand up out of the map. The
spacing grows with the map so the dots stay at MAX_POINTS at most: 1.2 m on a
2.2 x 1.7 km map, 6.6 m on a 12 x 9 km city. They come grouped by the 1 km
square they lie in (app/patches.py's grid), so the view can hide a square's
dots when its patch of the real model is in.
"""
import json
import math
import struct

import numpy as np
from scipy import ndimage as ndi

MAX_POINTS = 2_500_000      # the map's dots at most (about 38 MB to the browser, 30 MB on the graphics card)
MIN_SPACING_M = 0.5         # and never closer than this
OBJECT_POINTS = 400_000     # the objects' dots at most
OBJECT_COLOUR = (96, 150, 84)


def spacing_m(width_m, height_m):
    """The dots' spacing for a map this size: MAX_POINTS at most, MIN_SPACING_M at least."""
    return max(MIN_SPACING_M, math.sqrt(max(width_m * height_m, 1.0) / MAX_POINTS))


def _ellipsoid(rng, n, w, h, d):
    """n points spread over the surface of the ellipsoid with these sides (x, y, z), centred at 0."""
    u = rng.normal(size=(n, 3))
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return u * np.array([w / 2, h / 2, d / 2])


def build(road, colours, mpp, patch_m, sidewalk_m=None, sidewalk_h=None, block_h=None,
          sidewalk_rgb=(150, 150, 150), block_rgb=(150, 150, 150), objects=(), seed=7):
    """
    The dots as bytes: a little-endian uint32 n, n bytes of JSON (spaces to a
    multiple of 4), then the positions (float32 x, y, z in the model's metres,
    the map's middle at 0, y up) and the colours (uint8 r, g, b).

    road: the mask (bool, H x W); colours: the texture (H' x W' x 3 uint8, any
    size, read in proportion); mpp: the mask's metres a pixel; sidewalk_m: how
    far from the road the sidewalks reach (None: no sidewalks); sidewalk_h,
    block_h: their heights (None: none of them); objects: [(x, y, angle, w, d,
    h, stand)] in metres in image axes (x right, y down), angle in radians.
    The JSON: count, spacing_m, size_m, patch_m, max_h (the highest dot) and
    cells: [{"id": "i_j", "start", "count"}] (the 1 km squares, in order).
    """
    rng = np.random.default_rng(seed)
    H, W = road.shape
    Wm, Hm = W * mpp, H * mpp
    s = spacing_m(Wm, Hm)
    nx, nz = max(1, int(math.ceil(Wm / s))), max(1, int(math.ceil(Hm / s)))
    gi, gj = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(nz, dtype=np.float32))
    x = ((gi.ravel() + rng.random(nx * nz, dtype=np.float32)) * s).astype(np.float32)
    z = ((gj.ravel() + rng.random(nx * nz, dtype=np.float32)) * s).astype(np.float32)
    keep = (x < Wm) & (z < Hm)
    x, z = x[keep], z[keep]
    px = np.clip((x / mpp).astype(np.int64), 0, W - 1)
    pz = np.clip((z / mpp).astype(np.int64), 0, H - 1)
    on_road = road[pz, px]
    y = np.zeros(len(x), np.float32)
    rgb = np.zeros((len(x), 3), np.uint8)
    ch, cw = colours.shape[:2]
    rgb[on_road] = colours[np.clip((z[on_road] / Hm * ch).astype(np.int64), 0, ch - 1),
                           np.clip((x[on_road] / Wm * cw).astype(np.int64), 0, cw - 1)]
    off = ~on_road
    keep = on_road.copy()
    if off.any() and (sidewalk_h is not None or block_h is not None):
        if sidewalk_h is not None and sidewalk_m:
            far = ndi.distance_transform_edt(~road)[pz[off], px[off]] * mpp       # metres to the road
            walk = far <= sidewalk_m
        else:
            walk = np.zeros(int(off.sum()), bool)
        idx = np.flatnonzero(off)
        if sidewalk_h is not None:
            y[idx[walk]] = sidewalk_h
            rgb[idx[walk]] = sidewalk_rgb
            keep[idx[walk]] = True
        if block_h is not None:
            y[idx[~walk]] = block_h
            rgb[idx[~walk]] = block_rgb
            keep[idx[~walk]] = True
    # a little light and dark over the paving, as its grain (one flat colour reads as a cardboard)
    shade = (0.9 + 0.2 * rng.random(len(x), dtype=np.float32))[:, None]
    rgb = np.where(on_road[:, None], rgb, np.clip(rgb * shade, 0, 255)).astype(np.uint8)
    x, y, z, rgb = x[keep], y[keep], z[keep], rgb[keep]
    # the objects: a cloud on the ellipsoid filling each one's box, about as dense as the map's dots
    if objects:
        so = max(0.25, s / 2)
        sizes = []
        for (ox, oy, ang, w, d, h, stand) in objects:
            a, b, c = max(w, 0.2) / 2, max(h, 0.2) / 2, max(d, 0.2) / 2
            area = 4 * math.pi * ((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) ** (1 / 1.6) / 3   # Knud Thomsen
            sizes.append(max(8, int(area / (so * so))))
        k = min(1.0, OBJECT_POINTS / max(sum(sizes), 1))
        parts = []
        for (ox, oy, ang, w, d, h, stand), n in zip(objects, sizes):
            p = _ellipsoid(rng, max(8, int(n * k)), max(w, 0.2), max(h, 0.2), max(d, 0.2))
            ca, sa = math.cos(ang), math.sin(ang)
            px_, pz_ = p[:, 0] * ca - p[:, 2] * sa, p[:, 0] * sa + p[:, 2] * ca
            parts.append(np.column_stack([ox + px_, stand + h / 2 + p[:, 1], oy + pz_]))
        if parts:
            P = np.concatenate(parts).astype(np.float32)
            inside = (P[:, 0] >= 0) & (P[:, 0] < Wm) & (P[:, 2] >= 0) & (P[:, 2] < Hm)
            P = P[inside]
            x, y, z = np.concatenate([x, P[:, 0]]), np.concatenate([y, P[:, 1]]), np.concatenate([z, P[:, 2]])
            rgb = np.concatenate([rgb, np.tile(np.array(OBJECT_COLOUR, np.uint8), (len(P), 1))])
    # grouped by their 1 km square, row by row
    ncx = max(1, int(math.ceil(Wm / patch_m - 1e-9)))
    ci = np.clip((x // patch_m).astype(np.int64), 0, ncx - 1)
    cj = np.clip((z // patch_m).astype(np.int64), 0, max(0, int(math.ceil(Hm / patch_m - 1e-9)) - 1))
    key = cj * ncx + ci
    order = np.argsort(key, kind="stable")
    x, y, z, rgb, key = x[order], y[order], z[order], rgb[order], key[order]
    cells = []
    if len(key):
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)]
        for a, b in zip(starts, ends):
            k_ = int(key[a])
            cells.append({"id": f"{k_ % ncx}_{k_ // ncx}", "start": int(a), "count": int(b - a)})
    pos = np.column_stack([x - Wm / 2, y, z - Hm / 2]).astype("<f4")
    head = json.dumps({"count": int(len(x)), "spacing_m": round(s, 3), "size_m": [round(Wm, 2), round(Hm, 2)],
                       "patch_m": patch_m, "max_h": round(float(y.max()) if len(y) else 0.0, 2),
                       "cells": cells}).encode()
    head += b" " * (-len(head) % 4)
    return struct.pack("<I", len(head)) + head + pos.tobytes() + np.ascontiguousarray(rgb, np.uint8).tobytes()


def read(data):
    """The header and arrays back from build()'s bytes (for tests)."""
    n = struct.unpack_from("<I", data, 0)[0]
    head = json.loads(data[4:4 + n])
    c = head["count"]
    pos = np.frombuffer(data, "<f4", c * 3, 4 + n).reshape(c, 3)
    rgb = np.frombuffer(data, np.uint8, c * 3, 4 + n + c * 12).reshape(c, 3)
    return head, pos, rgb
