"""
Looks for the 3D tab's pictures: colour lookup tables (LUTs) and saved looks.

A LUT maps each colour of the finished picture to another, the way film
stocks, camera profiles and colour grades are passed around. Two common kinds
are read:
- .cube files (Adobe/Resolve format): 3D (LUT_3D_SIZE) or 1D (LUT_1D_SIZE),
  with an optional DOMAIN_MIN / DOMAIN_MAX;
- Hald CLUT pictures (.png): a square picture whose pixels list the table,
  as shipped by RawTherapee, darktable and G'MIC film simulation packs.
Whatever comes in is stored as one canonical 3D .cube, which is what the
browser reads.

A look is a name, its settings (exposure, contrast, ...) and optionally a LUT.
Your saved looks and imported LUTs live in the workspace (looks/); looks and
LUTs that ship with the program are in app/looks.
"""
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

BUNDLED = Path(__file__).with_name("looks")
MAX_3D = 65            # the largest 3D table kept (65^3 entries, as Resolve's largest)
SIZE_1D_AS_3D = 33     # a 1D table becomes a 3D one of this size


def parse_cube(text):
    """
    A .cube file's table: (size, domain_min, domain_max, data), data an
    (size^3, 3) float32 array with red changing fastest, then green, then
    blue (the file's own order). A 1D table is turned into a 3D one.
    """
    size3 = size1 = None
    dmin, dmax = np.zeros(3, np.float32), np.ones(3, np.float32)
    rows = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0].upper()
        if head == "TITLE":
            continue
        if head == "LUT_3D_SIZE":
            size3 = int(line.split()[1])
        elif head == "LUT_1D_SIZE":
            size1 = int(line.split()[1])
        elif head in ("DOMAIN_MIN", "DOMAIN_MAX", "LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
            v = [float(x) for x in line.split()[1:]]
            if head.endswith("INPUT_RANGE"):
                dmin, dmax = np.full(3, v[0], np.float32), np.full(3, v[1], np.float32)
            elif head == "DOMAIN_MIN":
                dmin = np.array(v[:3], np.float32)
            else:
                dmax = np.array(v[:3], np.float32)
        elif re.match(r"^[-+0-9.eE]", head):
            v = line.split()
            if len(v) >= 3:
                rows.append((float(v[0]), float(v[1]), float(v[2])))
        # anything else (a keyword this reader does not know) is skipped
    data = np.asarray(rows, np.float32)
    if size3:
        if not 2 <= size3 <= 256:
            raise ValueError("LUT_3D_SIZE %d is out of range" % size3)
        if len(data) != size3 ** 3:
            raise ValueError("a %d^3 table needs %d rows, the file has %d" % (size3, size3 ** 3, len(data)))
        if size3 > MAX_3D:
            data, size3 = _resample3d(data, size3, MAX_3D), MAX_3D
        return size3, dmin, dmax, data
    if size1:
        if len(data) != size1:
            raise ValueError("a 1D table of %d needs %d rows, the file has %d" % (size1, size1, len(data)))
        return SIZE_1D_AS_3D, dmin, dmax, _from_1d(data, SIZE_1D_AS_3D)
    raise ValueError("not a .cube table (no LUT_3D_SIZE or LUT_1D_SIZE)")


def _from_1d(curve, n):
    """A 1D table (one curve per channel) as an n^3 3D table."""
    x = np.linspace(0.0, 1.0, n)
    src = np.linspace(0.0, 1.0, len(curve))
    per = [np.interp(x, src, curve[:, c]) for c in range(3)]
    b, g, r = np.meshgrid(x, x, x, indexing="ij")          # blue slowest, red fastest
    idx = np.rint(np.stack([r, g, b], -1).reshape(-1, 3) * (n - 1)).astype(int)
    return np.stack([per[0][idx[:, 0]], per[1][idx[:, 1]], per[2][idx[:, 2]]], -1).astype(np.float32)


def _resample3d(data, n, m):
    """An n^3 table resampled to m^3 (trilinear)."""
    from scipy.ndimage import map_coordinates
    cube = data.reshape(n, n, n, 3)                           # [b, g, r, channel]
    x = np.linspace(0, n - 1, m)
    b, g, r = np.meshgrid(x, x, x, indexing="ij")
    out = [map_coordinates(cube[..., c], [b.ravel(), g.ravel(), r.ravel()], order=1) for c in range(3)]
    return np.stack(out, -1).astype(np.float32)


def parse_hald(data):
    """
    A Hald CLUT picture (PNG): level L, a square of L^3 x L^3 pixels listing an
    L^2-sized table in the same order as a .cube (red fastest). Returns
    (size, data) as parse_cube.
    """
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    level = int(round(w ** (1.0 / 3.0)))
    if w != h or level ** 3 != w or level < 2:
        raise ValueError("not a Hald CLUT picture (it must be square, %d x %d is not a cube's side)" % (w, h))
    arr = np.asarray(im.convert("RGB"), np.float32)
    size = level * level
    table = arr.reshape(-1, 3)[: size ** 3] / 255.0
    if size > MAX_3D:
        table, size = _resample3d(table, size, MAX_3D), MAX_3D
    return size, table.astype(np.float32)


def cube_text(title, size, data, dmin=(0, 0, 0), dmax=(1, 1, 1)):
    """A canonical 3D .cube file."""
    head = ['TITLE "%s"' % str(title).replace('"', "'")[:80], "LUT_3D_SIZE %d" % size,
            "DOMAIN_MIN %.6g %.6g %.6g" % tuple(dmin), "DOMAIN_MAX %.6g %.6g %.6g" % tuple(dmax)]
    body = "\n".join("%.6f %.6f %.6f" % tuple(r) for r in np.asarray(data, np.float32))
    return "\n".join(head) + "\n" + body + "\n"


def read_lut(name, raw):
    """Any LUT file (by its name) as (size, canonical .cube text). Raises ValueError if it is not one."""
    low = name.lower()
    title = Path(name).stem
    if low.endswith(".png") or raw[:8] == b"\x89PNG\r\n\x1a\n":
        size, data = parse_hald(raw)
        return size, cube_text(title, size, data)
    if low.endswith(".cube") or b"LUT_3D_SIZE" in raw[:20000] or b"LUT_1D_SIZE" in raw[:20000]:
        size, dmin, dmax, data = parse_cube(raw.decode("utf-8", "replace"))
        return size, cube_text(title, size, data, dmin, dmax)
    raise ValueError("%s is not a LUT: use a .cube file or a Hald CLUT .png" % name)


# ------------------------------------------------------------------ storage
SETTINGS = ("film", "exposure", "contrast", "brightness", "highlights", "shadows", "saturation", "warmth", "tint",
            "split", "split_shadows", "split_highlights", "fade", "vignette", "sharpen", "grain", "glow",
            "lut", "lut_mix")


def clean_settings(p):
    """A look's settings, only the known ones: numbers as numbers, colours as #rrggbb."""
    out = {}
    for k in SETTINGS:
        if k not in (p or {}):
            continue
        v = p[k]
        if k == "film":
            if v in ("agx", "neutral", "aces"):
                out[k] = v
        elif k in ("split_shadows", "split_highlights"):
            if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v):
                out[k] = v.lower()
        elif k == "lut":
            if v is None or (isinstance(v, str) and re.fullmatch(r"[0-9a-z]{12}", v)):
                out[k] = v
        else:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                out[k] = max(-1000.0, min(1000.0, f))
    return out


class Store:
    """Your saved looks (looks/<id>.json) and LUTs (looks/luts/<id>.cube) in the workspace, and the bundled ones."""

    def __init__(self, folder):
        self.dir = Path(folder)
        self.luts = self.dir / "luts"
        self.luts.mkdir(parents=True, exist_ok=True)

    # LUTs: ids are 12 characters; bundled ones are named from their file
    def _bundled_luts(self):
        out = {}
        for f in sorted(BUNDLED.glob("*")) if BUNDLED.exists() else []:
            if f.suffix.lower() in (".cube", ".png") and f.is_file():
                out["b" + hashlib.sha1(f.name.encode()).hexdigest()[:11]] = f
        return out

    def lut_list(self):
        out = []
        for lid, f in self._bundled_luts().items():
            out.append({"id": lid, "name": f.stem.replace("_", " "), "source": "bundled"})
        for f in sorted(self.luts.glob("*.json"), key=lambda p: p.stat().st_mtime):
            meta = json.loads(f.read_text())
            out.append({"id": f.stem, "name": meta.get("name", f.stem), "size": meta.get("size"), "source": "yours"})
        return out

    def lut_text(self, lid):
        if not re.fullmatch(r"[0-9a-z]{12}", str(lid)):
            return None
        bundled = self._bundled_luts()
        if lid in bundled:
            f = bundled[lid]
            return read_lut(f.name, f.read_bytes())[1]
        f = self.luts / ("%s.cube" % lid)
        return f.read_text() if f.exists() else None

    def add_lut(self, name, raw):
        size, text = read_lut(name, raw)
        # named by the table alone (not its title): the same LUT imported twice is kept once
        lid = hashlib.sha1(text.split("\n", 1)[1].encode()).hexdigest()[:12]
        (self.luts / ("%s.cube" % lid)).write_text(text)
        title = Path(name).stem.replace("_", " ")[:80] or "LUT"
        (self.luts / ("%s.json" % lid)).write_text(json.dumps({"name": title, "size": size}))
        return {"id": lid, "name": title, "size": size, "source": "yours"}

    def delete_lut(self, lid):
        if not re.fullmatch(r"[0-9a-f]{12}", str(lid)):
            return False
        gone = False
        for ext in (".cube", ".json"):
            f = self.luts / (lid + ext)
            if f.exists():
                f.unlink()
                gone = True
        return gone

    # looks
    def look_list(self):
        out = []
        for f in sorted(BUNDLED.glob("*.json")) if BUNDLED.exists() else []:
            try:
                lk = json.loads(f.read_text())
                out.append({"id": "b" + hashlib.sha1(f.name.encode()).hexdigest()[:11], "name": lk.get("name", f.stem),
                            "settings": clean_settings(lk.get("settings")), "source": "bundled"})
            except (ValueError, OSError):
                continue
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            lk = json.loads(f.read_text())
            out.append({"id": f.stem, "name": lk.get("name", f.stem), "settings": clean_settings(lk.get("settings")),
                        "source": "yours"})
        return out

    def save_look(self, name, settings, lid=None):
        lid = lid if lid and re.fullmatch(r"[0-9a-f]{12}", str(lid)) else hashlib.sha1(
            (str(name) + json.dumps(settings, sort_keys=True) + str(np.random.random())).encode()).hexdigest()[:12]
        name = (str(name).strip() or "My look")[:80]
        (self.dir / ("%s.json" % lid)).write_text(json.dumps({"name": name, "settings": clean_settings(settings)}))
        return {"id": lid, "name": name, "settings": clean_settings(settings), "source": "yours"}

    def delete_look(self, lid):
        if not re.fullmatch(r"[0-9a-f]{12}", str(lid)):
            return False
        f = self.dir / ("%s.json" % lid)
        if f.exists():
            f.unlink()
            return True
        return False

    def import_file(self, name, raw):
        """A LUT (.cube, Hald .png) or a look downloaded from here (.json, its LUT inside)."""
        if name.lower().endswith(".json"):
            try:
                lk = json.loads(raw.decode("utf-8"))
            except ValueError:
                raise ValueError("%s is not a look file" % name)
            settings = dict(lk.get("settings") or {})
            lut = None
            if lk.get("lut_cube"):
                lut = self.add_lut((lk.get("lut_name") or lk.get("name") or "LUT") + ".cube", lk["lut_cube"].encode())
                settings["lut"] = lut["id"]
            else:
                settings["lut"] = None
            return {"look": self.save_look(lk.get("name") or Path(name).stem, settings), "lut": lut}
        return {"lut": self.add_lut(name, raw)}
