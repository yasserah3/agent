"""
The speed-up files of the textures, let go by themselves so the workspace does not grow.

Each Generate texture keeps two files in workspace/artifacts that only Update 2D
texture on that same texture reads, to skip making them again:
  - gen_<id>_geometry.cache: the mask cleaned, its junctions and streets (tens to
    hundreds of MB, more on big maps);
  - gen_<id>_surface.npz: the material and wear before their settings.
Without them Update 2D texture makes them again, the same (the same seed lays the same
patches), only slower. Nothing else reads them: not the training (the libraries and
trees), not the satellite analysis's confidence, not the dataset.

They are deleted:
  - when the server starts: the session before's;
  - when a texture is made: every other texture's;
  - when another mask is loaded (an alpha map, coordinates, another street): those of
    textures of other masks;
  - an hour after they were last used (made or read).
A file written in the last two minutes is left (a texture being made just now).
"""

import os
import re
import shutil
import threading
import time
from pathlib import Path

KINDS = ("geometry.cache", "surface.npz")
IDLE_S = 3600.0               # unused this long: deleted
FRESH_S = 120.0               # written this recently: left alone (still being made)
SWEEP_S = 600.0               # how often the idle ones are looked for
_NAME = re.compile(r"^gen_([0-9a-f]{12})_(geometry\.cache|surface\.npz)$")
_lock = threading.Lock()


def files(artifacts):
    """(generation id, path) of every speed-up file in the artifacts folder."""
    try:
        entries = list(os.scandir(artifacts))
    except OSError:
        return []
    out = []
    for e in entries:
        m = _NAME.match(e.name)
        if m and e.is_file():
            out.append((m.group(1), Path(e.path)))
    return out


def used(path):
    """Mark a speed-up file as just used (read), so the hour starts again."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def release(artifacts, keep=None, idle=None):
    """
    Delete the speed-up files, except those of the generations in keep (a set of ids, or
    a function id -> bool) and those written in the last FRESH_S. idle: only those unused
    for at least this many seconds. Returns (files deleted, bytes freed).
    """
    now = time.time()
    keep_fn = keep if callable(keep) else (lambda gid, k=set(keep or ()): gid in k)
    n, freed = 0, 0
    with _lock:
        for gid, path in files(artifacts):
            try:
                st = path.stat()
            except OSError:
                continue
            age = now - st.st_mtime
            if age < FRESH_S or (idle is not None and age < idle) or keep_fn(gid):
                continue
            try:
                path.unlink()
            except OSError:              # open elsewhere (Windows): next time
                continue
            n += 1
            freed += st.st_size
    return n, freed


def clear_leftovers(artifacts):
    """Folders a texture made in tiles left behind when it was stopped half way (app/tiled.py)."""
    freed = 0
    tmp = Path(artifacts) / "tiles_tmp"
    if tmp.is_dir():
        for d in tmp.iterdir():
            if d.is_dir() and d.name.startswith("tiles_"):
                freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
    return freed


def start_sweeper(artifacts, interval=SWEEP_S, idle=IDLE_S):
    """In the background: every interval, the speed-up files unused for idle seconds go."""
    def run():
        while True:
            time.sleep(interval)
            try:
                n, freed = release(artifacts, idle=idle)
                if n:
                    print("  Freed %s: %d speed-up file(s) of textures unused for an hour." % (size_text(freed), n))
            except Exception as e:                  # noqa: BLE001 - never stop the server for this
                print("  Speed-up files not swept:", repr(e))
    t = threading.Thread(target=run, daemon=True, name="cache-sweeper")
    t.start()
    return t


def size_text(b):
    return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB" if b >= 1e6 else "under 1 MB"
