"""
The agent's memory.

Three things live here, in one SQLite file:

1. steps      - an append-only log of everything that happens, in order.
                Nothing is ever deleted or overwritten. This is the record you
                asked for: every upload, every analysis, every decision, every
                correction, with its reason.

2. situations - the keyed lookup. A key is a short signature of a situation
   + attempts  (for example "material | open road | 3.2 m from edge | no junction").
                Under that key sits every route that was tried there, whether it
                worked, and why it failed. Before acting, the agent looks up the
                key and skips routes already known to fail, so the same mistake
                is not repeated. Lookup cost does not grow as the agent learns.

3. trees      - the current decision trees, rebuilt from the log.

The log is the source of truth. Situations and trees can always be rebuilt from
it, so nothing learned can be lost.

Files are recorded by their place in the workspace (the folder holding this
database), such as artifacts/tile_x.png, not by a full path, so the workspace
can be moved or copied to another computer and still find everything. A full
path recorded before (or on another computer) is found again under this
workspace by the part from its workspace folder on (artifacts, uploads ...);
on start the database's full paths are rewritten that way once.
"""

import json
import re
import sqlite3
import time
from pathlib import Path

_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])")      # a full path, Windows or not
# the workspace's own folders: a full path recorded elsewhere is found again from one of these on
FOLDERS = ("artifacts", "uploads", "objects", "packages", "looks", "skies", "decals", "errors")

SCHEMA = """
CREATE TABLE IF NOT EXISTS steps (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS steps_kind ON steps(kind);

CREATE TABLE IF NOT EXISTS images (
    id       TEXT PRIMARY KEY,
    role     TEXT NOT NULL,
    name     TEXT NOT NULL,
    path     TEXT NOT NULL,
    width    INTEGER NOT NULL,
    height   INTEGER NOT NULL,
    sha256   TEXT NOT NULL,
    ts       REAL NOT NULL,
    meta     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS situations (
    key        TEXT PRIMARY KEY,
    tree       TEXT NOT NULL,
    features   TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen  REAL NOT NULL,
    seen       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS attempts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    key      TEXT NOT NULL,
    route    TEXT NOT NULL,
    outcome  TEXT NOT NULL,           -- accepted | rejected | flagged
    reason   TEXT,
    ts       REAL NOT NULL,
    step_id  INTEGER,
    FOREIGN KEY(key) REFERENCES situations(key)
);
CREATE INDEX IF NOT EXISTS attempts_key ON attempts(key);

CREATE TABLE IF NOT EXISTS trees (
    name    TEXT PRIMARY KEY,
    data    TEXT NOT NULL,
    updated REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tree_versions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    version INTEGER NOT NULL,
    source  TEXT NOT NULL,
    data    TEXT NOT NULL,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tree_versions_name ON tree_versions(name);

CREATE TABLE IF NOT EXISTS trained_pairs (
    id          TEXT PRIMARY KEY,
    mask_id     TEXT NOT NULL,
    photo_id    TEXT NOT NULL,
    mask_name   TEXT NOT NULL,
    photo_name  TEXT NOT NULL,
    scale       REAL NOT NULL,
    road_px     INTEGER NOT NULL,
    junctions   INTEGER NOT NULL,
    groups      TEXT NOT NULL,
    patch_files TEXT NOT NULL,
    created     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS library_index (
    part       TEXT NOT NULL,
    pair_id    TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    pixels     INTEGER NOT NULL,
    patches    INTEGER NOT NULL,
    size_px    INTEGER NOT NULL,
    confidence REAL NOT NULL,
    weight     REAL NOT NULL,
    npz        TEXT NOT NULL,
    PRIMARY KEY (part, pair_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id      TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,
    path    TEXT NOT NULL,
    ts      REAL NOT NULL,
    meta    TEXT NOT NULL
);
"""


class Memory:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.path.parent.resolve()           # the workspace
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self.moved = self._portable()

    # ---------------------------------------------------------------- paths
    def rel(self, p):
        """A path as recorded: its place in the workspace (forward slashes) when it lies in it, else as given."""
        if p is None or p == "":
            return p
        s = str(p)
        try:
            return Path(s).resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return s

    def resolve(self, p):
        """
        A recorded path as a full path on this computer: a place in the
        workspace under it; a full path recorded on another computer, or before
        the workspace moved, found again under this workspace by the part from
        its workspace folder on (artifacts/..., uploads/...).
        """
        if p is None or p == "":
            return p
        s = str(p)
        parts = [x for x in re.split(r"[\\/]+", s) if x]
        if not _ABS.match(s):
            return str(self.root.joinpath(*parts)) if parts else str(self.root)
        if Path(s).exists():
            return s
        for i in range(len(parts) - 2, -1, -1):          # the last workspace folder in it, not the file itself
            if parts[i] in FOLDERS:
                return str(self.root.joinpath(*parts[i:]))
        return s

    def _meta_out(self, meta):
        """Meta as recorded: its full paths in the workspace made places in it."""
        return {k: (self.rel(v) if isinstance(v, str) and _ABS.match(v) else v) for k, v in (meta or {}).items()}

    def _meta_in(self, meta):
        """Meta as read: its paths (full, or places in the workspace under a path key) made full paths here."""
        out = {}
        for k, v in meta.items():
            if isinstance(v, str) and v and (_ABS.match(v) or ((k == "npz" or k == "nomark" or k.endswith("path"))
                                                                and re.match(r"(?:%s)[\\/]" % "|".join(FOLDERS), v))):
                v = self.resolve(v)
            out[k] = v
        return out

    def _portable(self):
        """
        Once, on start: every full path in the database that lies in this
        workspace (or did, on another computer) becomes its place in it.
        Returns how many were rewritten.
        """
        n = 0

        def fix(v):
            if not isinstance(v, str) or not _ABS.match(v):
                return v
            r = self.rel(self.resolve(v))
            return v if _ABS.match(r) else r
        for table, key, col in (("images", "id", "path"), ("artifacts", "id", "path")):
            for row in self.db.execute(f"SELECT {key}, {col} FROM {table}").fetchall():
                new = fix(row[col])
                if new != row[col]:
                    self.db.execute(f"UPDATE {table} SET {col}=? WHERE {key}=?", (new, row[key])); n += 1
        for row in self.db.execute("SELECT part, pair_id, npz FROM library_index").fetchall():
            new = fix(row["npz"])
            if new != row["npz"]:
                self.db.execute("UPDATE library_index SET npz=? WHERE part=? AND pair_id=?",
                                (new, row["part"], row["pair_id"])); n += 1
        for row in self.db.execute("SELECT id, patch_files FROM trained_pairs").fetchall():
            pf = json.loads(row["patch_files"])
            new = {k: fix(v) for k, v in pf.items()}
            if new != pf:
                self.db.execute("UPDATE trained_pairs SET patch_files=? WHERE id=?", (json.dumps(new), row["id"])); n += 1
        for row in self.db.execute("SELECT id, meta FROM artifacts").fetchall():
            m = json.loads(row["meta"])
            new = {k: fix(v) for k, v in m.items()}
            if new != m:
                self.db.execute("UPDATE artifacts SET meta=? WHERE id=?", (json.dumps(new), row["id"])); n += 1
        self.db.commit()
        return n

    def _migrate(self):
        """Add columns introduced after a database was created, keeping its data."""
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(library_index)")}
        for col, ddl in (("rejections", "INTEGER NOT NULL DEFAULT 0"),
                         ("acceptances", "INTEGER NOT NULL DEFAULT 0"),
                         ("cooldown", "INTEGER NOT NULL DEFAULT 0")):
            if col not in have:
                self.db.execute(f"ALTER TABLE library_index ADD COLUMN {col} {ddl}")

    # ---------------------------------------------------------------- steps
    def record(self, kind: str, summary: str, payload: dict | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO steps (ts, kind, summary, payload) VALUES (?,?,?,?)",
            (time.time(), kind, summary, json.dumps(payload or {})),
        )
        self.db.commit()
        return cur.lastrowid

    def steps(self, limit: int = 200, kind: str | None = None):
        if kind:
            rows = self.db.execute(
                "SELECT * FROM steps WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM steps ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    # ---------------------------------------------------------------- images
    def add_image(self, image_id, role, name, path, width, height, sha256, meta=None):
        self.db.execute(
            "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?)",
            (image_id, role, name, self.rel(path), width, height, sha256, time.time(),
             json.dumps(meta or {})),
        )
        self.db.commit()

    def image(self, image_id):
        r = self.db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return dict(r) | {"path": self.resolve(r["path"]), "meta": json.loads(r["meta"])} if r else None

    def images(self, role=None):
        rows = (self.db.execute("SELECT * FROM images WHERE role=? ORDER BY ts DESC", (role,))
                if role else
                self.db.execute("SELECT * FROM images ORDER BY ts DESC")).fetchall()
        return [dict(r) | {"path": self.resolve(r["path"]), "meta": json.loads(r["meta"])} for r in rows]

    # ------------------------------------------------------- situations (KV)
    def touch_situation(self, key: str, tree: str, features: dict):
        now = time.time()
        row = self.db.execute("SELECT key FROM situations WHERE key=?", (key,)).fetchone()
        if row:
            self.db.execute(
                "UPDATE situations SET last_seen=?, seen=seen+1 WHERE key=?", (now, key)
            )
        else:
            self.db.execute(
                "INSERT INTO situations (key, tree, features, first_seen, last_seen, seen)"
                " VALUES (?,?,?,?,?,1)",
                (key, tree, json.dumps(features), now, now),
            )
        self.db.commit()

    def add_attempt(self, key, route, outcome, reason=None, step_id=None):
        self.db.execute(
            "INSERT INTO attempts (key, route, outcome, reason, ts, step_id) VALUES (?,?,?,?,?,?)",
            (key, route, outcome, reason, time.time(), step_id),
        )
        self.db.commit()

    def recall(self, key: str) -> dict:
        """Everything already known about this exact situation."""
        rows = self.db.execute(
            "SELECT route, outcome, reason, ts FROM attempts WHERE key=? ORDER BY id", (key,)
        ).fetchall()
        tried, rejected, accepted = [], [], []
        for r in rows:
            if r["route"] not in tried:
                tried.append(r["route"])
            if r["outcome"] == "rejected":
                rejected.append({"route": r["route"], "reason": r["reason"]})
            elif r["outcome"] == "accepted":
                accepted.append(r["route"])
        return {"key": key, "tried": tried, "rejected": rejected,
                "accepted": accepted, "attempts": [dict(r) for r in rows]}

    def situation_count(self):
        return self.db.execute("SELECT COUNT(*) c FROM situations").fetchone()["c"]

    # ---------------------------------------------------------------- trees
    def save_tree(self, name, data, source="unknown"):
        """
        Saves a tree and keeps every earlier version.

        Nothing learned is ever thrown away. Stage B adds to what priming
        created; if a later run makes things worse, the earlier version is still
        there and can be restored.
        """
        row = self.db.execute(
            "SELECT MAX(version) v FROM tree_versions WHERE name=?", (name,)).fetchone()
        version = (row["v"] or 0) + 1
        now = time.time()
        self.db.execute(
            "INSERT INTO tree_versions (name, version, source, data, ts) VALUES (?,?,?,?,?)",
            (name, version, source, json.dumps(data), now))
        self.db.execute("INSERT OR REPLACE INTO trees VALUES (?,?,?)",
                        (name, json.dumps(data), now))
        self.db.commit()
        return version

    def tree_history(self, name=None):
        rows = (self.db.execute(
            "SELECT id, name, version, source, ts FROM tree_versions WHERE name=? ORDER BY version",
            (name,)) if name else self.db.execute(
            "SELECT id, name, version, source, ts FROM tree_versions ORDER BY name, version")).fetchall()
        return [dict(r) for r in rows]

    def tree_version(self, name, version):
        r = self.db.execute("SELECT data FROM tree_versions WHERE name=? AND version=?",
                            (name, version)).fetchone()
        return json.loads(r["data"]) if r else None

    def restore_tree(self, name, version):
        data = self.tree_version(name, version)
        if data is None:
            return None
        self.save_tree(name, data, source=f"restored from version {version}")
        return data

    def tree(self, name):
        r = self.db.execute("SELECT * FROM trees WHERE name=?", (name,)).fetchone()
        return json.loads(r["data"]) if r else None

    def tree_names(self):
        return [r["name"] for r in self.db.execute("SELECT name FROM trees").fetchall()]

    # -------------------------------------------------------- trained pairs
    def add_trained_pair(self, pair_id, mask, photo, scale, road_px, junctions,
                         groups, patch_files):
        self.db.execute(
            "INSERT OR REPLACE INTO trained_pairs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (pair_id, mask["id"], photo["id"], mask["name"], photo["name"], scale,
             road_px, junctions, json.dumps(groups), json.dumps({k: self.rel(v) for k, v in patch_files.items()}),
             time.time()))
        self.db.commit()

    def _patch_files(self, raw):
        return {k: self.resolve(v) for k, v in json.loads(raw).items()}

    def trained_pairs(self):
        rows = self.db.execute("SELECT * FROM trained_pairs ORDER BY created").fetchall()
        return [dict(r) | {"groups": json.loads(r["groups"]),
                           "patch_files": self._patch_files(r["patch_files"])} for r in rows]

    def trained_pair(self, pair_id):
        r = self.db.execute("SELECT * FROM trained_pairs WHERE id=?", (pair_id,)).fetchone()
        return (dict(r) | {"groups": json.loads(r["groups"]),
                           "patch_files": self._patch_files(r["patch_files"])}) if r else None

    def delete_trained_pair(self, pair_id):
        rec = self.trained_pair(pair_id)
        if not rec:
            return None
        self.db.execute("DELETE FROM trained_pairs WHERE id=?", (pair_id,))
        self.db.commit()
        return rec

    def find_trained_pair(self, mask_id, photo_id, scale):
        r = self.db.execute(
            "SELECT id FROM trained_pairs WHERE mask_id=? AND photo_id=? AND abs(scale-?)<1e-6",
            (mask_id, photo_id, scale)).fetchone()
        return r["id"] if r else None

    # -------------------------------------------------------- library index
    # One ordered list per part of the road (junction, edge, open). Each row is
    # one pair's library for that part. The agent reads it top to bottom: the
    # first usable, not-rejected row is used, the next one only if that fails.
    # Reordering after a correction rewrites this list, never the patches.

    def set_index(self, part, rows, keep_confidence=True):
        # a rebuild must not wipe what corrections taught: keep each pair's
        # confidence, its counts, and whether it is sitting out a generation
        old = {r["pair_id"]: r for r in self.index(part)} if keep_confidence else {}
        self.db.execute("DELETE FROM library_index WHERE part=?", (part,))
        for r in rows:
            o = old.get(r["pair_id"], {})
            conf = o.get("confidence", r.get("confidence", 0.6))
            self.db.execute(
                "INSERT INTO library_index (part, pair_id, rank, pixels, patches, size_px, "
                "confidence, weight, npz, rejections, acceptances, cooldown) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (part, r["pair_id"], 0, r["pixels"], r["patches"], r["size_px"],
                 conf, r["pixels"] * conf, self.rel(r["npz"]),
                 o.get("rejections", 0), o.get("acceptances", 0), o.get("cooldown", 0)))
        self._rerank(part)
        self.db.commit()

    def _rerank(self, part):
        rows = self.db.execute(
            "SELECT pair_id FROM library_index WHERE part=? ORDER BY weight DESC, pixels DESC",
            (part,)).fetchall()
        for i, r in enumerate(rows, start=1):
            self.db.execute("UPDATE library_index SET rank=? WHERE part=? AND pair_id=?",
                            (i, part, r["pair_id"]))

    def index(self, part):
        rows = self.db.execute(
            "SELECT * FROM library_index WHERE part=? ORDER BY rank", (part,)).fetchall()
        return [dict(r) | {"npz": self.resolve(r["npz"])} for r in rows]

    def reject_pair(self, part, pair_id):
        """
        Halve the confidence, and sit out the next generation.

        Halving means the drop follows the evidence: a pair only slightly ahead
        falls after one rejection, one with far more road needs two or three.
        Sitting out one generation is what lets you see the next library at
        once; after that the order decides again.
        """
        r = self.db.execute("SELECT confidence, pixels FROM library_index WHERE part=? AND pair_id=?",
                            (part, pair_id)).fetchone()
        if not r:
            return None
        conf = round(max(0.01, r["confidence"] * 0.5), 4)
        self.db.execute("UPDATE library_index SET confidence=?, weight=?, "
                        "rejections=rejections+1, cooldown=1 WHERE part=? AND pair_id=?",
                        (conf, r["pixels"] * conf, part, pair_id))
        self._rerank(part)
        self.db.commit()
        return conf

    def accept_pair(self, part, pair_id):
        r = self.db.execute("SELECT confidence, pixels FROM library_index WHERE part=? AND pair_id=?",
                            (part, pair_id)).fetchone()
        if not r:
            return None
        conf = round(min(0.98, r["confidence"] + 0.05), 4)
        self.db.execute("UPDATE library_index SET confidence=?, weight=?, "
                        "acceptances=acceptances+1, cooldown=0 WHERE part=? AND pair_id=?",
                        (conf, r["pixels"] * conf, part, pair_id))
        self._rerank(part)
        self.db.commit()
        return conf

    def tick_cooldowns(self, part):
        """A generation has used this part: whoever sat it out is back in."""
        self.db.execute("UPDATE library_index SET cooldown=MAX(0, cooldown-1) WHERE part=?", (part,))
        self.db.commit()

    def drop_from_index(self, pair_id):
        parts = [r["part"] for r in self.db.execute(
            "SELECT DISTINCT part FROM library_index WHERE pair_id=?", (pair_id,)).fetchall()]
        self.db.execute("DELETE FROM library_index WHERE pair_id=?", (pair_id,))
        for part in parts:
            self._rerank(part)
        self.db.commit()

    # ------------------------------------------------------------ artifacts
    def add_artifact(self, artifact_id, kind, path, meta=None):
        self.db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?)",
                        (artifact_id, kind, self.rel(path), time.time(), json.dumps(self._meta_out(meta))))
        self.db.commit()

    def _artifact(self, r):
        return dict(r) | {"path": self.resolve(r["path"]), "meta": self._meta_in(json.loads(r["meta"]))} if r else None

    def latest_artifact(self, kind):
        return self._artifact(self.db.execute("SELECT * FROM artifacts WHERE kind=? ORDER BY ts DESC LIMIT 1",
                                              (kind,)).fetchone())

    def artifact(self, artifact_id):
        return self._artifact(self.db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone())

    # ------------------------------------------------------------- summary
    def summary(self):
        c = self.db.execute
        return {
            "steps": c("SELECT COUNT(*) v FROM steps").fetchone()["v"],
            "images": c("SELECT COUNT(*) v FROM images").fetchone()["v"],
            "situations": c("SELECT COUNT(*) v FROM situations").fetchone()["v"],
            "attempts": c("SELECT COUNT(*) v FROM attempts").fetchone()["v"],
            "rejected": c("SELECT COUNT(*) v FROM attempts WHERE outcome='rejected'").fetchone()["v"],
            "accepted": c("SELECT COUNT(*) v FROM attempts WHERE outcome='accepted'").fetchone()["v"],
            "flagged": c("SELECT COUNT(*) v FROM attempts WHERE outcome='flagged'").fetchone()["v"],
            "trees": self.tree_names(),
            "trained_pairs": self.db.execute(
                "SELECT COUNT(*) v FROM trained_pairs").fetchone()["v"],
            "tree_versions": self.db.execute(
                "SELECT COUNT(*) v FROM tree_versions").fetchone()["v"],
            "tree_versions": len(self.tree_history()),
        }


def situation_key(tree: str, features: dict) -> str:
    """
    Turns a set of measurements into a short, stable key.

    Values are bucketed on purpose: two spots 3.1 m and 3.3 m from the edge are
    the same situation, and should share what was learned. Without bucketing,
    every pixel would be its own key and the agent would never recognise that it
    had seen this case before.
    """
    parts = [tree]
    for name in sorted(features):
        v = features[name]
        if isinstance(v, bool):
            parts.append(f"{name}={'y' if v else 'n'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{name}={_bucket(name, float(v))}")
        else:
            parts.append(f"{name}={v}")
    return "|".join(parts)


def _bucket(name: str, v: float) -> str:
    if name.endswith("_m"):                      # metres: 0.5 m steps
        return f"{round(v * 2) / 2:.1f}"
    if name.endswith("_px"):                     # pixels: 2 px steps
        return str(int(round(v / 2) * 2))
    if 0.0 <= v <= 1.0:                          # ratios: tenths
        return f"{round(v, 1):.1f}"
    return str(int(round(v)))
