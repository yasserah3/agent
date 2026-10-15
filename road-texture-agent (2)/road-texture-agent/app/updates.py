"""Per-generation settings snapshots and an append-only texture update history."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

MARKING_ENGINE = '2026.10.12-placement4'


def paths(root, gid):
    return Path(root) / f"gen_{gid}_settings.json", Path(root) / f"gen_{gid}_updates.jsonl"


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temporary, path)


def differences(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        out = []
        for key in sorted(before.keys() | after.keys()):
            out += differences(before.get(key), after.get(key), prefix + '.' + key if prefix else key)
        return out
    return [] if before == after else [{'field': prefix, 'before': before, 'after': after}]


def initialise(root, gid, settings):
    snapshot, history = paths(root, gid)
    write(snapshot, {'generation': gid, 'revision': 0, 'settings': settings, 'marking_engine': MARKING_ENGINE})
    history.write_text(json.dumps({'revision': 0, 'at': datetime.now(timezone.utc).isoformat(),
                                  'action': 'generated', 'changes': [], 'parts': []}) + '\n', encoding='utf-8')


def load(root, gid):
    snapshot, _ = paths(root, gid)
    return json.loads(snapshot.read_text(encoding='utf-8')) if snapshot.exists() else None


def commit(root, gid, settings, changes, parts):
    previous = load(root, gid)
    revision = (previous or {}).get('revision', 0) + 1
    event = {'revision': revision, 'at': datetime.now(timezone.utc).isoformat(),
             'action': 'updated', 'changes': changes, 'parts': sorted(set(parts))}
    snapshot, history = paths(root, gid)
    with history.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(event, sort_keys=True) + '\n')
    write(snapshot, {'generation': gid, 'revision': revision, 'settings': settings, 'marking_engine': MARKING_ENGINE})
    return event


def since(root, gid, revision=-1):
    _, history = paths(root, gid)
    if not history.exists():
        return []
    return [event for line in history.read_text(encoding='utf-8').splitlines()
            if line and (event := json.loads(line))['revision'] > revision]
