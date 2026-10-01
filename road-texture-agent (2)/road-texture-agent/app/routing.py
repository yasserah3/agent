"""
Choosing which route to use, and remembering what was rejected.

This is the piece that makes a correction stick. Before generation paints a
part of the road, it builds the key for that situation and asks memory what has
already been tried there. A route you rejected for that situation is skipped,
and the next untried one is used instead. It is never offered again unless you
restore it.
"""

from app.memory import situation_key

TREE_FOR = {"material": "material", "wear": "noise"}


def index_routes(tree: dict) -> dict:
    """Every route in a tree, by id, including the primed fallbacks."""
    found = {}

    def walk(node):
        if not isinstance(node, dict):
            return
        if "id" in node and "action" in node:
            found[node["id"]] = node
            primed = node.get("primed_fallback")
            if isinstance(primed, dict):
                found[primed["id"]] = primed
            return
        for child in (node.get("children") or {}).values():
            walk(child)

    walk(tree)
    return found


def features_for(group: str, junction_type: str = None, edge_m: float = None) -> dict:
    """
    What the agent knows about this spot, at the granularity it can act on.

    The key is the part of the road, and for a junction its type. The exact
    distance from the kerb is deliberately left out: generation chooses one
    route per part of the road, so a key holding a per-pixel distance would
    never match the key generation asks under, and a correction would silently
    do nothing. That was a real bug, caught in testing.
    """
    f = {"part": group, "junction": group == "junction"}
    if group == "junction" and junction_type:
        f["type"] = junction_type.replace(" ", "_").replace("(", "").replace(")", "")
    return f


def candidates_for(route: dict, routes: dict) -> list:
    """The route to try first, then its fallbacks, in order."""
    out, seen = [], set()
    for rid in [route["id"]] + list(route.get("fallbacks") or []):
        if rid in seen:
            continue
        seen.add(rid)
        r = routes.get(rid)
        if r:
            out.append(r)
    primed = route.get("primed_fallback")
    if isinstance(primed, dict) and primed["id"] not in seen:
        out.append(primed)
    return out


def choose(mem, tree: dict, tree_name: str, group: str,
           junction_type: str = None, edge_m: float = None) -> dict:
    """
    Pick the route to use here, skipping anything rejected for this situation.

    Returns the chosen route, the key it was chosen under, and the full list of
    candidates with what memory knows about each.
    """
    routes = index_routes(tree)
    base = _route_for_group(tree, tree_name, group)
    if base is None:
        return {"key": None, "route": None, "candidates": [], "recall": None}

    features = features_for(group, junction_type, edge_m)
    key = situation_key(tree_name, features)
    recall = mem.recall(key)
    rejected = {r["route"] for r in recall["rejected"]}

    cands = candidates_for(base, routes)
    chosen = next((c for c in cands if c["id"] not in rejected), None)
    exhausted = chosen is None
    if exhausted:
        chosen = cands[0] if cands else base       # nothing left: keep the first

    return {
        "key": key, "features": features, "route": chosen, "exhausted": exhausted,
        "candidates": [{"id": c["id"], "confidence": c.get("confidence"),
                        "rejected": c["id"] in rejected,
                        "chosen": c["id"] == chosen["id"],
                        "lesson": c.get("lesson")} for c in cands],
        "recall": recall,
    }


def _route_for_group(tree: dict, tree_name: str, group: str):
    if tree_name == "material":
        if group == "junction":
            return tree["children"]["yes"]
        branch = tree["children"]["no"]["children"]
        return branch["under 1.5 m"] if group == "edge" else branch["1.5 m or more"]
    if tree_name == "noise":
        return tree["children"]["yes"] if group == "junction" else tree["children"]["no"]
    return None


def record_outcome(mem, trees: dict, tree_name: str, key: str, features: dict,
                   route_id: str, outcome: str, reason: str = None):
    """
    Write the decision to memory and update the route's own record.

    Accepting raises the route's confidence a little; rejecting lowers it. Both
    are recorded in the step log with the reason you gave, so the tree can
    always explain itself later.
    """
    mem.touch_situation(key, tree_name, features)
    step = mem.record("review", f"{route_id} {outcome} at {key}"
                      + (f": {reason}" if reason else ""),
                      {"key": key, "route": route_id, "outcome": outcome,
                       "reason": reason, "features": features, "tree": tree_name})
    mem.add_attempt(key, route_id, outcome, reason, step)

    tree = trees.get(tree_name)
    if tree:
        routes = index_routes(tree)
        r = routes.get(route_id)
        if r is not None:
            r["visits"] = r.get("visits", 0) + 1
            if outcome == "accepted":
                r["successes"] = r.get("successes", 0) + 1
                r["confidence"] = round(min(0.98, r.get("confidence", 0.5) + 0.05), 2)
            elif outcome == "rejected":
                r["failures"] = r.get("failures", 0) + 1
                r["confidence"] = round(max(0.05, r.get("confidence", 0.5) - 0.1), 2)
            notes = r.setdefault("notes", [])
            if reason:
                notes.append(f"{outcome} at {key}: {reason}")
            mem.save_tree(tree_name, tree, source=f"review: {route_id} {outcome}")
    return mem.recall(key)


# ---------------------------------------------------------------------------
# Choosing from the ordered library index.
#
# Each part of the road (junction, kerb, open road) has one ordered list of
# pairs. The agent reads it top to bottom and uses the first pair that has
# enough patches and has not been rejected for this situation. Only if that
# pair is rejected does it move to the next. If every pair is rejected, the
# library from priming is the last resort.
# ---------------------------------------------------------------------------

MIN_USABLE = 6


def pair_route_id(part, pair_id):
    return f"material.{part}@{pair_id}"


def choose_pair(mem, trees, part, junction_type=None, prefer_pair=None, labels=None):
    """
    prefer_pair keeps a street consistent: once open road has picked a pair,
    the kerb and junction try that same pair first, so the kerb and the middle
    of a road come from the same photo.
    """
    labels = labels or {}
    features = features_for(part, junction_type)
    key = situation_key("material", features)
    recall = mem.recall(key)
    rejected = {r["route"] for r in recall["rejected"]}

    rows = [r for r in mem.index(part) if r["patches"] >= MIN_USABLE]
    cands = []
    for r in rows:
        rid = pair_route_id(part, r["pair_id"])
        # a pair is skipped only while it sits out the generation after being
        # rejected; after that its weight, lowered by the rejection, decides
        cands.append({"id": rid, "pair_id": r["pair_id"], "rank": r["rank"],
                      "confidence": r["confidence"], "weight": round(r["weight"]),
                      "patches": r["patches"], "size_px": r["size_px"], "npz": r["npz"],
                      "rejections": r.get("rejections", 0),
                      "acceptances": r.get("acceptances", 0),
                      "sitting_out": r.get("cooldown", 0) > 0,
                      "label": labels.get(r["pair_id"], r["pair_id"]),
                      "rejected": r.get("cooldown", 0) > 0})

    # the library from priming, always last
    tree = trees.get("material")
    base = _route_for_group(tree, "material", part) if tree else None
    primed = None
    if base is not None:
        primed = base.get("primed_fallback") or (base if str(base.get("source", "")).startswith("stage A") else None)
    if primed:
        lib = (primed.get("params") or {}).get("library")
        art = mem.artifact(lib) if lib else None
        pid = f"material.{part}.primed"
        cands.append({"id": pid, "pair_id": None, "rank": len(cands) + 1,
                      "confidence": primed.get("confidence", 0.5), "weight": 0,
                      "patches": (art or {}).get("meta", {}).get("count", 0),
                      "size_px": (art or {}).get("meta", {}).get("size_px"),
                      "npz": (art or {}).get("meta", {}).get("npz"),
                      "label": "from priming", "rejected": pid in rejected})

    chosen = None
    if prefer_pair:
        chosen = next((c for c in cands if c["pair_id"] == prefer_pair and not c["rejected"]), None)
    if chosen is None:
        chosen = next((c for c in cands if not c["rejected"]), None)
    exhausted = chosen is None
    if exhausted and cands:
        chosen = cands[0]
    for c in cands:
        c["chosen"] = chosen is not None and c["id"] == chosen["id"]

    return {"key": key, "features": features, "part": part, "chosen": chosen,
            "candidates": cands, "exhausted": exhausted, "recall": recall}


def record_pair_outcome(mem, part, key, features, route_id, outcome, reason=None):
    """A correction on a pair's look moves that pair in the order for this part."""
    mem.touch_situation(key, "material", features)
    step = mem.record("review", f"{route_id} {outcome} at {key}" + (f": {reason}" if reason else ""),
                      {"key": key, "route": route_id, "outcome": outcome, "reason": reason,
                       "features": features, "part": part})
    mem.add_attempt(key, route_id, outcome, reason, step)
    moved = None
    if "@" in route_id and outcome in ("accepted", "rejected"):
        pair_id = route_id.split("@", 1)[1]
        before = [r["pair_id"] for r in mem.index(part)]
        conf = (mem.reject_pair if outcome == "rejected" else mem.accept_pair)(part, pair_id)
        rows = mem.index(part)
        after = [r["pair_id"] for r in rows]
        row = next((r for r in rows if r["pair_id"] == pair_id), {})
        moved = {"pair_id": pair_id, "confidence": conf,
                 "rejections": row.get("rejections", 0),
                 "rank_before": before.index(pair_id) + 1 if pair_id in before else None,
                 "rank_after": after.index(pair_id) + 1 if pair_id in after else None}
    return {"recall": mem.recall(key), "moved": moved}
