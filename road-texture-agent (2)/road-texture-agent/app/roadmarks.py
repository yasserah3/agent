"""
Road markings by the road marking guide: UTTIPEC's "Guidelines for Road
Markings" (Delhi Development Authority, November 2009), which follows IRC 35,
the Indian Roads Congress's code of practice for road markings.

Two things live here:

- the lines' sizes and colours (the Generate tab's Road markings): the centre
  line between the two directions, the lane lines, the edge lines, where
  lines stop before a junction, and highways (app/lanes.py has which line goes
  where): a street at least 18 m wide, 3 lanes each way, whose lines run on
  through the junctions where side streets join it and through T-junctions,
  with no crossing, stop line or arrow laid across it there (the side street
  keeps its own);
- every junction's approaches laid square to their kerbs: a crossing or a stop
  line goes across the street where its kerbs say it runs (decals.kerb_frame),
  not where its centre line, bending into the junction, points;
- the markings the guide draws at junctions and roundabouts, as presets of the
  Decals panel: the pedestrian crossing, the stop line, the give way line, an
  arrow in each lane, the word STOP, the cyclist crossing, and a yellow box
  (not in the guide, said so). Each is drawn from its measurements as a
  picture (so it lies like any decal: in the texture, the Decals layer and the
  3D model) and laid where the junction calls for it: plan() reads every
  junction's streets, their lanes and which way the traffic on them goes, which
  road is the main one, and which junctions are roundabouts (a ring around a
  round island: its entries get a give way line, the ring none) or big open
  junctions (the yellow box).

Every measurement has the guide's value ("guide"; IRC 35's where the guide's
text gives no number, "irc"; ours where neither does, "ours"), a range, and a
group of others it keeps in proportion with ("link"): with the chain on,
changing one sets the others of its group to the guide's values times the same
factor, so the design keeps the guide's proportions. ui/app.js has the same
few lines (chain).
"""
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw

WHITE = (236, 234, 226)          # road paint
YELLOW = (232, 172, 36)          # the guide's yellow
SOURCES = {"guide": "the guide (UTTIPEC, Delhi 2009)", "irc": "IRC 35, the code the guide follows",
           "ours": "not in the guide: our choice"}

# ------------------------------------------------------------------ the lines
# (the centre line between the two directions, the lines between lanes going the
# same way, the edge lines of wide roads, and where lines end before a junction)
LINE_FIELDS = [
    {"key": "centre.dash_m", "group": "centre", "label": "Dash", "guide": 3.0, "min": 0.3, "max": 30.0, "link": "len",
     "src": "guide", "note": "3 m dashes with 4.5 m gaps on two-lane roads"},
    {"key": "centre.gap_m", "group": "centre", "label": "Gap", "guide": 4.5, "min": 0.3, "max": 40.0, "link": "len",
     "src": "guide"},
    {"key": "centre.width_m", "group": "centre", "label": "Width", "guide": 0.15, "min": 0.02, "max": 1.0, "link": "wid",
     "src": "guide", "note": "150 mm"},
    {"key": "double.width_m", "group": "double", "label": "Each line", "guide": 0.10, "min": 0.02, "max": 0.5,
     "link": "wid", "src": "guide", "note": "two 100 mm lines, 100 mm apart"},
    {"key": "double.gap_m", "group": "double", "label": "Between", "guide": 0.10, "min": 0.0, "max": 1.0, "link": "wid",
     "src": "guide"},
    {"key": "lane.dash_m", "group": "lane", "label": "Dash", "guide": 1.5, "min": 0.3, "max": 30.0, "link": "len",
     "src": "guide", "note": "1.5 m dashes with 3 m gaps on four-lane roads"},
    {"key": "lane.gap_m", "group": "lane", "label": "Gap", "guide": 3.0, "min": 0.3, "max": 40.0, "link": "len",
     "src": "guide"},
    {"key": "lane.width_m", "group": "lane", "label": "Width", "guide": 0.10, "min": 0.02, "max": 1.0, "link": "wid",
     "src": "irc", "note": "100 mm"},
    {"key": "edge.width_m", "group": "edge", "label": "Width", "guide": 0.15, "min": 0.02, "max": 1.0, "link": "wid",
     "src": "guide", "note": "150 mm, 200 mm on multi-lane roads without a median"},
    {"key": "edge.inset_m", "group": "edge", "label": "From the kerb", "guide": 0.15, "min": 0.0, "max": 2.0,
     "link": "wid", "src": "guide", "note": "about 150 mm from the edge"},
    {"key": "setback_m", "group": "ends", "label": "Lines end before a junction with no stop line", "guide": 2.0,
     "min": 0.0, "max": 30.0, "src": "ours", "note": "where no preset stop line or crossing sets it"},
    {"key": "highway.min_m", "group": "highway", "label": "From width", "guide": 18.0, "min": 6.0, "max": 200.0,
     "src": "ours", "note": "a street at least this wide is a highway: 18 m holds 3 + 3 lanes of 3 m"},
    {"key": "highway.lanes", "group": "highway", "label": "Lanes each way", "guide": 3, "min": 1, "max": 8,
     "int": True, "src": "ours", "note": "a highway's lanes each way, widened to fill the road; the Lanes tool "
                                         "gives a highway its own"},
]
LINE_OPTS = [
    {"key": "rules", "label": "Lines", "default": "guide",
     "choices": [["guide", "By the road marking guide"], ["simple", "Every line dashed alike (as before)"]]},
    {"key": "drive", "label": "Traffic drives on the", "default": "right",
     "choices": [["right", "Right"], ["left", "Left"]]},
    {"key": "colours", "label": "Colours", "default": "guide",
     "choices": [["guide", "Guide colours (yellow centre line)"], ["white", "All white"]]},
    {"key": "sizes", "label": "Dash and line sizes", "default": "set",
     "choices": [["set", "As set below"], ["learned", "Learned from your training images"]]},
    {"key": "highways", "label": "Highways", "default": "on",
     "choices": [["on", "Lines run on past side streets and T-junctions"],
                 ["off", "None: every junction stops the lines (as before)"]]},
]

# ------------------------------------------------------------------ the presets
EDGE_GAP_M = 0.2               # between a junction's edge and the first marking laid there
NO_CROSSING_M = 1.0            # a stop or give way line with no crossing: this far from the edge (ours)
KERB_MARGIN_M = 0.15           # a line across the road stops this short of the kerb
CENTRE_MARGIN_M = 0.1          # and of the centre line
BIG_M2 = 300.0                 # a big open junction: more than this between its edges (ours)
OPEN_X = 2.0                   # and this many times a plain crossing of its two widest streets (ours)
BOX_ROAD_SHARE = 0.85          # a box only on a junction whose area is road (not one running off it)
CELL_M = 20.0                  # placing: the markings already laid are looked up in cells this big
FRAME_NEAR_M = 12.0            # markings this near a junction's edge are laid square to its kerbs

PRESETS = {
    "zebra": {
        "label": "Pedestrian crossing (zebra)", "short": "Crossing",
        "note": "Across the street at every junction edge, kerb to kerb. At roundabouts a car length back from the "
                "give way line.",
        "fields": [
            {"key": "stripe_m", "label": "Stripe width", "guide": 0.5, "min": 0.1, "max": 2.0, "link": "stripe",
             "src": "guide", "note": "equally spaced white stripes, 500 mm"},
            {"key": "gap_m", "label": "Gap between stripes", "guide": 0.5, "min": 0.1, "max": 2.0, "link": "stripe",
             "src": "guide", "note": "equally spaced: the gap is the stripe's width"},
            {"key": "length_m", "label": "Crossing width (along the road)", "guide": 3.0, "min": 1.0, "max": 10.0,
             "range": [2.0, 4.0], "src": "guide", "note": "between 2 and 4 m, never narrower than the footpath"},
            {"key": "rb_back_m", "label": "At roundabouts: back from the give way line", "guide": 5.0, "min": 0.0,
             "max": 30.0, "src": "ours", "note": "one car length (IRC 65 practice)"},
        ],
        "opts": [{"key": "junctions", "label": "At junctions", "default": True},
                 {"key": "roundabouts", "label": "At roundabout entries", "default": True}],
    },
    "stop": {
        "label": "Stop line", "short": "Stop line",
        "note": "Across the lanes coming into the junction, behind the crossing.",
        "fields": [
            {"key": "width_m", "label": "Line width", "guide": 0.2, "min": 0.05, "max": 1.0, "link": "w",
             "src": "guide", "note": "200 mm"},
            {"key": "gap_m", "label": "Double line: between the lines", "guide": 0.3, "min": 0.05, "max": 1.5,
             "link": "w", "src": "guide", "note": "two 200 mm lines 300 mm apart"},
            {"key": "dist_m", "label": "Before the crossing", "guide": 2.5, "guide_if": {"signals": 1.0}, "min": 0.0,
             "max": 20.0, "range": [2.0, 3.0], "range_if": {"signals": [1.0, 1.0]}, "src": "guide",
             "note": "2 to 3 m at junctions without traffic lights, 1 m with them"},
        ],
        "opts": [{"key": "double", "label": "Double line (junctions with a STOP sign)", "default": False},
                 {"key": "signals", "label": "The junctions have traffic lights", "default": False},
                 {"key": "minor_only", "label": "Only on side streets (where a main road goes through)",
                  "default": False}],
    },
    "giveway": {
        "label": "Give way line", "short": "Give way",
        "note": "Two broken lines across the lanes coming in: at roundabout entries, and on side streets if asked.",
        "fields": [
            {"key": "seg_m", "label": "Dash", "guide": 0.6, "min": 0.1, "max": 3.0, "link": "dash", "src": "guide",
             "note": "600 mm dashes with 300 mm gaps"},
            {"key": "gap_m", "label": "Gap", "guide": 0.3, "min": 0.05, "max": 3.0, "link": "dash", "src": "guide"},
            {"key": "width_m", "label": "Line width", "guide": 0.2, "min": 0.05, "max": 1.0, "link": "w",
             "src": "guide", "note": "200 mm lines, 300 mm apart"},
            {"key": "spacing_m", "label": "Between the two lines", "guide": 0.3, "min": 0.05, "max": 1.5, "link": "w",
             "src": "guide"},
            {"key": "dist_m", "label": "Before the crossing", "guide": 2.5, "min": 0.0, "max": 20.0,
             "range": [2.0, 3.0], "src": "guide", "note": "as the stop line"},
        ],
        "opts": [{"key": "roundabouts", "label": "At roundabout entries", "default": True},
                 {"key": "minor", "label": "On side streets at junctions", "default": False}],
    },
    "arrows": {
        "label": "Lane arrows", "short": "Arrows",
        "note": "In each lane coming into a junction, the way that lane may go, read from the junction's streets: "
                "the left lane may turn left, the right lane right, the others go straight on.",
        "fields": [
            {"key": "length_m", "label": "Arrow length", "guide": 3.5, "min": 1.0, "max": 12.0, "link": "size",
             "range": [3.5, 5.0], "src": "irc", "note": "3.5 m up to 50 km/h, 5 m above; elongated along the road"},
            {"key": "width_m", "label": "Head width", "guide": 0.6, "min": 0.2, "max": 3.0, "link": "size",
             "src": "ours", "note": "in proportion with the length"},
            {"key": "first_m", "label": "First arrow from the stop line", "guide": 15.0, "min": 0.0, "max": 200.0,
             "src": "guide", "note": "the nearest 15 m from the stop line"},
            {"key": "every_m", "label": "Then every", "guide": 15.0, "min": 3.0, "max": 200.0, "src": "guide",
             "note": "each next one 15 m before the one ahead"},
            {"key": "count", "label": "Arrows in each lane", "guide": 4, "min": 1, "max": 8, "int": True,
             "src": "guide", "note": "normally four in sequence"},
        ],
        "opts": [{"key": "type", "label": "Arrows", "default": "auto",
                  "choices": [["auto", "As the junction's streets allow"], ["S", "Straight on"], ["L", "Left"],
                              ["R", "Right"], ["SL", "Straight or left"], ["SR", "Straight or right"],
                              ["LR", "Left or right"], ["SLR", "Any way"]]},
                 {"key": "junctions", "label": "At junctions", "default": True},
                 {"key": "roundabouts", "label": "At roundabout entries", "default": True}],
    },
    "stopword": {
        "label": "STOP word", "short": "STOP",
        "note": "The word STOP in each lane before a stop line, elongated so drivers read it at a low angle.",
        "fields": [
            {"key": "height_m", "label": "Letter height (along the road)", "guide": 1.25, "min": 0.3, "max": 6.0,
             "link": "size", "range": [1.25, 2.5], "src": "irc", "note": "1.25 m up to 50 km/h, 2.5 m up to 100"},
            {"key": "width_m", "label": "Word width", "guide": 1.8, "min": 0.3, "max": 6.0, "link": "size",
             "src": "ours", "note": "in proportion with the height"},
            {"key": "dist_m", "label": "Before the stop line (top of the word)", "guide": 2.5, "min": 0.0,
             "max": 20.0, "range": [2.0, 3.0], "src": "guide", "note": "2 to 3 m from the line"},
        ],
        "opts": [{"key": "minor_only", "label": "Only on side streets", "default": False}],
    },
    "cycle": {
        "label": "Cyclist crossing", "short": "Cycle crossing",
        "note": "Two lines across the road beside the pedestrian crossing, as wide apart as the cycle track.",
        "fields": [
            {"key": "width_m", "label": "Line width", "guide": 0.1, "min": 0.03, "max": 0.5, "src": "guide",
             "note": "two 100 mm white lines"},
            {"key": "spacing_m", "label": "Track width (between the lines)", "guide": 2.0, "min": 0.5, "max": 6.0,
             "range": [1.0, 3.0], "src": "guide", "note": "the cycle track's width, 1 to 3 m"},
            {"key": "gap_m", "label": "Space to the pedestrian crossing", "guide": 0.5, "min": 0.0, "max": 5.0,
             "src": "ours"},
        ],
        "opts": [{"key": "junctions", "label": "At junctions", "default": True},
                 {"key": "roundabouts", "label": "At roundabout entries", "default": True}],
    },
    "box": {
        "label": "Yellow box (not in the guide)", "short": "Yellow box",
        "note": "Criss-cross yellow lines inside big open junctions: do not stop here. Not in the guide; common "
                "practice.",
        "fields": [
            {"key": "line_m", "label": "Line width", "guide": 0.15, "min": 0.03, "max": 0.6, "link": "w",
             "src": "ours"},
            {"key": "border_m", "label": "Border width", "guide": 0.15, "min": 0.03, "max": 0.6, "link": "w",
             "src": "ours"},
            {"key": "spacing_m", "label": "Between the diagonals", "guide": 1.5, "min": 0.3, "max": 10.0,
             "src": "ours"},
            {"key": "big_m2", "label": "Big junction: more than (m²)", "guide": BIG_M2, "min": 0.0, "max": 100000.0,
             "src": "ours", "unit": "m²"},
            {"key": "open_x", "label": "and bigger than a plain crossing (×)", "guide": OPEN_X, "min": 1.0, "max": 20.0,
             "src": "ours", "unit": "×"},
        ],
        "opts": [{"key": "all", "label": "At every junction, not only big ones", "default": False}],
    },
}
PRESET_ORDER = ["zebra", "stop", "giveway", "arrows", "stopword", "cycle", "box"]
# "The junction set": what Add the junction markings adds
JUNCTION_SET = ["zebra", "stop", "giveway", "arrows"]


def spec():
    """Everything the page needs to show the options: the lines' and the presets' fields."""
    return {"lines": {"fields": LINE_FIELDS, "opts": LINE_OPTS}, "presets": PRESETS, "order": PRESET_ORDER,
            "junction_set": JUNCTION_SET, "sources": SOURCES}


# ------------------------------------------------------------------ guide values, ranges and the chain
def guide_value(f, opts=None):
    """A field's guide value, which may depend on an option (a stop line with traffic lights: 1 m)."""
    v = f["guide"]
    for o, alt in (f.get("guide_if") or {}).items():
        if (opts or {}).get(o):
            v = alt
    return v


def guide_range(f, opts=None):
    r = f.get("range")
    for o, alt in (f.get("range_if") or {}).items():
        if (opts or {}).get(o):
            r = alt
    return r


def _clamp(f, v):
    v = min(f["max"], max(f["min"], float(v)))
    return int(round(v)) if f.get("int") else round(v, 4)


def chain(fields, values, key, new, opts=None):
    """
    values with key set to new and, if key belongs to a link group, every other
    field of the group set to its guide value times new / key's guide value.
    """
    out = dict(values)
    f = next(x for x in fields if x["key"] == key)
    out[key] = _clamp(f, new)
    g = f.get("link")
    base = guide_value(f, opts)
    if not g or not base:
        return out
    k = out[key] / base
    for h in fields:
        if h.get("link") == g and h["key"] != key:
            out[h["key"]] = _clamp(h, guide_value(h, opts) * k)
    return out


def changed_from_guide(fields, values, opts=None):
    """The fields not at the guide's value: [(key, value, guide value)]."""
    return [(f["key"], values.get(f["key"]), guide_value(f, opts)) for f in fields
            if abs(float(values.get(f["key"], guide_value(f, opts))) - float(guide_value(f, opts))) > 1e-6]


def _values(fields, raw, opts=None):
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for f in fields:
        try:
            v = float(raw[f["key"]])
            if not math.isfinite(v):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            v = guide_value(f, opts)
        out[f["key"]] = _clamp(f, v)
    return out


def _opts(spec_opts, raw):
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for o in spec_opts:
        v = raw.get(o["key"], o["default"])
        if "choices" in o:
            out[o["key"]] = v if v in [c[0] for c in o["choices"]] else o["default"]
        else:
            out[o["key"]] = bool(v)
    return out


# ------------------------------------------------------------------ the lines' settings
def lines_config(raw, learned_share=None, learned_ratio=None):
    """
    The lines' settings as the page sends them (marks: {"rules", "drive",
    "colours", "sizes", "values": {key: value}}), checked, and what laying them
    needs per kind of line: {"rules", "drive", "colours", "learned",
    "width_ratio", "edge_inset", "setback_m", "kinds": {kind: {"cycle",
    "share", "width", "yellow", "double_gap"}}, "values", "exact": dashes keep
    their length and only the gaps stretch to fit a street, "highway": the
    highways' settings for app/lanes.py ({"min_m", "lanes"}) or None (Highways
    off)}. learned_share,
    learned_ratio: the dash's share of its cycle and the line's share of its
    street's width learned from training (sizes "learned").
    """
    raw = raw if isinstance(raw, dict) else {}
    o = _opts(LINE_OPTS, raw)
    v = _values(LINE_FIELDS, raw.get("values"))
    learned = o["sizes"] == "learned"
    yellow = o["colours"] == "guide" and o["rules"] == "guide"

    def dashed(dash, gap, width, yel=False):
        cycle = dash + gap
        share = float(learned_share) if learned and learned_share else dash / cycle
        return {"cycle": cycle, "share": share, "width": None if learned and learned_ratio else width, "yellow": yel}
    kinds = {
        "dash": dashed(v["lane.dash_m"], v["lane.gap_m"], v["lane.width_m"]),
        "cdash": dashed(v["centre.dash_m"], v["centre.gap_m"], v["centre.width_m"], yellow),
        "solid": {"width": None if learned and learned_ratio else v["edge.width_m"], "yellow": False},
        "csolid": {"width": None if learned and learned_ratio else v["centre.width_m"], "yellow": yellow},
        "cdouble": {"width": None if learned and learned_ratio else v["double.width_m"], "yellow": yellow,
                    "double_gap": v["double.gap_m"]},
    }
    highway = ({"min_m": float(v["highway.min_m"]), "lanes": int(v["highway.lanes"])}
               if o["highways"] == "on" else None)
    return {"rules": o["rules"], "drive": o["drive"], "colours": o["colours"], "learned": learned, "highway": highway,
            "width_ratio": learned_ratio if learned else None,
            "edge_inset": v["edge.inset_m"] + v["edge.width_m"] / 2 if o["rules"] == "guide" else None,
            "setback_m": v["setback_m"], "kinds": kinds, "values": v, "opts": o, "fixed_width_m": v["lane.width_m"],
            "exact": not learned}


def legacy_config(cycle_m, dash_share, width_m, width_ratio, setback_m):
    """The lines as laid before road markings had settings: every line dashed alike, white."""
    d = {"cycle": float(cycle_m), "share": float(dash_share), "width": None if width_ratio else float(width_m),
         "yellow": False}
    w = {"width": None if width_ratio else float(width_m), "yellow": False}
    return {"rules": "simple", "drive": "right", "colours": "white", "learned": bool(width_ratio), "highway": None,
            "width_ratio": width_ratio, "edge_inset": None, "setback_m": float(setback_m),
            "kinds": {"dash": d, "cdash": dict(d), "solid": w, "csolid": dict(w), "cdouble": dict(w, double_gap=0.1)},
            "values": None, "opts": None, "legacy": True, "fixed_width_m": float(width_m)}


def kind_style(cfg, kind, street_m):
    """A kind of line's paint on a street street_m wide: {"width" (metres), "cycle", "share", "yellow", "double_gap"}."""
    from app import lanes as LN
    k = dict(cfg["kinds"].get(kind) or cfg["kinds"]["dash"])
    if k.get("width") is None:
        k["width"] = LN.line_width(street_m, cfg.get("width_ratio"), 0.15)
    return k


def main_cycle(cfg, lay):
    """The dash cycle and share a street's dashes follow (its dashed lines are all of one kind)."""
    for _, kind in lay["lines"]:
        if kind in ("dash", "cdash"):
            k = cfg["kinds"][kind]
            return k["cycle"], k["share"]
    k = cfg["kinds"]["dash"]
    return k["cycle"], k["share"]


# ------------------------------------------------------------------ preset layers
def layer_check(raw):
    """A preset layer as the page sends it: {"preset", "values", "opts", "chain"}, checked; None if not one."""
    if not isinstance(raw, dict) or raw.get("preset") not in PRESETS:
        return None
    P = PRESETS[raw["preset"]]
    opts = _opts(P["opts"], raw.get("opts"))
    return {"preset": raw["preset"], "decal": "preset:" + raw["preset"], "values": _values(P["fields"], raw.get("values"), opts),
            "opts": opts, "chain": bool(raw.get("chain", True)), "off": bool(raw.get("off", False))}


def new_layer(preset):
    """A preset layer at the guide's values."""
    return layer_check({"preset": preset})


def names():
    """Each preset's name, for the decal summaries."""
    return {"preset:" + k: P["label"] for k, P in PRESETS.items()}


# ------------------------------------------------------------------ the pictures
PX_PER_M = 50                   # a picture's pixels per metre (2 cm each)
SS = 4                          # drawn this many times finer, then reduced: smooth edges
MAX_PX = 2048


def _raster(w_m, h_m, polys, rgb):
    """
    A picture w_m x h_m metres (its top the marking's front) with polygons (in
    metres, x across from its left edge, y along from its bottom) filled in rgb,
    the rest see-through.
    """
    ppm = min(PX_PER_M, MAX_PX / max(w_m, h_m, 1e-6))
    W, H = max(2, int(round(w_m * ppm))), max(2, int(round(h_m * ppm)))
    sx, sy = W * SS / w_m, H * SS / h_m
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for poly in polys:
        if isinstance(poly, (list, tuple)):
            d.polygon([(x * sx, (h_m - y) * sy) for x, y in poly], fill=255)
            continue
        for g in getattr(poly, "geoms", [poly]):                       # a shapely shape: its polygons, holes cut
            if g.geom_type != "Polygon" or g.is_empty:
                continue
            d.polygon([(x * sx, (h_m - y) * sy) for x, y in g.exterior.coords], fill=255)
            for hole in g.interiors:
                d.polygon([(x * sx, (h_m - y) * sy) for x, y in hole.coords], fill=0)
    a = im.reduce(SS)
    out = Image.new("RGBA", (W, H), tuple(rgb) + (0,))
    out.putalpha(a)
    return out


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def zebra_size(road_m, v):
    """How many stripes fit across a road road_m wide, and the crossing's width across: (n, width)."""
    period = v["stripe_m"] + v["gap_m"]
    usable = max(0.0, road_m - 2 * KERB_MARGIN_M)
    n = max(1, int((usable + v["gap_m"]) // period))
    return n, n * v["stripe_m"] + (n - 1) * v["gap_m"]


def giveway_size(span_m, v):
    period = v["seg_m"] + v["gap_m"]
    n = max(1, int((max(0.0, span_m) + v["gap_m"]) // period))
    return n, n * v["seg_m"] + (n - 1) * v["gap_m"]


# arrows: drawn at 3.5 m long with a 0.6 m head, then stretched to the size asked (they are
# elongated along the road, so stretching keeps their look). x across from the stem, y along.
_A_L, _A_W, _A_S = 3.5, 0.6, 0.15          # length, head width, stem width


def _arrow_shape(kind):
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    parts = []
    s2 = _A_S / 2
    if "S" in kind:
        parts.append(Polygon(_rect(-s2, 0.0, s2, 2.3)))
        parts.append(Polygon([(-_A_W / 2, 2.3), (_A_W / 2, 2.3), (0.0, _A_L)]))
    turns = [t for t in "LR" if t in kind]
    if turns:
        straight = "S" in kind
        y0, y1 = (1.0, 1.8) if straight else (1.45, 2.25)       # where the branch leaves the stem, and ends
        parts.append(Polygon(_rect(-s2, 0.0, s2, y0 + 0.05)))
        for t in turns:
            sx = -1.0 if t == "L" else 1.0
            end = (sx * 0.5, y1)
            parts.append(LineString([(0.0, y0), (0.0, y0 + 0.05), end]).buffer(s2, cap_style=2, join_style=2))
            ux, uy = sx * 0.5, y1 - y0 - 0.05
            n = math.hypot(ux, uy)
            ux, uy = ux / n, uy / n
            px, py = -uy, ux
            hl = 0.95
            parts.append(Polygon([(end[0] + px * _A_W / 2, end[1] + py * _A_W / 2),
                                  (end[0] - px * _A_W / 2, end[1] - py * _A_W / 2),
                                  (end[0] + ux * hl, end[1] + uy * hl)]))
            parts.append(Polygon([(end[0] + px * s2 - ux * 0.05, end[1] + py * s2 - uy * 0.05),
                                  (end[0] - px * s2 - ux * 0.05, end[1] - py * s2 - uy * 0.05),
                                  (end[0] - px * s2 + ux * 0.05, end[1] - py * s2 + uy * 0.05),
                                  (end[0] + px * s2 + ux * 0.05, end[1] + py * s2 + uy * 0.05)]))
    return unary_union(parts)


def arrow_size(kind, v):
    """An arrow's picture across and along (metres): as wide as its branches reach, either way from the stem."""
    g = _arrow_shape(kind)
    x0, _, x1, _ = g.bounds
    reach = max(abs(x0), abs(x1))
    kx = v["width_m"] / _A_W
    return 2 * reach * kx + 0.04, v["length_m"]


def _arrow_picture(kind, v, rgb):
    from shapely import affinity
    w, h = arrow_size(kind, v)
    g = affinity.scale(_arrow_shape(kind), xfact=v["width_m"] / _A_W, yfact=v["length_m"] / _A_L, origin=(0, 0))
    g = affinity.translate(g, xoff=w / 2)
    return _raster(w, h, [g], rgb)


# block letters for the words: strokes in a 1 x 1 box (x across, y up), drawn as thick lines
_LETTERS = {
    "S": [[(1, .88), (.85, 1), (.15, 1), (0, .86), (0, .62), (.15, .5), (.85, .5), (1, .38), (1, .14), (.85, 0),
           (.15, 0), (0, .12)]],
    "T": [[(0, 1), (1, 1)], [(.5, 1), (.5, 0)]],
    "O": [[(.15, 0), (.85, 0), (1, .15), (1, .85), (.85, 1), (.15, 1), (0, .85), (0, .15), (.15, 0)]],
    "P": [[(0, 0), (0, 1), (.82, 1), (1, .86), (1, .64), (.82, .5), (0, .5)]],
}


def _word_picture(text, w_m, h_m, rgb):
    """A word in block letters, w_m across and h_m along (elongated), its top the word's top."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    n = len(text)
    gap = 0.25                               # between letters, in letter widths
    lw = w_m / (n + gap * (n - 1))
    stroke = min(lw * 0.3, h_m * 0.2)
    polys = []
    for i, ch in enumerate(text):
        x0 = i * lw * (1 + gap)
        for line in _LETTERS.get(ch, []):
            pts = [(x0 + stroke / 2 + px * (lw - stroke), stroke / 2 + py * (h_m - stroke)) for px, py in line]
            polys.append(LineString(pts).buffer(stroke / 2, cap_style=2, join_style=2))
    return _raster(w_m, h_m, [unary_union(polys)], rgb)


def _box_picture(w, h, v, rgb):
    """A yellow box: a border and criss-cross diagonals inside it."""
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union
    b, lw, sp = v["border_m"], v["line_m"], v["spacing_m"]
    outer, inner = box(0, 0, w, h), box(b, b, w - b, h - b)
    parts = [outer.difference(inner)]
    step = sp * math.sqrt(2)                 # along x, for diagonals sp apart
    k = -h
    while k < w + h:
        for (x0, y0, x1, y1) in ((k, 0, k + h, h), (k + h, 0, k, h)):
            parts.append(LineString([(x0, y0), (x1, y1)]).buffer(lw / 2, cap_style=2).intersection(inner))
        k += step
    return _raster(w, h, [unary_union([p for p in parts if not p.is_empty])], rgb)


def colour_of(preset, colours):
    return YELLOW if preset == "box" and colours == "guide" else WHITE


def variant_id(li, layer, variant, colours):
    """A placed marking's picture: its layer's preset and values, the variant (stripes, arrow kind, size) and colours."""
    h = hashlib.sha1(json.dumps([layer["preset"], layer["values"], layer["opts"], colours], sort_keys=True)
                     .encode()).hexdigest()[:10]
    return f"rm_{layer['preset']}_{variant}_{h}"


def picture(layer, variant, colours):
    """The picture of one variant of a preset layer (PIL RGBA, its top the marking's front)."""
    p, v = layer["preset"], layer["values"]
    rgb = colour_of(p, colours)
    if p == "zebra":
        n = int(variant[1:])
        w = n * v["stripe_m"] + (n - 1) * v["gap_m"]
        per = v["stripe_m"] + v["gap_m"]
        return _raster(w, v["length_m"], [_rect(i * per, 0, i * per + v["stripe_m"], v["length_m"]) for i in range(n)], rgb)
    if p == "stop":
        w, lw = 2.0, v["width_m"]
        if variant == "double":
            h = 2 * lw + v["gap_m"]
            return _raster(w, h, [_rect(0, 0, w, lw), _rect(0, h - lw, w, h)], rgb)
        return _raster(w, lw, [_rect(0, 0, w, lw)], rgb)
    if p == "giveway":
        n = int(variant[1:])
        per = v["seg_m"] + v["gap_m"]
        w = n * v["seg_m"] + (n - 1) * v["gap_m"]
        lw, h = v["width_m"], 2 * v["width_m"] + v["spacing_m"]
        polys = [_rect(i * per, y, i * per + v["seg_m"], y + lw) for i in range(n) for y in (0.0, h - lw)]
        return _raster(w, h, polys, rgb)
    if p == "arrows":
        return _arrow_picture(variant, v, rgb)
    if p == "stopword":
        return _word_picture("STOP", v["width_m"], v["height_m"], rgb)
    if p == "cycle":
        w, lw, h = 2.0, v["width_m"], v["spacing_m"] + v["width_m"]
        return _raster(w, h, [_rect(0, 0, w, lw), _rect(0, h - lw, w, h)], rgb)
    if p == "box":
        bw, bh = (float(x) for x in variant[1:].split("x"))
        return _box_picture(bw, bh, v, rgb)
    raise ValueError(p)


def pictures(placements, layers, colours):
    """{decal id: picture} for every preset marking placed (placements' "rm" says which variant)."""
    out = {}
    for p in placements:
        if p.get("rm") and p["decal"] not in out:
            L = layers[p["layer"]]
            if L.get("preset"):
                out[p["decal"]] = picture(L, p["rm"], colours)
    return out


def variant_names(placements, layers):
    """{decal id: a name for its material}: the preset's short name and the variant."""
    out = {}
    for p in placements:
        if p.get("rm") and p["decal"] not in out:
            L = layers[p["layer"]]
            short = PRESETS[L["preset"]]["short"]
            v = p["rm"]
            out[p["decal"]] = (f"{short} {v[1:]} stripes" if L["preset"] == "zebra" else
                               f"{short} {v}" if L["preset"] in ("arrows", "stop") else
                               f"{short} {v[1:]}" if L["preset"] in ("giveway", "box") else short)
    return out


# ------------------------------------------------------------------ roundabouts
def roundabouts(det, mpp):
    """
    The roundabouts: a round island (a hole in the road, 2 to 60 m across its
    radius, round: it fills most of the circle round its middle) ringed by road.
    [{"cx", "cy", "r" (the island's radius), "R" (the ring's outer radius), mask
    pixels}].
    """
    from scipy import ndimage as ndi
    road = det["road"]
    H, W = road.shape
    lab, n = ndi.label(~road)
    out = []
    if not n:
        return out
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None or sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W:
            continue                                         # touches the picture's edge: a block, not an island
        hole = lab[sl] == i
        area = float(hole.sum())
        r = math.sqrt(area / math.pi)
        if not (2.0 / mpp <= r <= 60.0 / mpp):
            continue
        ys, xs = np.nonzero(hole)
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
        rmax = float(np.hypot(ys + sl[0].start - cy, xs + sl[1].start - cx).max()) + 0.5
        if area / (math.pi * rmax * rmax) < 0.78:
            continue                                         # not round
        # ringed by road: a little way out from the island, all round, is road
        t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        rr = rmax + max(1.5, 1.0 / mpp)
        px_, py_ = np.round(cx + rr * np.cos(t)).astype(int), np.round(cy + rr * np.sin(t)).astype(int)
        ok = (px_ >= 0) & (px_ < W) & (py_ >= 0) & (py_ < H)
        if ok.mean() < 0.99 or road[py_[ok], px_[ok]].mean() < 0.95:
            continue
        # the ring's outer edge: walking out along rays, the median distance to off the road
        outs = []
        for a in t:
            d = rr
            while d < rmax + 40.0 / mpp:
                x, y = int(round(cx + d * math.cos(a))), int(round(cy + d * math.sin(a)))
                if not (0 <= x < W and 0 <= y < H) or not road[y, x]:
                    break
                d += 0.5
            outs.append(d)
        out.append({"cx": float(cx), "cy": float(cy), "r": float(rmax), "R": float(np.percentile(outs, 25))})
    return out


# ------------------------------------------------------------------ the junctions' approaches
def _unit(x, y):
    n = math.hypot(x, y) or 1.0
    return x / n, y / n


def _turn(d, e):
    """Straight, left or right from heading d to exit e (image axes, y down), or None for a U turn."""
    dot = d[0] * e[0] + d[1] * e[1]
    cross = d[0] * e[1] - d[1] * e[0]
    ang = math.degrees(math.atan2(cross, dot))
    if abs(ang) > 150:
        return None
    if abs(ang) <= 35:
        return "S"
    return "L" if cross < 0 else "R"


def arrow_kinds(moves, n):
    """Each lane's arrow, left to right as the driver sees them, for the moves the junction allows."""
    if not moves or n < 1:
        return []
    if n == 1:
        return ["".join(m for m in "SLR" if m in moves)]
    if "S" in moves:
        out = ["S"] * n
        if "L" in moves:
            out[0] = "SL"
        if "R" in moves:
            out[-1] = "SR"
        return out
    if "L" in moves and "R" in moves:
        k = (n + 1) // 2
        return ["L"] * k + ["R"] * (n - k)
    return ["L" if "L" in moves else "R"] * n


class _Street:
    """A street's centreline from one of its ends, out along it: where a point a distance out and across lies."""

    def __init__(self, path_yx, near_last, px):
        from app import generation as G
        P = np.array(G._smooth(path_yx, window=7), float)[:, ::-1]            # (x, y)
        if near_last:
            P = P[::-1]
        self.P, self.px = P, px
        seg = np.hypot(*np.diff(P, axis=0).T) if len(P) > 1 else np.zeros(0)
        self.arc = np.concatenate([[0.0], np.cumsum(seg)])
        tg = np.gradient(P, axis=0) if len(P) > 1 else np.array([[1.0, 0.0]])
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        self.T = tg                                                          # out along the street
        self.length_m = float(self.arc[-1]) * px

    def at(self, a_px):
        a = min(max(a_px, 0.0), float(self.arc[-1]))
        x, y = np.interp(a, self.arc, self.P[:, 0]), np.interp(a, self.arc, self.P[:, 1])
        tx, ty = _unit(np.interp(a, self.arc, self.T[:, 0]), np.interp(a, self.arc, self.T[:, 1]))
        return x, y, tx, ty


def _road_share(poly, road):
    """How much of a polygon (mask pixels) is road."""
    H, W = road.shape
    x0, y0, x1, y1 = poly.bounds
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1)) + 1), min(H, int(math.ceil(y1)) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    im = Image.new("L", (x1 - x0, y1 - y0), 0)
    rings = [poly] if poly.geom_type == "Polygon" else list(getattr(poly, "geoms", []))
    for g in rings:
        if g.geom_type == "Polygon":
            ImageDraw.Draw(im).polygon([(x - x0, y - y0) for x, y in g.exterior.coords], fill=255)
    inside = np.array(im) > 0
    n = int(inside.sum())
    return float(road[y0:y1, x0:x1][inside].sum()) / n if n else 0.0


THROUGH_IN_LINE = 35.0         # a junction's arm runs along a highway's centreline within this many degrees
FRAME_M = (8.0, 30.0)          # a street's frame at a junction edge is read from this much of it (least, most)


def _through_arms(j, edges, cidx, px):
    """
    The arms of a junction a highway runs on through (its centreline passes
    the junction's middle, and the arm runs along it): nothing is laid across
    them. A highway whose centreline stops here (a crossroads of highways) is
    an ordinary arm.
    """
    if cidx is None or not cidx.ok:
        return []
    d, i = cidx.tree.query([j["cx"], j["cy"]])
    if d > max(2.0, 0.5 * cidx.W[i]) / px:
        return []
    out = []
    cos_in = math.cos(math.radians(THROUGH_IN_LINE))
    for ai, e in enumerate(edges):
        if e is None:
            continue
        x, y, ox, oy, _ = e
        d, k = cidx.tree.query([x, y])
        if d > max(2.0, 0.35 * cidx.W[k]) / px:
            continue
        # the centreline's way there
        k0, k1 = max(0, k - 3), min(len(cidx.P) - 1, k + 3)
        t = cidx.P[k1] - cidx.P[k0]
        n = float(np.hypot(*t))
        if n > 0 and abs(float(t @ np.array([ox, oy]))) / n >= cos_in:
            out.append(ai)
    return out


def plan(det, mpp, found, chosen, rules="simple", drive="right", edge_inset=None, big_m2=BIG_M2, open_x=OPEN_X,
         hw=None):
    """
    Every junction's approaches, from the mask's junctions (app/junctions.py),
    their edges (app/decals.py) and the streets' lanes (found, chosen: as
    app/generation.py reads them, mask pixels): [{"j", "arm", "x", "y" (the
    edge, centred between the kerbs), "ox", "oy" (out along the street),
    "road_m" (its width there), "role": "major", "minor", "equal" (no main road),
    "through" (a highway's arm where it runs on through the junction: nothing
    laid on it), "entry" (into a roundabout) or "ring" (the roundabout's own
    road), "big", "lanes": [(x0, x1) across, left to right as the driver coming
    in sees them], "half": (x0, x1) the lanes coming in, "moves": the ways out
    ("S", "L", "R"), "street": its _Street or None, "frame": the street's
    middle and way at the edge read from its kerbs (decals.kerb_frame), "canon":
    whether coming in is the street's canonical way}], the roundabouts and
    junction boxes, and "chains": the highways' whole centrelines their lines are
    laid along (app/highways.py). hw: the highways' settings (app/lanes.py),
    None: none.
    """
    from app import decals as DC
    from app import lanes as LN
    from shapely.geometry import Polygon
    px = max(mpp, 1e-6)
    road_mask = det["road"]
    junctions = [j for j in det["junctions"] if j["type"] != "interchange (flagged)"]
    rbs = roundabouts(det, mpp)
    # which street each segment is, and its lanes
    lays = {s: LN.layout(w, chosen.get(s), rules=rules, edge_inset=edge_inset, hw=hw) for s, (_, w) in found.items()}
    approaches, boxes = [], []
    # the highways' whole centrelines: their lines run on through the junctions they pass
    from app import highways as HWY
    chains = HWY.chains(det, mpp, hw) if hw else []
    cidx = LN.ChainIndex(chains, px) if chains else None
    for ji, j in enumerate(junctions):
        edges = []
        for ai, arm in enumerate(j["arms"]):
            e = DC._edge(j, arm, det) if not arm.get("off_image") else None
            edges.append(e)
        # each arm's street: its centreline from the junction out, its lanes, and its frame
        info = []
        for ai, (arm, e) in enumerate(zip(j["arms"], edges)):
            inf = {"seg": None, "street": None, "lay": None, "canon": True, "frame": None, "edge": e,
                   "width_m": 0.0}
            info.append(inf)
            if e is None:
                continue
            x, y, ox, oy, half = e
            s = arm.get("segment")
            if s in found:
                path = found[s][0]
                pa = np.asarray(path, float)
                first, last = pa[0][::-1], pa[-1][::-1]
                near_last = math.hypot(last[0] - x, last[1] - y) < math.hypot(first[0] - x, first[1] - y)
                st = _Street(path, near_last, px)
                # the street's canonical way (app/lanes.py): coming in is it when its canonical end is here
                inf.update(seg=s, street=st, lay=lays[s], canon=near_last == LN.canonical(first, last),
                           width_m=float(found[s][1]))
                room = float(st.arc[-1]) * 0.8
            else:
                room = FRAME_M[1] / px
            length = min(max(3.0 * half, FRAME_M[0] / px), FRAME_M[1] / px, max(room, 4.0))
            inf["frame"] = DC.kerb_frame(road_mask, x, y, ox, oy, half, length)
            inf["frame_m"] = length * px
        # a roundabout's junction: its middle on the ring
        rb = None
        for ri, r in enumerate(rbs):
            if math.hypot(j["cx"] - r["cx"], j["cy"] - r["cy"]) <= r["R"] + 2.0 / px:
                rb = ri
        roles = ["equal"] * len(j["arms"])
        through = _through_arms(j, edges, cidx, px) if rb is None else []
        if rb is not None:
            r = rbs[rb]
            ru = _unit(j["cx"] - r["cx"], j["cy"] - r["cy"])
            for ai, e in enumerate(edges):
                if e is None:
                    continue
                roles[ai] = "entry" if e[2] * ru[0] + e[3] * ru[1] >= 0.6 else "ring"
        else:
            # the main road: the two arms most in line (within 35 degrees), the widest such pair
            ws = [2 * e[4] * px if e else 0.0 for e in edges]
            pairs = []
            for a in range(len(edges)):
                for b in range(a + 1, len(edges)):
                    if edges[a] is None or edges[b] is None:
                        continue
                    dev = abs(((j["arms"][a]["angle"] - j["arms"][b]["angle"]) % 360) - 180)
                    if dev <= 35:
                        pairs.append((min(ws[a], ws[b]), a, b))
            if pairs:
                pairs.sort(reverse=True)
                best = pairs[0]
                tie = any(p[0] >= best[0] - 1.0 and {p[1], p[2]}.isdisjoint({best[1], best[2]}) for p in pairs[1:])
                if not tie:
                    roles = ["major" if ai in (best[1], best[2]) else "minor" for ai in range(len(edges))]
            # a highway running on through it: its lines go on, the side streets give way to it
            if through:
                roles = ["through" if ai in through else "minor" for ai in range(len(edges))]
        # the junction's area between its edges: big open junctions get the yellow box
        corners = []
        for e in edges:
            if e is None:
                continue
            x, y, ox, oy, half = e
            corners += [(x - oy * half, y + ox * half), (x + oy * half, y - ox * half)]
        area_m2, poly = 0.0, None
        if len(corners) >= 3:
            c = np.array(corners)
            ang = np.arctan2(c[:, 1] - j["cy"], c[:, 0] - j["cx"])
            poly = Polygon(c[np.argsort(ang)])
            if not poly.is_valid:
                poly = poly.buffer(0)
            area_m2 = float(poly.area) * px * px
        # big: more than big_m2 and open_x times a plain crossing of its two widest streets (a
        # crossing of two wide streets is not big for being wide), or five streets or more; and
        # mostly road (edges found across a wide road or the picture's edge can enclose blocks);
        # never where a highway runs on through
        ws = sorted((2 * e[4] * px for e in edges if e), reverse=True)
        cross_m2 = ws[0] * ws[1] if len(ws) >= 2 else 0.0
        on_road = _road_share(poly, road_mask) if poly is not None and not poly.is_empty else 0.0
        big = (rb is None and not through and on_road >= BOX_ROAD_SHARE
               and ((area_m2 >= big_m2 and area_m2 >= open_x * cross_m2) or len(j["arms"]) >= 5))
        if rb is None and poly is not None and not poly.is_empty:
            boxes.append({"j": ji, "poly": poly, "area_m2": area_m2, "big": big, "cross_m2": cross_m2,
                          "on_road": on_road, "through": bool(through)})
        for ai, (arm, e) in enumerate(zip(j["arms"], edges)):
            if e is None:
                continue
            x, y, ox, oy, half = e
            inf = info[ai]
            fr = inf["frame"]
            road_m = 2 * (fr[4] if fr else half) * px
            st, lay, canon = inf["street"], inf["lay"], inf["canon"]
            if lay is None:
                lay = LN.layout(road_m, None, rules=rules, edge_inset=edge_inset, hw=hw)
            spans = LN.lane_spans(lay)
            right_side = "side2" if canon else "side1"                      # the right of the way in
            left_side = "side1" if canon else "side2"
            side = right_side if drive == "right" else left_side
            lanes = spans[side]
            k = road_m / max(lay["width_m"], 1e-6)                          # the layout's metres to the edge's
            lanes = [(a0 * k, a1 * k) for a0, a1 in lanes]
            # left to right as the driver coming in sees them: +x is the right of the canonical way
            lanes = sorted(lanes) if canon else sorted(lanes, key=lambda t: -t[0])
            half_span = (min(min(t) for t in lanes), max(max(t) for t in lanes)) if lanes else None
            approaches.append({"j": ji, "arm": ai, "x": x, "y": y, "ox": ox, "oy": oy, "road_m": road_m,
                               "role": roles[ai], "rb": rb, "big": big, "lanes": lanes, "half": half_span,
                               "street": st, "frame": fr, "frame_m": inf.get("frame_m", 0.0), "canon": canon,
                               "lay": lay, "reach_m": j["r"] * px + road_m + 5.0, "exits": None})
    # the ways out of each approach
    by_j = {}
    for a in approaches:
        by_j.setdefault(a["j"], []).append(a)
    by_rb = {}
    for a in approaches:
        if a["role"] == "entry":
            by_rb.setdefault(a["rb"], []).append(a)
    for a in approaches:
        if a["role"] in ("ring", "through"):
            continue
        d = (-a["ox"], -a["oy"])
        moves = set()
        if a["role"] == "entry":
            r = rbs[a["rb"]]
            d = _unit(r["cx"] - a["x"], r["cy"] - a["y"])
            for b in by_rb.get(a["rb"], []):
                if b is a or not _leaves(b, drive):
                    continue
                m = _turn(d, _unit(b["x"] - r["cx"], b["y"] - r["cy"]))
                if m:
                    moves.add(m)
        else:
            for b in by_j.get(a["j"], []):
                if b is a or b["role"] == "ring" or not _leaves(b, drive):
                    continue
                m = _turn(d, (b["ox"], b["oy"]))
                if m:
                    moves.add(m)
        a["moves"] = moves
    return {"approaches": approaches, "roundabouts": rbs, "boxes": boxes, "chains": chains}


def _leaves(b, drive):
    """Whether traffic can leave the junction along this approach's street (it has lanes going out)."""
    lay = b["lay"]
    if lay["kind"] == "none":
        return True
    a1, a2 = lay["sides"]
    # going out is the street's canonical way when coming in is not; its lanes are on its right (left)
    out_canon = not b["canon"]
    side = (a2 if out_canon else a1) if drive == "right" else (a1 if out_canon else a2)
    return side > 0


def _ways(layers, preset):
    """The preset layers of a kind that are on, and the first of them (its values set the distances)."""
    ls = [(li, L) for li, L in layers if L["preset"] == preset and not L.get("off")]
    return ls, (ls[0][1] if ls else None)


def place(det, mpp, layers, found, chosen, cfg, bridges=None):
    """
    Where every preset marking goes: (placements, ends). layers: [(index in
    the decal layers, preset layer)]. cfg: the lines' settings (lines_config:
    rules, drive, colours). placements: as app/decals.py's place gives them,
    with "rm" (which picture of the preset). ends: where each junction's
    markings ask the lines to stop ([{"x", "y", "ox", "oy", "m", "reach_m"}],
    app/lanes.py's end_setbacks).
    """
    from app import decals as DC
    px = max(mpp, 1e-6)
    road = det["road"]
    H, W = road.shape
    colours = cfg.get("colours", "white")
    drive = cfg.get("drive", "right")
    box0 = next((L["values"] for _, L in layers if L["preset"] == "box"), {})
    P = plan(det, mpp, found, chosen, cfg.get("rules", "simple"), drive, cfg.get("edge_inset"),
             box0.get("big_m2", BIG_M2), box0.get("open_x", OPEN_X), cfg.get("highway"))
    placements, ends = [], []
    # each layer's laid so far (none on another), in cells of the map so a marking
    # checks only those near it: {layer: {"cells": {(i, j): [p]}, "w", "h" (the largest)}}
    mine = {}
    cell = max(1.0, CELL_M / px)

    def lay(li, L, variant, x, y, ux, uy, w, h, at):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H) or not road[yi, xi] or DC._in_bridge(x, y, bridges):
            return None
        n = math.hypot(ux, uy) or 1.0
        p = {"layer": li, "decal": variant_id(li, L, variant, colours), "rm": variant,
             "x": round(float(x), 2), "y": round(float(y), 2), "ux": round(ux / n, 5), "uy": round(uy / n, 5),
             "w_m": round(float(w), 3), "h_m": round(float(h), 3), "at": at}
        c = DC._corners(p, mpp, 0.98)
        g = mine.setdefault(li, {"cells": {}, "w": 0.0, "h": 0.0})
        reach = math.hypot(g["w"] + p["w_m"], g["h"] + p["h_m"]) / px      # the furthest one that could touch it
        ci, cj, r = int(p["x"] // cell), int(p["y"] // cell), int(math.ceil(reach / cell))
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                for q in g["cells"].get((i, j), ()):
                    if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) < math.hypot(q["w_m"] + p["w_m"], q["h_m"] + p["h_m"]) / px \
                            and DC._overlap(c, DC._corners(q, mpp, 0.98)):
                        return None
        g["cells"].setdefault((ci, cj), []).append(p)
        g["w"], g["h"] = max(g["w"], p["w_m"]), max(g["h"], p["h_m"])
        placements.append(p)
        return p

    def spot(a, d_m, x_m):
        """The point d_m out from the edge and x_m across (the canonical way's right), and the way in there."""
        fr = a.get("frame")
        if fr is not None and d_m <= min(a["frame_m"], FRAME_NEAR_M):
            # near the edge: square to the kerbs, from the middle between them
            fx, fy, fox, foy, _ = fr
            nx, ny = (foy, -fox) if a["canon"] else (-foy, fox)
            return fx + fox * d_m / px + nx * x_m / px, fy + foy * d_m / px + ny * x_m / px, -fox, -foy
        st = a["street"]
        ex, ey = a["x"], a["y"]
        if st is not None:
            off0 = (st.P[0][0] - ex) * a["ox"] + (st.P[0][1] - ey) * a["oy"]       # the street's end, past the edge
            along = d_m / px - off0
            if along > 0:
                cx, cy, tx, ty = st.at(along)
                # its centre there, moved by how far the edge's middle lies off the street's centreline
                nx, ny = (ty, -tx) if a["canon"] else (-ty, tx)
                return cx + nx * x_m / px, cy + ny * x_m / px, -tx, -ty
        nx, ny = (a["oy"], -a["ox"]) if a["canon"] else (-a["oy"], a["ox"])
        return (ex + a["ox"] * d_m / px + nx * x_m / px, ey + a["oy"] * d_m / px + ny * x_m / px, -a["ox"], -a["oy"])

    zebras, _ = _ways(layers, "zebra")
    stops, stop0 = _ways(layers, "stop")
    gws, gw0 = _ways(layers, "giveway")
    arrows, _ = _ways(layers, "arrows")
    words, word0 = _ways(layers, "stopword")
    cycles, cyc0 = _ways(layers, "cycle")
    for a in P["approaches"]:
        if a["role"] in ("ring", "through"):
            continue                                           # a roundabout's ring, a highway running on
        rb = a["role"] == "entry"
        where = "roundabouts" if rb else "junctions"
        zs = [(li, L) for li, L in zebras if L["opts"][where]]
        cs = [(li, L) for li, L in cycles if L["opts"][where]]
        far = 0.0                                              # the furthest edge of anything across the road
        line_far = None
        has_in = bool(a["lanes"])

        def across(items, d0, depth, kind):
            for li, L in items:
                v = L["values"]
                if kind == "zebra":
                    n, w = zebra_size(a["road_m"], v)
                    var = f"n{n}"
                else:
                    w, var = max(0.5, a["road_m"] - 2 * KERB_MARGIN_M), "c"
                x, y, ux, uy = spot(a, d0 + depth / 2, 0.0)
                lay(li, L, var, x, y, ux, uy, w, depth, "intersection")

        def line_in(items, d0, kind):
            """A stop or give way line across the lanes coming in, d0 out from the edge."""
            if not a["half"]:
                return None
            x0, x1 = a["half"]
            lo, hi = sorted((x0, x1))
            # short of the kerb, and of the centre line (unless the road is one way in)
            road_half = a["road_m"] / 2
            lo = lo + (KERB_MARGIN_M if lo <= -road_half + 1e-3 else CENTRE_MARGIN_M)
            hi = hi - (KERB_MARGIN_M if hi >= road_half - 1e-3 else CENTRE_MARGIN_M)
            span = hi - lo
            if span <= 0.3:
                return None
            depth_out = None
            for li, L in items:
                v = L["values"]
                if kind == "stop":
                    depth = v["width_m"] * 2 + v["gap_m"] if L["opts"]["double"] else v["width_m"]
                    var, w = ("double" if L["opts"]["double"] else "single"), span
                else:
                    depth = 2 * v["width_m"] + v["spacing_m"]
                    n, w = giveway_size(span, v)
                    var = f"n{n}"
                x, y, ux, uy = spot(a, d0 + depth / 2, (lo + hi) / 2)
                lay(li, L, var, x, y, ux, uy, w, depth, "intersection")
                depth_out = max(depth_out or 0.0, depth)
            return depth_out

        cursor = EDGE_GAP_M
        if rb:
            gw = [(li, L) for li, L in gws if L["opts"]["roundabouts"]]
            if gw and has_in:
                dep = line_in(gw, cursor, "giveway")
                if dep:
                    line_far = cursor + dep
                    far = max(far, line_far)
            start = (line_far or cursor) + (zs[0][1]["values"]["rb_back_m"] if zs else 0.0)
            if zs:
                ln = zs[0][1]["values"]["length_m"]
                across(zs, start, ln, "zebra")
                far = max(far, start + ln)
                start += ln
            if cs:
                v = cs[0][1]["values"]
                d0 = start + v["gap_m"]
                across(cs, d0, v["spacing_m"] + v["width_m"], "cycle")
                far = max(far, d0 + v["spacing_m"] + v["width_m"])
        else:
            if cs:
                v = cs[0][1]["values"]
                across(cs, cursor, v["spacing_m"] + v["width_m"], "cycle")
                cursor += v["spacing_m"] + v["width_m"] + v["gap_m"]
                far = max(far, cursor - v["gap_m"])
            crossing = False
            if zs:
                ln = zs[0][1]["values"]["length_m"]
                across(zs, cursor, ln, "zebra")
                cursor += ln
                far = max(far, cursor)
                crossing = True
            minor = a["role"] == "minor"
            gw = [(li, L) for li, L in gws if L["opts"]["minor"] and minor]
            st = [(li, L) for li, L in stops if not L["opts"]["minor_only"] or minor]
            if has_in and (gw or st):
                items, kind = (gw, "giveway") if gw else (st, "stop")
                v = items[0][1]["values"]
                d0 = cursor + (v["dist_m"] if crossing else NO_CROSSING_M)
                dep = line_in(items, d0, kind)
                if dep:
                    line_far = d0 + dep
                    far = max(far, line_far)
                    if kind == "stop" and has_in:
                        for li, L in words:
                            if L["opts"]["minor_only"] and not minor:
                                continue
                            v = L["values"]
                            for (l0, l1) in a["lanes"]:
                                if abs(l1 - l0) < v["width_m"] * 0.8:
                                    continue
                                x, y, ux, uy = spot(a, line_far + v["dist_m"] + v["height_m"] / 2, (l0 + l1) / 2)
                                lay(li, L, "STOP", x, y, ux, uy, v["width_m"], v["height_m"], "edge")
        # arrows in each lane coming in
        if has_in and a["moves"] is not None:
            base = line_far if line_far is not None else (far or EDGE_GAP_M)
            st = a["street"]
            room = (st.length_m + (st.P[0][0] - a["x"]) * a["ox"] * px + (st.P[0][1] - a["y"]) * a["oy"] * px
                    if st is not None else 30.0)
            one_way = a["lay"]["kind"] != "none" and 0 in a["lay"]["sides"]
            room = room - 5.0 if one_way else room / 2
            for li, L in arrows:
                if not L["opts"]["roundabouts" if rb else "junctions"]:
                    continue
                v = L["values"]
                kinds = ([L["opts"]["type"]] * len(a["lanes"]) if L["opts"]["type"] != "auto"
                         else arrow_kinds(a["moves"], len(a["lanes"])))
                for (l0, l1), kind in zip(a["lanes"], kinds):
                    if not kind:
                        continue
                    w, h = arrow_size(kind, v)
                    if w > abs(l1 - l0) * 1.15:
                        continue                                  # the lane is too narrow for it
                    for k in range(int(v["count"])):
                        tip = base + v["first_m"] + k * v["every_m"]
                        if tip + h > room:
                            break
                        x, y, ux, uy = spot(a, tip + h / 2, (l0 + l1) / 2)
                        lay(li, L, kind, x, y, ux, uy, w, h, "edge")
        if far > 0:
            # from the middle and way the markings were laid by (the kerbs' frame), else the edge
            ex, ey, eox, eoy = a["frame"][:4] if a.get("frame") else (a["x"], a["y"], a["ox"], a["oy"])
            ends.append({"x": round(ex, 2), "y": round(ey, 2), "ox": round(eox, 5), "oy": round(eoy, 5),
                         "m": round(far, 3), "reach_m": round(a["reach_m"], 2), "half_m": round(a["road_m"] / 2, 2)})
    # the yellow box inside big open junctions (or every junction)
    for li, L in [(li, L) for li, L in layers if L["preset"] == "box" and not L.get("off")]:
        v = L["values"]
        for b in P["boxes"]:
            if not (b["big"] or L["opts"]["all"] and b["on_road"] >= BOX_ROAD_SHARE):
                continue
            rect = b["poly"].buffer(-0.3 / px).minimum_rotated_rectangle if b["poly"].area > 0 else None
            if rect is None or rect.is_empty or rect.geom_type != "Polygon":
                continue
            c = np.array(rect.exterior.coords)[:4]
            e1, e2 = c[1] - c[0], c[2] - c[1]
            l1, l2 = float(np.hypot(*e1)) * px, float(np.hypot(*e2)) * px
            if min(l1, l2) < 2.0:
                continue
            # its sides rounded to half a metre (one picture serves boxes of about the same size)
            w_m, h_m = round(l1 * 2) / 2, round(l2 * 2) / 2
            cx, cy = c.mean(axis=0)
            ux, uy = _unit(*e2)
            lay(li, L, f"b{w_m:g}x{h_m:g}", cx, cy, ux, uy, w_m, h_m, "junction")
    # the highways' centrelines their lines are laid along (app/lanes.py, app/highways.py)
    ends.extend(dict(c, chain=True) for c in P["chains"])
    return placements, ends, P


def line_ends(det, mpp, layers, found, chosen, cfg):
    """Only where the lines stop (place()'s ends, with the highways' centrelines)."""
    return place(det, mpp, layers, found, chosen, cfg)[1]


def summary(plan_):
    """For the log: how many junction approaches of each kind, roundabouts and big junctions."""
    roles = {}
    for a in plan_["approaches"]:
        roles[a["role"]] = roles.get(a["role"], 0) + 1
    return {"approaches": roles, "roundabouts": len(plan_["roundabouts"]),
            "big_junctions": sum(1 for b in plan_["boxes"] if b["big"]),
            "highways": len(plan_.get("chains") or []),
            "highway_km": round(sum(c.get("length_m", 0.0) for c in plan_.get("chains") or []) / 1000.0, 1)}
