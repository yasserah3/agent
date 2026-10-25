# OSM tool handoff v2: training images for the Road Texture Agent (Riyadh and Madinah)

For whoever updates the OSM tool (`Riyadh_OSM_Tool`, OSM_Overlay_Tool v1.6.1). This
replaces the handoff of 15 October 2026 (v1). Everything in v1 is still here, in
sections 4 to 6 and 8; what changed is listed first.

## 0. What changed since v1

1. **Two cities**: Riyadh (`RYD`) and Madinah (`MED`). The tool becomes
   city-generic: a city is a row in `cities.json`, nothing city-specific stays in
   the code.
2. **Madinah is in UTM zone 37N (EPSG:32637)**, Riyadh in 38N (EPSG:32638). The
   tool currently writes 32638 into every exact square. A Madinah square made in
   32638 comes out turned about 2.2° and 0.35% too big, so the EPSG must come from
   the city.
3. **Riyadh stays exactly as it is**: its grid, its 40 IDs and its downloads are
   kept. Re-running the Riyadh batch must reuse everything and give the same PNGs.
4. **New file `NAME_str16.png`**: the street map of the same square at zoom 16.
   The agent now takes the streets from zoom 16, where Google draws no buildings
   over them.
5. **The three folders are defined**: folder 2 (Details) is 100 m squares where
   the markings are, and folder 3 (formerly "Riyadh") is now "City", made of 5 km
   squares, one set per city.
6. **Tunnels are left out** of the OSM overlay and of the road length. This
   matters in Madinah, where roads run under the plazas around the Haram.
7. **ID examples corrected**: the 500 m IDs have 5 digits (`RYD500_06094`).

## 1. Cities

| Code | City | Window: left, right, top, bottom (WGS 84) | UTM zone | EPSG | Status |
|---|---|---|---|---|---|
| `RYD` | Riyadh | 46.4832, 47.0222, 24.9823, 24.5375 | 38N | 32638 | as now, unchanged |
| `MED` | Madinah | 39.47, 39.77, 24.61, 24.35 | 37N | 32637 | new (window to be confirmed on the map) |

The Madinah window is a proposal: about 30 × 29 km around the Prophet's Mosque
(24.467 N, 39.611 E), from Mount Uhud and the airport in the north to Quba and
beyond in the south. Adjust it on the map before generating the grid. Its UTM zone
stays 37N for any window between 36° and 42° E.

The rule for any city: `zone = floor((centre longitude + 180) / 6) + 1`,
`EPSG = 32600 + zone` (northern hemisphere). Never use a neighbouring zone.

`cities.json` (new, next to `app.py`):

```json
{
  "RYD": {"name": "Riyadh", "window": [46.4832, 47.0222, 24.9823, 24.5375], "epsg": 32638,
          "grids": ["riyadh_grid.csv", "riyadh_500m_grid.csv"]},
  "MED": {"name": "Madinah", "window": [39.47, 39.77, 24.61, 24.35], "epsg": 32637,
          "grids": ["madinah_500m_grid.csv"]}
}
```

Window order is left, right, top, bottom, as in the tool's fields.

## 2. Changes in the tool's code (v1.6.1 → v1.7)

| File | Change |
|---|---|
| `cities.json` | New: the table above. |
| `Generate_500m_Grid.py` | Take `--city MED` (window, EPSG and prefix from `cities.json`) instead of the hard-coded Riyadh bounds and 32638. Write `<City>_500m_Grid_Package/` with `<city>_500m_grid.csv` (plus two new columns, `city` and `epsg`), `Selected_40_500m_Coordinates.csv`, `Batch_40_<City>_500m_IDs.txt`, the two GeoJSON files and `grid_metadata.json` (with `crs` set to the city's EPSG). Same selection rule as Riyadh: at least 20 occupied 100 m bins, 3,000 m of roads and 1,500 m of residential roads; the top 40. Leave out tunnel ways (section 7). The Riyadh grid is **not** regenerated. |
| `exact_square.py` | The EPSG comes from the grid record (`epsg` column; when missing, 32638, so Riyadh's existing CSVs work unchanged). It is used in all four places that now say 32638: the transform onto the square, the `.prj`, the metadata `crs`, and the text of `MASK_TOOL_INPUTS.txt`. |
| `batch.py` | Pass the record's EPSG to the square export. Download the zoom 16 street map after the zoom 20 one (section 3). The 1 km `RYD_` rule stays. |
| `app.py` | Load every grid CSV listed in `cities.json`. The hint text becomes "`RYD500_06094` or `MED500_00001`". The default export name comes from the city. |
| download fingerprint | Must not change for Riyadh. Add new keys to `config` only when they differ from their old default, so the 40 Riyadh downloads are reused, not downloaded again. |
| new: **Export training folders** | A button that copies finished batch outputs into the three folders, named as in section 4. It only copies, so it can run again at any time. |
| new: Details and City grids | Section 3. |

Tests to add (alongside the existing ones):
- A Madinah square's `.prj` is EPSG:32637, and its `.pgw` origin is
  `col × 500 + step / 2`, `row × 500 + 500 − step / 2`. (Check value: the Prophet's
  Mosque, 24.4672 N 39.6112 E, is at 561,940 E, 2,706,090 N in EPSG:32637.)
- A Riyadh square exported again is byte-identical to the v1.6.1 export.
- The tool refuses an ID whose grid record has an EPSG different from its city's.
- `NAME_sat.png` and `NAME_str.png` have the same size and the same `.pgw`.

## 3. The three folders and how their areas are chosen

| Folder | Areas, per city | Zoom (m a pixel) | Image size |
|---|---|---|---|
| `1_Layout` | the 40 selected 500 m squares | 20 (0.136) | about 3,700 px |
| `2_Details` | 100 m squares inside the 40, where the markings are | 22 (0.034), or 21 (0.068) | about 2,950 px (1,470 at zoom 21) |
| `3_City` | 5 km squares over the city window, those with roads | 16 (2.17) | about 2,300 px |

Ground resolution is the same in both cities at a given zoom (both are near 24.5° N).

**Details (folder 2).** Each 500 m square is cut into 5 × 5 cells of 100 m (the
same 100 m bins the grid generator already counts). A cell is scored by what OSM
marks in it:
- junctions: nodes shared by 3 or more motor-road ways;
- `highway=crossing`, `highway=traffic_signals`, `highway=stop`, `highway=give_way`;
- `amenity=parking`, and ways tagged `parking:lane` or `parking:both`.

Take up to 5 cells per square, the highest scores first, and only cells scoring at
least 1. The ID is the parent ID plus the cell, counted from the square's
south-west corner (column, then row, each 0 to 4): `MED500_00012_c4r0`.

**City (folder 3).** These are 5 km UTM squares (multiples of 5,000 m) inside the
city window, kept when they contain motor roads. IDs follow the 500 m generator's
pattern and order: `RYD5000_00001`, `MED5000_00001`. That gives about 30 squares
for Madinah and about 100 for Riyadh.

A job stays well under the tool's 1,600-tile limit in all three: about 250 tiles
for a 500 m square at zoom 20, 150 for 100 m at zoom 22, and 100 for 5 km at zoom 16.

Folder layout:

```
OSM_Training/
  1_Layout/   RYD500_06094_sat.png  RYD500_06094_str.png  RYD500_06094_str16.png ...
              MED500_00001_sat.png  ...
  2_Details/  RYD500_06094_c2r3_sat.png  RYD500_06094_c2r3_str.png ...  MED500_00012_c4r0_...
  3_City/     RYD5000_00001_sat.png  RYD5000_00001_str.png ...  MED5000_00001_...
```

Both cities go in the same three folders; the city comes from the ID's prefix and
from `NAME.json`.

## 4. Files of one area

`NAME` is the area's ID: letters, digits, `_` and `-` only (for example
`RYD500_06094`, `MED500_00012_c4r0`, `MED5000_00003`).

| File | Required | What it is |
|---|---|---|
| `NAME_sat.png` | yes | the satellite map (Google Satellite, map id 1), RGB, 8 bits, PNG |
| `NAME_str.png` | yes | the street map without labels (Google Street Map, map id 66), RGB, PNG, at the folder's zoom |
| `NAME_sat.pgw` + `NAME_sat.prj` | yes | the satellite's world file and projection |
| `NAME_str.pgw` + `NAME_str.prj` | yes | the street map's (the same values as the satellite's) |
| `NAME_str16.png` + `.pgw` + `.prj` | recommended in `1_Layout` (new) | the street map (id 66) of the same square at zoom 16, reprojected onto the same square at its own pixel size (about 2.2 m, 230 px for 500 m), with its own world file |
| `NAME_osm.png` | no | the OSM overlay, transparent PNG, same size as `NAME_sat.png` |
| `NAME.json` | no, but wanted | the area's record (section 6) |

Rules:

- `NAME_sat.png` and `NAME_str.png` have **the same width and height and cover the
  same area**, pixel for pixel: their world files hold the same six numbers.
- **PNG only** (no JPEG): JPEG blurs the street map's colours and the thin paint
  lines.
- **One resampling at most.** Use either the native crop in EPSG:3857, or the
  exact UTM square, which is resampled once (bilinear) at the native pixel size, as
  v1.6 already does. Never scale to another size afterwards.
- **The street map as Google draws it**: coloured, with its buildings. Do not
  threshold it, recolour it or use the roads-only custom style for it. The agent
  reads Google's classic colours and its 2024 colours, and it reads the buildings
  from the zoom 20 street map. (`street_roads.png`, the thresholded map, is not
  used by the agent.)
- Missing tiles (white or black squares) are left as they came; the agent finds
  them. A list in `NAME.json` is welcome.

## 5. World file (`.pgw`) and projection (`.prj`)

The standard six lines, as the tool already writes them:

```
pixel width in metres            (A)
0                                (D)
0                                (B)
minus the pixel height in metres (E)
x of the CENTRE of the top-left pixel   (C)
y of the CENTRE of the top-left pixel   (F)
```

The `.prj` holds the projection's WKT, one of:

- **EPSG:3857** (Web Mercator, as the Map Combiner's "TIFF in metres"); the agent
  turns Mercator metres into ground metres with the latitude.
- **EPSG:32638** (UTM 38N: Riyadh's exact squares) or **EPSG:32637** (UTM 37N:
  Madinah's exact squares); ground metres as they are.

## 6. `NAME.json`

```json
{
  "id": "MED500_00012_c4r0",
  "city": "MED",
  "folder": "2_Details",
  "parent": "MED500_00012",
  "bounds": {"left": 39.6108004, "right": 39.6117915, "top": 24.4672874, "bottom": 24.4663803},
  "zoom": 22,
  "crs": "EPSG:32637",
  "square": {"size_m": 100, "utm_x0": 561900, "utm_y0": 2706000},
  "satellite_map_id": 1,
  "street_map_id": 66,
  "street16": null,
  "osm_left_out": ["tunnel"],
  "missing_tiles": [[0, 0, 256, 256]],
  "downloaded_utc": "2026-10-25T08:00:00Z",
  "osm_tool_version": "1.7"
}
```

- `bounds`: the square's enclosing longitude/latitude rectangle.
- `square`: its south-west corner in the city's UTM metres, and its size. (This
  example is the 100 m cell at the Prophet's Mosque: UTM 37N 561,900 E,
  2,706,000 N, in the 500 m square at column 1123, row 5412.)
- `missing_tiles`: boxes in the PNG's pixels (left, top, right, bottom) where a
  tile did not come.
- `street16`: the zoom 16 map's file name, or `null` when it was not exported
  (recommended for `1_Layout` only).

## 7. OSM overlay and road length: what is left out

As in v1.6, the following are left out: unpaved surfaces, tracks, paths,
construction, proposed, and area roads.

New, for every city:
- **Tunnels are left out of the overlay and the road length**: ways tagged
  `tunnel=yes`, `tunnel=building_passage` or `covered=yes`. On the satellite they
  are plazas, buildings or ground, not road. The many roads under the plazas around
  the Haram in Madinah would otherwise be drawn across the plazas.
- **Bridges are kept** (`bridge=yes`): they are road on the satellite. A list of
  bridge ways in `NAME.json` is welcome.
- Record what was left out in `NAME.json` (`osm_left_out`). Changing this must not
  change the download fingerprint (section 2).

## 8. What the agent does with each pair

1. Checks the pair: same size, same world file; a pair that does not match is
   listed and skipped.
2. Makes the alpha from `NAME_str.png`, with the same code as the Coordinates input.
   It reads both of Google's colour schemes. When `NAME_str16.png` is there, it
   takes the streets from it (no buildings cut them) and the buildings from
   `NAME_str.png`.
3. Works on `NAME_sat.png` inside the alpha's roads, with missing tiles cut out:
   - folder 1: the road colours, widths and asphalt;
   - folder 2: the paint and the markings' sizes (stop lines, crossings, parking);
   - folder 3: the road classes and widths across each city.
4. Keeps the city of every area, so it can learn what Riyadh and Madinah share and
   where they differ.
5. Keeps some areas aside to check what it learned. Neighbouring squares, and a
   square with its detail cells, stay on the same side, so nothing is checked
   against an area it has already seen.

## 9. Madinah: things to expect in its images

- **The centre**: wide pedestrian plazas around the Prophet's Mosque, with roads in
  tunnels under them (hence section 7).
- **Dark lava fields (harrat)** border parts of the city. On the satellite their
  basalt is as dark as asphalt. The agent only reads the satellite inside the
  street map's roads, but a few squares where roads cross lava ground are useful
  training. If the road-length rule does not pick any, swap one or two of the 40
  for such squares by hand.
- **Mountains** (Uhud to the north, Ayr to the south): roads on slopes and long
  shadows at low sun.
- The same Google maps (ids 1 and 66) and zooms as Riyadh, and the same GCC PBF
  (`gcc-states-261011.osm.pbf` covers both cities), so no new downloads or settings.

Map data: © OpenStreetMap contributors, ODbL 1.0.
