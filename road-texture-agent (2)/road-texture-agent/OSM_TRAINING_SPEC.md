# OSM training: what the OSM tool exports for the Road Texture Agent

The Road Texture Agent's **Train > OSM training** tab reads three folders, by their
paths (nothing is copied):

| Folder | What is in it | Suggested zoom |
|---|---|---|
| 1. Layout | the 40 selected areas (500 m squares) | 20 |
| 2. Details | zoomed-in crops where the markings are | 21 or 22 |
| 3. Riyadh | the city, zoomed out | 15 to 17 |

Every folder holds pairs of images of the same area, matched by name.

## Files of one area

`NAME` is the area's ID: letters, digits, `_` and `-` only (for example `RYD500_0123`).

| File | Required | What it is |
|---|---|---|
| `NAME_sat.png` | yes | the satellite map (Google Satellite, map id 1), RGB, 8 bits, PNG |
| `NAME_str.png` | yes | the street map without labels (Google Street Map, map id 66), RGB, PNG |
| `NAME_sat.pgw` + `NAME_sat.prj` | yes | the satellite's world file and projection |
| `NAME_str.pgw` + `NAME_str.prj` | yes | the street map's (the same values as the satellite's) |
| `NAME_osm.png` | no | the OSM overlay, transparent PNG, same size |
| `NAME.json` | no | the area's record (below) |

Rules:

- `NAME_sat.png` and `NAME_str.png` have **the same width and height and cover the
  same area**, pixel for pixel: their world files hold the same six numbers.
- **PNG only** (no JPEG): JPEG blurs the street map's colours and the thin paint
  lines.
- **No resizing** after the Map Combiner: the native pixels, cropped only.
- Missing tiles (white squares) are left as they came; the agent finds them. A
  list in `NAME.json` (below) is welcome.

## World file (`.pgw`) and projection (`.prj`)

The standard six lines, as the OSM tool already writes:

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
- **EPSG:32638** (UTM zone 38N, the exact 500 m and 1 km squares); ground metres
  as they are.

## `NAME.json` (optional)

```json
{
  "id": "RYD500_0123",
  "bounds": {"left": 46.6702, "right": 46.6735, "top": 24.7198, "bottom": 24.7178},
  "zoom": 20,
  "crs": "EPSG:3857",
  "satellite_map_id": 1,
  "street_map_id": 66,
  "missing_tiles": [[x0, y0, x1, y1]],
  "downloaded_utc": "2026-10-15T08:00:00Z",
  "osm_tool_version": "1.7"
}
```

`missing_tiles`: boxes in the PNG's pixels (left, top, right, bottom) where a tile
did not come.

## What the agent does with each pair

1. Checks the pair: same size, same world file; a pair that does not match is
   listed and skipped.
2. Makes the alpha from `NAME_str.png` (the same code as the Coordinates input:
   the street map's white and yellow roads).
3. Works on `NAME_sat.png` inside the alpha's roads (missing tiles cut out):
   folder 1 the road colours, widths and asphalt; folder 2 the paint and the
   markings' sizes; folder 3 the road classes and widths across the city.
4. Keeps a few areas aside to check what it learned.
