# Road Texture Agent

A standalone, local tool. Nothing is sent anywhere, and no external AI service is used.

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
python server.py
```

Then open **http://127.0.0.1:8765** in your browser.

On Windows, if `python` is not recognised, install Python from python.org with
"Add python.exe to PATH" ticked, and turn off the Store aliases under
Settings > Apps > Advanced app settings > App execution aliases.

## What works today

- **Uploads.** Every image is stored in `workspace/uploads` and recorded in memory.
- **Isolated noise.** Material with noise minus material alone, computed on the server.
  A size mismatch is refused with a clear message, and the refusal is recorded too.
- **Junction detection.** Cleaning, road widths, centreline, junctions, arms,
  classification and an overlay image. Tested on a real mask: 13 junctions found
  (8 crossroads, 1 skewed crossroads, 3 T-junctions, 1 interchange flagged).
- **Memory.** The step log, the keyed situation cache, attempts and trees.

- **Stage A priming.** The three methods run on the three reference images:

  | Method | What it measures |
  |---|---|
  | 1, neighbour pixels | tone, local contrast, grain size, whether the surface has a direction |
  | 2, far pixel pairs | repetition: dash length, gap length and cycle, measured rather than guessed |
  | 3, random rectangles | a patch library that generation draws from, so wear varies instead of repeating |

  It then creates the three trees (material, wear, lines), each asking about
  junctions first, and saves them to memory. Runs in about a second.

- **Stage B training.** An aligned mask and photo pair is sorted into three
  groups using the junction geometry: inside a junction, within 1.5 m of the
  kerb, and open road. Each group is measured on its own and gets its own patch
  library, so the trees learn that a kerb is not the middle of the road.

  Painted markings are cut out of the photo before anything is sampled. Only
  the lines tree knows where a line belongs, so paint left in the material or
  wear libraries would be scattered at random across the road as bright specks.
  Any patch that even clips a line is rejected.

  Patch size follows the measured road width and kerb band, because a fixed
  size finds nothing on a narrow road, and shrinks further if clean patches
  cannot be found between the paint and the kerb. If the road is under 20 px wide the
  result says so, since there is no surface detail to learn at that scale.

- **Generation.** A new mask is filled in three layers:

  1. *material*, built from the patch library for each part of the road, placed
     with random orientation and feathered where patches meet;
  2. *wear*, laid over it at the strength you choose;
  3. *markings*, dashes along each street's centreline, stopping 2 m before
     every junction so each street starts and ends on a whole dash.

  Each layer is saved on its own, so you can look at them separately. The dash
  to gap ratio comes from the line image measured in priming; the cycle length
  in metres and the marking width are yours to set.

## Memory tab and the ordered libraries

Every pair keeps its own library for each part of the road: junction, kerb and
open road. Libraries are never mixed, so a road never shows a patchwork of
differently lit photos.

For each part there is one **ordered list** of pairs. The agent reads it top to
bottom: it uses the first pair that has enough patches and has not been
rejected for that part, and only moves down when that one is rejected. If every
pair is rejected, the library from priming is the last resort.

- Order is by weight: road pixels the pair supplied, times its confidence.
- Open road picks first. The kerb and the junctions then try the same pair, so
  one road's parts come from one photo.
- Rejecting a pair **halves its confidence** and makes it **sit out the next
  generation**, so you see the next library straight away. After that it is
  back in the list at its new weight. A pair only slightly ahead falls after one
  rejection; one that supplied far more road needs two or three before it moves
  down, which is how repeated rejections push it down rather than one.
- Accepting a pair raises its confidence by 0.05 and ends any sit-out.
- Reordering rewrites only the list, never the patches, so it is instant.
- Adding pairs measures only the new ones. Deleting a pair removes its files and
  its place in every list.

The Memory tab shows every trained pair and the three ordered lists.

## 3D model (stage 1: flat road)

After generating, **Download 3D model (GLB)** builds a road model from that
result:

- The outline is traced from the same smooth distance field the texture uses,
  so the mesh edge sits exactly on the painted road edge. On test data the mesh
  and texture road areas agreed within 0.1%.
- City blocks and interchange loops become holes in the road surface.
- Units are metres, Y is up, and the model is centred on the origin. Image right
  is +X, image down is +Z, so it reads like the picture from above.
- The generated texture is embedded as a JPEG, capped at 8192 pixels, the
  largest Unreal accepts without extra settings.
- Near-flat sliver triangles are dropped, so every triangle faces up.

Blender: File > Import > glTF 2.0. Unreal: drag the file into the Content
Browser. Both keep the metre scale.

### Quad mesh, straightened (the default)

- Each street is rebuilt from its centreline. The centre and width are measured
  on the smooth distance field against both kerbs, to a quarter pixel, because
  a skeleton is one whole pixel wide and sits off centre on even-width roads.
  The kerbs are then offsets of the centreline, so a straight street gets two
  straight, parallel edges.
- Rows run across the road every *row spacing* metres, with a centre vertex, so
  each row is two quads.
- Junctions are quad patches: a rim along the junction edge, two rings stepping
  inward, and a centre fan. Corners are curves that follow both kerbs. They share
  vertices with the street ends, so there are no cracks.
- **Straightness** 0 to 100: 0 follows the mask closely, 100 gives each street
  one constant width and a strongly smoothed centreline.
- GLB stores triangles only, so each quad is written as two triangles sharing a
  diagonal. In Blender, Tab into edit mode and use Face > Triangles to Quads
  (Alt+J) to see the quads. A quad bent the other way is split on its other
  diagonal, so every quad stays a rejoinable pair.
- Interchanges stay as traced triangles for now, since they are multi-level.
- The texture's road colour is carried a few pixels out into the background, so
  a straightened edge never shows a dark seam.

On a test mask of 28 streets and 12 junctions, the quad mesh matched the road
area within about 1.5% beyond pixel rounding, with every quad intact.

*Traced outline* is still available: it follows the mask exactly, as triangles.

Next stages: repeating materials along each road with dashes as their own
strips (sharp at any zoom), then kerbs, pavements and camber.

## Material tiles (stage 2, part 1)

Small seamless squares of road surface that will repeat along every street in
3D, so the road is sharp up close however large the city. By default a tile is
1024 px covering 4 × 4 m, about 4 mm per pixel.

- **Close-up detail comes from the priming close-up**, the "material with noise"
  photo. The training pairs are photographed from far above and hold no detail
  at this scale. Set how wide that photo is in real life in the Memory tab.
- **Each part's tone and contrast come from training**: the pair at the top of
  each part's order. Parts with no usable pair use the priming tone.
- **Built against visible repetition**: tiles wrap seamlessly, large blotches and
  extreme spots are removed (they are what the eye notices repeating), and
  several variants are made per part.
- Two scores per tile: **seam**, the difference across the wrap-around edge
  against the tile's own neighbour differences (1 means invisible), and
  **noticeable**, medium-sized features against fine grain (lower is harder to
  spot repeating). On test data: seam 1.01 to 1.05, noticeable 0.06 to 0.07,
  against 1.22 and 0.09 for a plain crop of the same photo.
- Rebuilt automatically whenever the libraries change, or by hand from the
  Memory tab with new settings. Old tile sets are deleted.

## Tiled 3D export (stage 2, part 2): the default

*Tiled material, sharp up close* is now the default in the 3D model settings.

- **Tiles laid along each street** with texture coordinates in metres along and
  across it, so the tile follows every bend. Junctions use a flat layout.
- **Every street and junction** gets its own tile variant, offset, rotation and
  flip.
- **Tone per part through vertex colours.** All tiles share one tone; kerb,
  open road and junction tones are vertex colours, so they fade into each other
  across a quad. Giving each part its own tone in its own material made a hard
  line at every boundary.
- **The large variation layer**, from the generated image with the dashes
  removed and blurred to several metres, is multiplied into the same vertex
  colours. Set its strength with *Large variation*.
- **Dashes are their own mesh**, 1 cm above the road, with the generation's
  spacing and line width, so they are crisp at any distance.
- **Interchanges** keep their traced triangles, with the open-road tile.
- **A built-in repetition check**: along the longest street, the saved model is
  rendered from above and the similarity one tile apart is compared with nearby
  distances. A visible repeat shows as a peak. On test data: 0.13 with tiles
  alone, 0.002 with the variation layer.

The variation and part tones are stored as glTF vertex colours (COLOR_0), which
the glTF standard multiplies into the base colour. If an importer leaves the
road evenly toned, connect them in the material: multiply the Color Attribute
(Blender) or Vertex Color (Unreal) into Base Color.

## Sidewalks

Built automatically with the tiled 3D export, as two more meshes: **Sidewalk**
(the paving, with a kerb stone along its road edge) and **Kerb** (the vertical
face down to the road, facing the road).

- **Height**: 10 cm by default. The road stays at 0.
- **Width**: 30% of the road's width, kept between 1.5 m and 2.5 m.
- **Never overlapping**: before each row, the free space beyond the kerb is
  measured on the distance field out to the next road, and a sidewalk takes at
  most half of it. The measurement first leaves the street's own road, since
  near a junction the straightened kerb sits inside the painted road's flare.
- **Corners** wrap around each block, narrowed where the curve would fold over.
- **No sidewalks** on interchanges, or on roads wider than 20 m.
- **Paving material**: from an optional paving close-up loaded in the Prime
  panel (set its real width in the Memory tab). If the photo repeats, a grid of
  slabs or bricks, the tile is cut to a whole number of repeats starting on a
  joint line, so the grid stays aligned and the joins fall between slabs. If
  not, the asphalt method is used. Without a photo, sidewalks are light
  concrete. Slabs run along each street, parallel to the kerb.
- **Distance along** the sidewalk is measured on its kerb, not the street's
  centreline, so paving is not squeezed on the inside of bends.

Known limit: block corners are a few straight segments, the same points as the
road's junction corners, so they read as slightly faceted up close.

## Tight groups of junctions, and no holes

Some places have several junctions a few metres apart: slip roads cutting the
corners of a crossroads, a Y with a triangular island, a roundabout, or a
junction cut off by the image border. There the centreline breaks into short
pieces and overlapping junctions, and patching each junction alone left holes,
patches joined to the wrong roads, and the real crossing belonging to no
junction at all.

- **Groups**: junctions around the same small island (under 2,500 m²), or
  nearly touching, are handled as one area; groups whose areas nearly touch are
  merged. Junctions at the image border are handled the same way.
- **Inside a group**, junctions get no patch and short links get no strip. The
  road is covered by the traced outline, exactly as in the mask, laid 3 mm below
  the road surface. Streets leaving the group keep their quad strips on top, so
  wherever a strip reaches it shows, and wherever it cannot, the fill shows.
- **A final check**: after the mesh is built, it is drawn at the mask's size and
  compared with the road. Any uncovered piece bigger than a few pixels is filled
  the same way. What remains are single-pixel slivers along edges.
- The console and the export report list every group and filled piece.

Limits: inside a group the road is triangles, not quads, and has no dashes or
sidewalks yet. Roundabouts are filled this way too; building their ring as a
curved strip of quads is a possible later step.

## Bridges

In the Generate tab, **Add bridge** places a red rectangle over the image. Drag
it to move it, a corner to resize it, the round handle to rotate it. Place it
along the road that should go over. Bridges are saved with their mask.

- **The rectangle selects the road running through it lengthwise.** Only that
  road is lifted, with its sidewalks, kerbs and dashes. Roads crossing it stay
  on the ground.
- **The crossing inside the rectangle becomes an over-and-under crossing**: the
  bridge road continues as one unbroken piece at full height, and the crossing
  road continues underneath, each with its own sidewalks.
- **Height profile along the road**: ground, a ramp of *ramp length* (120 m by
  default) rising to *height* (5 m by default), full height across the
  rectangle, then a ramp of the same length back down. The ramp is a smooth
  S-curve, flat where it leaves the ground and where it reaches full height, so
  there is no kink. Its steepest point is 1.5 times the average slope: 6.25%
  for 5 m over 120 m. Measured on an exported model: 6.23%.
- **The road is followed outward through junctions**, always taking the
  straightest way on, until each ramp has its full length. If the road ends
  first, that ramp is shorter and the console says how steep it became; above
  8% it warns.
- **Side streets meeting a ramp** rise to meet it, falling back to the ground
  over 30 m, so they meet it at its height instead of at a step.
- **Deck edges**: a side face down each edge and an underside, 1 m below the
  road surface by default, so the bridge is not a see-through ribbon.
- **Preview**: after each move, grey lines show where the road ramps and red
  where it is at full height, with the ramp lengths and steepest slope in the
  console. A rectangle with no road running along it says so.

Not yet: parapets and pillars. Junctions passed on a ramp are tilted with it but
have no deck edges.

`app/preview3d.py` renders a saved model from above, for checking results
without Blender.

## Nothing learned is ever lost

- Every tree is **versioned**. Stage A is version 1, Stage B is version 2, and
  both stay in `tree_versions`. `GET /api/memory/tree-history` lists them and
  `POST /api/memory/restore-tree` puts an earlier one back.
- Inside each trained route, the primed route is **kept as its fallback**, so a
  rejection during review falls back to what priming learned.
- The step log records every run, including failures.

## Output size and smooth edges

- **Output size** 1×, 2× or 4× the mask. The mask is cleaned at its own size
  first, then enlarged with smooth edges, so junctions and dividers come out the
  same at every size. 2× takes about three times as long as 1×.
- **Smooth edges.** The road shape is not enlarged as pixels. The mask becomes a
  signed distance field, where every pixel holds its distance to the road edge,
  positive inside and negative outside. Distance changes smoothly even where the
  mask steps, so it is smoothed, enlarged, and the edge is drawn exactly where the
  distance is zero, with a one-pixel fade into the background. Fonts are scaled
  the same way. The result is continuous curves at any output size, where
  blurring the pixels only softened the steps and made them bigger at 2× and 4×.
- Each patch carries a mask of which of its pixels were really road in the
  training photo, so off-road pixels are never pasted into the road. Libraries
  trained before this existed are handled by tone.
- Junction, kerb and open road blend into each other over a few pixels, so there
  is no hard line where two libraries meet.
- Where patches overlap, each pixel mostly belongs to the patch whose centre is
  nearest, and patches only blend in a band in the middle of their overlap.
  Averaging patches everywhere washed their grain out: at quality 3 only 69% of
  the photo's grain survived. With the middle band it keeps 100%, and the joins
  stay hidden. A stronger setting brings back visible square edges, so the band
  is set at the middle value that keeps both.

- **Texture is sized by real-world scale, not by pixels.** Each library is
  resized by the ratio between the scale it was trained at and the output scale.
  A library trained at 0.25 m per pixel and generated at 1.1 is shrunk to 0.23 of
  its size, so the grain averages out as it would from that height; at 4× output
  it is shown at 0.91 of its size, with the road four times wider in pixels.
  Before this, patches were pasted at their training pixel size whatever the
  output, which made the grain far too big and the surface look like static.
  This makes the training scale matter: enter the real metres per pixel for each
  pair in the Train tab. The Memory tab shows the scale each pair used.

Surface detail is still limited by the training photos: enlarging the output
sharpens edges and markings, but asphalt detail only improves with training
pairs where roads are 40 px wide or more.

## Smooth markings

Dashes are drawn as smooth strokes, the same way the road edges are: every
pixel near a dash gets its distance to the dash's centreline, and its coverage is
how much of it falls inside the stroke, so the edge fades over one pixel instead
of stepping. The centreline is always smoothed a little first, so dashes follow
the road's direction rather than the skeleton's pixel steps; "Lines are not
aligned" smooths it further.

The viewer shows the zoom percentage in its toolbar, with − and + buttons,
Fit, and 100% for real size. Past 100% the percentage turns amber and a note
says the image is being enlarged, since that is when it starts to look soft.
The viewer scales smoothly when zoomed in, and only shows hard pixel squares
beyond 800%, for inspecting individual pixels. Zooming past 100% still enlarges
the image rather than adding detail: for close-up work, generate at 2× or 4×.

## Line width

By default, lines are drawn as a **share of each street's width**, learned from
the training pairs. During training, the painted lines in each photo are
measured along their own centreline and compared with the road's width; the
pairs are combined with a weighted median. At generation, every street gets
lines at that share of its own measured width, so main roads get wider lines
and side streets narrower ones, and the proportion is the same at 1×, 2× and 4×
and does not depend on the scale being exactly right.

The Memory tab shows the learned share. Choosing *Fixed, in metres* in the
Generate panel overrides it. If no training photo showed painted lines, the
fixed width is used and the console says so.

## Surface grain and colour

- **Surface grain**, 0 to 100, controls the fine detail inside the material.
  The material is split into its broad tone and its grain, and only the grain is
  scaled: 0 is clean, 50 is exactly as learned, 100 is twice as rough. The road
  keeps its colour and its larger light and dark areas at every setting.
- **Wear** is separate: the darker marks laid over the surface. For a truly
  clean road, lower both.
- **Match my material colour** shifts the road's tone to the one measured on
  the material-alone image during priming, keeping the grain.

## What is wrong? corrections

The options unlock as soon as a result exists. Without clicking a spot they
apply to open road; click a spot first to target a junction or kerb. Settings
corrections change one setting and generate again, keeping the same library and
seed so nothing else moves:

| Option | What it changes |
|---|---|
| Material is too noisy | 15 less surface grain |
| Material looks too flat | 15 more surface grain |
| Lines are not aligned with the road | the centreline is averaged before the dashes are placed, so they follow the road axis instead of the skeleton's diagonal steps |
| Image has bad quality | quality up one step, to 3 at most: patches overlap more and are placed with less jitter, which softens the joins |
| Material looks wrong, try the next library | the only option that changes the library: the current one sits out and the next in the order is used |

The options stay greyed out until a result exists, and the apply button stays
greyed out until an option is chosen. After applying, the choice is cleared so
the same correction is not applied twice by accident. Every correction is recorded in memory
with its reason.

## Known limits

- Patch seams can show at high zoom on wide roads. Larger overlap or
  match-based patch choice would reduce it.
- Cloverleaf interchanges are flagged, and get ordinary road material and
  markings. They are multi-level, so they belong with the 3D work.
- Review and correction are next: the buttons exist but do not yet feed the
  memory.

## Memory

`workspace/memory.db` is a SQLite file with five tables:

| Table | What it holds |
|---|---|
| `steps` | Every action in order. Append only, nothing deleted. |
| `images` | Every uploaded image, with size and checksum. |
| `situations` | One row per distinct situation, keyed. |
| `attempts` | Every route tried for a situation, with outcome and reason. |
| `trees` | The current decision trees. |

The key is a short signature such as
`material|edge_m=3.0|junction=n|contrast=0.4`. Values are bucketed on purpose,
so two spots 3.1 m and 3.3 m from the edge count as the same situation and share
what was learned. Before acting, the agent looks up the key and skips routes
already known to fail there, which is what stops it repeating a mistake.

The step log is the source of truth. Situations and trees can be rebuilt from it.

Read it yourself at any time:

```
sqlite3 workspace/memory.db "SELECT id, kind, summary FROM steps ORDER BY id;"
```

## Junction rules

Fixed rules, not learned:

- Black strips inside a road narrower than **6 px** are dividers, so the two
  sides are one road. Anything wider stays two separate roads.
- A branch that ends near the image edge is a road leaving the picture, never
  pruned as noise.
- Junction types come from the number of arms and the angles between them.
- Cloverleaf interchanges are detected and **flagged**, not handled. They are
  multi-level, so they belong with the 3D work later.

Learned later: how junction surfaces look (material and wear), from junction
areas in your training pairs.

## Layout

```
server.py            the local server
app/memory.py        step log, keyed situations, trees
app/junctions.py     junction detection and classification
app/curves.py        placements' maths: curves, packages, random spaces, ids, space cells
app/placements.py    where each copy of a placement goes, for the 3D model and inner streets
app/streets.py       inner streets drawn between objects, built into the street mask
app/images.py        loading, hashing, noise isolation
ui/index.html        the interface
ui/app.js            the interface logic
workspace/           created on first run: uploads, artifacts, memory.db
```

## Markings painted into the road texture

Under 3D model, **Markings** can be *Separate strips on the road* (a mesh of
thin quads 1 cm above the road) or *Painted into the road texture* (no extra
geometry).

Painted markings use a **marked road texture** holding exactly one dash cycle
along the street, a dash then a gap, with the dash along its centre and the
road's own grain. On each street, the quads where the dashes run get this
texture, laid along the street so each repeat is exactly one cycle, starting at
the setback: the dashes land exactly where the strips would be. Near junctions,
where there are no dashes, the ordinary road texture continues.

- The grain joins itself where the texture repeats: the cycle is rarely a whole
  number of tiles, so the tiles are stretched slightly along the street to fit
  (two 4 m tiles stretched 12% make 9 m, invisible in grain).
- One texture per line width, rounded to 5 cm, so learned widths still apply.
- The texture is capped at 2048 pixels on its longer side, about 8 mm per pixel,
  so dashed streets are slightly softer close up than the 4 mm road tile.
- Dashes on bridge crossings stay as strips.

On the test mask: all 1,203 dashes painted, no markings mesh, 9,124 fewer
triangles (7%), same positions as the strips.

## Interchanges and complex junctions: real roads, fill only at knots

Inside group areas (tight junction groups, interchanges, border junctions) the
road is no longer one traced fill. The centreline is re-read at full detail:

- **Knots** are the points where pieces really meet. From each knot, every road
  is followed outward and cut where it has separated from the others and is back
  to its own normal width, so a right-angle crossing gets a small knot and a
  shallow merge a long, tapering one.
- **Every piece between knots is a real road strip**: quads, kerb band,
  sidewalks, dashes. Ramps and slip roads are smoothed over a few metres only,
  so they follow their curves. Roads entering an area continue as one piece.
- **The fill covers only the knots**, with its outline smoothed so it does not
  wobble, and still sits 3 mm below the road. The final bare-road check still
  runs afterwards.
- **No centre dashes on narrow pieces** (under 80% of the typical road width:
  single-lane slip roads and ramps), and inside groups a piece needs room for
  at least two dashes.
- **Sidewalks**: none inside highway interchanges; along roads in urban groups,
  with scraps shorter than 8 m dropped.
- **Painted markings** use at most three line widths, so many short pieces do
  not each create a texture.

On the test masks, about half of each group area is now real road strips; the
rest is knots. Different pieces overlap on about 0.2% of the road area, in
slivers where strip ends meet at a knot.

Next: sidewalks that wrap around knots instead of stopping at them, and
the remaining overlap slivers.

## Sidewalks along the kerb line (replaces per-street sidewalks)

Sidewalks are no longer built per street and joined at corners. The road
surface is assembled from the exact shapes of every street strip, junction
patch and knot fill, and its edge, the kerb line, is walked around each block
and island:

- **One continuous sidewalk per block or island.** Where two roads merge, their
  kerbs are one line and the sidewalk follows it; at the nose of an island both
  sides narrow and meet at the tip. No scraps and no gaps at knots.
- **Same curves as the road**: the kerb line is the road mesh's own edge.
- **Same width rules**: 30% of the nearby road, 1.5 to 2.5 m, never more than
  half the space to the next road, and narrowed on the inside of tight curves
  until the outer edge no longer folds over.
- **Runs to the image border**: where the kerb line meets the edge of the image,
  the sidewalk is carried right to it and cut there.
- **No sidewalks** on highway interchanges (unchanged, for a later decision) or
  on roads wider than 20 m.
- Paving is laid along each kerb, so slabs stay parallel to it.

With bridges, sidewalks use the previous per-street method for now: roads
crossing at two heights need sidewalks at two heights, which one ground-level
kerb line cannot describe.

Known: small kinks where a straightened strip meets a knot's traced outline.

### Kerb line, second pass

- **The kerb line is straightened** like the roads: Taubin smoothing on each
  ring of the road surface (pull towards neighbours, then push back slightly
  more, so wobble goes but curves keep their size). More passes the higher the
  straightness. Points on the image border stay put.
- **Knot fills follow the straightened kerb**, so the road's edge and the
  sidewalk's edge are the same line.
- **Rows are adaptive**: a new row whenever the kerb has turned 6 degrees or run
  2 m, and always exactly at a sharp corner. Dense on curves, sparse on straights.
- **Inner edges cannot fold**: for every block and island the inner edge is the
  block's own shape shrunk by the sidewalk width, and kerb points are paired
  with it moving only forwards. Blocks at the image border use the same method,
  with the frame enlarged so the border itself never gets a sidewalk. Islands
  too small to shrink are paved completely.
- **Fan rows kept**: rows fanning round an inner corner meet at one point, so
  they are really triangles. The export used to drop such quads entirely,
  leaving a hole; it now keeps the half with area. This applies to roads too.

Measured on the user's mask: folds went from 24% of sidewalk area (a bug that
rotated the inner edge around each island) to 0.03%.

### Sidewalks on every road (default)

Sidewalks now go on every road, including interchanges and roads wider than
20 m. The previous behaviour, leaving interchanges and very wide roads without
sidewalks, is an option under 3D model: *Skip interchanges and roads over 20 m*,
off by default.

### Kerb snapped to the road

The road surface used for the kerb line was grown by a hair to merge touching
pieces and never shrunk back, so every kerb sat about 2 cm (at 1.1 m per pixel)
outside the road, visible close up as a dark line. It is now shrunk back, and
anything inside the kerb line that the strips and patches do not cover is part
of the road fill where there is a real gap, and kerb points sitting just
outside the road (within 20 cm) are snapped exactly onto its edge. Measured on
both test masks: no kerb point more than 1 mm from the road.

## Mesh detail: Full or Optimised

Under 3D model, **Mesh detail**:

- **Full**: a row across each street every 2 m, 4 quads across.
- **Optimised**: rows only where they are needed, 3 quads across (kerb band,
  road, kerb band). A row is kept where the street turns 2 degrees, where its
  width changes by 5%, at the start and end of the painted dashes (so the marked
  texture lines up), every 4 m within reach of a bridge (so ramps stay smooth),
  and at least every 20 m (the variation layer that hides the tile repeating is
  stored per vertex). Junctions, knots and interchanges keep their detail.
  Sidewalks and kerbs get a row every 4 degrees of turn, up to 20 m apart.

On the user's mask: rows along streets 6,343 to 1,160; triangles 152,929 to
48,727 (68% fewer); file 12.5 MB to 6.4 MB. All 1,244 dashes still painted,
repetition check 0.004, no kerb gap, sidewalk folds 0.03%; rendered from above
the two differ on 0.76% of pixels.

## Updating to a new version

Everything learned lives in **`workspace`**: trained pairs, libraries, priming,
tiles, corrections and saved bridges. Updates replace only the program files, so
you never need to retrain or delete anything.

**Easiest (Windows):** stop the server, then drag the new
`road-texture-agent.zip` onto **`update.bat`** (or double-click it to use the
newest zip in your Downloads folder). It backs up `workspace` to a dated zip,
replaces `app`, `ui`, `server.py`, `requirements.txt` and `README.md`, installs
any new packages, and leaves `workspace` untouched.

**By hand:** copy those same files and folders over the old ones and keep
`workspace`.

Rarely, an update changes what is measured during training; the update notes
say so when it happens, usually for specific pairs only. When tiles change,
press Rebuild tiles once in the Memory tab.

## Blocks and islands

Under 3D model, **Blocks and islands** (on by default) fills every area between
roads as a flat plane with the sidewalk paving, at the sidewalk height.

- The shape is everything inside the image that is not road, using the same
  straightened kerb line as the sidewalks.
- **Blocks snap to the sidewalk**: the whole sidewalk (paving, kerb stone, paved
  islands) is cut out of each block, so they meet edge to edge at exactly the
  sidewalk height, with nothing underneath. Measured: 0.01 m2 of overlap
  (rounding) against about 5,300 m2 of sidewalk in a test window.
- **Kerb faces wherever a block meets the road directly**: with sidewalks off,
  and wherever a sidewalk run is missing (image border, runs too short). The
  image border gets no face.
- Small uncovered gaps remain, about 0.02% of the area, where pieces meet.
- Blocks are a **grid of quads** aligned across the whole map: 8 m cells at Full
  mesh detail, 16 m at Optimised. Cells cut by the kerb line are clipped and
  split into a few small triangles so the edge follows the curve exactly.
  (A first version used long thin triangles; the variation layer blended along
  them showed as diagonal streaks.)
- Blocks get their own gentle weathering: the variation layer blurred over about
  25 m at a third of its strength. Inside a block there is no road in the
  generated image, only the nearest road's colour copied outwards, which made
  stripes.
- On the test mask (Optimised): 43 blocks, 5,896 quads plus small edge
  triangles, covering 93% of the image; road and blocks together cover all of it.

## Workspace location

The server prints which `workspace` it uses at startup and in the console, and
warns when it had to create a new, empty one. To keep the workspace somewhere
else (so updates can never touch it), put its full path into `workspace.txt`
next to `server.py`, or set the `RTA_WORKSPACE` environment variable.

## Objects (scatter)

In the Generate tab, **Objects** sits above Foliage (not active yet) and Bridges.

- **Import** a GLB, OBJ or FBX. Model it with its **front facing +Y** (Blender's
  green arrow). The **↻** button on a layer turns an object a quarter turn
  within its rectangle, for objects modelled facing another way.
- Objects keep **their own axes**, the ones Blender shows on the object
  (Transform orientation: Local), whatever the object was rotated to in its
  scene. The white arrow on the map points to the object's own +Y, and at
  rotation 0 that is the top of the map. Only the object's turn about the up
  axis is undone, so it still stands as it did: a model carrying Blender's
  rotation X 90 is not laid down. When the file holds several separate objects
  (walls, windows and so on), the axes of the main one, the one with the most
  geometry, are used, and every other part keeps its place relative to it. The
  console says which axes were used after each import. OBJ files store no
  object axes, so an OBJ keeps the axes of the scene it was exported from: use
  GLB or FBX to keep an object's own axes.
- Nothing turns an object away from its axes: the earlier squaring-up by the
  tightest rectangle around the footprint (up to 45 degrees) is gone, and so is
  a bug that laid FBX files from Blender on their side. Objects imported
  before this change keep their old orientation: import them again.
- Each object is brought to metres, Y up, its footprint centred and its base at
  height 0, and becomes a layer showing its real size. FBX files write units inconsistently between programs (a test file
  came in at 150 m), so each layer has a **scale** to correct it; a size over
  60 m is flagged.
- **Add selected object to the map** places a rectangle at the object's real
  size, relative to the streets. Drag its corners to add copies: it snaps to
  whole copies and never goes below one. Blue between copies is the gap; set
  **Gap along X** and **Gap along Y** for the selected placement. The round
  handle rotates.
- Copies **stand only on blocks and sidewalks**: in the viewport a copy on the
  road is crossed out, and the export leaves it out.
- In the GLB each object is stored **once** and placed as instances, so many
  copies cost almost nothing in file size. Copies stand at the block and
  sidewalk height.
- Placements are saved with their mask, like bridges.

### Curved placements

For copies along a curved street, select a placement and press **Bend into a
curve**. The rectangle becomes a dashed line through its middle, with a point at
each end, holding the same copies as before.

- **Bend it**: double-click the line (or press **Add point**, which adds one in
  the middle of the longest stretch) and drag the new point. The line always
  passes through every point and bends smoothly between them (a centripetal
  Catmull-Rom curve, which never loops or overshoots). Double-click a point in
  the middle to remove it. Drag an end point to make the line longer or shorter.
- **Copies fill the line** at the *Gap along the line*, the run centred on it,
  so a longer line holds more copies. *Rows* adds parallel rows beside the line,
  *Gap between rows* spaces them.
- **Copies turn with the curve**: each copy's width runs along the line and its
  front, its +Y, faces one side of it; the small white arrows show the fronts.
  **Flip side** turns every copy to face the other side, for example towards the
  street. The ↻ quarter turn of the object still applies on top.
- Drag the blue band to move the whole curve. **Make straight again** turns it
  back into a rectangle from its first point to its last, facing the same way.
- What the map shows is what the 3D model gets: the map and the export use the
  same curve steps (`ui/app.js` and `app/curves.py`). Copies on the road are left
  out, as with rectangles. On a tight bend, copies on the inside of the curve
  can touch; add a gap or move the line outwards.

### Packages: several objects in one placement

A package mixes several objects, for example different buildings along one
street. Under Objects, press **Create package**, then **Import object** in it
(several files at once are fine). Each object becomes a **slot** with its own
colour, **weight**, **scale** and **↻** quarter turn; × removes it. Rename the
package by editing its name. Objects in a package do not appear as layers.

- **Place on the map** puts the package in the middle of the view. It works
  like a layer's rectangle: drag it to move it, a corner to make it longer or
  add rows, the round handle to rotate it, and **Bend into a curve** for curved
  streets, with the same points, **Flip side** and **Rows** as a layer.
- **The mix**: each spot gets a random object from the slots. The weight sets
  how often: 2 is picked about twice as often as 1, 0 never. Two neighbours are
  never the same object when the package has more than one, so a heavy weight
  cannot fill a row on its own. **Shuffle the mix** picks a new random order.
  The mix is saved with the placement and never changes on its own; making a
  row longer only adds to its end.
- **Spacing**: each object takes its own width along the row, with the same gap
  between every pair; a row holds as many as fit in its length, the run
  centred. Rows are as deep as the deepest object.
- **Fronts in line**: every object's front sits on the row's front edge, so a
  row of different buildings has one street line; deeper objects reach further
  back.
- On the map, copies are coloured by slot and each has a small arrow at its
  front. The panel lists how many of each object a placement holds.
- The map and the 3D export use the same steps and the same random numbers
  (`ui/app.js` and `app/curves.py`), so the model holds exactly the mix the map
  shows. In the GLB each object is still stored once and placed as instances.
- Packages live in `workspace/packages`, their objects in `workspace/objects`.
  Deleting a package deletes its objects and its placements.

GLB objects with a plain colour (no texture) now keep that colour; before, they
came in grey.

### Random spaces

Under **Random spaces** in a selected placement's panel, set a **min** and
**max** in metres for **X** (the spaces along the rows) and **Y** (the spaces
between rows), then press **Randomize spaces**.

- Every space along a row becomes its own random distance between the X min and
  max, and every space between two rows its own one between the Y min and max.
  A row moves as a whole, so its fronts stay in line.
- Each press gives a new random set. The set is saved with the placement and
  never changes on its own. Changing a min or max keeps the set and stretches it
  to the new range. **Even spaces** goes back to the gaps above, which are greyed
  out while spaces are random; the ranges are kept for next time.
- It works for layers and packages, straight and curved. A curve or a package
  holds as many objects as fit with the random spaces. A layer's rectangle keeps
  its number of copies, so each row gets its own length, centred, and the
  rectangle is as long as the longest row. Dragging a corner counts copies and
  rows with the middle of the range.
- The random spaces have their own seed, apart from a package's mix: Shuffle
  does not change the spaces, and Randomize spaces does not change the mix. The
  map and the 3D export use the same random numbers, as with packages.

### Inner streets and islands

Select a placement and press **Draw inner streets**. The spaces between its
objects split into cells:

- **x**: the space between two neighbours in a row (`x1-2`: row 1, after column 2);
- **y**: the space between two rows, beside the objects (`y1-1`: rows 1 and 2, first);
- **j**: a junction, the small cell where a space in a row meets the space
  between the rows.

Click a cell to make it a street (green), or press and drag across several;
whatever the first cell becomes, the others do too, so the same drag can take
streets back (grey). Press **Done drawing streets** (or Esc) to leave; outside
this mode the spaces look as before. Works for layers and packages, straight
and curved, with even or random spaces.

When you press **Generate**, the drawn streets become streets in a copy of the
street mask, so everything after it treats them like the main streets:

- **Road, kerbs and sidewalks**: the road runs down the middle of each space,
  with a sidewalk of the **Sidewalk width** (2 m by default) between it and the
  objects, wrapping round each object's corner on a curve. A space needs room
  for its road between two sidewalks; the console names spaces too narrow.
- **Rounded corners** to the **Corner radius** (4 m by default, at most three
  sidewalk widths, never closer to an object than half a sidewalk).
- **Joining the streets**: a street reaching the edge of the placement goes on
  straight to the nearest street, up to 50 m, unless that would run through an
  object of any placement; otherwise it ends at the edge.
- **Islands**: the areas between streets become islands, like any city block:
  raised to sidewalk height and paved like the sidewalks, the objects standing
  on them. Specks of island under 40 m² left between streets become road.
- **Markings**: tick **Markings on inner streets** for the markings of the main
  streets, as strips or painted in, whichever the 3D model is set to; untick it
  for inner streets with no markings at all.
- Inner streets never run over an object, even where placements overlap.

After changing inner streets (or moving a placement that has them), press
Generate again: the 3D export says so if you forget.

### Turning single objects

Every object in a placement has an id, **row-column** counted from the top
left: `1-1` is the first object of the front row, the first along the line.
Press **Turn single objects** to see them; click an object to pick it,
Shift-click to pick more. **↺ 90°** and **↻ 90°** turn the picked objects a
quarter turn, **Rotation** sets an angle (degrees, clockwise on the map), and
**Reset picked objects** puts them back. The turns are kept with the placement
by id and apply in the 3D model; a turned object that reaches the road is left
out like any other.

Reading FBX: binary FBX only (Blender's default), meshes, UVs, model
transforms, unit scale and axis settings, material colour and embedded textures.
ASCII FBX, rigs and animation are not read. OBJ textures need their MTL and
image files, which a single-file import does not include, so OBJ objects come
in untextured.

## When something goes wrong

An unexpected server error now shows its real cause in the console, the error
and the line where it happened, instead of "500 Internal Server Error". The full
report is saved in `workspace/errors/error-<date>-<time>.txt`; send that file
when reporting a problem.

A problem with one object no longer stops the export: it is left out, with a
console line naming it and saying why.
