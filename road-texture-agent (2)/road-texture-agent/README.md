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

After generating, **Download 3D model** builds a road model from that
result. A window in the middle of the screen asks first:

- **Format**: **GLB** (glTF binary, one file with its textures inside, as
  before), **FBX** (binary FBX 7.4, as Blender and the FBX SDK write it) or
  **OBJ** (with its .mtl). FBX and OBJ come as a zip: the model and a
  `textures` folder it refers to; unzip it, then import the model. All three
  hold the same model: metres, Y up, every material with its colour, normal
  (bump) and roughness pictures, the variation layer as vertex colours, decals
  with their transparency. Checked by importing each into Blender 5.0: the
  same objects, bounds and materials (base colour, roughness and normal map
  connected) from all three.
- **Combine streets and kerbs**: on, the streets, sidewalks and kerbs (and
  bridge decks) are one object, "Streets", each part keeping its materials;
  the blocks and islands and the markings stay objects of their own. Off,
  each is separate (Road, Sidewalk, Kerb...).
- **Combine objects**: on, every placed object (all copies of all objects) is
  one object, "Objects". Off, each copy is its own object: in GLB and FBX the
  copies share one mesh (instances, small files; Blender makes them linked
  duplicates); OBJ has no instances, so each copy is written out.

The choices are kept in this browser. The model itself is built once and
shared (the Top view and the 3D tab use it); the format is written from it
(app/formats.py).

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

## Big maps and markings at coarse scales

**Markings, worked out in pixels.** Every length in metres becomes pixels by
dividing by the scale (m per px): at 1.1 m per px, 12 m is 11 px and 15 cm is
0.14 px.

- **Dash cycle** (one dash and its gap): lane lines repeat every 9 to 12 m in
  most countries (3 m dash + 6 m gap in European towns, 3 m + 9 m in the US and
  the Gulf). Keep it above about 6 px, or the dashes run together into a solid
  line. Generate says so when it is too short: a 1 m cycle at 1.1 m per px is
  under a pixel, every street becomes one long line, and it lays tens of
  thousands of dashes (slow).
- **Dash length**: how long each painted dash is (the rest is the gap); empty
  takes it from your training line image, or 60% of the cycle. Typical: 3 m.
- **Line width** (fixed): real lane lines are 0.10 to 0.15 m (edge lines 0.15 to
  0.20 m). A line narrower than a pixel is drawn one pixel wide but only as
  strong as the share of the pixel it covers (a 15 cm line in a 1.1 m pixel at
  14%), the way an aerial photo at that height shows it, instead of being
  widened to a whole pixel (1.1 m, seven times too wide).
- For crisp markings at a coarse scale, the 3D model's markings as strips are
  real geometry at their real size, whatever the texture's pixel size.

Recommended at 1.1 m per px: Dash cycle 12, Dash length 3, Line width fixed
0.15, Quality 1 (at this scale the patch joins Quality smooths are under a
pixel; Quality 3 doubles the time for no visible change).

**Speed.** Generate used to go over the whole picture once for every junction
and every street (finding junction arms, cutting junction discs, the junction
and group maps, each street's dashes). On a city-sized map (thousands of
junctions, millions of pixels) that was billions of steps: a 3.7 × 2.5 km map
(3328 × 2304 px, 1064 junctions) took 18 minutes. Each step now works on that
junction's or street's own pixels, the material and wear patches are laid only
where they are used (not on the background), each patch's orientations are
prepared once: the same map takes about 2 minutes (1 minute at Quality 1),
with exactly the same picture, pixel for pixel.

The **3D scene** had the same kind of problem: for every kerb point, block
edge, junction group, lamp and dash it asked the whole city's road, kerb line,
sidewalk or lamps (distances to the whole road outline, shrinking a block that
holds the whole city as a hole for every sidewalk run, uniting every sidewalk
piece, comparing each new lamp with every lamp so far, re-reading every dash so
far for each dash). Now an index of the outline's edges answers distances and
nearest points, each sidewalk run shrinks only the part of its block round it,
the kerb apron is worked out tile by tile, each block is cut by the sidewalk
near it, lamps look only at lamps nearby. Streets, junctions, sidewalks, kerbs
and markings come out identical; blocks and the fill under the kerbs within
0.02% (the same shapes, triangulated a little differently). On the test map
above, the 3D model went from over an hour (it never finished) to minutes.

After the model is written, a check draws about 60 m of the longest street from
above to see whether the tile repeat shows. It used to work through every
pixel of each triangle's bounding box, and the road fill under the kerbs has
long thin triangles crossing the whole stretch (millions of pixels for a few
hundred covered): on the test map that one check took about 2 minutes. It now
goes row by row through only the pixels each triangle crosses, and reads only
the road from the file: about 3 seconds, the same picture pixel for pixel.

**Each 3D model is built once.** After Generate, the Top view builds the 3D
model; Generate 3D scene (3D tab) and Export 3D model ask for the same model.
Each used to build it again, even while the top view was still building it:
twice the time on a big map, and two builds writing the same file, so the 3D
tab could load one half written. Now a request for a model already being built
waits for that build (its console line follows it, marked *shared*), and one
already built with the same texture, settings, materials, objects and tiles is
answered at once (kept for half an hour, as long as its file has not been
rebuilt with other settings since). Each model is written under its own name
and put in place whole, so a model is never read half written.

**Big scenes in the 3D tab**: a 3.7 × 2.5 km city is millions of triangles and
thousands of street lamps. Mesh detail *Optimised* (Generate tab, 3D model)
keeps far fewer rows along straight streets; Bake light *Whole place* covers it
at about 1.8 m a pixel, so bake *What you see* close to where you look.

## Time so far and time left, in the console

Every long job has a line in the console that updates about once a second:
what it is doing now, how far along it is, the time so far and about how long
is left. For example:

    Building the 3D model: kerb lines and sidewalks · 41% · 1 min 05 s so far · about 1 min 34 s left

When it ends, the line says how long it took in all (green), or that it stopped
(red), with the error below it as before. The jobs with a line:

- Generate tab: **Generate** (the texture), the **Top view**, **Export 3D model**,
  finding junctions.
- Train tab: **Train**, **Prime**, building or rebuilding the material tiles.
- 3D tab: **Generate 3D scene** and **Update view** (the server's stages, then
  loading the model into the view with its own time left), and **Bake light**
  (compiling the shader, then samples done with the time left from their pace,
  then clearing the noise).

Pressing Generate 3D scene while the Top view is still building the same model
does not start a second build: its line follows the one under way, marked
*shared* (see Big maps).

How the time left is worked out: each job is a list of stages, each with its
share of the time, measured on the 3.7 × 2.5 km test map (for the 3D model,
for example, the blocks take about a quarter of it and the kerb lines and
sidewalks about a sixth). Stages that go through many pieces (rows of patches,
streets, junctions, kerb rings, tiles, blocks, training pairs) report how far
they are; the others are taken to go at the pace of the stages before them.
The time left is the time so far scaled by what is left, so it settles after
the first few seconds ("working out the time left" until then) and is a guide,
not a promise: a map with more blocks than streets, say, will spend longer in
that stage than the shares expect.

Render photo has no line: it keeps getting sharper until you stop it, so it has
no end to count down to.

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
app/curves.py        placements' maths: curves and curve lines, mirroring, alignment, packages, random spaces, ids, space cells
app/placements.py    where each copy of a placement goes, for the 3D model and inner streets
app/streets.py       inner streets drawn between objects, built into the street mask
app/surface.py       bump and roughness maps for the material tiles (scanned, or from a tile's grain)
app/library.py       the scanned material library (app/scans): tiles with colour, bump and roughness lined up
app/islands.py       the islands between the roads: their numbers and outlines, slots, the islands texture
app/decals.py        decals: where each layer's go (intersections, streets, junctions, junction edges), painted into the texture, 3D quads
app/formats.py       the 3D model as FBX (binary 7.4) and OBJ, and parts combined, read from the GLB
app/looks.py         the Look panel's LUTs (.cube, Hald CLUT) and saved looks; app/looks holds bundled ones
app/images.py        loading, hashing, noise isolation
ui/index.html        the interface
ui/app.js            the interface logic
ui/view3d.js         the 3D tab: the scene, sky, sun, lamps, live view and photo render (bundled in ui/vendor)
ui/sky_blender.js    Blender 5's sky (Multiple Scattering), ported to JavaScript; ui/sky_worker.js runs it off the page
ui/denoise.js        Intel Open Image Denoise, ported to WebGL2, for the photo render
ui/look.js           the Look panel: film response, adjustments, LUTs and presets for the finished picture
ui/topview.js        the Generate tab's Top view: the 3D model drawn from straight above
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
- **Full width round sharp corners**: where a block has a sharp corner, such as
  where an inner street meets a main street, the sidewalk keeps its width all
  the way round it and its inner edge turns a square corner, instead of
  tapering to nothing at the corner. The corner itself is kept exactly, not
  cut off.
- **Road right up to the kerb**: the kerb runs straight from one row of the
  sidewalk to the next, which on the inside of a bend or across a corner lies
  a little off the road's own edge. The ground left in front of it is filled
  with road (laid like the knot fills, with the kerb band's tone), so no gap
  shows at the foot of the kerb.
- **Runs to the image border**: where the kerb line meets the edge of the image,
  the sidewalk is carried right to it and cut there.
- **No sidewalks** on highway interchanges (unchanged, for a later decision) or
  on roads wider than 20 m.
- Paving is laid as one continuous pattern per paved area (see Blocks and
  islands); the kerb stone is laid along each kerb.

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

### Kerb line traced from the mask

The kerb line used to be the edge of the rebuilt road mesh (street strips,
junction patches and knot fills put together) smoothed with up to 73 passes.
Where those pieces met, it picked up their joins: notches in the kerb, S-shaped
jogs where a strip ended a little off a patch, bulges where a fill stuck out,
and corners rounded far more than the mask's. The mask itself was right.

Now the kerb line comes from the mask's own outline:

- **Traced from the mask** (the same outline the sidewalk cover uses), joined
  with the exact shapes of inner streets, so it is where the mask says the road
  ends.
- **Straight lines through the pixel steps**: the traced outline follows the
  pixels in stairs; it is replaced by straight lines that are never more than
  0.75 px from it, then smoothed lightly (4 to 20 passes with Straightness,
  instead of up to 73), which rounds only the joins.
- **No snapping back onto the mesh's edge** (that put the strips' joins back in).
  The road fill in front of the kerb still closes any gap.
- **No doubled kerb faces on blocks**: cutting a block by the sidewalks round it
  could leave hairline slivers along the kerb (a second kerb face a few
  millimetres from the first). Block pieces thinner than 6 cm are removed.

Measured on the 3.7 × 2.5 km test map (1.1 m per px), distance from each kerb
point to the mask's edge: before, median 0.13 m, 13% of the kerb over 0.5 m
off, the worst 22.9 m; now, median 0.02 m, the worst 0.14 m. Doubled kerb faces
on blocks went from 95 km to 2.3 km. Natural wiggle in the mask (a street that
really bends a little) is kept; that is the mask's own shape.

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

## Street, sidewalk and kerb materials

In the Generate tab, under Settings, **Materials** chooses what each part is
made of, before you press Generate texture:

- **Streets**, **Sidewalks**, **Kerbs**: each shows its material on a small
  ball and its name. Click one (or *Change*) and a window opens with every
  material it can use, each on a ball with its name under it: *Your tiles*
  (the material tiles built from your training, as before) first, then the
  **scanned materials** from the bundled library (`app/scans`), grouped by
  kind: real surfaces scanned by Poly Haven, free to use (CC0), each with its
  own measured colour, bump and roughness. The balls are 1 m across and show
  each material at its real size, lit so its bump and shine show. Click one
  to choose it; Esc or a click outside closes the window. Your choices are
  kept in this browser for the next time.
  - Streets: three asphalts (fine grey, new and dark, old and worn).
  - Sidewalks: four pavings (grey blocks, square slabs, hexagons, red brick),
    two cobblestone fans (below), the concretes or the asphalts.
  - Kerbs: two concretes (exposed aggregate, brushed). Bridge decks use it too.
  - **Squares, blocks and islands**: the paved areas between the roads. By
    default *Same as sidewalks* (as before: the blocks and islands paved as
    their sidewalks, the paving running on from them). Or a material of their
    own, such as **cobblestone fans**: granite setts laid in overlapping arcs,
    the classic pattern of old squares (Helsinki's Senate Square, Lisbon,
    Prague). Two real scans, CC0 from Poly Haven by Rob Tuytel: *grey granite*
    (arcs repeating every 1.2 m, setts about 8 cm) and *dark and worn* (2.5 m).
    Each square's fans are turned to its main street, as its paving always is;
    the sidewalks round it keep their own paving. Also the **desert and
    ground** materials (below). The small islands paved over completely (too
    small for a ring of sidewalk) take this material too.
- **Streets** get fresh tiles made from the scan, the way the tiles from your
  training are made (app/tiles.py), so a street never shows the scan
  repeating every couple of metres: patches of the scan half a metre across,
  placed at random, turned and flipped, blended into 4 m tiles that join
  themselves on every edge, their blotches evened out. As many variants as
  your own tiles, different ones for open road, the kerb band and junctions
  (9 with the default 3 variants), and each street picks one with its own
  offset, turn and flip, as before. Colour, bump and roughness are placed
  together patch by patch, so they still line up, and a turned or flipped
  patch has its slopes turned with it (checked: the bump's slopes agree with
  the colour's to a median of 1.000). The scan's broad light and dark, shine
  and undulation are evened out first, so the patches blend unseen; the
  large-scale variation comes from the Large variation layer, as for your
  tiles. Brushed concrete's patches only make half turns, so its brush lines
  keep one direction. Made once per material (about 15 s) and kept in the
  workspace.
- **Sidewalks and kerbs** keep the scan itself (paving's joints must stay on
  their grid): its colour, bump and roughness are laid over the 4 m
  material tile together: repeated a whole number of times each way (a 2 m
  scan twice), stretched by the little it takes to fit, all three the same
  way, so **every stone and joint has its bump exactly on it**. Checked: the
  colour's edges and the bump line up at zero pixels of offset, on the tiles
  and on the painted-dash textures made from them.
- **Match the tone of your tiles** (off by default): a scanned material takes
  on the mean colour of what it replaces, so it fits your trained look. The
  large-scale variation (stains, wear) still comes from your generated
  texture either way. Off, each material keeps its own colour. On streets,
  open road, the kerb band and junctions each take their own tile's tone.
- **Generate texture** lays the street material into the texture: at its
  real size, each pixel the average of the ground it covers (at a metre or
  so per pixel that is its colour and tone, not its stones), with your
  trained material's lighter and darker areas over 2 m and more kept on it,
  then wear and markings as always. The log names it.
- **Top view**: once the texture is made, the 3D model is built with the
  chosen materials and drawn from straight above, as a flat map (about
  24 cm per pixel on a 1 km place): streets, sidewalks, kerbs, blocks,
  islands, markings and objects. It shows by itself when ready, and the
  *Top view* layer button switches between it and the texture. Clicking a
  spot on it inspects the same spot of the texture.
- **The 3D tab** uses the same choices: *Update view* rebuilds the scene with
  them (see 3D tab).
- Painted dashes are painted on the scanned asphalt too, smoother and
  slightly raised.
- The export log names the materials used. Each material in the GLB also
  records where its bump came from (glTF extras).

## Surface detail

Under 3D model, **Surface detail** gives every road, sidewalk, kerb, block
and bridge material a bump map and a roughness map, so the sun and the lamps
catch the surface: the asphalt's stones and pores, the joints of paving.

- **Scanned** (the default): scanned materials use their own measured maps.
  Your own tiles get the library's scanned bump and roughness where they have
  no pattern of their own (asphalt: the fine grey asphalt; plain concrete:
  the exposed-aggregate concrete), laid at the scan's real size. Its stones
  do not line up with the stones in your tile's colour; from street height
  that is hard to see. Paving tiles never take a scan, since its joints
  would fall in the wrong places: they use their own grain.
- **From each tile's grain**: your tiles' maps are worked out from their own
  colour (`app/surface.py`): the bump from the fine grain only, details up to
  about 7 mm, about 0.8 mm deep on asphalt, lighter grain proud and darker
  sunk; larger changes of tone (stains, patches, wear) are not relief and are
  left out, their edges softened so a stain does not grow a rim. Roughness
  follows the tone gently: asphalt about 0.86, paving 0.78, concrete 0.80.
  An estimate from the colour, lined up with it, not a measurement.
- **Off**: colour only, a smaller file.
- Road paint is smoother (0.55, also the paint strips) and painted dashes have
  a slight raised edge, whatever the source.
- The maps go into the GLB as standard glTF normal and metallic-roughness
  textures, with tangents, so Blender, the 3D tab and the photo render all
  read them the same way. This was checked with raised test dots lit from the
  east and from the north: the side facing the light is brighter in three.js,
  in the path tracer and in Blender 5.
- Pictures used by several materials are stored once in the GLB. In a
  990 × 770 m test the model is about 3.4 to 5 MB with surface detail and
  2.2 MB without.

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
- **One paving pattern per paved area**: each area between roads (a block or
  an island) has one layout shared by its sidewalks, its paved islands and the
  block itself: one grid, turned to the area's main street direction, one
  paving variant and one broad weathering tone. The pattern and the tone run
  straight across every edge between them, with no seam, no mitred "picture
  frame" round islands and no change of angle at corners. This holds for your
  own tiles and for library materials. On a curved street the grid stays
  straight (as one continuous paved area would be); the kerb stone still
  follows the kerb.
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

### Island materials: a material for each island

Under Blocks and islands, **Island materials** gives islands materials of
their own, island by island:

- **Every island has a number.** Generate texture finds the islands (every
  area between the roads of at least 4 m², city blocks and traffic islands
  alike, from the same cleaned mask the texture is made from) and numbers them
  in reading order: by each island's topmost pixel, top to bottom, then left
  to right. The same mask always gets the same numbers. **Show islands** draws
  each island's outline with its number on the map. Your city map has 321.
- **Material slots.** *Add material slot* adds one: a ball with its material
  (click it to choose: the window shows the 9 **desert and ground** scans
  first, then the pavings, concretes and asphalts the sidewalks can have, or
  *As the other islands*), a **Pick** button and a remove button (×). Each
  slot has its own colour on the map.
- **Pick**: press it, then click islands on the map. A click puts the island
  in this slot (taking it out of any other); a second click takes it out.
  Press Pick again or Esc to finish. Under each slot are its islands by
  number (such as `3, 7, 12-15`): you can also type numbers there and press
  Enter, handy on a map with hundreds of islands.
- Islands in no slot keep the **Squares, blocks and islands** material (by
  default the sidewalk paving), as before.
- A slot remembers its islands by a point inside each, not by number, so if
  the islands change (an inner street splits one) the point still finds the
  island it is in. The slots are kept on the server with the mask's picture:
  loading the same mask again brings them back (press Generate texture to
  number its islands again).
- **In the 3D model** (Download 3D model, the Top view and the 3D tab): each
  picked island's block is laid with its slot's material, island by island
  (looked up for every triangle, so two islands that share a paved area each
  keep their own), and a small paved island takes it too. The sidewalks round
  an island keep their paving; the island's material starts at the
  sidewalk's inner edge.
- **Desert and ground** (sand, gravel, dry earth, cracked mud, red sand: 9
  CC0 scans from Poly Haven, listed in `app/scans/README.md`) is laid as fresh
  4 m tiles made from the scan's patches, like the streets, so the scan never
  repeats; in its own colour (Match the tone does not grey it); with broad
  light and dark over it (drifts and damper patches, features about 3, 10
  and 35 m across, ±10%) so a large sandy block shows no tile repeat from
  above.
- **Generate islands texture** makes a texture of only the islands: the same
  size as the road texture, each island laid with its material at real size
  along its main kerb direction, the roads transparent (PNG with alpha). Its
  edge is the exact counterpart of the road texture's smooth edge, so the two
  laid together leave no gap and no overlap. It is quick: it does not touch
  the roads (5 s on a 900 × 700 mask once the material tiles exist; the
  first use of a desert material makes its tiles, a few seconds each).
  - Ground is mixed from two of its tile variants by a smooth random field,
    and paving keeps one variant and grid per island, as in the 3D model.
  - At coarse scales (1.1 m per pixel) a texture pixel is wider than a whole
    tile: each pixel then holds the material's average over it (no
    flicker or moiré), with the broad light and dark on top.
  - **Layers**: *Islands* shows the islands texture, *Roads + islands* the
    road texture with the islands under it. *Save islands* and *Save roads
    + islands* download them (`<mask>_islands.png`,
    `<mask>_roads_and_islands.png`). The Top view is rebuilt with the island
    materials.

## Decals

Pictures laid on the streets: arrows, crossings, manhole covers, stains,
patches. In the Generate tab, **Decals**, **Import decal (PNG)** adds one as a
layer (a PNG with a transparent background; WebP and JPEG work too). The top
of the picture is its front, its width goes across the road and its length
along it, in proportion. A layer can use any of these four at once:

- **Place intersections**: across the street right where it meets a
  junction, its top towards the junction (a crossing, a stop line). With one
  of these three (one at a time):
  - **Place at junctions**: a random **Share of edges (%)** of the junctions'
    edges.
  - **At all edges**: every edge of every junction.
  - **Even edges**: two opposite edges of every junction: the two streets
    most in line (the through road of a T, either road of a crossroads).

  The junction's edge is where the street really meets it: walking along the
  street towards the junction, the place where the road starts to widen (the
  kerb corners begin), so the decal sits right there, not back at the
  junction's middle. **Back from the edge (m)** moves it further along the
  street. **Fit the road's width** scales it to the road's width there (a
  crossing from kerb to kerb). A short street between two junctions gets one,
  not one from each end; two crossings may touch at a junction's corner.
- **Place on streets**: anywhere on the road, streets and junctions alike, at
  random, lying along the road facing either way. **Weight**: more weight,
  more decals; 50 is about one per 400 m² of road, 100 twice that, 0 none.
- **Place on junctions**: inside the junctions only (T, Y, crossroads and any
  other kind), at random. **Weight**: 100 is one per junction on average, 50
  one in two junctions, 200 two each.
- **Place junction edge**: next to the junction, against the **Right** or the
  **Left** kerb (one of the two; as seen driving along the street towards the
  junction), 0.3 m from the kerb (plus half a mask pixel), its top towards
  the junction (an arrow before a junction). Behind a crossing when there is
  one. **Weight**: 100 is one at every street end, 50 at half of them, 200
  two, one behind the other.

**Width (m)** is its size across the road (not used when fitted). **Turn
randomly** (on streets and junctions) turns it any way, for manhole covers
and stains. Decals never lie where the road is narrower than them, never two of
a layer on top of each other, and the later kinds keep clear of the earlier
ones (crossings first, then junction-edge decals, then junctions, then
streets). Layers saved before this (Place randomly) come back as Place
intersections (with a junction choice) or Place on streets.

The layers are kept on the server for every map. **Generate texture** lays
them, and **Place decals** lays them again after a change (quickly: the
texture as generated is kept, so they never pile up). The places come from the
generation's mask (its junctions and street centre lines), the same for the
texture and the 3D model; decals over a bridge's raised road are left out.

Under 3D model, **Decals** (beside Markings) sets how they lie:

- **Separate objects on the road**: in the 3D model they are an object of
  their own, "Decals" (flat quads 1.5 cm above the road, see-through
  materials), and in 2D a layer of their own (**Decals** in Layers), the road
  texture staying clean.
- **Painted into the road texture**: painted into the generated road texture
  (only on the road), and in the 3D model part of the road object (just above
  its surface: tiles repeat, so they cannot be painted into them). With
  Combine streets and kerbs they are in "Streets".

On the 900 × 700 test crop (1.1 m per pixel): 965 decals in under 4 s (257
fitted crossings at all intersection edges, 251 arrows at the right kerb at
weight 100, 119 oil stains in junctions at weight 150, 338 manhole covers on
streets at weight 30).

## Workspace location

The server prints which `workspace` it uses at startup and in the console, and
warns when it had to create a new, empty one. To keep the workspace somewhere
else (so updates can never touch it), put its full path into `workspace.txt`
next to `server.py`, or set the `RTA_WORKSPACE` environment variable.

## Objects (scatter)

In the Generate tab, **Objects** sits above **Foliage** (below) and Bridges.

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

### Duplicate and flip a placement

The two buttons side by side under **Add selected object to the map** work on
the selected placement (a layer or a package, straight or curved):

- **Duplicate placement** makes a copy with everything in it: gaps, random
  spaces, the package mix, curve lines, single turns and inner streets. The copy
  lands just behind the original, moved by its whole depth and a gap so the two
  never overlap, and is selected, ready to drag where it goes.
- **Flip placement** gives the mirror image, left to right along the rows: the
  order of the objects in each row reverses (a package's mix and random spaces
  too), a curve bends the other way, and a single object's turn mirrors (30°
  becomes -30°). Fronts still face the same side, and the models themselves are
  not mirrored. Every object keeps its id, so its turn and the inner streets
  around it stay with it. Press it again to flip back. **Flip side** (in a
  curve's panel) is different: it turns the copies to face the other side of
  the line.

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
  can touch; add a gap or move the line outwards, or give the rows their own
  curve lines (below).

### Curve lines

With one line every row runs parallel to it. To shape the rows differently, for
example the front row hugging a curved street and the back row straight along
the block behind, press **Add curve line** under **Rows**.

- **Spread over the rows**: the first line shapes the front row, the last line
  the back row, and any others are spread evenly between. A row between two
  lines follows a blend of the two, point by point: with 5 rows and 2 lines,
  row 2 is 3/4 of line 1 and 1/4 of line 2, row 3 half of each, row 4 a quarter
  and three quarters. With as many lines as rows, each row has its own line.
  Each line is drawn on the row it shapes, dashed, with its points.
- **At most one line per row.** Add curve line is greyed out when there are as
  many lines as rows; with fewer rows than lines, the lines are spread over the
  rows again. **Remove curve line** takes one away, the rest spread over the
  rows again (one line left becomes the middle line).
- **A point added goes on every line**, at the same place along each (Add point
  or a double-click on any line); removing a point removes it from every line.
- **Dragging a point**: along its line, only that point moves; across its line,
  the points at the same place on the other lines move with it, so the rows bend
  together and keep their spacing. A drag in any other direction does both: the
  along part moves the point alone, the across part moves the whole column.
  Along and across are taken at the point itself, from its own line's direction.
- Each row keeps its depth and the gaps between rows (even or random), and each
  row has its own length along its own line, so rows can hold different numbers
  of copies. Inner street cells follow each row's line; a space between two
  rows runs from one row's line to the next.
- **Make straight again** uses the middle of the lines.

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

**Road width**, **Sidewalk width**, **Corner radius** and **Markings on inner streets** are
under Draw inner streets, kept with the placement, whether drawing or not.
Click a cell to make it a street (green), or press and drag across several;
whatever the first cell becomes, the others do too, so the same drag can take
streets back (grey). Press **Done drawing streets** (or Esc) to leave; outside
this mode the spaces look as before. Works for layers and packages, straight
and curved, with even or random spaces.

When you press **Generate**, the drawn streets become streets in a copy of the
street mask, so everything after it treats them like the main streets:

- **Road, kerbs and sidewalks**: the road is a straight strip down the middle
  of each space, with sidewalks between it and the objects on either side. A
  street across the rows runs straight on through each junction drawn on its
  way, and so does one along the rows; at a T or a corner the street stops at
  the far side of the one it meets.
- **The road comes first**: it gets its **Road width** (6 m, two lanes, by
  default) before the sidewalks get any room; then the sidewalks get their
  **Sidewalk width** from what is left, and any more space widens the road. A
  space narrower than the road is road from side to side, without sidewalks.
  Nothing in the layout moves: a street is as wide as the space it is drawn
  in. The panel gives the narrowest space and the road it gets.
- **Small streets**: every drawn street is kept in the street mask, at least a
  pixel wide and unbroken. A road narrower than about 3 mask pixels (3.3 m at
  1.1 m per pixel) shows only faintly in the generated texture, as a band
  with the islands as holes; Generate says which. The 3D model does not
  depend on that: it lays every inner street at its exact size.
- **Rounded corners** to the **Corner radius** (4 m by default, at most three
  sidewalk widths, never closer to an object than half a sidewalk).
- **Joining the streets**: a street reaching the edge of the placement goes on
  straight to the nearest street, at both ends, up to **Join streets up to**
  (200 m by default, per placement; 0 never joins), unless that would run
  through an object of any placement; otherwise it ends at the edge, and
  Generate says which ends did not join and why. Set it before pressing
  Generate.
- **Islands**: the areas between streets become islands, like any city block:
  raised to sidewalk height and paved like the sidewalks, the objects standing
  on them. Specks of island under 40 m² left between streets become road.
- **Markings**: tick **Markings on inner streets** for the markings of the main
  streets, as strips or painted in, whichever the 3D model is set to; untick it
  for inner streets with no markings at all.
- Inner streets never run over an object, even where placements overlap.
- **In the 3D model** the inner streets are laid exactly as drawn, not traced
  back from the mask: straight edges, the kerb line and sidewalks along them,
  islands with rounded corners, and where they meet a main street, a junction
  like any other. The kerb line keeps their edges exactly as drawn (the main
  streets' kerbs are smoothed as before). With markings ticked, each street
  gets its dashes along its centre, running on past side streets and stopping
  where streets cross or end (a stretch too short for a dash gets none).
  They follow the 3D panel's markings setting like the main streets' dashes:
  strips, or painted into the road texture, each marked inner street then
  carrying them in a strip of road down its middle that reaches its edges.

After changing inner streets (or moving a placement that has them), press
Generate again: the 3D export says so if you forget.

### Turning single objects

Every object in a placement has an id, **row-column** counted from the top
left: `1-1` is the first object of the front row, the first along the line.
Press **Turn single objects** to see them: each object shows its id (at 30% of
its smaller side, so small objects get small labels) and a **green arrow** from
its centre to its front, its +Y like Blender's green axis, turning with it.
Click an object to pick it, Shift-click to pick more. **↺ 90°** and **↻ 90°**
turn the picked objects a quarter turn, **Rotation** sets an angle (degrees,
clockwise on the map), and **Reset picked objects** puts them back. The turns
are kept with the placement by id and apply in the 3D model.

**The spaces stay as set around a turned object.** A turned object takes its
whole turned outline in its row: its neighbours move along the row to keep the
gap, and when it gets deeper its row gets deeper, so the rows behind move back
by the difference and the gap between rows stays as set. Its front stays on
the row's front line, and the front row never moves. On a curve or in a
package, a wider turned object can mean one object fewer on the line. Inner
street cells follow the new layout, so after turning objects in a placement
with inner streets, press Generate again.

### Draw spaced: objects made empty spots

Under Single objects, **Draw spaced** turns objects of the grid into spaces:
click an object and it becomes an **empty spot** (its outline dashed on the
map); click a spot again to bring its object back; press and drag across
several to empty (or bring back) them all at once. Every other object stays
exactly where it is: the grid, its rows and gaps do not change. **Bring every
object back** fills them all again. The empty spots are kept with the
placement by id (row-column) and apply everywhere: the 3D model, the Top view
and the 3D tab leave those objects out, and an inner street joining a street
beyond the placement is no longer blocked by them. The placement's note counts
them.

### Objects alignment

By default every row faces the same way. For rows that face each other across
an inner street, a selected placement's panel has **Objects alignment**: three
switches (press again to switch off) and a checkbox, kept with the placement.

- **Alternate rows**: row 1 stays as placed, row 2 is turned round, row 3 is as
  row 1, row 4 turned round, and so on. Rows 2 and 3 then face each other, as
  do rows 4 and 5.
- **Flip last row**: the last row faces the other way from where it faces now
  (with Alternate rows on as well, it is turned back).
- **Flip columns**: the objects at either end of each row face out of that end:
  the left column turned 270° and the right column 90°, measured clockwise
  from row 1's direction (the object's arrow). They face out in every row,
  turned round or not, so a row's left end always faces left. One object
  alone in its row is not turned.
- **Exclude rows from columns**: the first and last rows are left out of Flip
  columns, so the four corner objects face the same way as their row and only
  the rows between have turned ends.

The panel shows which way each row faces (↑ as placed, ↓ turned round). A row
turned round lines up its objects' fronts on its back edge, the side it now
faces. Turned ends take their turned outline, so the spaces stay as set: the
row gets longer or deeper, or, on a curve or in a package, one object fewer
may fit. On a curve the ends face out along the line. **Flip placement**
mirrors all of it (the ends still face out), **Duplicate** copies it, and an
object's own turn in **Turn single objects** adds to it. The map and the 3D
model use the same steps.

Reading FBX: binary FBX only (Blender's default), meshes, UVs, model
transforms, unit scale and axis settings, material colour and embedded textures.
ASCII FBX, rigs and animation are not read. OBJ textures need their MTL and
image files, which a single-file import does not include, so OBJ objects come
in untextured.

## Foliage

Trees, bushes and other plants, in the **Foliage** panel under Objects. It
works like Objects, with only what plants need:

- **Import plant** (GLB, OBJ or FBX) makes a layer, with its scale and ↻
  quarter turn; select it and press **Add selected plant to the map**.
- **Plant packages** mix several plants in one placement, with weights, as
  object packages do (**Create package**, **Import plant**, **Place on the
  map**, **Shuffle the mix**).
- Placements are dragged, sized and rotated like objects' and can be bent into
  a curve (with **Add point**, **Flip side** and **Make straight again**).
  **Duplicate placement** and **Flip placement** sit under Add selected plant.
  Plants are green on the map. Random spaces, curve lines, inner streets,
  turning single objects and objects alignment are for objects only.

**Random transform**: each plant can vary on its own, between a **min** and a
**max**: **Scale** (×, the same in every direction), **Rotation** (degrees,
clockwise) and **Offset** (metres, in a random direction). A new placement
starts with no variation (scale 1 to 1, rotation and offset 0 to 0). The random
values are kept with the placement; changing a range keeps them and stretches
them to it, and **New random set** picks new ones.

- **Overlap on** (the default): plants keep their spots, laid out from their
  real size and the gaps, and vary on them, so big plants may overlap their
  neighbours, as in a real garden.
- **Overlap off**: each plant takes the room of its scaled and turned outline,
  plus its offset on every side, so no two plants overlap; rows get longer or
  hold fewer plants.

Plants on the road are left out, as objects are. The 3D export places every
plant as an instance with its own scale and rotation, exactly as on the map.

## 3D tab

The **3D** tab shows the whole place in 3D, to look around:

- **Wheel**: zooms towards the point under the mouse, a share of the way each
  notch, so it gets there and never stalls. It stops at eye height (1.6 m);
  more notches then carry you on along the street.
- **Left drag** orbits round the ground in the middle of the view (the point
  is set where you look each time you grab the view, not a fixed point in the
  middle of the place); **right drag** (or Shift + drag) pans, at the pace of
  that ground.
- **W A S D** or the **arrows** walk, **Q / E** go down and up, **Shift** is
  four times faster; the pace suits the height (slow in the street, fast from
  above). **Double-click** brings that spot to the middle of the view.
- **B** switches baked light off and on, to compare. **Reset view** goes back.
- **Ground plane**: the land round the place, its position (X east, Y up, Z
  south, in metres; the streets are at Y 0, the land 30 cm under them by
  default), its turn round the up axis, its width and length, its colour (or
  the time of day's), and whether it shows. Kept in the browser; Render photo
  and Bake light trace it too.

- **Update view** builds the scene again with the current materials and 3D
  model settings, keeping the camera, the time of day and the look. When the
  materials or settings (or the texture) have changed since the scene was
  built, the button lights up and a note says so.
- **Generate 3D scene** builds the last generated texture (Generate it in the
  Generate tab first): streets and inner streets, sidewalks and kerbs, blocks
  and islands, markings, and your objects and plants, exactly the model the
  GLB export makes, with the 3D model settings of the Generate tab (sidewalks,
  blocks, markings painted or as strips, mesh detail). Around the place the
  land runs on to the horizon and fades into the haze; it lies 30 cm under the
  streets, so it never shows through the blocks or islands.
- **Street lamps** stand along every sidewalk, one about every 30 m, the two
  sides of a street alternating, set 45 cm in from the kerb with the arm
  reaching over the road. Never on a corner, on a sidewalk under 0.8 m, or in
  an object. They are in the 3D view and the photo render only, not in the GLB.
- **Dawn, Day, Night** set the time:
  - *Dawn*: a low sun in the east, long warm shadows, a pink and gold sky, the
    lamps still on;
  - *Day*: a high sun from the south-west, a clear blue sky;
  - *Night*: moonlight, a starry sky, and the street lamps lighting the streets
    in warm pools of light.
- **Light** sets, for every time of day:
  - *Sun strength* (0 to 200%): how bright the sun is, or the moon at night;
  - *Street lamps* (0 to 300%, 0 is off) and *Lamp colour*: how bright the
    lamps are when they are on, and the colour of their light;
  - *Wet roads* (0 to 100%, 0 is dry): streets, sidewalks and kerbs darken
    (paint less), turn glossy so the lamps and the sky shine in them, and their
    fine grain fills in. At night the lamps' pools stretch into reflections.
  - *Puddles* (with Wet roads above 0): standing water in patches a few metres
    across, smooth as a mirror and a little darker, on the streets, sidewalks
    and squares. Render photo has them in the same places, traced: the water
    over the wet surfaces is a clear coat (glossier the wetter they are), the
    puddles a mirror-smooth coat that fills the grain.
  - *Live reflections* (on): wet streets mirror the scene, not only the sky:
    the buildings, the lamp posts and their lit heads, your objects, the clouds.
    The scene is drawn a second time, mirrored under the street, at half size,
    and the street surfaces show it where the sky would shine in them, blurred
    as much as they are rough and strongest at a glancing view (as real
    reflections are). Untick it for speed on a slow graphics card. Nothing
    changes while the roads are dry. Render photo traces its own reflections.
  They are kept in the browser for the next time, and apply to Render photo too.
- **The sky** at dawn and by day is Blender's own physical sky, the Sky
  Texture of Blender 5 (Multiple Scattering), ported to JavaScript: sunlight
  scattered by the air and by haze, absorbed by ozone, the sun's disc seen
  through the air. It gives the colour of the sky, the colour and strength of
  the sunlight and the light of the sky, so the balance of sun and sky is that
  of Blender and Cycles; it matches Blender 5.0.1 to within 1%. It is worked
  out once per time of day, in the background, in a second or two.
- **Sky: physical or HDRI** (the Sky panel). *Physical sky* is the one above,
  set by Dawn, Day and Night. Or pick a photographed sky (an HDRI): four come
  with the program, CC0 from Poly Haven (`ui/skies`): **Partly cloudy day**,
  **Sunset with clouds**, **Overcast** and **Cloudy night**. What the camera
  sees is a sharp 4k picture of the sky; the light is the HDR picture itself:
  - **its sun** is found in it (the brightest spot, less the sky round it):
    its direction, colour and strength become the sun's light, casting the
    shadows, and it is taken out of the sky's light so it is not counted
    twice. A sky without a sharp sun (overcast, a sun behind clouds, night)
    lights softly from all round; the panel says which.
  - **the rest of the sky** lights the scene from all round (its blue from
    above, its warm glow from the horizon at sunset), and the haze takes the
    colour of its horizon.
  - **how bright**: HDR pictures do not say how bright their sky really was,
    so each is brought to the light of the time of day it stands for (a sunny
    day, an overcast day, dusk), and a night sky is kept dim under the street
    lamps, which are on at sunset and at night as with the physical sky.
  - **Rotation** turns the sky round, its sun and shadows with it; **Strength**
    makes it brighter or darker, in stops. Double-click either to reset.
  - **Import HDRI…** uses your own: an equirectangular (2:1) panorama, `.hdr`
    or `.exr` (or a `.jpg` or `.png` photo, without HDR light), kept in the
    workspace (`workspace/skies`). Set **Light like** (Day, Overcast, Sunset
    or dawn, Night) so it is exposed right and the lamps are on when they
    should be; a file name with "night", "sunset" or "overcast" sets it.
    **Delete** removes one of yours.
  Render photo and Bake light trace the HDRI's light too (its sun as a light,
  the sky from all round, the sharp picture in view). A time of day (Dawn, Day,
  Night) goes back to the physical sky. The sky in use is kept in the browser.
- **The live view** is drawn with that sky, sunlight with shadows (sharp close
  up, covering everything when zoomed out), the sky's light from all round,
  the haze in the sky's own colour at the horizon, soft contact shadows where
  surfaces meet (ambient occlusion), a glow round bright lights, and filmic
  colour. The lamps nearest to where the camera looks light the scene for
  real; the others show their pool of light on the ground.
- **Render photo** traces the light through the scene (three-gpu-pathtracer):
  light bouncing between surfaces, soft shadows from the sun and the sky, and
  every street lamp as a real light. The picture gets sharper with every
  sample for as long as the camera stays put; the panel counts the samples.
  Moving the camera, or **Back to live view**, returns to the live view.
  Setting up takes a few seconds on a big map.
  - **Denoise** (on unless you untick it): the grain is cleared by Intel Open
    Image Denoise, the denoiser of Blender and Cycles, ported to run on the
    graphics card in the browser: the same neural network and trained weights,
    giving the same picture as OIDN itself (to within 0.1%). It works from
    the photo and the colour and direction of the surfaces in view, so edges
    and textures stay sharp. A decal's see-through parts show the road under
    them in that colour too (as in the photo), never the picture's black.
  - Decals in the photo: their clear parts are always left out (an alpha
    test on a copy of the picture's alpha), so what the graphics card makes of
    the picture's own alpha cannot turn them into black rectangles; soft
    edges and half see-through paint stay as see-through as in the live view. It runs after 4 samples, and again each time the
    samples have grown four times over (16, 64, 256…); the panel says which
    one it shows. Save image denoises the latest samples first.
  - **If the photo comes out empty** (some graphics cards' drivers handle the
    path tracer differently): after the first samples the photo checks
    itself. If the path tracer's shader did not compile, or the picture has
    no light at all, it tries again without the light tree (and remembers
    that for this card if that works). If the picture has light but the glow
    and look turn it black, it shows the photo without them. If nothing
    works it goes back to the live view and says exactly what failed: the
    shader's error, the graphics card, and whether it can draw, filter and
    blend float pictures. A single invalid pixel (not a number) is shown as
    one dark dot and never spreads.
  - **Windows (DirectX 11)** allows a shader 16 textures at once. The path
    tracer uses exactly 16; the light tree keeps its data in the lights'
    texture rather than one of its own (with its own, 17 textures, the photo
    could not render on Windows at all).
  - **Light tree**: with hundreds of lamps, each point of the picture picks
    the lights likely to light it (near, facing it, their beam towards it),
    as Cycles' light tree does, instead of any lamp at random, so night photos
    sharpen many times sooner.
  - **Compiling**: the first photo in a session compiles the path tracer's
    shader for the graphics card, which on Windows (DirectX) can take a
    minute or more. Meanwhile the live view stays and the panel counts the
    seconds; the samples start once it is done.
- **Save image** saves what the view shows, live or the photo, as a PNG,
  with its look.

### Baked light

**Bake light** (under Photo render) is what games do with their lighting:
the light is traced once, slowly and properly, and stored on the surfaces as
a light map; afterwards the view only reads it back, so you move round freely
with the traced light. It traces the light as Render photo does (the sun's
soft shadows, the sky's light, light bounced off kerbs and blocks, every
lamp's pool at night, the shadows of the lamp poles), on the ground from high
above. It does not need a photo first, and it does not keep the photo as a
picture: a picture is right only from where it was taken, while the light
stored on the ground is right from anywhere. With **Baked light** ticked, the
live view takes the light of the streets, sidewalks, islands and blocks from
that map.

How to use it: set the time of day and the lights, look at the part you want
(or choose the whole place), press **Bake light**, wait for "Baked in … s"
(a message on the view says it is on), then move round. Press **B** to see
the difference: the sun's light is the same live and traced, so by day the
change is in the soft shadows, the light under and between things and the
overall tone; at dawn and at night (lamp pools, long shadows) it is large.

- **Bake**: *What you see*, the ground in the view now: close up 10 cm a
  pixel (sharp shadows), coarser the more you see; or *the whole place*, at
  20 cm a pixel or coarser on a big place (the map is at most 2048 pixels a
  side). Look round inside what you baked: outside it the live light shows.
- **Quality**: how many samples are traced before the noise is cleared by
  Open Image Denoise: Draft (32), Good (128), Best (512), Very high (1024),
  Ultra (2048). The panel counts them and the time left; the view stays live
  meanwhile, and **Stop baking** stops it. On a good graphics card a Good bake
  takes about a minute; each step up takes about twice (Ultra 16 times) as long.
- **Resolution**: the light map's longest side, **2048 px** (sharper shadow
  edges and lamp pools, as before) or **1024 px** (about a quarter of the time
  and memory, softer). The bake's note says which it has.
- **What it keeps**: the light reaching each spot, not the surface's colour
  (the traced picture divided by the colour of what it saw). So Wet roads, and
  the materials' colours, still change live over a bake. The shine (the sky
  and the lamps in wet roads, highlights) stays live too, and follows the
  camera as it should. The live contact shadows are lightened, and the
  lamps' painted pools hidden, while a bake shows: both are in the bake.
- **Walls and objects**: *What you see* (the default) also traces the view
  from where the camera stands, every surface in it: walls, kerb faces, the
  lamp posts, your buildings and objects, and the ground near by, as sharp as
  the view. *All round you* traces four views round the camera (a little more
  than a quarter turn each), so you can turn round in the street and the walls
  behind you are baked too (about four times longer). *None* bakes the ground
  only, as before. Each surface checks that the bake saw that very surface
  (the same distance from where it was made, and facing it), so a wall never
  takes the light of the one in front of it; what the views did not see
  (round a corner, the far side of a building) keeps the live light. At night
  this is where the lamps' light on the house fronts comes from.
- **The ground from above**: each spot checks that the bake saw that very
  surface (the same height), so a road under a bridge never takes the
  bridge's light; walls are left to the views above.
- **A bake belongs to its light**: the time of day, the sun's strength and the
  lamps' strength and colour. Change one and the live light shows (the note
  says so) until you bake again; the last three bakes are kept, so going back
  to Day after baking Night shows the Day bake again. Generate 3D scene or
  Update view starts without bakes (the scene changed).

### Look

The **Look** panel adjusts the finished picture, as a photo editor would. It
applies to the live view and to Render photo alike, at once and without
rendering again (a photo keeps sharpening while you change its look), and Save
image keeps it.

- **Look**: a built-in preset (Neutral, Natural, Warm evening, Cool morning,
  Cinematic (teal and orange), Faded film, Vivid, Black and white), a look that
  came with the program, or one of yours. Change anything and the look becomes
  *Custom*, based on the one you picked.
- **Film response**: how the scene's light becomes the picture's tones. *AgX*
  (as Blender; the default, and what the 3D tab always used), *Neutral*
  (Khronos PBR Neutral: truer, stronger colours) or *ACES filmic* (contrasty).
- **Exposure** (in stops, before the film response, so highlights still roll
  off softly), **Contrast** and **Brightness** (black and white stay put),
  **Highlights** and **Shadows** (bring back a bright sky, see into the shade),
  **Saturation** (0: black and white), **Warmth** and **Tint** (white balance,
  at the same brightness), **Split toning** with its two colours (one in the
  shadows, one in the highlights), **Fade** (blacks lifted, as an old print),
  **Vignette**, **Sharpen**, **Film grain** and **Glow** (the glow round bright
  lights, 100% being the time of day's own; it now glows in photos too).
  Double-click a slider to put it back.
- **Lens**: the camera's focal length on a 35 mm frame: 25 mm (the view as it
  always was), 18 mm very wide, 35, 50 (as the eye), 85 (portrait) or 135 mm
  (telephoto, zoomed in). It changes the view itself, and how much depth of
  field blurs.
- **Depth of field** (off unless ticked): what is at the focus distance stays
  sharp, nearer and further blurs, as a camera lens does. **Aperture** is the
  lens's f-stop (f/1.4 blurs a lot, f/16 keeps more sharp; below f/1 is more
  than a real lens: towards f/0.005 a whole street looks like a model, the
  miniature look). **Focus**: on the middle of the view (found when the camera
  stops, then eased to, as an autofocus does; the distance shows beside it) or
  at a fixed distance; **Pick focus in the view** then a click sets it to that
  spot. The live view blurs from the scene's depth; Render photo traces the
  lens itself (its path tracer's physical camera, the same focal length,
  aperture and focus), so the photo's blur is the real thing, bokeh included.
  As with a real camera, a wide lens keeps a street sharp; for visible blur
  use a longer lens, focus close, or a small f-number.
- **LUT**: a colour lookup table, the usual way film emulations and colour
  grades are shared. **Import…** takes `.cube` files (3D or 1D, as from
  Resolve, Premiere, Photoshop and most LUT packs) and Hald CLUT pictures
  (`.png`, as RawTherapee's and darktable's film simulation packs); tables over
  65³ are resampled to 65³. The LUT is applied right after the film response,
  then your adjustments on top; **LUT strength** mixes it in. **×** removes a
  LUT of yours.
- **Save** keeps the settings (and the LUT) as a look of yours, under the
  name typed; saving under the name of one of your looks updates it.
  **Delete** removes one. Your looks and LUTs are kept in the workspace
  (`workspace/looks`), so updates never touch them.
- **Download** saves the look as a `.look.json` file with its LUT inside, to
  keep or to give to someone; **Import…** takes such a file back.
- Looks and LUTs put in `app/looks` come with the program for every workspace
  (see `app/looks/README.md`).

The current look is kept in the browser for the next time.

Everything is bundled in `ui/vendor`, so the tab works offline and nothing
needs installing:
- three.js r186 with the add-ons used, three-mesh-bvh and three-gpu-pathtracer
  (MIT licence); the path tracer has the light tree added (marked in its file);
- `blender-sky`: the licences of Blender's sky code ported in
  `ui/sky_blender.js` (MIT, and Apache 2.0 for Cycles' sky lookup);
- `oidn`: Open Image Denoise's trained weights (Apache 2.0), run by
  `ui/denoise.js`, a port of OIDN 2.4's network (Apache 2.0).
- `app/scans` (server side): twenty scanned materials from Poly Haven (eleven street
  surfaces, nine desert and ground), CC0
  (public domain), listed with their authors in `app/scans/README.md`;
- `ui/skies`: four HDRI skies from Poly Haven, CC0, listed with their authors
  in `ui/skies/README.md`; HDR and EXR files are read by three.js's HDRLoader
  and EXRLoader (with fflate, MIT).

The Look panel's presets are written for this program (no third-party LUTs
are bundled); LUTs you import keep their own licences.

A browser with WebGL 2 is needed (any current Chrome, Edge, Firefox or
Safari); the photo render and its denoising are much faster on a real
graphics card.

## When something goes wrong

An unexpected server error now shows its real cause in the console, the error
and the line where it happened, instead of "500 Internal Server Error". The full
report is saved in `workspace/errors/error-<date>-<time>.txt`; send that file
when reporting a problem.

A problem with one object no longer stops the export: it is left out, with a
console line naming it and saying why.
