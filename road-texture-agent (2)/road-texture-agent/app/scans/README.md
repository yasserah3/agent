# Scanned materials

Measured colour, bump (normal) and roughness of real street surfaces and desert ground, all
CC0 (public domain) from Poly Haven (https://polyhaven.com/license), listed in
`library.json` with their real size, source page and authors.

They are used in two ways (app/library.py, app/surface.py):

- **As a material** (Generate tab, 3D model: Street, Sidewalk and Kerb
  surface). The scan's own colour, bump and roughness are laid over the 4 m
  material tile together: repeated a whole number of times each way (so the
  tile still joins itself) and stretched by the little it takes to fit, all
  three in exactly the same way, so every stone's bump sits on that stone.
  With "Match the tone of your tiles" its colour takes on the tone of your
  trained tiles.
- **On islands** (Generate tab, Blocks and islands: island material slots,
  and the Squares chip). Desert and ground (`kind: ground`) is laid as fresh
  tiles made from the scan's patches, like the streets, so the scan does not
  repeat; it keeps its own colour.
- **As bump and roughness over your own tiles** (Surface detail: Scanned):
  the entry marked `bump_default` for asphalt and for concrete. Paving never,
  since its joints must line up with the paving in the colour.

Files per material: `<id>_diff.jpg` (colour, sRGB), `<id>_normal.jpg` (OpenGL
convention, green up, as glTF wants), `<id>_rough.jpg` (roughness, grey, 0
smooth to 1 rough). 1024 px, re-encoded from Poly Haven's 1K JPGs. Also
`<id>_height.jpg` (Poly Haven's displacement, 1K PNG, stretched so 0.5% to
99.5% of it spans 0 to 1; `relief_m` in `library.json` is how many metres that
is) and `<id>_ao.jpg` (ambient occlusion, 1K PNG, scaled so the open surface is
white), grey, quality 85. They give the parallax and occlusion of the 3D tab and
the photo render.

To add a material: put its maps here and add an entry to
`library.json` (`id`, `name`, `kind`: asphalt, paving, concrete or ground, `size_m`,
the file names (`colour`, `normal`, `roughness`, and optionally `height` with
its `relief_m` and `ao`), `source`, `authors`, `licence`). Only add maps whose
licence allows bundling.

| Material | Poly Haven | Size | Authors |
|---|---|---|---|
| Asphalt, fine grey | [Asphalt Pit Lane](https://polyhaven.com/a/asphalt_pit_lane) | 2.0 m | Dimitrios Savva |
| Asphalt, new and dark | [Clean Asphalt](https://polyhaven.com/a/clean_asphalt) | 2.1 m | Dimitrios Savva |
| Asphalt, old and worn | [Asphalt 04](https://polyhaven.com/a/asphalt_04) | 4.04 m | Jenelle van Heerden, Sergej Majboroda |
| Concrete pavers, grey blocks | [Concrete Pavers 02](https://polyhaven.com/a/concrete_pavers_02) | 2.0 m | Amal Kumar |
| Square concrete slabs | [Square Concrete Pavers](https://polyhaven.com/a/square_concrete_pavers) | 1.8 m | Amal Kumar |
| Hexagonal concrete paving | [Hexagonal Concrete Paving](https://polyhaven.com/a/hexagonal_concrete_paving) | 1.6 m | Stephan Seeliger |
| Red brick pavement | [Brick Pavement](https://polyhaven.com/a/brick_pavement) | 2.0 m | Charlotte Baglioni |
| Cobblestone fans, grey granite | [Floor Pattern 02](https://polyhaven.com/a/floor_pattern_02) | 1.2 m | Rob Tuytel |
| Cobblestone fans, dark and worn | [Patterned Cobblestone](https://polyhaven.com/a/patterned_cobblestone) | 2.5 m | Rob Tuytel |
| Concrete, exposed aggregate | [Concrete Floor 01](https://polyhaven.com/a/concrete_floor_01) | 2.0 m | Rob Tuytel |
| Concrete, brushed | [Brushed Concrete 03](https://polyhaven.com/a/brushed_concrete_03) | 2.0 m | Amal Kumar |
| Sand, pale desert | [Dense Sand](https://polyhaven.com/a/dense_sand) | 1.8 m | Dimitrios Savva |
| Sand, pale with patches | [Sand 01](https://polyhaven.com/a/sand_01) | 1.5 m | Rob Tuytel |
| Sand with gravel | [Gravelly Sand](https://polyhaven.com/a/gravelly_sand) | 2.48 m | Dario Barresi |
| Gravel, sandy brown | [Sandy Gravel 02](https://polyhaven.com/a/sandy_gravel_02) | 2.53 m | Dario Barresi |
| Gravel, pale grey | [Sandy Gravel](https://polyhaven.com/a/sandy_gravel) | 2.1 m | Charlotte Baglioni |
| Dry ground with stones | [Dry Ground Rocks](https://polyhaven.com/a/dry_ground_rocks) | 4.0 m | Rob Tuytel |
| Dry earth, brown | [Brown Mud Dry](https://polyhaven.com/a/brown_mud_dry) | 1.3 m | Rob Tuytel |
| Cracked dry mud | [Mud Cracked Dry 03](https://polyhaven.com/a/mud_cracked_dry_03) | 1.5 m | Dario Barresi, Dimitrios Savva |
| Red sand | [Red Sand](https://polyhaven.com/a/red_sand) | 3.0 m | Rohit Seervi |
