# Scanned surface maps

Measured bump (normal) and roughness maps of real surfaces, used by Surface
detail = Scanned (app/surface.py) for road asphalt and plain concrete (kerbs,
bridge decks, concrete sidewalks). Paving keeps maps from its own tile, so its
joints line up with the joints in its colour.

For each kind, `asphalt` and `concrete`, a small JSON file names the maps:

```json
{
  "name": "Poly Haven asphalt_02",
  "source": "https://polyhaven.com/a/asphalt_02 (CC0)",
  "size_m": [2.0, 2.0],
  "normal": "asphalt_normal.jpg",
  "roughness": "asphalt_rough.jpg"
}
```

- `normal`: the OpenGL-convention normal map (green up), as glTF wants
  (Poly Haven `nor_gl`, ambientCG `NormalGL`).
- `roughness`: the roughness map, greyscale, 0 smooth to 1 rough.
- `size_m`: the width and height the maps cover in metres (the asset's
  real-world size). They are repeated a whole number of times over each
  material tile and stretched by the little it takes to fit.
- The maps must tile seamlessly, as the scans on these sites do.

Without a JSON file for a kind, its surfaces fall back to maps worked out from
each tile's own grain. Only use maps whose licence allows bundling (CC0).
