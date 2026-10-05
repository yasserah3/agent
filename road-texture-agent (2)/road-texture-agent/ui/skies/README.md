# HDRI skies

The photographed skies of the 3D tab's Sky panel, all CC0 (public domain)
from Poly Haven (https://polyhaven.com/license), the "pure sky" versions
(the sky only, no ground: the 3D tab has its own).

Files per sky: `<id>.jpg`, the sky as the camera sees it, 4096 x 2048, its
light squeezed into 8 bits (stored as t^(1/2.2) with t = xk / (1 + xk), k in
`skies.json`, so the 3D tab unsqueezes it exactly; up to about 1000 times
the sky's middle brightness); `<id>_light.hdr`, Poly Haven's own 1K HDR, the
light (its sun found in it, see ui/view3d.js); `<id>_thumb.jpg`, the panel's
preview. `skies.json` lists them with their kind of light (day, sunset,
overcast, night), which sets the exposure and the street lamps.

| Sky | Poly Haven | Authors |
|---|---|---|
| Partly cloudy day | [Kloofendal 48d Partly Cloudy (Pure Sky)](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky) | Greg Zaal, Jarod Guest |
| Sunset with clouds | [Belfast Sunset (Pure Sky)](https://polyhaven.com/a/belfast_sunset_puresky) | Dimitrios Savva, Greg Zaal, Jarod Guest |
| Overcast | [Kloofendal Overcast (Pure Sky)](https://polyhaven.com/a/kloofendal_overcast_puresky) | Greg Zaal |
| Cloudy night | [Kloppenheim 07 (Pure Sky)](https://polyhaven.com/a/kloppenheim_07_puresky) | Greg Zaal, Jarod Guest |

To add one: make the three files the same way (the script that made these is
described above), and add an entry to `skies.json`. Your own skies are better
imported from the 3D tab (Import HDRI…): they are kept in the workspace.
