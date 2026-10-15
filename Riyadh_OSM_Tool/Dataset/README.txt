40 selected 500 m Riyadh squares, from gcc-states-261011.osm.pbf.
Use updated OSM_Overlay_Tool v1.6 in this ZIP: it recognizes RYD500_ IDs.
Start_OSM_Overlay.bat installs pyproj once for exact-square reprojection.
Load/build your PBF index, set zoom 20, Import Batch_40_Riyadh_500m_IDs.txt,
then Start / resume batch and choose an output folder.

All download envelopes are inside your specified bounds:
Left 46.4832, Right 47.0222, Top 24.9823, Bottom 24.5375.
The exact squares are in EPSG:32638, 500x500 projected metres.
UTM southwest corner = (col*500, row*500).
The coordinate CSV is the enclosing longitude/latitude rectangle.
GeoJSON contains the exact square boundaries.
Batch v1.6 downloads rectangles then reprojects to exact squares.

Selection uses actual clipped OSM motor-road segments, not random IDs.
Road length and 100m subcell coverage favor dense urban street layouts.
OSM coverage does not guarantee paved surfaces or complete satellite imagery.
Original road-containing candidate grid has 7938 squares.
These new IDs are separate from the previous RYD_1km IDs.
Adjacent squares should remain in one split when training/evaluating a model.
