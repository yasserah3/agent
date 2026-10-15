"""
Copied from the Riyadh OSM tool v1.6.1 (Riyadh_OSM_Tool/OSM_Overlay_Tool in this
repository), only their imports made relative (core.py reads
the PBF through pbf_reader.py only when it builds the index, so protobuf is needed only then): core.py (Web Mercator, the PBF road
index and its queries, the overlay drawing), pbf_reader.py (reads .osm.pbf offline)
and satellite.py (the Universal Maps Downloader bridge and reading a raster's
georeferencing). The coordinates input (app/coords.py) uses them; the OSM tool goes
on in its own folder, so its later versions do not change the agent.
"""
