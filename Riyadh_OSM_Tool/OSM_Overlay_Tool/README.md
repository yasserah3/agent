# OSM + Satellite Tool 1.1

Standalone Windows desktop tool for creating geographically positioned road-centreline images from an OSM PBF and controlling your installed Universal Maps Downloader for satellite previews. It does not modify the road-texture project, your PBF, or training data. Asphalt-boundary estimation is a later stage.

## Start

1. Extract this folder somewhere writable, separate from the road-texture project.
2. Install standard Python 3.11 or newer from https://www.python.org/downloads/windows/ with the Python launcher and Tcl/Tk enabled.
3. Double-click `Start_OSM_Overlay.bat`. On its first run it creates a private virtual environment and downloads protobuf, NumPy and Pillow. Later runs use these installed packages. The OSM reader performs no network requests; satellite downloads are performed by Universal Maps Downloader when you click its download button.
4. Browse to your `gcc-states-261011.osm.pbf`, then click **Build / load road index**.

First import makes two streaming passes: cache node coordinates in a temporary SQLite database, then build road geometry and its spatial index. It does not hold all nodes or imagery in RAM. Allow several minutes and a few GB of free local disk space for the GCC file. Subsequent launches reuse the SQLite spatial index in `%LOCALAPPDATA%\OSMOverlayTool\cache`. Changing the source path, file size, modification timestamp or sampled content creates a different cache. This signature is not a full-file cryptographic checksum. PBF input is opened for reading only. Cancellation is checked between node batches and road ways. Standard zlib/raw PBF and uncompressed OSM XML are supported; other PBF compression and history/diff files are not supported.

## Coordinates and preview

- Enter Left longitude, Right longitude, Top latitude and Bottom latitude in WGS 84 decimal degrees. Saudi values are positive.
- The optional **Load grid ID** button accepts IDs from the previous PDF (`RYD_0001` through `RYD_4581`). The included coordinate table is generated from that same study window, not an official Riyadh boundary.
- Choose width and let the tool calculate height to preserve the geographic aspect in EPSG:3857, or enter the exact width and height of your satellite export.
- **Size from zoom** estimates pixel dimensions using standard 256 px Web Mercator tiles. It does not download anything or guarantee source detail. The export is not snapped to online tile edges.
- Default filter shows motor roads. The other-highway checkbox includes paths and other mapped highway types; proposed/construction roads and highway polygons are excluded from the index.
- Click **Preview selected roads**. Cyan lines are OSM way paths. The reduced preview fits the viewport; export keeps the requested resolution. Maximum image area is 80 million pixels, maximum dimension 16,000 pixels.

OSM highway classes do not describe physical road width. Road surface, sidewalk, median and marking masks are later stages. Missing node references break a polyline into valid parts; the tool does not invent lines across missing geometry.

## Export

Click **Export overlay bundle**, then select an output parent folder. Each result is saved in a new named folder; a numbered suffix avoids replacing an earlier export.

| File | Contents |
| --- | --- |
| overlay.png | Cyan road paths on a transparent background |
| overlay.pgw | Pixel-to-map transformation, using pixel-centre convention |
| overlay.prj | EPSG:3857 coordinate reference definition |
| preview.png | The same paths over a dark background, not satellite imagery |
| preview.pgw / preview.prj | Matching georeferencing for the preview |
| roads.geojson | Clipped road segments and original tags in WGS 84 |
| metadata.json | Bounds, dimensions, source signature, resolution and attribution |

Keep each PNG with its matching `.pgw` and `.prj` sidecars. The coordinate metadata also supports the future satellite comparison tool. GeoJSON is divided into clipped two-point segments; road properties and OSM IDs remain attached.

## Matching Universal Maps Downloader imagery

### Download and preview

Scroll down the left controls to **5. Universal Maps Downloader**.

1. Choose `downloader.exe` in your Universal Maps Downloader installation. `combiner.exe` must be in the same folder. The application is not bundled. Close any already-running downloader task before starting a task here.
2. Select the satellite map source, or enter a custom map type ID from the installed version's list. Preset IDs checked against the vendor's published list: Google Satellite = 1, Bing Satellite = 6, ArcGIS World Imagery = 43. The tool does not choose a source automatically. IDs, available zoom and imagery coverage can vary with provider/version.
3. Enter the coordinate fields above, set zoom 19 or 20, and choose a download folder. The Width/Height fields control the OSM/export grid, not the downloader's native pixel count. The downloader uses coordinates and zoom.
4. Click **Download satellite + preview**. The tool invokes the documented downloader command, waits for it to exit, then invokes Map Combiner to produce a TIFF in metres. It reads the combined image and opens the Satellite view. The trial downloader is limited to zoom 13; a suitable licensed installation is needed for higher zoom.
5. If the PBF index is loaded, the roads for these same coordinates are retrieved automatically. Switch between **OSM**, **Satellite**, and **Overlay**, and use **Line opacity** to inspect alignment.

One job is limited to an estimated 1,600 standard 256-pixel tiles to prevent accidentally requesting a whole city at high zoom. Keep your first job to a small square. This is an estimate; custom tile schemes can differ. Each click makes a new numbered job folder rather than overwriting old images. To reuse an existing download, choose **Load existing satellite image**. Downloader path, source and parent folder are remembered locally in `%LOCALAPPDATA%\OSMOverlayTool\satellite_settings.json`.

Every job keeps its `.umd` file, combined image, `download_job.json`, and separate downloader/combiner console logs. This tool does not extract credentials, bypass provider restrictions, or replace the application's licence. Use a source whose terms allow your intended downloads and training use. Missing tiles or poor native imagery detail still need visual checking.

**Cancel** stops the process launched for this job, preserving its partial files. If the downloader launches a detached secondary process, stopping that process may require the downloader's own controls. The helper handles process exit codes and missing task/combined files; a silent provider failure can still result in incomplete imagery, so inspect the preview.

### Geographic alignment

The preview reads actual georeferencing from supported GeoTIFF tags or a PNG/JPG/TIFF world file plus `.prj`. North-up EPSG:3857 and EPSG:4326 are supported. EPSG:4326 is resampled into the EPSG:3857 preview; other projections and rotated imagery are shown without a registered overlay. The thumbnail is limited to 1,600 pixels per side to keep the viewport responsive; the original downloaded image is preserved unchanged. Images above 100 million pixels must be exported as a smaller area before loading.

If geographic metadata is absent or unsupported, Satellite view still shows the image, but Overlay view is disabled. For an existing image that you have **already cropped to the exact entered bounds in EPSG:3857**, tick the explicit exact-crop fallback before loading it. Do not use this fallback to label an untrimmed downloader mosaic as if it matched the requested envelope.

Raster previews use the road preview's geographic bounds, not a simple stretch of the satellite image. This does not correct real OSM/imagery positional errors. Manual registration tools and paired full-resolution satellite export are later stages.

The overlay is north-up **EPSG:3857**, for exactly the entered geographic envelope. Use satellite imagery without labels. The actual downloaded mosaic can cover whole online tiles extending beyond the requested coordinates. Therefore, the PDF envelope and requested zoom alone do not guarantee a pixel-for-pixel match.

Use the satellite raster's actual bounds, CRS and dimensions, or geographically crop/reproject both rasters onto one common grid. Do not simply resize a wider satellite mosaic until its dimensions match the overlay. EPSG:4326 outputs need reprojection, not just a change of coordinate label. Same bounds in different projections are not the same pixel grid. Positional errors and different source dates can still require visual review.

PDF grid envelopes enclose 1 km UTM squares. This tool exports the envelope, **not** the final exact 1 km UTM square. Exact square cropping will be part of the later pairing stage. You can recover the square from E km / N km values in the PDF: multiply by 1000 for its southwest corner; add 1000 to obtain east and north boundaries, EPSG:32638.

## Later stages

1. Side-by-side comparison and saved manual registration corrections.
3. Editable road-surface masks, separating mapped measurements, estimates and reviewed boundaries.

Neither visual place recognition nor a trained street-layout model is included in this first tool.

## Validation status

Verified: desktop UI startup and controls, PDF grid lookup, coordinate validation, XML and zlib PBF decoding, dense delta nodes and coordinate offsets, clipping roads whose endpoints lie outside the selected region, missing-node gaps, road filtering, cache reuse, transparent PNG output and world-file pixel positioning.

Full GCC import validation could not finish in the development environment: automatic approval review blocked the earlier run after detecting an unexpected external Visual Studio telemetry attempt. The OSM component uses an offline streaming PBF/XML reader instead of the initial native OSM importer. Its complete runtime and disk requirements on the full GCC file remain unverified. Start with one preview after import and check its bounds before batch use.

Satellite bridge checks use controlled local fixtures: command argument ordering, download/combine sequencing, missing-output errors, job manifests, TIFF/world-file georeferencing, expanded-mosaic cropping and unsupported metadata handling. Live Universal Maps Downloader/provider operation was not tested because it is not installed in the development environment. The feature uses the vendor's documented command-line interface; the first real job verifies compatibility with your installed version. No live satellite download was performed during development. Version 1.1 keeps the existing OSM cache schema and signature unchanged, so an existing compatible index can be reused.

## Sources and attribution

Map data: OpenStreetMap contributors, ODbL 1.0. Preserve attribution in derived products: https://www.openstreetmap.org/copyright

OSM PBF schema: https://wiki.openstreetmap.org/wiki/PBF_Format

Universal Maps Downloader fields: https://www.allmapsoft.com/umd/support.htm

Universal Maps Downloader CLI and map IDs: https://www.allmapsoft.com/umd/commandline.htm

GeoTIFF metadata: https://docs.ogc.org/is/19-008r4/19-008r4.html

Mercator pixel formula: https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system

## Version 1.2: Export aligned pair
Load/build the existing PBF index, select coordinates, and download or load satellite imagery with geographic metadata. Click **Export aligned pair...** below the satellite load button.

The new export reads the ORIGINAL downloaded image, not the reduced preview. It crops on native source pixel boundaries, then renders OSM into that exact native grid. It does not rescale either file or visually snap OSM paths to road edges. Actual export bounds may extend less than one source pixel beyond the requested bounds. Export-width/height fields do not control this operation; the line-width setting does apply.

The output folder contains:
- satellite.png: original-resolution crop.
- overlay.png: matching transparent cyan centrelines.
- comparison.png: combined view for alignment review.
- matching .pgw and .prj geographic sidecars for all three PNGs.
- roads.geojson, pair_metadata.json and MASK_TOOL_INPUTS.txt.

Load satellite.png and overlay.png into Road Mask Tool. Copy the ground metres/pixel value from MASK_TOOL_INPUTS.txt. This is a local ground-scale approximation, not the EPSG:3857 projected scale. Non-square pixels are flagged; reproject them to a square metric grid before using the current scalar-scale mask tool.

Unknown georeferencing, out-of-coverage selections, nearly uniform imagery and changed source dimensions are rejected. Manual fallback remains available ONLY when the loaded image is already an exact EPSG:3857 crop of the entered coordinates; it cannot establish the extent of a whole-tile UMD mosaic. Manually assigned bounds are recorded in metadata. Do not use the fallback to guess bounds.

Existing OSM cache schema is unchanged. Extract this release into a separate folder or replace program files after closing the previous version. Never remove the existing cache. This export works with registered EPSG:3857 and EPSG:4326 rasters; rotated or other coordinate systems need prior conversion.

## Version 1.3: roads-only styled map
New **Download roads-only map (UMD)** prepares roads_only_style.json and a DOWNLOAD_INSTRUCTIONS.txt with the selected bounds and zoom. It copies the style to your clipboard, opens Universal Maps Downloader and opens the instructions folder. Select Google Maps (Customized Stylers and Languages), Custom Parameters > normal > Import, paste the style, then Accept. Enter the supplied coordinates/zoom and download. Combine to georeferenced TIFF in metres. Official instructions: https://allmapsoft.com/umd/customized-google-maps.htm

Style import and starting this styled download remain manual in UMD: its documented CLI has no style-import argument. No map type ID is guessed, and no unverified task format is generated. Downloading satellite via the existing bridge remains available. Live styled download and Windows UI need testing with your installation.

Use **Load / clear roads-only map**, then **Export aligned set**. The set includes satellite.png, overlay.png, street_map.png (styled grayscale map) and street_roads.png (white/black thresholded map) plus matching sidecars and metadata. The satellite's native crop defines the output grid. Street map alignment requires geographic metadata, complete coverage and matching CRS; coloured street maps and unregistered screenshots are rejected. When no street map is loaded, the existing pair export still works. Reload the street map when switching areas. Map styling widths are not measured asphalt boundaries. Existing Mask2Former v2 does not yet consume street_roads.png; it is an additional exported guidance input for the next mask integration.

Existing OSM/PBF indexes are reused: their schema has not changed. No new Python dependencies were added. Tests cover native 4000-pixel road alignment, bounds/CRS rejection and style job generation, alongside the existing aligned-pair checks. Only previews are reduced; exports retain native satellite pixels.

## Version 1.4: automatic standard street map downloads
Choose **Google Street Map (English)** (UMD ID 69) or **Google Street Map (without labels)** (ID 66), then **Download selected map + preview**. Coordinates, zoom and destination come from the tool; no DOWNLOAD_INSTRUCTIONS.txt copying or custom style import is required for these standard types. Python starts downloader.exe, waits for completion, starts combiner.exe and loads the combined TIFF. Street jobs use a separate _street task name; the loaded satellite is retained. A Street map preview tab shows the downloaded map. Export aligned set saves it as street_map_downloaded.png with matching sidecars on the native satellite crop. Georeferencing and complete coverage are required.

These are normal colored maps. English includes labels; without labels removes labels according to the provider's rendering. Neither is the custom white-on-black roads-only mask. The earlier custom-style workflow is still available separately. Standard street maps are saved as images, not silently converted into road alpha masks. Existing Mask2Former v2 still requires a separate update to consume these maps.

IDs verified against https://www.allmapsoft.com/providers/1.htm . Tests use a simulated downloader/combiner to verify both IDs, coordinate order, zoom, separate jobs and full-size colored raster export. Existing alignment/style tests and compilation pass. Live Windows UMD/provider downloads still require testing on your PC.

## Version 1.4.1: direct automatic street buttons
The manual **Download roads-only map (UMD)** button was removed from the main interface to avoid confusing it with standard automatic downloads. Use **Download English street map** or **Download street map without labels**. These directly select ID 69 or 66 and call the same automatic downloader/combiner bridge as satellite downloads. No instructions file is involved. They use the currently entered coordinates, zoom and download folder and are disabled while a job runs. Output is the combined TIFF in the job folder; the Street map tab previews it. Custom monochrome maps can still be loaded using Load / clear roads-only map. The earlier README sections describing the manual download button refer to prior versions.

A direct-callback test verifies that both new buttons set the correct ID and invoke the automatic path. Live UMD execution is not available in this environment. If UMD opens an empty form despite using these direct buttons, preserve the job's download_job.json and downloader_console.log for troubleshooting the installed executable/version.

## Version 1.5: batch IDs queue
1. Copy Batch_IDs_example.txt, replace the examples with your desired grid IDs (one per line), and save as a text file. Blank lines and # comments are accepted. Duplicate IDs are removed; unknown IDs reject the import before downloading.
2. Choose downloader.exe, build/load the existing PBF index, set zoom and OSM line/filter settings.
3. Click Import batch IDs.txt, then Start / resume batch, and choose a batch output folder.

Every ID uses coordinates from riyadh_grid.csv and runs Google satellite (ID 1), Google street map without labels (ID 66), then native aligned export. This exports the grid coordinate envelope, not a new exact 1 km UTM square. The street map remains colored and is not an alpha mask. OSM needs the loaded index and selected road filter.

Outputs live under output/RYD_ID/settings-key/: download jobs, batch_job.json and the ID_aligned folder containing satellite.png, street_map_downloaded.png, overlay.png, comparison.png, geographic sidecars, roads.geojson and metadata. The settings-key prevents mixing outputs with different zoom, bounds, line/filter or index file changes. Output dimensions follow native satellite pixels.

Cancel current operation stops the running download between process polls or checks between pipeline stages. Rerun with the same IDs, settings and output folder to resume. Completed exports with valid required PNGs are skipped; saved georeferenced satellite/street downloads are reused after alignment failures. Failed IDs are logged and later IDs continue. batch_summary.json reports completed, skipped and failed IDs for the latest run. Retrying an interrupted downloader creates a new download job; partial tiles are not merged automatically. No files are overwritten by the download/export helpers.

Tests simulate downloads and cover ordering, ID validation, failure continuation, reuse/resume, settings changes and cancellation. Existing bridge and aligned export tests also pass. Live 40-image UMD execution remains to be tested on your PC. Existing PBF index schema is unchanged; no model weights or new dependencies are required.

## Version 1.6: road-rich 500 m grid from your PBF
An additional riyadh_500m_grid.csv is loaded alongside the existing 1 km grid; original RYD_ IDs are preserved. New IDs begin RYD500_. Import Batch_40_Riyadh_500m_IDs.txt, choose zoom 20, load/build the PBF index and Start / resume batch. Set the zoom explicitly: IDs do not override it. The launcher uses a new dependency marker to install pyproj into existing environments once.

These squares are exactly 500 x 500 projected metres in WGS84 / UTM zone 38N (EPSG:32638), anchored at multiples of 500 m. Download bounds are outward-rounded geographic envelopes that wholly fit longitude 46.4832–47.0222 and latitude 24.5375–24.9823. UMD downloads satellite/street TIFFs for the envelope. For RYD500_ batch jobs the final aligned set is reprojected onto the exact UTM square; OSM paths are rerendered. Final imagery uses bilinear reprojection, not pixel-for-pixel native copying; original TIFFs remain. Output spacing approximates source ground sampling, so zoom 20 should yield about 3,700 pixels per side near Riyadh, depending on source sampling. Higher pixel counts cannot restore missing detail. Original 1 km ID batch behavior is unchanged.

The additional Dataset folder provides all 7,938 eligible road-containing squares, the selected 40 coordinate table, exact square GeoJSON boundaries and source/selection metadata. Selection ranks occupied 100m subcells, residential/urban road length and total motor-road length. All selected squares occupy 25/25 subcells and have >=8,068m of mapped road segments. Length counts mapped carriageways separately. Explicit unpaved surfaces, tracks, paths, construction/proposed and area roads are excluded; unknown surface is retained. OSM geometry is not reviewed asphalt ground truth. Nodes are region-filtered with a 0.02 degree buffer; segments with either endpoint outside it are omitted.

Use batch mode for exact-square reprojection. Single-image Export aligned set remains the existing native envelope export. EPSG:32638 final exports use pair_metadata.json plus .pgw/.prj and square_boundary.geojson; the older standalone image loader supports only EPSG:3857/4326. Load original UMD TIFFs for preview. The exported masks' consumer can use the supplied dimensions and scale metadata.

Tests cover exact 500m UTM extents, common output dimensions, raster reprojection, OSM location, sidecars, batch reuse/cancellation and existing aligned pairs. Live downloading was not performed here. Existing white/missing source tiles are not repaired by this update; inspect a sample before downloading all 40. No training has been run.

## Version 1.6.1: exact-square export for original 1 km IDs
Batch mode now recognizes the original RYD_ IDs as 1000 m UTM squares using their existing col/row values. RYD500_ IDs remain 500 m. Both download envelopes first and then export satellite, street map and OSM on an exact shared EPSG:32638 square. The export limit is 80 million pixels: 1km at zoom 20 is generally within it; higher zoom can exceed it. Existing outputs are not overwritten. Changed square settings have a new batch settings key, so old envelope outputs are not treated as exact-square completions. Single-image export remains the native geographic envelope workflow.

The road index is extracted from the selected PBF by Build index and cached separately. It stores road geometry/tags, not a 1km or 500m grid. The grid CSV files are precomputed spatial grids bundled with this release, including coordinates and UTM col/row. They do not regenerate automatically when another PBF is loaded. The grid definitions are metric geometry; PBF roads determine eligibility and ranking. Generate_500m_Grid.py is an optional developer script to regenerate 500m candidates from a supplied PBF. The selected 40 still originate from gcc-states-261011.osm.pbf. Rebuilding a road index from a newer PBF does not update these bundled candidate IDs or rankings.

Tests confirm both 500m square exports and 1km mesh spacing/coverage, and verify original IDs dispatch 1000m finalization. Live downloads remain untested here.
