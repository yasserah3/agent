"""Read-only OSM indexing and georeferenced centreline exports."""
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

VERSION = '1.6.1'
SCHEMA = 2
RADIUS = 6378137.0
MOTOR = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
         'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
         'tertiary_link', 'residential', 'unclassified', 'living_street', 'service'}
WKT = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'


class Cancelled(Exception):
    pass


def project(lon, lat):
    return RADIUS * math.radians(lon), RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def unproject(x, y):
    return math.degrees(x / RADIUS), math.degrees(2 * math.atan(math.exp(y / RADIUS)) - math.pi / 2)


def validate_bounds(left, right, top, bottom):
    values = (left, right, top, bottom)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Coordinates must be finite numbers.')
    if not -180 <= left < right <= 180:
        raise ValueError('Left must be less than Right, within -180 to 180.')
    if not -85.05112878 <= bottom < top <= 85.05112878:
        raise ValueError('Bottom must be less than Top, within Web Mercator latitude limits.')
    return values


def extent(bounds):
    left, right, top, bottom = validate_bounds(*bounds)
    x0, y0 = project(left, bottom)
    x1, y1 = project(right, top)
    return x0, y0, x1, y1


def dimensions(bounds, width, height=None):
    ex = extent(bounds)
    if height is None:
        height = max(1, round(width * (ex[3] - ex[1]) / (ex[2] - ex[0])))
    if not 64 <= width <= 16000 or not 64 <= height <= 16000 or width * height > 80_000_000:
        raise ValueError('Use 64-16000 pixels per side and no more than 80 million pixels total.')
    return width, height


def source_signature(source):
    source = Path(source).resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        digest.update(stream.read(1024 * 1024))
        stream.seek(max(0, stat.st_size - 1024 * 1024))
        digest.update(stream.read())
    return dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sample_sha256=digest.hexdigest(), schema=SCHEMA)


def cache_path(source, directory):
    signature = source_signature(source)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    return Path(directory) / (key + '.sqlite'), signature


def cache_valid(path, signature):
    if not Path(path).exists():
        return False
    try:
        with closing(sqlite3.connect(path)) as db:
            meta = json.loads(db.execute("SELECT value FROM metadata WHERE key='info'").fetchone()[0])
        return meta['source'] == signature and meta['complete']
    except (sqlite3.Error, ValueError, KeyError, TypeError):
        return False


def build_index(source, cache_directory, progress=lambda message: None, cancelled=lambda: False):
    """Two offline streaming passes with temporary SQLite node locations."""
    from .pbf_reader import node_batches, ways      # (here: only reading the PBF needs protobuf)
    destination, signature = cache_path(source, cache_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cache_valid(destination, signature):
        progress('Using the existing road index.')
        return destination
    with tempfile.TemporaryDirectory(prefix='import-', dir=destination.parent) as temp:
        temp = Path(temp)
        db = sqlite3.connect(temp / 'roads.sqlite')
        db.executescript('CREATE TABLE roads(id INTEGER PRIMARY KEY, highway TEXT, tags TEXT, parts TEXT);'
                         'CREATE VIRTUAL TABLE bounds USING rtree(id,minx,maxx,miny,maxy);'
                         'CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);')
        count = 0
        skipped = 0

        node_db=sqlite3.connect(temp/'nodes.sqlite')
        node_db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE nodes(id INTEGER PRIMARY KEY,lon REAL,lat REAL);')
        try:
            progress('Pass 1/2: caching node coordinates on disk...')
            last_position=0
            for batch,position in node_batches(source):
                if cancelled():raise Cancelled('Import cancelled.')
                node_db.executemany('INSERT INTO nodes VALUES(?,?,?)',batch)
                if position-last_position>8_000_000:
                    node_db.commit();last_position=position
                    progress(f'Pass 1/2: {position/signature["size"]:.0%} of source scanned.')
            node_db.commit()
            progress('Pass 2/2: building road geometry and spatial index...')
            for way_id,tags,refs in ways(source):
                if cancelled():
                    raise Cancelled('Import cancelled. The source PBF was not changed.')
                highway=tags.get('highway')
                if not highway or highway in {'proposed','construction'} or tags.get('area')=='yes':continue
                positions={}
                for start in range(0,len(refs),900):
                    chunk=refs[start:start+900]
                    rows=node_db.execute('SELECT id,lon,lat FROM nodes WHERE id IN ('+','.join('?' for _ in chunk)+')',chunk)
                    positions.update((i,(x,y)) for i,x,y in rows)
                parts, current = [], []
                for node_id in refs:
                    point=positions.get(node_id)
                    if point is None:
                        if len(current) >= 2:
                            parts.append(current)
                        current = []
                        skipped += 1
                        continue
                    lon,lat=point
                    if abs(lat) > 85.05112878:
                        if len(current) >= 2:
                            parts.append(current)
                        current = []
                    else:
                        current.append([lon, lat])
                if len(current) >= 2:
                    parts.append(current)
                if not parts:
                    continue
                xs = [p[0] for part in parts for p in part]
                ys = [p[1] for part in parts for p in part]
                db.execute('INSERT INTO roads VALUES(?,?,?,?)',
                           (way_id, highway, json.dumps(tags), json.dumps(parts, separators=(',', ':'))))
                db.execute('INSERT INTO bounds VALUES(?,?,?,?,?)', (way_id, min(xs), max(xs), min(ys), max(ys)))
                count += 1
                if count % 5000 == 0:
                    db.commit()
                    progress(f'Indexing roads: {count:,}. First import scans the entire file.')

            if cancelled():
                raise Cancelled('Import cancelled.')
            info = dict(source=signature, complete=True, indexed_ways=count,
                        missing_node_references=skipped, built_utc=datetime.now(timezone.utc).isoformat())
            db.execute('INSERT INTO metadata VALUES(?,?)', ('info', json.dumps(info)))
            db.commit()
        finally:
            node_db.close()
            db.close()
        os.replace(temp / 'roads.sqlite', destination)
    progress(f'Index ready: {count:,} road ways. Missing node references: {skipped:,}.')
    return destination


def clip_segment(a, b, ex):
    x0, y0, x1, y1 = ex
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0., 1.
    for p, q in ((-dx, a[0]-x0), (dx, x1-a[0]), (-dy, a[1]-y0), (dy, y1-a[1])):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                low = max(low, t)
            else:
                high = min(high, t)
            if low > high:
                return None
    if high - low < 1e-12:
        return None
    return [[a[0]+low*dx, a[1]+low*dy], [a[0]+high*dx, a[1]+high*dy]]


def query_roads(index, bounds, include_minor=False):
    left, right, top, bottom = validate_bounds(*bounds)
    ex = extent(bounds)
    result = []
    with closing(sqlite3.connect(index)) as db:
        rows = db.execute('SELECT r.id,r.highway,r.tags,r.parts FROM roads r JOIN bounds b ON r.id=b.id '
                          'WHERE b.maxx>=? AND b.minx<=? AND b.maxy>=? AND b.miny<=?',
                          (left, right, bottom, top))
        for way_id, highway, tags, raw in rows:
            if not include_minor and highway not in MOTOR:
                continue
            segments = []
            for part in json.loads(raw):
                points = [project(*p) for p in part]
                for a, b in zip(points, points[1:]):
                    segment = clip_segment(a, b, ex)
                    if segment:
                        segments.append(segment)
            if segments:
                result.append(dict(id=way_id, highway=highway, tags=json.loads(tags), segments=segments))
    return result


def render(roads, bounds, width, height, line_width=3, transparent=True):
    dimensions(bounds, width, height)
    x0, y0, x1, y1 = extent(bounds)
    im = Image.new('RGBA', (width, height), (0,0,0,0) if transparent else (20,35,48,255))
    draw = ImageDraw.Draw(im)
    sx, sy = width / (x1-x0), height / (y1-y0)
    for road in roads:
        for a, b in road['segments']:
            pts = [((p[0]-x0)*sx-0.5, (y1-p[1])*sy-0.5) for p in (a,b)]
            draw.line(pts, fill=(0,220,240,255), width=line_width)
    return im


def export_bundle(directory, name, roads, bounds, width, height, line_width, index):
    if not roads:
        raise ValueError('No roads in this selection. Adjust the bounds or road filter.')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Name must contain only letters, numbers, underscores or hyphens.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    folder = directory / name
    # A unique result directory prevents replacing an earlier export.
    suffix = 2
    while folder.exists():
        folder = directory / f'{name}_{suffix}'
        suffix += 1
    folder.mkdir()
    image = render(roads, bounds, width, height, line_width, True)
    image.save(folder/'overlay.png')
    background = Image.new('RGBA', image.size, (20,35,48,255))
    background.alpha_composite(image)
    background.convert('RGB').save(folder/'preview.png')
    x0,y0,x1,y1 = extent(bounds)
    sx,sy = (x1-x0)/width, (y1-y0)/height
    world = '\n'.join(f'{v:.14f}' for v in (sx,0,0,-sy,x0+sx/2,y1-sy/2))+'\n'
    for stem in ('overlay','preview'):
        (folder/(stem+'.pgw')).write_text(world, encoding='ascii')
        (folder/(stem+'.prj')).write_text(WKT, encoding='ascii')
    features = []
    for road in roads:
        coords = [[unproject(*p) for p in s] for s in road['segments']]
        features.append(dict(type='Feature', properties=dict(osm_id=road['id'], **road['tags']),
                             geometry=dict(type='MultiLineString', coordinates=coords)))
    (folder/'roads.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features),ensure_ascii=False),encoding='utf8')
    with closing(sqlite3.connect(index)) as db:
        cache_info=json.loads(db.execute("SELECT value FROM metadata WHERE key='info'").fetchone()[0])
    metadata=dict(tool_version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                  bounds_wgs84=dict(zip(('left','right','top','bottom'),bounds)), crs='EPSG:3857',
                  projected_bounds=list(extent(bounds)), width=width,height=height,
                  projected_metres_per_pixel=[sx,sy],line_width_pixels=line_width,osm_way_count=len(roads),
                  source=cache_info['source'],missing_node_references=cache_info['missing_node_references'],
                  attribution='Map data © OpenStreetMap contributors, ODbL 1.0',
                  notes='Centreline overlay only. No road widths or asphalt mask inferred. Geographic bounds must match the satellite raster actual bounds and CRS; requested downloader bounds may differ.')
    (folder/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    return folder
