"""Native-resolution geographic pair exports, independent of preview thumbnails."""
import json, math, shutil
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageStat
from core import query_roads, unproject, extent, WKT, Cancelled, VERSION

WGS84='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'

def export_pair(directory,name,satellite,bounds,index,line_width=3,include_minor=False,
                cancelled=lambda:False,progress=lambda text:None,street_path=None,standard_path=None):
    geo=satellite.get('geo')
    if not geo or geo['crs'] not in (3857,4326):
        raise ValueError('Load satellite with supported geographic metadata first. Unknown bounds cannot be aligned.')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Export name must use letters, numbers, underscores or hyphens.')
    if not isinstance(line_width,int) or not 1<=line_width<=30:raise ValueError('Line width must be 1-30 pixels.')
    x0,y0,x1,y1=geo['extent'];sw,sh=geo['source_size']
    sx,sy=(x1-x0)/sw,(y1-y0)/sh
    request=extent(bounds) if geo['crs']==3857 else (bounds[0],bounds[3],bounds[1],bounds[2])
    a,b,c,d=request
    # Refuse partial coverage; never pad unavailable areas or stretch to fit.
    eps=max(sx,sy)*1e-6
    if a<x0-eps or b<y0-eps or c>x1+eps or d>y1+eps:
        raise ValueError('Selected bounds extend beyond the satellite image. Download full coverage or choose bounds inside it.')
    snap=lambda v: round(v) if abs(v-round(v))<1e-6 else v
    left=max(0,math.floor(snap((a-x0)/sx)));right=min(sw,math.ceil(snap((c-x0)/sx)))
    top=max(0,math.floor(snap((y1-d)/sy)));bottom=min(sh,math.ceil(snap((y1-b)/sy)))
    w,h=right-left,bottom-top
    if w<1 or h<1 or w*h>80_000_000:raise ValueError('Choose a nonempty crop of at most 80 million pixels.')
    actual=(x0+left*sx,y1-bottom*sy,x0+right*sx,y1-top*sy)
    ax,by,cx,dy=actual
    if geo['crs']==3857:
        lon0,lat0=unproject(ax,by);lon1,lat1=unproject(cx,dy)
    else:lon0,lat0,lon1,lat1=actual
    actual_bounds=(lon0,lon1,lat1,lat0)
    progress('Reading the original satellite pixels and matching OSM roads...')
    with Image.open(satellite['path']) as source:
        if source.size!=(sw,sh):raise ValueError('Satellite file dimensions changed. Reload it before exporting.')
        raster=source.crop((left,top,right,bottom)).convert('RGB')
    if max(ImageStat.Stat(raster).stddev)<8:
        raise ValueError('Image is nearly uniform; check for unavailable-map placeholder tiles before exporting.')
    roads=query_roads(index,actual_bounds,include_minor)
    if not roads:raise ValueError('No OSM roads found in the satellite crop. Check bounds and filters.')
    if cancelled():raise Cancelled('Cancelled before export.')
    overlay=Image.new('RGBA',(w,h),(0,0,0,0));draw=ImageDraw.Draw(overlay)
    for road in roads:
        for segment in road['segments']:
            points=[unproject(*p) if geo['crs']==4326 else p for p in segment]
            draw.line([((x-ax)/sx-.5,(dy-y)/sy-.5) for x,y in points],fill=(0,220,240,255),width=line_width)
    midlat=(lat0+lat1)/2
    if geo['crs']==3857:
        ground=[sx*math.cos(math.radians(midlat)),sy*math.cos(math.radians(midlat))]
    else:
        phi=math.radians(midlat)
        ground=[sx*(111412.84*math.cos(phi)-93.5*math.cos(3*phi)+.118*math.cos(5*phi)),
                sy*(111132.92-559.82*math.cos(2*phi)+1.175*math.cos(4*phi)-.0023*math.cos(6*phi))]
    mean=sum(ground)/2
    square=abs(ground[0]-ground[1])/mean<.02
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    folder=directory/(name+'_aligned');suffix=2
    while folder.exists():folder=directory/f'{name}_aligned_{suffix}';suffix+=1
    folder.mkdir()
    try:
        progress('Saving aligned satellite, overlay and geographic sidecars...')
        raster.save(folder/'satellite.png');overlay.save(folder/'overlay.png')
        review=raster.convert('RGBA');review.alpha_composite(overlay);review.convert('RGB').save(folder/'comparison.png')
        world='\n'.join(f'{v:.14f}' for v in (sx,0,0,-sy,ax+sx/2,dy-sy/2))+'\n'
        for stem in ('satellite','overlay','comparison'):
            (folder/(stem+'.pgw')).write_text(world,encoding='ascii')
            (folder/(stem+'.prj')).write_text(WKT if geo['crs']==3857 else WGS84,encoding='ascii')
        features=[dict(type='Feature',properties=dict(osm_id=r['id'],**r['tags']),geometry=dict(type='MultiLineString',coordinates=[[unproject(*p) for p in s] for s in r['segments']])) for r in roads]
        (folder/'roads.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features),ensure_ascii=False),encoding='utf8')
        meta=dict(tool_version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
                  source_satellite=str(Path(satellite['path']).resolve()),source_osm_index=str(index),
                  crs=f"EPSG:{geo['crs']}",dimensions=[w,h],source_crop_pixel_box=[left,top,right,bottom],
                  requested_bounds_wgs84=list(bounds),actual_bounds_wgs84=list(actual_bounds),pixel_edge_extent=list(actual),
                  native_pixel_size=[sx,sy],approx_ground_metres_per_pixel_xy=ground,
                  mask_tool_metres_per_pixel=mean if square else None,
                  mask_tool_scale_note='Approximate local ground scale at crop centre. Reproject non-square pixels before using the scalar-scale mask tool.',
                  manually_assigned_georeferencing=bool(geo.get('manual')),native_resolution_preserved=True,
                  alignment='Shared geospatial grid only; no visual snapping or correction of OSM offsets.',
                  osm_attribution='Map data © OpenStreetMap contributors, ODbL 1.0',satellite_attribution='Retain source provider attribution and applicable terms.',review_required=True)
        if standard_path:
            from roads_style import align_roads
            street,_=align_roads(standard_path,meta,monochrome=False)
            street.save(folder/'street_map_downloaded.png')
            (folder/'street_map_downloaded.pgw').write_text(world,encoding='ascii')
            (folder/'street_map_downloaded.prj').write_text(WKT if geo['crs']==3857 else WGS84,encoding='ascii')
            meta['standard_street_map']=dict(source=str(Path(standard_path).resolve()),image='street_map_downloaded.png',roads_only=False)
        if street_path:
            from roads_style import align_roads
            styled,mask=align_roads(street_path,meta)
            styled.save(folder/'street_map.png');mask.save(folder/'street_roads.png')
            for stem in ('street_map','street_roads'):
                (folder/(stem+'.pgw')).write_text(world,encoding='ascii')
                (folder/(stem+'.prj')).write_text(WKT if geo['crs']==3857 else WGS84,encoding='ascii')
            meta['street_map']=dict(source=str(Path(street_path).resolve()),mask='street_roads.png',alignment='Geographic transform onto native satellite crop',physical_widths=False)
            with (folder/'MASK_TOOL_STREET_INPUTS.txt').open('w',encoding='utf8') as stream:
                stream.write('Street guidance: street_roads.png\nWhite roads, black background. Cartographic widths, not measured asphalt.\n')
        (folder/'pair_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
        scale=f'{mean:.8f}' if square else 'NON-SQUARE PIXELS: reproject before width estimation'
        (folder/'MASK_TOOL_INPUTS.txt').write_text('Satellite input: satellite.png\nCentreline input: overlay.png\nGround metres/pixel: '+scale+'\nBoth files have identical crop, dimensions and georeferencing.\nInspect comparison.png for OSM offsets before using them as segmentation seeds.\nNative pixels are preserved. Bounds are snapped outwards to source pixel edges.\n',encoding='utf8')
        if cancelled():raise Cancelled('Cancelled during export.')
    except Exception:
        shutil.rmtree(folder);raise
    return folder,meta
