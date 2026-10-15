"""Universal Maps Downloader bridge and geographic satellite previews."""
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from core import Cancelled, extent, project, unproject, validate_bounds

STREET_TYPES={'Google Street Map (English)':69,'Google Street Map (without labels)':66}
MAP_TYPES={**STREET_TYPES,'Google Satellite Maps':1,'Bing Satellite Maps':6,'ArcGIS Online World Imagery':43}


def commands(executable, folder, task_name, map_id, zoom, bounds):
    executable=Path(executable).resolve();folder=Path(folder).resolve()
    if executable.name.lower()!='downloader.exe' or not executable.is_file():
        raise ValueError('Select Universal Maps Downloader\\downloader.exe.')
    combiner=executable.with_name('combiner.exe')
    if not combiner.is_file():raise ValueError('combiner.exe must be beside downloader.exe.')
    if not task_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in task_name):
        raise ValueError('Task name must use letters, numbers, underscores or hyphens.')
    if not isinstance(map_id,int) or map_id<0:raise ValueError('Enter a valid Universal Maps Downloader map type ID.')
    if not isinstance(zoom,int) or not 1<=zoom<=23:raise ValueError('Zoom must be 1-23.')
    left,right,top,bottom=validate_bounds(*bounds)
    task=folder/(task_name+'.umd')
    download=[str(executable),task.name,str(map_id),str(zoom),*[f'{v:.10f}' for v in (left,right,top,bottom)],str(folder),'hide']
    combine=[str(combiner),str(task),'tif','meters']
    return download,combine,task


def tile_estimate(bounds,zoom):
    x0,y0,x1,y1=extent(bounds);world=2*math.pi*6378137.;step=world/2**zoom
    tx0=math.floor((x0+world/2)/step);tx1=math.ceil((x1+world/2)/step)-1
    ty0=math.floor((world/2-y1)/step);ty1=math.ceil((world/2-y0)/step)-1
    return max(1,tx1-tx0+1)*max(1,ty1-ty0+1)


def run_process(args,logfile,progress,cancelled):
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    with Path(logfile).open('wb') as stream:
        process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,shell=False,creationflags=flags)
        try:
            while process.poll() is None:
                if cancelled():
                    process.terminate()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise Cancelled('Stopped the downloader process. Partial files remain in its job folder.')
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f'{Path(args[0]).name} exited with code {process.returncode}. See {logfile}.')
        finally:
            if process.poll() is None:process.terminate()


def download(executable,parent,name,map_id,zoom,bounds,progress=lambda s:None,cancelled=lambda:False,runner=run_process):
    if tile_estimate(bounds,zoom)>1600:
        raise ValueError('This selection exceeds 1,600 standard tiles. Use a smaller area or lower zoom for one preview.')
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    # Validate before creating a job; never overwrite an existing job.
    commands(executable,parent,name,map_id,zoom,bounds)
    folder=parent/name;suffix=2
    while folder.exists():folder=parent/f'{name}_{suffix}';suffix+=1
    folder.mkdir()
    args,combine,task=commands(executable,folder,name,map_id,zoom,bounds)
    manifest=dict(status='downloading',requested_bounds_wgs84=dict(zip(('left','right','top','bottom'),bounds)),
                  map_type_id=map_id,zoom=zoom,estimated_standard_tiles=tile_estimate(bounds,zoom),
                  created_utc=datetime.now(timezone.utc).isoformat(),task=str(task),
                  note='Requested bounds are not necessarily the actual combined raster bounds.')
    file=folder/'download_job.json'
    def save():file.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    save()
    try:
        progress('Universal Maps Downloader is downloading tiles. Waiting for it to finish...')
        runner(args,folder/'downloader_console.log',progress,cancelled)
        if cancelled():raise Cancelled('Cancelled before combining.')
        if not task.is_file():raise RuntimeError('Downloader did not produce its .umd task file. Check installation, licence and download logs.')
        manifest['status']='combining';save()
        progress('Combining downloaded tiles into a TIFF in metres...')
        runner(combine,folder/'combiner_console.log',progress,cancelled)
        images=[p for p in folder.rglob('*') if p.suffix.lower() in {'.tif','.tiff'}]
        if not images:raise RuntimeError('No combined TIFF was found. Open this job in UMD Map Combiner and export TIFF manually, then use Load satellite image.')
        # Prefer the shallow combined image over any tile subdirectories.
        images.sort(key=lambda p:(len(p.relative_to(folder).parts),-p.stat().st_size))
        image=images[0]
        manifest.update(status='combined',image=str(image));save()
        return image,folder
    except Exception as error:
        manifest.update(status='stopped',error=str(error));save();raise


def _epsg_from_keys(keys):
    if not keys or len(keys)<4:return None
    geographic=None
    for i in range(4,len(keys)-3,4):
        key,location,count,value=keys[i:i+4]
        if key==3072 and location==0 and count==1:return 3857 if value in (3857,900913) else None
        if key==2048 and location==0 and count==1 and value==4326:geographic=4326
    if any(keys[i]==1024 and keys[i+3]==1 for i in range(4,len(keys)-3,4)):return None
    return geographic


def raster_geo(image,path):
    """Return actual pixel-edge bounds; reject unknown or rotated systems."""
    tags=getattr(image,'tag_v2',{})
    keys=tags.get(34735);epsg=_epsg_from_keys(keys)
    transform=tags.get(34264);scale=tags.get(33550);tie=tags.get(33922)
    if epsg and transform and len(transform)==16:
        if any(abs(transform[k])>1e-12 for k in (1,2,4,6)):return None
        sx,sy,x0,y1=transform[0],transform[5],transform[3],transform[7]
    elif epsg and scale and tie:
        sx,sy=float(scale[0]),-float(scale[1]);x0=float(tie[3])-float(tie[0])*sx;y1=float(tie[4])-float(tie[1])*sy
    else:
        candidates=[path.with_suffix(s) for s in ('.pgw','.jgw','.tfw','.wld')]
        candidates.append(Path(str(path)+'w'))
        world=next((p for p in candidates if p.is_file()),None)
        prj=path.with_suffix('.prj')
        if not world or not prj.is_file():return None
        wkt=prj.read_text(encoding='utf8',errors='replace').upper()
        if any(s in wkt for s in ('3857','900913','PSEUDO-MERCATOR','PSEUDO_MERCATOR','AUXILIARY_SPHERE','WEB_MERCATOR')):epsg=3857
        elif 'GEOGCS' in wkt and 'PROJCS' not in wkt and ('4326' in wkt or ('WGS' in wkt and '84' in wkt)):epsg=4326
        else:return None
        values=[float(v) for v in world.read_text().split()]
        if len(values)!=6:return None
        sx,ry,rx,sy,c,f=values
        if abs(rx)>1e-12 or abs(ry)>1e-12:return None
        x0,y1=c-sx/2,f-sy/2
    if sx<=0 or sy>=0:return None
    # PixelIsPoint tie points locate pixel centres rather than pixel edges.
    if keys and (transform or (scale and tie)):
        for i in range(4,len(keys)-3,4):
            if keys[i]==1025 and keys[i+1]==0 and keys[i+3]==2:x0-=sx/2;y1-=sy/2
    width,height=image.size
    ex=(x0,y1+height*sy,x0+width*sx,y1)
    if not all(math.isfinite(v) for v in ex):return None
    return dict(crs=epsg,extent=ex,source_size=(width,height))


def load_image(path,manual_bounds=None):
    path=Path(path)
    with Image.open(path) as source:
        if source.width*source.height>100_000_000:
            raise ValueError('Image exceeds 100 million pixels. Export a smaller area for preview.')
        geo=raster_geo(source,path)
        if manual_bounds is not None:
            validate_bounds(*manual_bounds)
            geo=dict(crs=3857,extent=extent(manual_bounds),source_size=source.size,manual=True)
        source.draft('RGB',(1600,1600))
        source.thumbnail((1600,1600),Image.Resampling.LANCZOS)
        preview=source.convert('RGB').copy()
    return dict(path=str(path),image=preview,geo=geo,
                note='Registered from image metadata.' if geo else 'No supported georeferencing: satellite preview only; overlay is disabled.')


def image_bounds(satellite):
    geo=satellite['geo']
    if not geo:return None
    x0,y0,x1,y1=geo['extent']
    if geo['crs']==3857:
        left,bottom=unproject(x0,y0);right,top=unproject(x1,y1)
    else:left,bottom,right,top=x0,y0,x1,y1
    return validate_bounds(left,right,top,bottom)


def preview_on_grid(satellite,bounds,size):
    """Align thumbnail to the OSM grid using actual georeferencing."""
    geo=satellite['geo'];im=satellite['image'];width,height=size
    if not geo:return im.copy()
    x0,y0,x1,y1=extent(bounds);sx0,sy0,sx1,sy1=geo['extent']
    def source_pixel(px,py):
        x=x0+(x1-x0)*px/width;y=y1-(y1-y0)*py/height
        if geo['crs']==4326:x,y=unproject(x,y)
        return ((x-sx0)/(sx1-sx0)*im.width,(sy1-y)/(sy1-sy0)*im.height)
    if geo['crs']==3857:
        a=source_pixel(0,0);b=source_pixel(width,height)
        return im.transform(size,Image.Transform.AFFINE,((b[0]-a[0])/width,0,a[0],0,(b[1]-a[1])/height,a[1]),Image.Resampling.BILINEAR,fillcolor=(20,35,48))
    # Latitude interpolation is nonlinear in Mercator: use narrow mesh strips.
    mesh=[]
    for start in range(0,height,8):
        end=min(height,start+8)
        ul=source_pixel(0,start);ll=source_pixel(0,end);lr=source_pixel(width,end);ur=source_pixel(width,start)
        mesh.append(((0,start,width,end),(*ul,*ll,*lr,*ur)))
    return im.transform(size,Image.Transform.MESH,mesh,Image.Resampling.BILINEAR,fillcolor=(20,35,48))
