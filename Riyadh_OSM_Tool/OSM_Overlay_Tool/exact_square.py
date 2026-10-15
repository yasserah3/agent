"""Reproject a newly exported envelope onto an exact UTM square."""
import json,math,shutil
from pathlib import Path
from PIL import Image,ImageDraw
from core import Cancelled

def square_mesh(meta,col,row,metres,size):
 from pyproj import Transformer
 n=size;x0=col*metres;y0=row*metres;step=metres/n
 transform=Transformer.from_crs(32638,int(meta['crs'].split(':')[-1]),always_xy=True)
 ax,by,cx,dy=meta['pixel_edge_extent'];w,h=meta['dimensions']
 def point(px,py):
  x,y=transform.transform(x0+px*step,y0+metres-py*step)
  return ((x-ax)/(cx-ax)*w,(dy-y)/(dy-by)*h)
 mesh=[]
 for y in range(0,n,64):
  for x in range(0,n,64):
   right=min(n,x+64);bottom=min(n,y+64)
   corners=[point(x,y),point(x,bottom),point(right,bottom),point(right,y)]
   if any(px<-.01 or py<-.01 or px>w+.01 or py>h+.01 for px,py in corners):raise ValueError('Downloaded envelope does not cover exact UTM square.')
   mesh.append(((x,y,right,bottom),tuple(v for p in corners for v in p)))
 return mesh,step

def finalize_square(folder,meta,col,row,metres,line_width,cancelled=lambda:False,progress=lambda s:None):
 from pyproj import CRS,Transformer
 folder=Path(folder)
 try:
  scale=meta.get('mask_tool_metres_per_pixel')
  if not scale:raise ValueError('Square export needs approximately square source ground pixels.')
  n=round(metres/scale)
  if n<1 or n*n>80_000_000:raise ValueError('Exact square exceeds export pixel limit; reduce zoom.')
  mesh,step=square_mesh(meta,col,row,metres,n)
  progress(f'Reprojecting onto exact {metres} m UTM square: {n} x {n} pixels...')
  for stem in ('satellite','street_map_downloaded','street_map','street_roads'):
   file=folder/(stem+'.png')
   if not file.exists():continue
   if cancelled():raise Cancelled('Cancelled during exact-square export.')
   with Image.open(file) as im:
    result=im.transform((n,n),Image.Transform.MESH,mesh,Image.Resampling.NEAREST if stem=='street_roads' else Image.Resampling.BILINEAR)
   result.save(file)
  overlay=Image.new('RGBA',(n,n),(0,0,0,0));draw=ImageDraw.Draw(overlay)
  transform=Transformer.from_crs(4326,32638,always_xy=True);x0=col*metres;y0=row*metres
  roads=json.loads((folder/'roads.geojson').read_text())
  for feature in roads['features']:
   for segment in feature['geometry']['coordinates']:
    points=[transform.transform(*p) for p in segment]
    draw.line([((x-x0)/step-.5,(y0+metres-y)/step-.5) for x,y in points],fill=(0,220,240,255),width=line_width)
  overlay.save(folder/'overlay.png')
  with Image.open(folder/'satellite.png') as im:
   comparison=im.convert('RGBA');comparison.alpha_composite(overlay);comparison.convert('RGB').save(folder/'comparison.png')
  world='\n'.join(f'{v:.14f}' for v in (step,0,0,-step,x0+step/2,y0+metres-step/2))+'\n'
  for file in folder.glob('*.png'):
   file.with_suffix('.pgw').write_text(world,encoding='ascii');file.with_suffix('.prj').write_text(CRS.from_epsg(32638).to_wkt(version='WKT1_GDAL'),encoding='ascii')
  inverse=Transformer.from_crs(32638,4326,always_xy=True)
  corners=[inverse.transform(x,y) for x,y in ((x0,y0),(x0+metres,y0),(x0+metres,y0+metres),(x0,y0+metres))]
  meta.update(envelope_export=dict(crs=meta['crs'],dimensions=meta['dimensions'],pixel_edge_extent=meta['pixel_edge_extent']),crs='EPSG:32638',dimensions=[n,n],pixel_edge_extent=[x0,y0,x0+metres,y0+metres],native_pixel_size=[step,step],approx_ground_metres_per_pixel_xy=[step,step],mask_tool_metres_per_pixel=step,native_resolution_preserved=False,resampling='Bilinear reprojection of imagery; original downloaded TIFFs retained. Binary street mask uses nearest neighbour.',exact_square_size_m=metres,utm_col=col,utm_row=row,actual_bounds_wgs84=[min(p[0] for p in corners),max(p[0] for p in corners),max(p[1] for p in corners),min(p[1] for p in corners)],mask_tool_scale_note='UTM projected metres/pixel; local ground scale differs slightly by projection scale factor.',alignment='Satellite and street raster reprojected onto shared exact UTM square; OSM rerendered on that grid.')
  (folder/'pair_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
  (folder/'square_boundary.geojson').write_text(json.dumps(dict(type='Feature',properties=dict(size_m=metres),geometry=dict(type='Polygon',coordinates=[corners+[corners[0]]]))))
  (folder/'MASK_TOOL_INPUTS.txt').write_text(f'Satellite: satellite.png\nOSM overlay: overlay.png\nStreet map: street_map_downloaded.png\nMetres/pixel (UTM): {step:.10f}\nExact square: {metres} x {metres} metres in EPSG:32638.\nImagery is reprojected, not enhanced. Check for missing source tiles before training.\n')
  if cancelled():raise Cancelled('Cancelled during exact-square export.')
  return folder,meta
 except Exception:
  shutil.rmtree(folder);raise
