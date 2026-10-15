"""UMD style preparation and geographic roads-only raster alignment."""
import json, math
from pathlib import Path
from PIL import Image
from satellite import raster_geo

STYLE=[
 {'featureType':'all','elementType':'geometry','stylers':[{'color':'#000000'}]},
 {'featureType':'all','elementType':'labels','stylers':[{'visibility':'off'}]},
 {'featureType':'road','elementType':'geometry','stylers':[{'visibility':'on'},{'color':'#ffffff'}]},
 {'featureType':'road','elementType':'geometry.stroke','stylers':[{'visibility':'off'}]},
 {'featureType':'road','elementType':'labels','stylers':[{'visibility':'off'}]}
]

def prepare_job(parent,name,bounds,zoom):
 from core import validate_bounds
 bounds=validate_bounds(*bounds)
 if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):raise ValueError('Use a simple export name.')
 if not 1<=zoom<=23:raise ValueError('Zoom must be 1-23.')
 folder=Path(parent)/(name+'_roads_style');i=2
 while folder.exists():folder=Path(parent)/f'{name}_roads_style_{i}';i+=1
 folder.mkdir(parents=True)
 text=json.dumps(STYLE,indent=2)
 (folder/'roads_only_style.json').write_text(text,encoding='utf8')
 l,r,t,b=bounds
 (folder/'DOWNLOAD_INSTRUCTIONS.txt').write_text(f'''Select Google Maps (Customized Stylers and Languages) in UMD.
Custom Parameters > normal > Import: paste roads_only_style.json contents; Accept.
Task name: {name}_roads
Left longitude: {l:.10f}
Right longitude: {r:.10f}
Top latitude: {t:.10f}
Bottom latitude: {b:.10f}
Zoom: {zoom}
Download, then Tools > Map Combiner > TIFF in metres with geographic metadata.
In OSM tool use Load roads-only map, then Export aligned set.
Keep the satellite loaded separately. Style import/download needs a live UMD test.
Drawn widths are map styling, not physical road measurements.
''',encoding='utf8')
 return folder,text

def align_roads(path,metadata,monochrome=True):
 path=Path(path)
 with Image.open(path) as source:
  if source.width*source.height>100_000_000:raise ValueError('Road map exceeds 100 million pixels.')
  geo=raster_geo(source,path)
  if not geo:raise ValueError('Road map needs a georeferenced TIFF or world file and .prj. Screenshots cannot establish alignment.')
  crs=int(metadata['crs'].split(':')[-1])
  if geo['crs']!=crs:raise ValueError('Road map and satellite must use the same CRS; export both TIFFs in metres.')
  x0,y0,x1,y1=geo['extent'];w,h=geo['source_size']
  ax,by,cx,dy=metadata['pixel_edge_extent'];ow,oh=metadata['dimensions']
  sx=(x1-x0)/w;sy=(y1-y0)/h;eps=max(sx,sy)*1e-6
  if ax<x0-eps or by<y0-eps or cx>x1+eps or dy>y1+eps:raise ValueError('Road map does not cover the complete satellite crop.')
  # Require the prepared monochrome map, not an arbitrary coloured basemap.
  sample=source.convert('RGB');sample.thumbnail((512,512))
  pixels=list(sample.getdata())
  if monochrome and sum(max(p)-min(p)>20 for p in pixels)>len(pixels)*.01:raise ValueError('Use the black/white roads-only style, not a coloured street map.')
  grey=source.convert('L' if monochrome else 'RGB')
  a=(cx-ax)/ow/sx;e=(dy-by)/oh/sy
  c=(ax-x0)/sx+(a-1)/2;f=(y1-dy)/sy+(e-1)/2
  aligned=grey.transform((ow,oh),Image.Transform.AFFINE,(a,0,c,0,e,f),Image.Resampling.NEAREST)
  if not monochrome:return aligned,None
  mask=aligned.point(lambda p:255 if p>=128 else 0)
  extrema=mask.getextrema()
  if extrema!=(0,255):raise ValueError('Road map is empty or uniform. Check style and unavailable tiles.')
  return aligned,mask
