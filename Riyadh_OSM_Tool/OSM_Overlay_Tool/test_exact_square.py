import tempfile,json
from pathlib import Path
from PIL import Image
from pyproj import Transformer
from exact_square import finalize_square
with tempfile.TemporaryDirectory() as d:
 folder=Path(d)/'aligned';folder.mkdir();col,row=1340,5470;metres=500;x0=col*metres;y0=row*metres
 t=Transformer.from_crs(32638,3857,always_xy=True);points=[t.transform(x,y) for x,y in ((x0,y0),(x0+500,y0),(x0+500,y0+500),(x0,y0+500))]
 ax=min(p[0] for p in points)-10;cx=max(p[0] for p in points)+10;by=min(p[1] for p in points)-10;dy=max(p[1] for p in points)+10
 meta=dict(crs='EPSG:3857',dimensions=[300,300],pixel_edge_extent=[ax,by,cx,dy],mask_tool_metres_per_pixel=2)
 for name in ('satellite','street_map_downloaded'):Image.new('RGB',(300,300),(80,120,160)).save(folder/(name+'.png'))
 to_ll=Transformer.from_crs(32638,4326,always_xy=True);line=[to_ll.transform(x0,y0+250),to_ll.transform(x0+500,y0+250)]
 (folder/'roads.geojson').write_text(json.dumps(dict(features=[dict(geometry=dict(coordinates=[line]))])))
 _,out=finalize_square(folder,meta,col,row,500,3)
 assert out['dimensions']==[250,250] and out['crs']=='EPSG:32638'
 assert out['pixel_edge_extent']==[x0,y0,x0+500,y0+500] and out['mask_tool_metres_per_pixel']==2
 with Image.open(folder/'overlay.png') as im:assert im.size==(250,250) and im.getpixel((125,125))[3]==255
 for name in ('satellite','street_map_downloaded','overlay','comparison'):
  with Image.open(folder/(name+'.png')) as im:assert im.size==(250,250)
  assert (folder/(name+'.pgw')).read_text()==(folder/'satellite.pgw').read_text()
 with Image.open(folder/'satellite.png') as im:assert im.getpixel((125,125))==(80,120,160)
print('PASS exact 500m UTM dimensions/extents, raster reprojection, OSM line position and matching sidecars')

from exact_square import square_mesh
col,row=670,2735;x0,y0=col*1000,row*1000
t=Transformer.from_crs(32638,3857,always_xy=True)
pts=[t.transform(x,y) for x,y in ((x0,y0),(x0+1000,y0),(x0+1000,y0+1000),(x0,y0+1000))]
meta={'crs':'EPSG:3857','dimensions':[1000,1000],'pixel_edge_extent':[min(p[0] for p in pts)-20,min(p[1] for p in pts)-20,max(p[0] for p in pts)+20,max(p[1] for p in pts)+20]}
mesh,spacing=square_mesh(meta,col,row,1000,1000)
assert spacing==1 and mesh[-1][0][-2:]==(1000,1000)
print('PASS exact 1km mesh coverage and pixel spacing')
