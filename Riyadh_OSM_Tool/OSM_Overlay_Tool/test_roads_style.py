import tempfile
from pathlib import Path
from PIL import Image,ImageDraw
from roads_style import prepare_job,align_roads
with tempfile.TemporaryDirectory() as d:
 p=Path(d);folder,text=prepare_job(p,'test',(46,46.001,25.001,25),19)
 assert (folder/'roads_only_style.json').exists()
 assert prepare_job(p,'test',(46,46.001,25.001,25),19)[0]!=folder
 im=Image.new('RGB',(4000,100),'black');ImageDraw.Draw(im).rectangle((1900,0,2000,99),fill='white');file=p/'roads.png';im.save(file)
 file.with_suffix('.pgw').write_text('1\n0\n0\n-1\n0.5\n99.5\n');file.with_suffix('.prj').write_text('EPSG:3857')
 meta={'crs':'EPSG:3857','pixel_edge_extent':[0,0,4000,100],'dimensions':[4000,100]}
 _,mask=align_roads(file,meta);assert mask.size==(4000,100) and mask.getpixel((1900,50))==255 and mask.getpixel((1899,50))==0
 try:align_roads(file,dict(meta,pixel_edge_extent=[-1,0,4000,100]));raise AssertionError('coverage accepted')
 except ValueError:pass
 try:align_roads(file,dict(meta,crs='EPSG:4326'));raise AssertionError('CRS accepted')
 except ValueError:pass
print('PASS style job, unique folders, 4000-pixel native alignment, coverage and CRS rejection')
