"""Offline alignment checks using known source pixels and a tiny SQLite road index."""
import tempfile,sqlite3,json
from pathlib import Path
import numpy as np
from PIL import Image
import satellite as SAT
from core import project,unproject,WKT
from aligned_pair import export_pair

def run():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td); index=p/'index.sqlite'
  db=sqlite3.connect(index)
  db.executescript('CREATE TABLE roads(id INTEGER,highway TEXT,tags TEXT,parts TEXT); CREATE VIRTUAL TABLE bounds USING rtree(id,minx,maxx,miny,maxy);')
  centre=project(46.75,25.1);x,y=centre
  for crs in (3857,4326):
   if crs==3857: ex=(x-128,y-128,x+128,y+128)
   else: ex=(46.74872,25.09872,46.75128,25.10128)
   left,bottom,right,top=ex
   coords=[unproject(x,y-200),unproject(x,y+200)]
   db.execute('DELETE FROM roads');db.execute('DELETE FROM bounds')
   db.execute('INSERT INTO roads VALUES(?,?,?,?)',(1,'residential','{}',json.dumps([coords])))
   db.execute('INSERT INTO bounds VALUES(?,?,?,?,?)',(1,min(a[0] for a in coords),max(a[0] for a in coords),min(a[1] for a in coords),max(a[1] for a in coords)));db.commit()
   arr=np.zeros((256,256,3),np.uint8);arr[:,:,0]=np.arange(256)[None,:];arr[:,:,1]=np.arange(256)[:,None];arr[:,:,2]=80
   path=p/f'source{crs}.png';Image.fromarray(arr).save(path)
   geo=dict(crs=crs,extent=ex,source_size=(256,256))
   # Deliberately use a tiny thumbnail; export must still read source pixels.
   sat=dict(path=str(path),geo=geo,image=Image.new('RGB',(8,8)))
   a=left+(right-left)*.25;b=bottom+(top-bottom)*.25;c=left+(right-left)*.75;d=bottom+(top-bottom)*.75
   if crs==3857:
    lo,bo=unproject(a,b);ri,to=unproject(c,d)
   else:lo,bo,ri,to=a,b,c,d
   folder,meta=export_pair(p,f'test{crs}',sat,(lo,ri,to,bo),index,2)
   assert meta['dimensions']==[128,128],meta['dimensions']
   assert np.array_equal(np.array(Image.open(folder/'satellite.png')),arr[64:192,64:192])
   overlay=np.array(Image.open(folder/'overlay.png'))
   assert overlay[:,:,3].sum()>0
   ys,xs=np.where(overlay[:,:,3]>0); assert abs(np.median(xs)-63.5)<=1
   assert (folder/'satellite.pgw').read_bytes()==(folder/'overlay.pgw').read_bytes()
   reloaded=SAT.load_image(folder/'satellite.png');assert reloaded['geo']['crs']==crs
   for v,z in zip(reloaded['geo']['extent'],meta['pixel_edge_extent']):assert abs(v-z)<1e-7
   repeated,_=export_pair(p,f'test{crs}',sat,(lo,ri,to,bo),index,2);assert repeated!=folder
   bad=dict(sat,geo=None)
   try:export_pair(p,'bad',bad,(lo,ri,to,bo),index);raise AssertionError('Unknown georeferencing accepted')
   except ValueError:pass
   try:export_pair(p,'bad',sat,(lo-1,ri,to,bo),index);raise AssertionError('Missing coverage accepted')
   except ValueError:pass
  db.close()
 print('PASS: native pixels, both CRSs, road positions, identical sidecars, reloaded georeferencing, safe repeated exports, missing coverage and metadata rejection')
if __name__=='__main__':run()
