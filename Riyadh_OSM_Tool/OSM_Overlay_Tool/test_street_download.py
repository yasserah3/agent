from pathlib import Path
import tempfile
from PIL import Image
import satellite as S
from roads_style import align_roads
with tempfile.TemporaryDirectory() as d:
 p=Path(d);exe=p/'downloader.exe';exe.touch();(p/'combiner.exe').touch()
 for label,mid in S.STREET_TYPES.items():
  calls=[]
  def runner(args,*unused):
   calls.append(args)
   if Path(args[0]).name=='downloader.exe':
    folder=Path(args[-2]);(folder/args[1]).touch()
   else:Image.new('RGB',(3000,20),(100,120,140)).save(Path(args[1]).with_suffix('.tif'))
  image,folder=S.download(exe,p,'RYD_3000_street',mid,19,(46.7086564,46.718653,24.602423,24.593283),runner=runner)
  assert calls[0][2]==str(mid) and calls[0][3]=='19'
  assert calls[0][4:8]==['46.7086564000','46.7186530000','24.6024230000','24.5932830000']
  assert image.exists() and len(calls)==2 and calls[1][-2:]==['tif','meters']
  image.with_suffix('.tfw').write_text('1\n0\n0\n-1\n0.5\n19.5\n');image.with_suffix('.prj').write_text('EPSG:3857')
  aligned,mask=align_roads(image,{'crs':'EPSG:3857','dimensions':[3000,20],'pixel_edge_extent':[0,0,3000,20]},monochrome=False)
  assert aligned.size==(3000,20) and aligned.getpixel((0,0))==(100,120,140) and mask is None
print('PASS both street IDs, automatic bounds/zoom, combiner invocation, separate jobs, native colored export')
# Test the direct UI callbacks without opening a Tk window.
from app import App
class Value:
 def __init__(self):self.value=''
 def set(self,v):self.value=v
 def get(self):return self.value
app=App.__new__(App);app.busy=False;app.map_type=Value();app.map_id=Value();seen=[]
app.download_satellite=lambda:seen.append((app.map_type.get(),app.map_id.get()))
for label in S.STREET_TYPES:app.download_street(label)
assert seen==[(label,str(mid)) for label,mid in S.STREET_TYPES.items()]
app.busy=True;app.download_street('Google Street Map (English)');assert len(seen)==2
print('PASS direct street buttons select IDs and invoke automatic download; busy guard')
