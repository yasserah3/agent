import tempfile,json
from pathlib import Path
from PIL import Image
from batch import run,read_ids
from core import Cancelled
with tempfile.TemporaryDirectory() as d:
 p=Path(d);exe=p/'downloader.exe';exe.touch();(p/'combiner.exe').touch();idx=p/'index';idx.write_text('index')
 grid={k:dict(left='46.7',right='46.701',top='24.601',bottom='24.6',col='670',row='2735') for k in ('RYD_0001','RYD_0002')}
 ids=p/'ids.txt';ids.write_text('# test\nryd_0001\nRYD_0001\nRYD_0002\n')
 assert read_ids(ids,grid)==list(grid)
 ids.write_text('UNKNOWN')
 try:read_ids(ids,grid);raise AssertionError()
 except ValueError:pass
 calls=[];fail=[True]
 def download(exe,folder,name,mid,*args,**kw):
  calls.append((name,mid));f=Path(folder)/(name+'.png');Image.new('RGB',(12,12),'white').save(f);return f,folder
 def load(path):return {'path':str(path),'geo':{'crs':3857}}
 def export(folder,name,*args,**kw):
  if name=='RYD_0001' and fail[0]:raise ValueError('simulated failure')
  out=Path(folder)/(name+'_aligned');out.mkdir(exist_ok=True)
  for name in ('satellite.png','overlay.png','street_map_downloaded.png'):Image.new('RGB',(12,12)).save(out/name)
  (out/'pair_metadata.json').write_text('{}');return out,{}
 finalized=[]
 def finalize(out,meta,col,row,size,*args):
  finalized.append(size);return out,meta
 kwargs=dict(download_fn=download,load_fn=load,export_fn=export,finalize_fn=finalize)
 first=run(list(grid),grid,exe,p/'out',idx,**kwargs)
 assert first['failed'] and first['completed']==['RYD_0002'] and len(calls)==4
 fail[0]=False
 second=run(list(grid),grid,exe,p/'out',idx,**kwargs)
 assert second['completed']==['RYD_0001'] and second['skipped']==['RYD_0002'] and len(calls)==4
 third=run(list(grid),grid,exe,p/'out',idx,**kwargs)
 assert len(third['skipped'])==2
 fourth=run(list(grid),grid,exe,p/'out',idx,zoom=18,**kwargs);assert len(calls)==8 and len(fourth['completed'])==2
 cancelled=run(list(grid),grid,exe,p/'out2',idx,cancelled=lambda:True,**kwargs);assert cancelled['cancelled']
 def stop(*args,**kwargs):raise Cancelled('test stop')
 stopped=run(list(grid),grid,exe,p/'out3',idx,download_fn=stop,load_fn=load,export_fn=export,finalize_fn=finalize);assert stopped['cancelled'] and not stopped['failed']
assert finalized and set(finalized)=={1000}
print('PASS 1km IDs dispatch exact UTM exports; IDs/deduplication, unknown IDs, sequential stages, failure continuation, stage reuse, resume skipping, settings invalidation and cancellation')
