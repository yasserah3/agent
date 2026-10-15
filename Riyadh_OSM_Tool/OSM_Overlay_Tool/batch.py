"""Sequential, resumable satellite/street/aligned batch jobs."""
import json, hashlib
from pathlib import Path
from PIL import Image
import satellite as SAT
from aligned_pair import export_pair
from core import Cancelled,validate_bounds

def read_ids(path,grid):
 ids=[];unknown=[]
 for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
  value=line.split('#',1)[0].strip().upper()
  if not value:continue
  if value not in grid:unknown.append(value)
  elif value not in ids:ids.append(value)
 if unknown:raise ValueError('Unknown IDs: '+', '.join(unknown[:20])+'. Use one grid ID per line.')
 if not ids:raise ValueError('The file contains no grid IDs.')
 return ids

def write_state(path,state):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2),encoding='utf8');temp.replace(path)

def image_ok(path):
 try:
  with Image.open(path) as im:im.verify()
  return True
 except (OSError,ValueError):return False

def run(ids,grid,exe,parent,index,zoom=19,line=4,include_minor=False,progress=lambda s:None,cancelled=lambda:False,download_fn=SAT.download,load_fn=SAT.load_image,export_fn=export_pair,finalize_fn=None):
 parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
 SAT.commands(exe,parent,'batch_check',1,zoom,tuple(float(grid[ids[0]][k]) for k in ('left','right','top','bottom')))
 info=Path(index).stat()
 base=dict(zoom=zoom,line=line,include_minor=include_minor,satellite_id=1,street_id=66,index=str(Path(index).resolve()),index_size=info.st_size,index_mtime_ns=info.st_mtime_ns)
 summary=dict(total=len(ids),completed=[],skipped=[],failed={},cancelled=False)
 for number,key in enumerate(ids,1):
  if cancelled():summary['cancelled']=True;break
  bounds=validate_bounds(*(float(grid[key][k]) for k in ('left','right','top','bottom')))
  square_size=int(grid[key].get('grid_size_m') or (1000 if key.startswith('RYD_') else 0))
  config=dict(base,bounds=bounds,square_size_m=square_size,utm_col=grid[key].get('col'),utm_row=grid[key].get('row'))
  fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
  folder=parent/key/fingerprint[:12];folder.mkdir(parents=True,exist_ok=True);file=folder/'batch_job.json'
  try:state=json.loads(file.read_text())
  except (OSError,ValueError):state={}
  if state.get('fingerprint')!=fingerprint:state=dict(id=key,fingerprint=fingerprint,config=config,status='pending')
  def report(message):progress(f'{number}/{len(ids)} {key}: {message}')
  try:
   output=Path(state.get('aligned','__missing__'))
   required=('satellite.png','overlay.png','street_map_downloaded.png')
   if state.get('status')=='complete' and (output/'pair_metadata.json').is_file() and all(image_ok(output/name) for name in required):
    summary['skipped'].append(key);report('already complete; skipped');continue
   for stage,mid in [('satellite',1),('street',66)]:
    if cancelled():raise Cancelled('Batch stopped.')
    path=state.get(stage)
    if path and image_ok(path) and load_fn(path).get('geo'):report('reusing '+stage)
    else:
     state['status']='downloading_'+stage;write_state(file,state)
     report('downloading '+stage)
     image,_=download_fn(exe,folder,key+'_'+stage,mid,zoom,bounds,progress=report,cancelled=cancelled)
     if not load_fn(image).get('geo'):raise ValueError(stage+' export has no supported geographic metadata.')
     state[stage]=str(Path(image).resolve());write_state(file,state)
   if cancelled():raise Cancelled('Batch stopped.')
   state['status']='aligning';write_state(file,state);report('exporting aligned set')
   output,meta=export_fn(folder,key,load_fn(state['satellite']),bounds,index,line,include_minor,cancelled,report,standard_path=state['street'])
   if square_size:
    if finalize_fn is None:
     from exact_square import finalize_square
     finalizer=finalize_square
    else:finalizer=finalize_fn
    output,meta=finalizer(output,meta,int(grid[key]['col']),int(grid[key]['row']),square_size,line,cancelled,report)
   state.update(status='complete',aligned=str(Path(output).resolve()));state.pop('error',None);write_state(file,state)
   summary['completed'].append(key);report('complete')
  except Cancelled as exc:
   state.update(status='cancelled',error=str(exc));write_state(file,state);summary['cancelled']=True;break
  except Exception as exc:
   state.update(status='failed',error=str(exc));write_state(file,state);summary['failed'][key]=str(exc);report('failed: '+str(exc))
  finally:write_state(parent/'batch_summary.json',summary)
 write_state(parent/'batch_summary.json',summary)
 return summary
