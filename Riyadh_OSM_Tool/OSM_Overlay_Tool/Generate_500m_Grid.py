import sys,csv,json,math,hashlib
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pbf_reader import node_batches,ways
from core import clip_segment
from pyproj import Transformer
if len(sys.argv)<2:raise SystemExit('Usage: python Generate_500m_Grid.py path_to_source.osm.pbf')
source=Path(sys.argv[1])
bounds=(46.4832,47.0222,24.9823,24.5375);left,right,top,bottom=bounds
forward=Transformer.from_crs(4326,32638,always_xy=True);inverse=Transformer.from_crs(32638,4326,always_xy=True)
nodes={};last=0
for batch,position in node_batches(source):
 for i,x,y in batch:
  if left-.02<=x<=right+.02 and bottom-.02<=y<=top+.02:nodes[i]=(x,y)
 if position-last>35_000_000:print('Node pass',round(position/source.stat().st_size*100),'%',flush=True);last=position
print('Regional nodes:',len(nodes),flush=True)
allowed={'motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','living_street','service'}
stats=defaultdict(lambda:{'length':0.,'ways':set(),'bins':set(),'residential_length':0.})
valid={}
def square(col,row):
 key=(col,row)
 if key not in valid:
  x,y=col*500,row*500
  corners=[inverse.transform(a,b) for a,b in ((x,y),(x+500,y),(x+500,y+500),(x,y+500))]
  lo=min(p[0] for p in corners);hi=max(p[0] for p in corners);bo=min(p[1] for p in corners);to=max(p[1] for p in corners)
  valid[key]=(lo,hi,to,bo,corners) if lo>=left and hi<=right and bo>=bottom and to<=top else None
 return valid[key]
count=0
for wid,tags,refs in ways(source):
 if tags.get('highway') not in allowed or tags.get('area')=='yes':continue
 if tags.get('surface') in {'unpaved','gravel','dirt','ground','sand','earth'}:continue
 for a,b in zip(refs,refs[1:]):
  if a not in nodes or b not in nodes:continue
  p=forward.transform(*nodes[a]);q=forward.transform(*nodes[b])
  for col in range(math.floor(min(p[0],q[0])/500),math.floor(max(p[0],q[0])/500)+1):
   for row in range(math.floor(min(p[1],q[1])/500),math.floor(max(p[1],q[1])/500)+1):
    if not square(col,row):continue
    x,y=col*500,row*500
    segment=clip_segment(p,q,(x,y,x+500,y+500))
    if not segment:continue
    a1,b1=segment;length=math.dist(a1,b1)
    if length<.01:continue
    st=stats[(col,row)];st['length']+=length;st['ways'].add(wid)
    if tags['highway'] in {'residential','living_street','tertiary','secondary','primary'}:st['residential_length']+=length
    for n in range(26):
     t=n/25;xx=a1[0]+(b1[0]-a1[0])*t;yy=a1[1]+(b1[1]-a1[1])*t
     st['bins'].add((min(4,max(0,int((xx-x)/100))),min(4,max(0,int((yy-y)/100)))))
 count+=1
 if count%30000==0:print('Road pass:',count,'motor-road ways;',len(stats),'cells',flush=True)
print('Road-rich candidates',len(stats),flush=True)
out=Path('Riyadh_500m_Grid_Package');out.mkdir(exist_ok=True)
records=[];features=[]
for ordinal,key in enumerate(sorted(stats,key=lambda k:(-k[1],k[0])),1):
 col,row=key;st=stats[key];lo,hi,to,bo,corners=square(col,row)
 rec=dict(id=f'RYD500_{ordinal:05d}',left=f'{math.floor(lo*1e7)/1e7:.7f}',right=f'{math.ceil(hi*1e7)/1e7:.7f}',top=f'{math.ceil(to*1e7)/1e7:.7f}',bottom=f'{math.floor(bo*1e7)/1e7:.7f}',col=col,row=row,grid_size_m=500,road_length_m=round(st['length'],2),osm_way_count=len(st['ways']),coverage_bins=len(st['bins']),residential_length_m=round(st['residential_length'],2))
 # Rounded download envelopes must also stay wholly within requested bounds.
 if float(rec['left'])<left or float(rec['right'])>right or float(rec['top'])>top or float(rec['bottom'])<bottom:continue
 records.append(rec)
 features.append(dict(type='Feature',properties=rec,geometry=dict(type='Polygon',coordinates=[corners+[corners[0]]])))
qualified=[r for r in records if r['coverage_bins']>=20 and r['road_length_m']>=3000 and r['residential_length_m']>=1500]
assert len(qualified)>=40,len(qualified)
selected=sorted(qualified,key=lambda r:(-r['coverage_bins'],-r['residential_length_m'],-r['road_length_m'],r['id']))[:40]
fields=list(records[0])
for filename,data in [('riyadh_500m_grid.csv',records),('Selected_40_500m_Coordinates.csv',selected)]:
 with (out/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
(out/'Batch_40_Riyadh_500m_IDs.txt').write_text('\n'.join(r['id'] for r in selected)+'\n')
(out/'riyadh_500m_grid.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features)))
(out/'Selected_40_500m_Squares.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[f for f in features if f['properties']['id'] in {r['id'] for r in selected}])))
meta=dict(source=source.name,bounds_wgs84=bounds,crs='EPSG:32638',square_size_m=500,grid_anchor='Multiples of 500 UTM metres',candidate_squares=len(records),selected_squares=40,selected_min_road_length_m=min(r['road_length_m'] for r in selected),selected_min_coverage_bins=min(r['coverage_bins'] for r in selected),method='Clipped motor-road length and occupied 100m subcells; excludes explicit unpaved surfaces, tracks, paths, construction, proposed and area roads.',limitation='OSM coverage is not visual proof of paved surfaces. Regional node filtering uses a 0.02 degree buffer; segments with either endpoint outside the buffer are omitted.',zoom_recommendation=20)
(out/'grid_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
