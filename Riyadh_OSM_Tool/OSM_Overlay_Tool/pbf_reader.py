"""Offline PBF/XML reader: no native OSM importer, subprocesses or network."""
import itertools
import struct
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from google.protobuf import descriptor_pb2 as D, descriptor_pool, message_factory

schema=D.FileDescriptorProto(name='overlay.proto',package='overlay',syntax='proto2')
def define(name,fields):
    msg=schema.message_type.add(name=name)
    for field,number,kind,repeated in fields:
        item=msg.field.add(name=field,number=number,label=3 if repeated else 1)
        if isinstance(kind,str):item.type=11;item.type_name='.overlay.'+kind
        else:item.type=kind
        if repeated and not isinstance(kind,str) and kind not in (9,12):item.options.packed=True
    return msg
define('BlobHeader',[('type',1,9,False),('datasize',3,5,False)])
define('Blob',[('raw',1,12,False),('raw_size',2,5,False),('zlib_data',3,12,False)])
define('Header',[('required_features',4,9,True)])
define('Strings',[('s',1,12,True)])
define('Node',[('id',1,18,False),('lat',8,18,False),('lon',9,18,False)])
define('Dense',[('id',1,18,True),('lat',8,18,True),('lon',9,18,True)])
define('Way',[('id',1,4,False),('keys',2,13,True),('vals',3,13,True),('refs',8,18,True)])
define('Group',[('nodes',1,'Node',True),('dense',2,'Dense',False),('ways',3,'Way',True)])
msg=define('Block',[('strings',1,'Strings',False),('groups',2,'Group',True),('granularity',17,5,False),('lat_offset',19,3,False),('lon_offset',20,3,False)])
msg.field[2].default_value='100'
pool=descriptor_pool.DescriptorPool();pool.Add(schema)
types={m.name:message_factory.GetMessageClass(pool.FindMessageTypeByName('overlay.'+m.name)) for m in schema.message_type}

def blocks(source):
    with Path(source).open('rb') as stream:
        while raw:=stream.read(4):
            if len(raw)!=4:raise ValueError('Truncated PBF block header.')
            size=struct.unpack('>I',raw)[0]
            if not 0<size<65536:raise ValueError('Invalid PBF header length.')
            header=types['BlobHeader'].FromString(stream.read(size))
            if not 0<header.datasize<=32*1024*1024:raise ValueError('Invalid PBF blob length.')
            raw=stream.read(header.datasize)
            if len(raw)!=header.datasize:raise ValueError('Truncated PBF blob.')
            blob=types['Blob'].FromString(raw)
            if blob.HasField('raw'):data=blob.raw
            elif blob.HasField('zlib_data'):data=zlib.decompress(blob.zlib_data)
            else:raise ValueError('PBF compression not supported; use standard raw/zlib OSM PBF.')
            if header.type=='OSMHeader':
                features=set(types['Header'].FromString(data).required_features)
                if features-{'OsmSchema-V0.6','DenseNodes'}:raise ValueError('Unsupported PBF required features: '+str(features))
            elif header.type=='OSMData':yield types['Block'].FromString(data),stream.tell()

def node_batches(source):
    if Path(source).suffix.lower()=='.osm':
        batch=[]
        for event,el in ET.iterparse(source,events=['end']):
            if el.tag=='node':
                batch.append((int(el.attrib['id']),float(el.attrib['lon']),float(el.attrib['lat'])));el.clear()
                if len(batch)==8000:yield batch,0;batch=[]
            elif el.tag in {'way','relation'}:el.clear()
        if batch:yield batch,0
        return
    for block,position in blocks(source):
        for group in block.groups:
            if group.HasField('dense'):
                d=group.dense
                ids=np.cumsum(d.id,dtype=np.int64)
                lon=(block.lon_offset+np.cumsum(d.lon,dtype=np.int64)*block.granularity)/1e9
                lat=(block.lat_offset+np.cumsum(d.lat,dtype=np.int64)*block.granularity)/1e9
                yield ((int(i),float(x),float(y)) for i,x,y in zip(ids,lon,lat)),position
            if group.nodes:
                yield ((n.id,(block.lon_offset+n.lon*block.granularity)/1e9,(block.lat_offset+n.lat*block.granularity)/1e9) for n in group.nodes),position

def ways(source):
    if Path(source).suffix.lower()=='.osm':
        for event,el in ET.iterparse(source,events=['end']):
            if el.tag=='way':
                tags={t.attrib['k']:t.attrib['v'] for t in el.findall('tag')}
                yield int(el.attrib['id']),tags,[int(n.attrib['ref']) for n in el.findall('nd')]
                el.clear()
            elif el.tag in {'node','relation'}:el.clear()
        return
    for block,_ in blocks(source):
        strings=[s.decode('utf8','replace') for s in block.strings.s]
        for group in block.groups:
            for way in group.ways:
                tags={strings[k]:strings[v] for k,v in zip(way.keys,way.vals)}
                if 'highway' in tags:
                    yield way.id,tags,list(itertools.accumulate(way.refs))
