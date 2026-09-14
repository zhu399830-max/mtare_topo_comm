"""Reuse the complete sealed V4 population's raw evidence, never recast rays."""
import gzip
import json
from .gse_joint_cached_interfaces_v1 import read_cached
from mtare_topo.governance_surface_material import read_pinned

RUN='results/gate3_semantics/gate3_20260907_gse_supplement_joint_v2_seed20260906'
SEAL=RUN+'/artifacts/evidence_sha256.txt'
SHA='d9cc2435ad928626cc53095503b0d94e8f231168a0893704e96bd52b40bc079f'


def compile_cache(root,scope):
    hashes={}
    for line in read_pinned(root,SEAL,SHA).decode().splitlines():
        h,p=line.split('  ',1)
        if p in hashes:raise ValueError('duplicate sealed V4 path')
        hashes[p]=h
    pins={}
    for task in scope['entries']:
        for row in task['observations']:
            key=row['task']+'_'+str(row['source_sequence_id'])
            path=RUN+'/artifacts/'+key+'.json.gz'
            if path not in hashes or key in pins:raise ValueError('V4 population missing or duplicate')
            pins[key]=dict(path=path,sha256=hashes[path])
    if len(pins)!=2676:raise ValueError('exact V4 population required')
    return pins


def read_v4(root,pin,source,opened):
    packed=read_pinned(root,pin['path'],pin['sha256']);opened[pin['path']]=pin['sha256']
    data=json.loads(gzip.decompress(packed))
    if ('raw_interfaces' in data)==('raw_interfaces_reference' in data):
        raise ValueError('one raw evidence storage form required')
    if 'raw_interfaces' in data:raw=data['raw_interfaces']
    else:raw,_=read_cached(root,data['raw_interfaces_reference'],source,opened)
    if raw['source']!=source:raise ValueError('V4 raw source mismatch')
    return raw,data['produced_targets']


def compare_versions(old,new):
    a,b=old['record'],new['record']
    if old['source_binding']!=new['source_binding']:raise ValueError('source changed')
    for key in a:
        if key not in ('openings','membership') and a[key]!=b[key]:
            raise ValueError('unexpected non-opening target change: '+key)
    old_by_position={tuple(x['position_m']):i for i,x in enumerate(a['openings'])}
    retained=[];added=[];changes=[]
    for j,item in enumerate(b['openings']):
        i=old_by_position.get(tuple(item['position_m']))
        if i is None:added.append(j);continue
        if item!=a['openings'][i]:raise ValueError('retained opening attributes changed')
        retained.append(i)
        for k,(before,after) in enumerate(zip(a['membership'][i],b['membership'][j],strict=True)):
            if before!=after:changes.append(dict(old_opening_index=i,new_opening_index=j,anchor_index=k,before=before,after=after))
    return dict(retained_old_openings=retained,added_new_openings=added,
        removed_old_openings=[i for i in range(len(a['openings'])) if i not in retained],
        membership_changes=changes,old_target_record_sha256=old['target_record_sha256'],
        new_target_record_sha256=new['target_record_sha256'],raw_source_reused=True,
        full_label_qualification=False)
