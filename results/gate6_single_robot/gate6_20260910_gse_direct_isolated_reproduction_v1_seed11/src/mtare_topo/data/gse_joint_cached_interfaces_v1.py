"""Reuse sealed raw interfaces; never reuse old targets as revised targets."""
import gzip
import json
from mtare_topo.governance_surface_material import read_pinned

CACHE_RUN='results/gate3_semantics/gate3_20260907_gse_supplement_joint_v1_seed20260906'
CACHE_SEAL=CACHE_RUN+'/artifacts/evidence_sha256.txt'
CACHE_SHA='a634ff3580e22d60d98b2c1192b481ddd3241254a0c041e9530fda7e73ce8f78'


def compile_cache(root,scope):
    seal=read_pinned(root,CACHE_SEAL,CACHE_SHA).decode()
    hashes={}
    for line in seal.splitlines():
        h,p=line.split('  ',1)
        if p in hashes:raise ValueError('duplicate sealed cache path')
        hashes[p]=h
    result={}
    for entry in scope['entries']:
        for row in entry['observations']:
            key=row['task']+'_'+str(row['source_sequence_id'])
            p=CACHE_RUN+'/artifacts/'+key+'.json.gz'
            if p in hashes:result[key]=dict(path=p,sha256=hashes[p])
    if len(result)!=512:raise ValueError('exact512 immutable raw interface cache required')
    return result


def read_cached(root,pin,source,opened):
    packed=read_pinned(root,pin['path'],pin['sha256']);opened[pin['path']]=pin['sha256']
    data=json.loads(gzip.decompress(packed));raw=data['raw_interfaces'];old=data['produced_targets']
    if raw['source']!=source:raise ValueError('cached interfaces from different observation')
    return raw,old


def compare_prior_targets(old,new):
    a=old['record'];b=new['record']
    if old['source_binding']!=new['source_binding']:raise ValueError('source binding drift')
    for key in a:
        if key not in ('openings','membership') and a[key]!=b[key]:
            raise ValueError('unexpected non-opening target change:'+key)
    used=[]
    for i,item in enumerate(b['openings']):
        candidates=[j for j,previous in enumerate(a['openings']) if item==previous]
        if len(candidates)!=1:raise ValueError('revised opening must retain an exact original reference')
        j=candidates[0]
        if a['membership'][j]!=b['membership'][i]:raise ValueError('unexpected retained-opening membership change')
        used.append(j)
    return dict(old_openings=len(a['openings']),new_openings=len(b['openings']),
        retained_old_indices=used,removed_from_positive_only=[j for j in range(len(a['openings'])) if j not in used],
        raw_source_reused=True,old_target_record_sha256=old['target_record_sha256'])
