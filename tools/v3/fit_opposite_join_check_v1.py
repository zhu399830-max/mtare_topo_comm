"""Verify saved141 inputs, composing motion only inside each traversal."""
import collections
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.evaluation.continuous_anchor_motion_v1 import compose_current_frames

ROOT=Path(__file__).resolve().parents[2]
RUN='results/gate3_semantics/gate3_20260909_gse_fit_opposite_input_export_v1_seed20260906'


def check():
    opened={}
    def read(p,h):
        raw=(ROOT/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source drift '+p)
        opened[p]=h;return raw
    inventory=json.loads(read('configs/v3/gate3/fit_opposite_inventory_v1.json',
                             'eab1bbcb8fddf3f1b6eabf4c04e63de5a91e6f0cfc3bea47de3f95cdaa4f0f02'))
    seal=read(RUN+'/artifacts/evidence_sha256.txt','3fbfe3e63b96c9b14be12088867834eb036b258afd0ff1c81a5ae03079b53b15')
    pins={p:h for h,p in (l.split('  ',1) for l in seal.decode().splitlines())}
    p=RUN+'/artifacts/input_manifest.json';manifest=json.loads(read(p,pins[p]))
    key=lambda s:(s['task'],s['source_sequence_id'],tuple(s['frame_rows']))
    index={}
    for task in manifest['tasks']:
        p=RUN+'/'+task['file']
        if pins[p]!=task['sha256']:raise ValueError('package binding mismatch')
        for i,(sid,frames) in enumerate(zip(task['source_sequence_ids'],task['source_frames'],strict=True)):
            k=(task['task'],sid,tuple(frames))
            if k in index:raise ValueError('duplicate new input')
            index[k]=dict(path=p,sha256=task['sha256'],row=i,container_rows=task['observations'])
    groups=collections.defaultdict(list)
    for s in inventory['rows']:
        ref=s['cached_input']
        if ref is not None and key(s) in index:raise ValueError('old/new overlap')
        if ref is None:ref=index.pop(key(s))
        else:
            ref=dict(ref)
            mp=ref['path'].split('/artifacts/')[0]+'/artifacts/input_manifest.json'
            old=json.loads(read(mp,inventory['metadata_sha256'][mp]))
            items=[o for o in old['observations'] if o['task']==s['task']]
            if key(items[ref['row']])!=key(s) or items[ref['row']]['split']!='fit':
                raise ValueError('cached row identity/split mismatch')
            ref['container_rows']=len(items)
        groups[(s['task'],s['traversal_id'])].append(dict(source=s,input=ref))
    if index:raise ValueError('extra new inputs')
    records=[];references=[];unique_frames=set()
    for (task,traversal),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['source']['sequence_row']);cache={};frames={};translations=[];yaws=[];histories=[]
        for r in rows:
            s,ref=r['source'],r['input'];p=ref['path']
            if p not in cache:
                raw=read(p,ref['sha256'])
                with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                    cache[p]=(raw,data['ranges_m'].copy(),data['valid_mask'].copy())
            raw,ranges,valid=cache[p]
            window=bind_feature_input(raw,expected_sha256=ref['sha256'],task=task,row=ref['row'],
                source_sequence_id=s['source_sequence_id'],frame_rows=s['frame_rows'],observation_count=ref['container_rows'])
            translations.append(window.translation_m);yaws.append(window.yaw_deg);histories.append(s['frame_rows'])
            for j,f in enumerate(s['frame_rows']):
                v=(ranges[ref['row'],j],valid[ref['row'],j]);unique_frames.add((task,f))
                if f in frames and any(not np.array_equal(a,b) for a,b in zip(frames[f],v)):raise ValueError('shared scan differs')
                frames[f]=v
            references.append(r)
        rotations,positions=compose_current_frames(translations,yaws,histories)
        records.append(dict(task=task,traversal_id=traversal,observations=len(rows),
            positions_in_first_frame_m=positions.tolist(),step_m=np.linalg.norm(np.diff(positions,axis=0),axis=1).tolist()))
    if len(references)!=141 or len(records)!=13 or len(unique_frames)!=193:raise ValueError('joined population mismatch')
    return dict(status='SAVED141_INPUTS_SHARED_FRAMES_AND_WITHIN_TRAVERSAL_MOTION_VERIFIED',
        observations=141,variant_traversals=13,unique_variant_frames=193,source_sha256=opened,
        references=references,routes=records,teacher_calls=0,model_calls=0,
        limitation='no cross-traversal stitching; not global trajectory, safety or label qualification')


if __name__=='__main__':print(json.dumps(check(),separators=(',',':')))

