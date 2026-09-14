"""Bound full-route references combining existing packages without copying."""
import json
from collections import defaultdict
from .multiview_input_reuse_v1 import compile_scope,INVENTORY,INVENTORY_SHA
from mtare_topo.governance_surface_material import read_pinned

EXPORT='results/gate3_semantics/gate3_20260909_gse_multiview_missing_input_export_v1_seed20260906'
SEAL='3bed30a77c9bfd890efa441a1affd506e781b183405a5c386450e18f8d251da3'


def compile_join(root):
    scope=compile_scope(root);opened=dict(scope['metadata_sha256'])
    def read(p,h):
        raw=read_pinned(root,p,h);opened[p]=h;return raw
    raw=read(EXPORT+'/artifacts/evidence_sha256.txt',SEAL)
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    p=EXPORT+'/artifacts/input_manifest.json';manifest=json.loads(read(p,pins[p]))
    if manifest['status']!='SIX_FIELD_INPUT_EXPORT_COMPLETE' or manifest['observations']!=2055:raise ValueError('export incomplete')
    def key(s):return s['task'],s['source_sequence_id'],tuple(s['frame_rows'])
    index={key(r['source']):r['selected_input'] for r in scope['reuse']}
    missing={key(r) for r in scope['missing']}
    for task in manifest['tasks']:
        path=EXPORT+'/'+task['file']
        if pins.get(path)!=task['sha256']:raise ValueError('package seal mismatch')
        n=task['observations']
        if n!=len(task['source_sequence_ids']) or n!=len(task['source_frames']):raise ValueError('package rows differ')
        for i,(s,f) in enumerate(zip(task['source_sequence_ids'],task['source_frames'])):
            k=(task['task'],s,tuple(f))
            if k not in missing or k in index:raise ValueError('duplicate or foreign window')
            index[k]=dict(input_path=path,input_sha256=task['sha256'],input_row=i,observation_count=n)
    inventory=json.loads(read(INVENTORY,INVENTORY_SHA));grouped=defaultdict(list)
    for r in inventory['observations']:
        if r['split']=='development':continue
        grouped[r['task']].append(dict(source=r,**index.pop(key(r))))
    if index or sum(map(len,grouped.values()))!=2259 or len(grouped)!=195:raise ValueError('full fit/cal population differs')
    tasks=[]
    for task,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:r['source']['decision_index'])
        if [r['source']['decision_index'] for r in rows]!=list(range(len(rows))):raise ValueError('missing route decision')
        tasks.append(dict(task=task,split=rows[0]['source']['split'],observations=rows))
    return dict(schema='multiview_joined_inputs_v1',metadata_sha256=opened,tasks=tasks,
        counts=dict(observations=2259,tasks=195,parents=65,fit=2079,calibration=180),
        labels=0,model_calls=0,development_payload=False,continuous_motion_verified=False)
