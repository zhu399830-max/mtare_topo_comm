"""Supplement-only feature scope, metadata reads and no teacher payloads."""
import json
import hashlib
from collections import defaultdict,Counter
from .gse_surface_feature_scope_v1 import compile_feature_scope as original_scope
from .gse_supplement_teacher_scope_v1 import INPUT,INPUT_SEAL
from mtare_topo.governance_surface_material import read_pinned


def compile_feature_scope(root):
    original=original_scope(root);opened=dict(original['metadata_sha256'])
    def read(path,h):
        raw=read_pinned(root,path,h);opened[path]=h;return raw
    seal=read(INPUT+'/artifacts/evidence_sha256.txt',INPUT_SEAL)
    pins={}
    for line in seal.decode().splitlines():
        h,p=line.split('  ',1)
        if p in pins:raise ValueError('duplicate input seal path')
        pins[p]=h
    mp=INPUT+'/artifacts/input_manifest.json';manifest=json.loads(read(mp,pins[mp]))
    if manifest['schema_version']!='gse_supplement_input_manifest_v1':raise ValueError('supplement schema drift')
    prior={(t['task'],r['source_sequence_id']) for t in original['tasks'] for r in t['observations']}
    task_splits={t['task']:t['observations'][0]['split'] for t in original['tasks']}
    grouped=defaultdict(list);frames=set();splits=Counter();seen=set();parents=set()
    for row in manifest['observations']:
        task=row['task'];key=(task,row['source_sequence_id'])
        if task not in task_splits or key in prior or key in seen or row['split']!=task_splits[task]:
            raise ValueError('scope overlap, duplicate or split drift')
        seen.add(key);parents.add(row['parent_id']);splits[row['split']]+=1
        frames.update((task,f) for f in row['frame_rows']);grouped[task].append(row)
    if (len(seen),len(frames),len(parents),len(grouped))!=(2676,13374,70,210):
        raise ValueError('exact supplemental population drift')
    tasks=[]
    for task,rows in sorted(grouped.items()):
        path=INPUT+'/artifacts/inputs/'+task+'.npz'
        tasks.append(dict(task=task,input_path=path,input_sha256=pins[path],observation_count=len(rows),
            observations=[dict(input_row=i,**r) for i,r in enumerate(rows)]))
    return dict(schema_version='gse_supplement_feature_source_plan_v1',metadata_sha256=opened,
        checkpoint=original['checkpoint'],tasks=tasks,
        counts=dict(parents=70,tasks=210,observations=2676,source_frames=13374,splits=dict(splits)),
        original_feature_observation_overlap=0,teacher_payload_reads=0)
