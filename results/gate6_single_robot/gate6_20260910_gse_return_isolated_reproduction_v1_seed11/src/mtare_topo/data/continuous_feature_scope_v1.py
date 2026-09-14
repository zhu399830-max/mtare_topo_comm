"""Bind three sealed input packages to the original frozen encoder."""
import hashlib
import json
from .gse_surface_feature_scope_v1 import TRAINING,CHECKPOINT_SHA
from .continuous_input_scope_v1 import PARENT,VARIANTS
from mtare_topo.governance_surface_material import read_pinned

INPUT='results/gate3_semantics/gate3_20260909_gse_continuous_model_input_export_v1_seed20260906'
SEAL_SHA='a9f3714946c379467a262eb6244135f14abc7b246f8ef46f79fc12e0acf5bdca'


def compile_feature_scope(root):
    opened={}
    def read(path,h):
        raw=read_pinned(root,path,h);opened[path]=h;return raw
    raw=read(INPUT+'/artifacts/evidence_sha256.txt',SEAL_SHA)
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    manifest_path=INPUT+'/artifacts/input_manifest.json'
    manifest=json.loads(read(manifest_path,pins[manifest_path]))
    if manifest['status']!='SIX_FIELD_INPUT_EXPORT_COMPLETE' or manifest['observations']!=30 or len(manifest['tasks'])!=3:
        raise ValueError('complete exact30 input required')
    tasks=[]
    for v in VARIANTS:
        name=PARENT+'__'+v
        candidates=[x for x in manifest['tasks'] if x['task']==name]
        if len(candidates)!=1:raise ValueError('unique expected task required')
        task=candidates[0];path=INPUT+'/'+task['file']
        if path!=INPUT+'/artifacts/inputs/'+name+'.npz' or pins[path]!=task['sha256']:
            raise ValueError('exact input file binding drift')
        if task['source_frames']!=[list(range(i,i+5)) for i in range(4012,4022)] or task['source_sequence_ids']!=list(range(158206,158216)):
            raise ValueError('fixed continuous windows differ')
        tasks.append(dict(task=name,input_path=path,input_sha256=task['sha256'],observations=[
            dict(input_row=i,source_sequence_id=s,frame_rows=task['source_frames'][i])
            for i,s in enumerate(task['source_sequence_ids'])]))
    return dict(schema_version='continuous_feature_scope_v1',metadata_sha256=opened,
        checkpoint=dict(path=TRAINING+'/artifacts/models/seed0/selected.pt',sha256=CHECKPOINT_SHA,seed=0,epoch=2),
        tasks=tasks,counts=dict(parents=1,tasks=3,physical_traversals=1,observations=30,unique_variant_frames=42),
        weights_read=False,scan_payloads_read=False,labels=0,training_steps=0)
