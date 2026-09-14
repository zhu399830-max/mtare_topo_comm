"""Exact fit/calibration input reuse and missing-only six-field access plan."""
import json
from collections import defaultdict,Counter
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_input import FIELDS,SEALS,expected_tasks
from .gse_surface_feature_scope_v1 import INPUT as ORIGINAL,INPUT_SEAL as ORIGINAL_SEAL
from .gse_supplement_teacher_scope_v1 import INPUT as SUPPLEMENT,INPUT_SEAL as SUPPLEMENT_SEAL
from .continuous_feature_scope_v1 import INPUT as CONTINUOUS,SEAL_SHA as CONTINUOUS_SEAL
from .gse_surface_input_export_v1 import plan_array_access,SurfaceInputReader

INVENTORY='configs/v3/gate3/multiview_coverage_inventory_v1.json'
INVENTORY_SHA='5f4b0277e8e4c53d01e1fb729f1d5566720c9d05aa7f515130aa40b3ef2dfeca'


def compile_scope(root):
    opened={}
    def read(p,h):
        raw=read_pinned(root,p,h);opened[p]=h;return raw
    def index(run,h):
        raw=read(run+'/artifacts/evidence_sha256.txt',h)
        return {p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    inventory=json.loads(read(INVENTORY,INVENTORY_SHA))
    active=[r for r in inventory['observations'] if r['split'] in ('fit','calibration')]
    if len(active)!=2259:raise ValueError('exact fit/cal2259 required')
    wanted={(r['task'],r['source_sequence_id']):r for r in active}
    existing={};bounds={}
    for run,h in ((ORIGINAL,ORIGINAL_SEAL),(SUPPLEMENT,SUPPLEMENT_SEAL)):
        pins=index(run,h);path=run+'/artifacts/input_manifest.json';doc=json.loads(read(path,pins[path]))
        grouped=defaultdict(list)
        for r in doc['observations']:grouped[r['task']].append(r)
        shards={r['task']:r for r in doc['task_shards']}
        for task,rows in grouped.items():
            # Only metadata from other partitions is seen; never their payloads.
            bounds[task]=(rows[0]['source_frame_count'],rows[0]['source_sequence_count'])
            for i,row in enumerate(rows):
                key=(task,row['source_sequence_id'])
                if key not in wanted:continue
                target=wanted[key]
                if target['frame_rows']!=row['frame_rows'] or target['split']!=row['split']:raise ValueError('old input identity mismatch')
                path=run+'/'+shards[task]['path']
                if pins.get(path)!=shards[task]['sha256']:raise ValueError('old input seal mismatch')
                existing.setdefault(key,[]).append(dict(input_path=path,input_sha256=pins[path],input_row=i,observation_count=len(rows)))
    pins=index(CONTINUOUS,CONTINUOUS_SEAL);path=CONTINUOUS+'/artifacts/input_manifest.json';doc=json.loads(read(path,pins[path]))
    for task in doc['tasks']:
        for i,sid in enumerate(task['source_sequence_ids']):
            key=(task['task'],sid)
            if key not in wanted:continue
            if wanted[key]['frame_rows']!=task['source_frames'][i]:raise ValueError('continuous input identity mismatch')
            path=CONTINUOUS+'/'+task['file']
            if pins.get(path)!=task['sha256']:raise ValueError('continuous seal mismatch')
            existing.setdefault(key,[]).append(dict(input_path=path,input_sha256=pins[path],input_row=i,observation_count=10))
    reuse=[];missing=[]
    for row in active:
        key=(row['task'],row['source_sequence_id'])
        if key in existing:
            # Keep alternate provenance; earliest original package has priority.
            reuse.append(dict(source=row,selected_input=existing[key][0],alternatives=existing[key][1:]))
        else:
            frames,sequences=bounds[row['task']]
            missing.append(dict(row,parent_id=row['task'].split('__')[0],source_frame_count=frames,source_sequence_count=sequences))
    sources=expected_tasks();tasks={r['task']:sources[r['task']] for r in missing};sealed={}
    for s in SEALS.values():
        for line in read(s['path'],s['sha256']).decode().splitlines():
            h,p=line.split('  ',1);sealed[p]=h
    files={};plans={}
    for task,source in sorted(tasks.items()):
        rows=[r for r in missing if r['task']==task]
        for role,names in FIELDS.items():
            prefix=source[role]
            for name in ('.zgroup','.zattrs'):
                p=prefix+'/'+name;read(p,sealed[p]);files[p]=sealed[p]
            selected=[f for r in rows for f in r['frame_rows']] if role=='sensor' else [r['sequence_row'] for r in rows]
            for name in names:
                prefix2=prefix+'/'+name;p=prefix2+'/.zarray';header=json.loads(read(p,sealed[p]));files[p]=sealed[p]
                plan=plan_array_access(header,name,role,selected);plans[prefix2]=plan
                for chunk in plan['chunk_keys']:
                    p=prefix2+'/'+chunk;files[p]=sealed[p]
    SurfaceInputReader(root,task_sources=tasks,selection=missing,sealed_keys=files,variable_population=True)
    return dict(schema='multiview_input_reuse_v1',status='METADATA_SCOPE_NOT_EXECUTION',metadata_sha256=opened,
        source_seals=SEALS,reuse=reuse,missing=missing,task_sources=tasks,input_files_sha256=files,array_access=plans,
        counts=dict(observations=len(active),reuse=len(reuse),missing=len(missing),missing_tasks=len(tasks),
            splits=dict(Counter(r['split'] for r in active)),missing_splits=dict(Counter(r['split'] for r in missing)),
            selected_unique_missing_variant_frames=len({(r['task'],f) for r in missing for f in r['frame_rows']}),
            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values())),
        restrictions=['development_payload_excluded','six_student_fields_only','no_structure_labels','no_training','no_new_worlds'])
