"""Metadata-only teacher source boundary for exact2259 fit/cal windows.

No construction/codebook contents, pose/membership chunks, or labels are read.
Sources remain teacher-only; student packages stay original six-field inputs.
"""
import json
from .multiview_joined_inputs_v1 import compile_join
from .multiview_input_reuse_v1 import compile_scope as input_scope
from .gse_supplement_teacher_scope_v1 import combine_plans
from .gse_surface_teacher_scope_v1 import FIELDS
from mtare_topo.governance_surface_input import SEALS,expected_tasks
from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_material import read_pinned


def compile_scope(root):
    joined=compile_join(root);base=input_scope(root);opened=dict(joined['metadata_sha256'])
    opened.update(base['metadata_sha256'])
    source=SEALS['sensor'];raw=read_pinned(root,source['path'],source['sha256']);opened[source['path']]=source['sha256']
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    bounds={r['task']:(r['source_frame_count'],r['source_sequence_count']) for r in base['missing']}
    sources=expected_tasks();files={};plans={};inputs={};entries=[]
    for task in joined['tasks']:
        name=task['task'];s=sources[name]
        if task['split'] not in ('fit','calibration'):raise ValueError('development teacher forbidden')
        count,seq=bounds[name]
        observations=[]
        for row in task['observations']:
            r=dict(row['source'],parent_id=s['parent_id'],source_frame_count=count,source_sequence_count=seq)
            observations.append(r);inputs[row['input_path']]=row['input_sha256']
        for field in FIELDS:
            prefix=s['sensor']+'/'+field;path=prefix+'/.zarray'
            raw=read_pinned(root,path,pins[path]);opened[path]=pins[path];files[path]=pins[path]
            plan=combine_plans(json.loads(raw),field,observations);plans[prefix]=plan
            for key in plan['chunk_keys']:
                path=prefix+'/'+key;files[path]=pins[path]
        documents={}
        for role,key in (('constructions','construction_path'),('codebooks','codebook_path')):
            path=P1A+'/artifacts/'+role+'/'+s['partition']+'/'+name+'.json'
            files[path]=pins[path];documents[key]=path
        entries.append(dict(task=name,split=task['split'],observations=observations,input_references=task['observations'],
            sensor_prefix=s['sensor'],**documents))
    return dict(schema='multiview_teacher_source_scope_v1',status='METADATA_ONLY_NOT_LABEL_AUTHORITY',
        entries=entries,file_sha256=files,input_sha256=inputs,array_access=plans,metadata_sha256=opened,
        counts=dict(observations=2259,tasks=195,parents=65,fit=2079,calibration=180,
            construction_files=195,codebook_files=195,
            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
            unique_variant_frames=sum(len(p['selected_rows']) for k,p in plans.items() if k.endswith('/yaw_deg'))),
        labels_generated=0,teacher_payload_reads=0,model_calls=0,
        restrictions=['student_teacher_separation','no_development_payload','no_C08_C10','no_training',
            'reuse_original_alignment_and_evidence_algorithms','unknown_not_negative','every_window_retained'])
