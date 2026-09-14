"""Teacher-only evidence source scope for the2676 newly exported observations."""
from collections import defaultdict
import math
from pathlib import Path

from mtare_topo.governance_identity_inventory import P1A
from mtare_topo.governance_surface_input import SEALS, expected_tasks
from .gse_surface_input_scope_v1 import checked_read, _object
from .gse_surface_teacher_scope_v1 import FIELDS, plan_field

INPUT = 'results/gate3_semantics/gate3_20260907_gse_supplement_input_v1_seed20260906'
INPUT_SEAL = 'b49727c02eb7265e40411e784c86c079fea4430b7464e51f68bf92c9d73c92a8'


def combine_plans(header,field,observations):
    if not observations:raise ValueError('nonempty fixed source selection required')
    plans=[plan_field(header,field,r['frame_rows'],r['source_frame_count']) for r in observations]
    rows=sorted({f for r in observations for f in r['frame_rows']})
    keys=sorted({k for p in plans for k in p['chunk_keys']},key=lambda k:int(k.split('.')[0]))
    count=header['shape'][0];chunk=header['chunks'][0]
    return dict(shape=header['shape'],chunks=header['chunks'],dtype=header['dtype'],selected_rows=rows,
        observation_rows=[r['frame_rows'] for r in observations],chunk_keys=keys,
        decoded_rows_including_collateral=sum(min(count,(int(k.split('.')[0])+1)*chunk)-int(k.split('.')[0])*chunk for k in keys),
        decoded_padded_bytes=len(keys)*math.prod(header['chunks'])*FIELDS[field][2])


def compile_supplement_teacher_scope(root):
    root=Path(root).resolve(strict=True);reads={};tasks=expected_tasks()
    seal=checked_read(root,INPUT+'/artifacts/evidence_sha256.txt',INPUT_SEAL,reads)
    mp=INPUT+'/artifacts/input_manifest.json'
    wanted={mp}|{INPUT+'/artifacts/inputs/'+t+'.npz' for t in tasks}
    inputs={}
    for line in seal.decode().splitlines():
        h,p=line.split('  ',1)
        if p in wanted:
            if p in inputs:raise ValueError('duplicate exported input')
            inputs[p]=h
    if set(inputs)!=wanted:raise ValueError('complete sealed input population required')
    manifest=_object(checked_read(root,mp,inputs[mp],reads))
    if manifest.get('schema_version')!='gse_supplement_input_manifest_v1':raise ValueError('input schema drift')
    by_task=defaultdict(list)
    for row in manifest['observations']:
        if row['task'] not in tasks:raise ValueError('protected/outside source task')
        source=tasks[row['task']]
        if row['parent_id']!=source['parent_id'] or row['variant']!=source['variant']:
            raise ValueError('input task identity mismatch')
        by_task[row['task']].append(row)
    if set(by_task)!=set(tasks) or sum(map(len,by_task.values()))!=2676:
        raise ValueError('frozen missing2676 population drift')
    raw=checked_read(root,SEALS['sensor']['path'],SEALS['sensor']['sha256'],reads)
    json_paths={P1A+'/artifacts/'+role+'/'+s['partition']+'/'+t+'.json'
                for t,s in tasks.items() for role in ('constructions','codebooks')}
    prefixes={s['sensor']+'/'+f+'/' for s in tasks.values() for f in FIELDS}
    candidates={}
    for line in raw.decode().splitlines():
        h,p=line.split(None,1)
        if p not in json_paths and not any(p.startswith(prefix) for prefix in prefixes):continue
        if p in candidates:raise ValueError('duplicate teacher source')
        candidates[p]=h
    files={p:candidates[p] for p in sorted(json_paths)};plans={};entries=[]
    for task,source in sorted(tasks.items()):
        observations=by_task[task]
        for field in FIELDS:
            prefix=source['sensor']+'/'+field;path=prefix+'/.zarray'
            header=_object(checked_read(root,path,candidates[path],reads))
            plan=combine_plans(header,field,observations);plans[prefix]=plan
            for path in [path,*(prefix+'/'+k for k in plan['chunk_keys'])]:files[path]=candidates[path]
        entries.append(dict(task=task,observations=observations,sensor_prefix=source['sensor'],
            input_path=INPUT+'/artifacts/inputs/'+task+'.npz',
            construction_path=P1A+'/artifacts/constructions/'+source['partition']+'/'+task+'.json',
            codebook_path=P1A+'/artifacts/codebooks/'+source['partition']+'/'+task+'.json'))
    for p,h in list(reads.items()):checked_read(root,p,h,{})
    return dict(schema='gse_supplement_teacher_scope_v1',status='PREPARATION_NOT_EXECUTION_AUTHORITY',
        entries=entries,file_sha256=files,input_sha256=inputs,array_access=plans,metadata_reads_sha256=reads,
        counts=dict(parents=70,tasks=210,observations=2676,selected_frames=13374,
            construction_files=210,codebook_files=210,input_npz_files=210,array_headers=630,
            source_chunk_files=sum(len(p['chunk_keys']) for p in plans.values()),
            decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),actual_payload_reads=0,labels=0),
        restrictions=['C01_C07_only','all_new_observations_no_support_filter','teacher_identity_never_student_input',
                      'multi_source_and_unknown_preserved','no_physical_root_labels','no_training'])
