"""Exact failed observation plus immediately preceding completed observation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

PREVIOUS='results/gate3_semantics/gate3_20260909_gse_v8_multiview_mechanism_v1_seed20260906'
SEAL='2e5ddeef503cbe00fd851e2342d2480a4cd4230b831233202a809f04580cc9f6'
TASK='S10_3d_complex_C04__c1_mixed'
IDS=(192653,192654)


def compile_scope(root):
    from mtare_topo.data.gse_supplement_teacher_scope_v1 import combine_plans
    from mtare_topo.data.gse_surface_teacher_scope_v1 import FIELDS
    root=Path(root).resolve(strict=True)
    raw=(root/PREVIOUS/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SEAL:raise ValueError('failed predecessor seal drift')
    pins={p:h for h,p in (l.split('  ',1) for l in raw.decode().splitlines())}
    opened={PREVIOUS+'/artifacts/evidence_sha256.txt':SEAL}
    def read(path,h):
        raw=(root/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source drift: '+path)
        opened[path]=h;return json.loads(raw)
    p=PREVIOUS+'/config/data_card.json';full=read(p,pins[p])['scope']
    entry=deepcopy(next(e for e in full['entries'] if e['task']==TASK))
    references={r['source']['source_sequence_id']:r for r in entry['input_references']}
    entry['observations']=[s for s in entry['observations'] if s['source_sequence_id'] in IDS]
    if tuple(s['source_sequence_id'] for s in entry['observations'])!=IDS:raise ValueError('exact pair required')
    if entry['observations'][0]['frame_rows'][1:]!=entry['observations'][1]['frame_rows'][:-1]:
        raise ValueError('pair must overlap by four original frames')
    entry['input_references']=[references[i] for i in IDS]
    arrays={};files={entry[k] for k in ('construction_path','codebook_path')}
    for field in FIELDS:
        prefix=entry['sensor_prefix']+'/'+field;p=prefix+'/.zarray'
        plan=combine_plans(read(p,full['file_sha256'][p]),field,entry['observations'])
        arrays[prefix]=plan;files.add(p);files.update(prefix+'/'+k for k in plan['chunk_keys'])
    containers={}
    for r in entry['input_references']:
        v=containers.setdefault(r['input_path'],dict(observation_count=r['observation_count'],selected_rows=[]))
        v['selected_rows'].append(r['input_row'])
    for v in containers.values():v['incidental_rows']=v['observation_count']-len(v['selected_rows'])
    p=PREVIOUS+'/logs/observations.jsonl'
    raw=(root/p).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pins[p]:raise ValueError('predecessor log drift')
    opened[p]=pins[p];logs=[json.loads(l) for l in raw.decode().splitlines()]
    completed=[l for l in logs if l['state']=='COMPLETED']
    if completed[-1]['source']['source_sequence_id']!=IDS[0] or logs[-1]['source']['source_sequence_id']!=IDS[1]:
        raise ValueError('not actual failed boundary')
    reference=PREVIOUS+'/artifacts/'+completed[-1]['evidence_file']
    return dict(schema='v8_batch_pair_scope_v1',status='DIAGNOSTIC_NOT_TRAINING',entries=[entry],
        file_sha256={p:full['file_sha256'][p] for p in sorted(files)},array_access=arrays,
        input_sha256={p:full['input_sha256'][p] for p in containers},input_container_population=containers,
        metadata_sha256=opened,counts=dict(observations=2,tasks=1,parents=1,unique_variant_frames=6,
            decoded_padded_bytes=sum(v['decoded_padded_bytes'] for v in arrays.values())),
        completed_reference=dict(path=reference,sha256=pins[reference]),
        restrictions=['exact_failed_pair_only','no_training','no_calibration','no_test','unknown_not_negative',
                      'no_retry','no_original_run_modification','compare_complete_preceding_output'])
