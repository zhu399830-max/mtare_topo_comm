"""Metadata-only exact19 complement; original141 evidence is never overwritten."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from v8_completed_union_v1 import RUNS

UNION='configs/v3/gate3/v8_completed_union_v1.json'
UNION_SHA='622b98c37920ae57fc06816eb62400ec599bf1fd7f2320bd3030956dd6357c46'


def compile_scope(root):
    from mtare_topo.data.gse_supplement_teacher_scope_v1 import combine_plans
    from mtare_topo.data.gse_surface_teacher_scope_v1 import FIELDS
    root=Path(root).resolve(strict=True);opened={}
    def read(path,sha):
        raw=(root/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('source drift: '+path)
        opened[path]=sha;return raw
    union=json.loads(read(UNION,UNION_SHA))
    run,seal_sha=RUNS[0]
    pins={p:h for h,p in (l.split('  ',1) for l in read(run+'/artifacts/evidence_sha256.txt',seal_sha).decode().splitlines())}
    card=run+'/config/data_card.json';full=json.loads(read(card,pins[card]))['scope']
    selected={(s['task'],s['source_sequence_id']):s for s in union['missing']}
    if len(selected)!=19:raise ValueError('exact19 required')
    entries=[];arrays={};files=set();containers={}
    for e in full['entries']:
        obs=[s for s in e['observations'] if (s['task'],s['source_sequence_id']) in selected]
        if not obs:continue
        if any(selected[(s['task'],s['source_sequence_id'])]!=s for s in obs):raise ValueError('source mismatch')
        entry=deepcopy(e);entry['observations']=obs
        ids={s['source_sequence_id'] for s in obs}
        entry['input_references']=[r for r in e['input_references'] if r['source']['source_sequence_id'] in ids]
        if len(entry['input_references'])!=len(obs):raise ValueError('input reference mismatch')
        entries.append(entry);files.update(entry[k] for k in ('construction_path','codebook_path'))
        for field in FIELDS:
            prefix=e['sensor_prefix']+'/'+field;p=prefix+'/.zarray'
            plan=combine_plans(json.loads(read(p,full['file_sha256'][p])),field,obs)
            arrays[prefix]=plan;files.add(p);files.update(prefix+'/'+k for k in plan['chunk_keys'])
        for ref in entry['input_references']:
            c=containers.setdefault(ref['input_path'],dict(observation_count=ref['observation_count'],selected_rows=[]))
            if c['observation_count']!=ref['observation_count']:raise ValueError('container mismatch')
            c['selected_rows'].append(ref['input_row'])
    actual={(s['task'],s['source_sequence_id']) for e in entries for s in e['observations']}
    if actual!=set(selected):raise ValueError('incomplete complement')
    sources=[s for e in entries for s in e['observations']]
    counts=dict(observations=len(sources),tasks=len(entries),
                parents=len({s['parent_id'] for s in sources}),
                unique_variant_frames=len({(s['task'],f) for s in sources for f in s['frame_rows']}))
    if counts!=dict(observations=19,tasks=3,parents=1,unique_variant_frames=31):raise ValueError('population drift')
    for c in containers.values():
        c['selected_rows']=sorted(set(c['selected_rows']))
        c['incidental_rows']=c['observation_count']-len(c['selected_rows'])
    return dict(schema='v8_remaining19_scope_v1',status='DIAGNOSTIC_NOT_TRAINING',entries=entries,
        file_sha256={p:full['file_sha256'][p] for p in sorted(files)},array_access=arrays,
        input_sha256={p:full['input_sha256'][p] for p in sorted(containers)},
        metadata_sha256=opened,input_container_population=containers,
        counts=dict(counts,decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in arrays.values())),
        restrictions=['exact_missing19_only','no_training','no_calibration','no_test','unknown_not_negative',
                      'no_original_run_modification','no_recompute_completed122','no_retry'])


if __name__=='__main__':
    scope=compile_scope(Path(__file__).resolve().parents[2])
    print(json.dumps(dict(counts=scope['counts'],files=len(scope['file_sha256']),
                         containers=scope['input_container_population']),indent=2))
