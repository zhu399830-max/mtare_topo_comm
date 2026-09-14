"""Teacher-only source metadata for289 existing calibration observations."""
import collections
import hashlib
import json
from pathlib import Path
from mtare_topo.data.gse_supplement_teacher_scope_v1 import combine_plans
from mtare_topo.data.gse_surface_teacher_scope_v1 import FIELDS
from mtare_topo.governance_surface_input import SEALS,expected_tasks
from mtare_topo.governance_identity_inventory import P1A


def compile_scope(root):
    root=Path(root);opened={}
    def read(p,h):
        raw=(root/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source metadata drift: '+p)
        opened[p]=h;return raw
    joined=json.loads(read('docs/figures/gse_graph/calibration_join_check_20260909.json',
                          '190eefc60fa512f2567baceccb0d108e855454018aaf35974bee18fd63284c15'))
    base=json.loads(read('configs/v3/gate3/calibration_missing_input_scope_v1.json',
                        'ccdbd30a3833c8b46e772d7f392c4966fa7d4191d554900ed9be4379535c8d97'))
    bounds={s['task']:(s['source_frame_count'],s['source_sequence_count']) for s in base['missing']}
    seal=SEALS['sensor'];pins={p:h for h,p in (l.split('  ',1) for l in read(seal['path'],seal['sha256']).decode().splitlines())}
    sources=expected_tasks();grouped=collections.defaultdict(list)
    for r in joined['references']:grouped[r['source']['task']].append(r)
    files={};plans={};inputs={};containers={};entries=[]
    for task,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:r['source']['sequence_row']);meta=sources[task];obs=[];refs=[]
        if meta['partition']!='c07':raise ValueError('calibration C07 only')
        for row in rows:
            s=dict(row['source']);s.pop('cached_input',None)
            s.update(parent_id=meta['parent_id'],variant=meta['variant'],source_frame_count=bounds[task][0],source_sequence_count=bounds[task][1])
            if s['split']!='calibration':raise ValueError('not calibration')
            obs.append(s);r=row['input'];p=r['path'];h=r['sha256']
            if p in inputs and inputs[p]!=h:raise ValueError('input hash conflict')
            inputs[p]=h
            refs.append(dict(source=s,input_path=p,input_sha256=h,input_row=r['row'],observation_count=r['container_rows']))
            c=containers.setdefault(p,dict(observation_count=r['container_rows'],selected_rows=[]))
            if c['observation_count']!=r['container_rows']:raise ValueError('container count conflict')
            c['selected_rows'].append(r['row'])
        for field in FIELDS:
            prefix=meta['sensor']+'/'+field;p=prefix+'/.zarray'
            header=json.loads(read(p,pins[p]));files[p]=pins[p];plan=combine_plans(header,field,obs);plans[prefix]=plan
            for k in plan['chunk_keys']:files[prefix+'/'+k]=pins[prefix+'/'+k]
        docs={}
        for folder,key in (('constructions','construction_path'),('codebooks','codebook_path')):
            p=P1A+'/artifacts/'+folder+'/c07/'+task+'.json';files[p]=pins[p];docs[key]=p
        entries.append(dict(task=task,split='calibration',observations=obs,input_references=refs,sensor_prefix=meta['sensor'],**docs))
    for c in containers.values():
        if len(set(c['selected_rows']))!=len(c['selected_rows']):raise ValueError('duplicate selected input')
        c['selected_rows'].sort();c['incidental_rows']=c['observation_count']-len(c['selected_rows'])
    allobs=[s for e in entries for s in e['observations']]
    counts=dict(observations=len(allobs),tasks=len(entries),parents=len({s['parent_id'] for s in allobs}),
                unique_variant_frames=len({(s['task'],f) for s in allobs for f in s['frame_rows']}),
                decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()))
    if [counts[k] for k in ('observations','tasks','parents','unique_variant_frames')]!=[289,12,4,397]:raise ValueError('population drift')
    return dict(schema='calibration289_teacher_scope_v1',status='METADATA_BOUND_NOT_EXECUTION',entries=entries,
        file_sha256=files,array_access=plans,input_sha256=inputs,metadata_sha256=opened,
        input_container_population=containers,counts=counts,
        restrictions=['calibration_only','teacher_never_student_input','no_test','unknown_not_negative','no_training',
                      'no_cross_traversal_stitching','historical_positive_conditioned_diagnostic_not_unbiased_evaluation'])


if __name__=='__main__':print(json.dumps(compile_scope(Path(__file__).resolve().parents[2]),separators=(',',':')))
