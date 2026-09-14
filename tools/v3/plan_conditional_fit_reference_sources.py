"""Metadata-only exact missing-field plan; reuse existing student caches."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json
from ai_junction_pilot import sha, write
from freeze_conditional_development_inputs import plan_rows
from mtare_topo.governance_surface_input import SEALS, expected_tasks
from mtare_topo.governance_identity_inventory import P1A

BINDING='docs/figures/gse_conditional_geometry_fit_v1/fit_cache_binding.json'
OUT='configs/v3/gate3/gse_conditional_fit_reference_source_scope_v1.json'
FIELDS=('sensor_xyz_m','yaw_deg','primitive_membership_code')


def main():
    if (ROOT/OUT).exists():raise FileExistsError('immutable source plan already exists')
    entries=json.loads((ROOT/BINDING).read_text())['entries']
    tasks=expected_tasks(); selected=sorted({e['task'] for e in entries})
    assert len(entries)==2880 and len(selected)==180
    assert all(tasks[t]['partition']=='fit' for t in selected)
    prefixes={tasks[t]['sensor']+'/'+f for t in selected for f in FIELDS}
    documents={P1A+'/artifacts/'+kind+'/fit/'+t+'.json' for t in selected for kind in ('constructions','codebooks')}
    pins={BINDING:sha(ROOT/BINDING)}; available={}
    for seal in SEALS.values():
        raw=(ROOT/seal['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seal['sha256']:raise ValueError('source seal drift')
        pins[seal['path']]=seal['sha256']
        for line in raw.decode().splitlines():
            digest,path=line.split(None,1)
            if path in documents or path.rsplit('/',1)[0] in prefixes:available[path]=digest
    plans={}; files={}
    for task in selected:
        rows=[e for e in entries if e['task']==task]; assert len(rows)==16
        indices=[i for e in rows for i in e['frame_rows']]
        files[task]={}
        for kind in ('constructions','codebooks'):
            path=P1A+'/artifacts/'+kind+'/fit/'+task+'.json'
            pins[path]=available[path];files[task][kind]=path
        for field in FIELDS:
            prefix=tasks[task]['sensor']+'/'+field;path=prefix+'/.zarray'
            raw=(ROOT/path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=available[path]:raise ValueError('header drift')
            plan=plan_rows(json.loads(raw),indices);plans[prefix]=plan;pins[path]=available[path]
            pins.update({prefix+'/'+key:available[prefix+'/'+key] for key in plan['chunk_keys']})
    scope=dict(schema_version='conditional_fit_missing_reference_scope_v1',status='METADATA_ONLY_NOT_EXPORT_AUTHORIZATION',
        entries=entries,observations=2880,parents=60,tasks=180,unique_variant_frames=14400,
        fields=list(FIELDS),task_files=files,array_plans=plans,input_sha256=pins,
        decoded_padded_bytes=sum(p['decoded_padded_bytes'] for p in plans.values()),
        array_headers_read=len(plans),document_payloads_read=0,array_chunks_read=0,
        student_cache_reuse=True,teacher_only=True,labels_generated=0,training_steps=0,
        caveats=['Source chunk collateral is not additional examples',
                 'Keep original float64 absolute poses; never student input',
                 'Source components remain construction-conditioned references, not semantic identity',
                 'Source scope is not permission for full reference generation or training'])
    write(ROOT/OUT,scope)
    print(json.dumps({k:v for k,v in scope.items() if k not in ('entries','task_files','array_plans','input_sha256')},indent=2))


if __name__=='__main__':main()
