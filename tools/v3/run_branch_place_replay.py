"""One causal branch-place replay of existing geometry, no new sensor or fit."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import sys
import time
import traceback
from mtare_topo.governance_branch_place_replay import SCHEMA,SLUG,POLICY,scope,validate_card

CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,obj,mode='x'):
    with p.open(mode) as f:json.dump(obj,f,indent=2,allow_nan=False)

def freeze():
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_operations=['audit'],authorized_gates=[6],scope='One sealed tunnel episode; branch-place replay, no new control or training',
        confirmation_reference='User ongoing end-to-end objective; next recorded branch management verification')
    card=dict(schema_version=SCHEMA,card_id=SLUG,scope=scope(ROOT),approval=a)
    assert validate_card(card).passed;write(ROOT/CARD,card)
    files=[CARD,'tools/v3/run_branch_place_replay.py','src/mtare_topo/governance_branch_place_replay.py','src/mtare_topo/topology/geometry_branch_places.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260910',slug=SLUG,seed=11,
        operation='audit',data_card=CARD,config_path=CARD,user_authorization=a,
        command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',sys.executable,'tools/v3/run_branch_place_replay.py','--execute'],
        question='Do the existing geometric continuations form persistent observation places and pending branch tasks?',
        method='Frozen geometry, causal repeated multi-direction observation; no physical junction-center claim.',
        baseline='Previous metric-anchor-only output; not a scored independent comparison.',
        estimated_cost=dict(compute='CPU only, 288 existing geometry observations',disk_gb=.1,wall_time_hours=.02),
        acceptance_criteria=['Exact292 rows/288 observations; no edge inferred; retain every update and unresolved task.',
            'Describe duplicate or ambiguous places without claiming ground-truth detection accuracy.'],
        expected_evidence=['place snapshot,288 decisions,summary,raw log,seal'],source_sha256={f:sha(ROOT/f) for f in files}))

def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());start=time.monotonic();error=None;rows=0;observations=0;registry=None
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    if spec!=json.loads((run/'config/run_spec.json').read_text()):raise ValueError('spec drift')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces
        card=json.loads((ROOT/CARD).read_text());assert validate_card(card).passed
        for p,h in spec['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('source drift')
        s=card['scope']
        if sha(ROOT/s['trace'])!=s['sha256']:raise ValueError('trace drift')
        registry=GeometryBranchPlaces(**POLICY)
        with (run/'artifacts/decisions.jsonl').open('x') as out:
            for line in (ROOT/s['trace']).open():
                rows+=1;r=json.loads(line);v=r.get('result')
                if not v:continue
                observations+=1
                result=registry.observe(v['geometry'],v['decision']['proposals'],navigation_node_id=v['decision']['node'])
                out.write(json.dumps(dict(source_scan=r['scan'],order=v['geometry']['timestamp'],result=result))+'\n')
        if (rows,observations)!=(292,288):raise ValueError('population mismatch')
        write(run/'artifacts/places.json',registry.snapshot())
    except Exception:error=traceback.format_exc()
    places=registry.snapshot()['places'] if registry else []
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,rows=rows,observations=observations,
        places=[dict(id=p['id'],state=p['state'],xyz=p['observer_xyz_m'],supports=len(p['orders']),tasks=len(p['frontiers'])) for p in places],
        verified_edges=0,elapsed_s=time.monotonic()-start,closed_loop=False,ground_truth_accuracy_measured=False)
    write(run/'metrics/summary.json',result);write(run/'logs/raw.json',result)
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
