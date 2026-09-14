"""One frozen-output diagnostic; no branch or exploration success claim."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import write,sha
import argparse,json,hashlib,sys,time,resource
NAME='gse_nonexclusive_support_v1';CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
FIT='results/gate3_semantics/gate3_20260913_gse_branch_core_fit_v1_seed0'
CACHE='results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'

def freeze():
    entries=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_branch_core_fit_v1.json').read_text())['scope']['entries']
    scope=dict(entries=entries,observations=12,parents=3,raw_frames=24,rays=135699,pairs=1815651,training_steps=0,model_forwards=0,maximum_anchors=32,thresholds=[.1,.9],teacher_in_inference=False,branch_identity_verified=False,
        sampling='All original12 overlapping windows,3 fit fragments, no new test population',split='C01/C06 fit only; no C07-C10 or robot test',teacher='None loaded; no new labels or independent performance scoring',
        anchors='Deterministic FPS on unit directions of all valid current rays; max32, no repeated padding, no GT or score selection',method='Direct saved anchor-ray high-score relation support only; no transitive closure, no anchor merge, repulsion retained separately. Old same-group scores are NOT calibrated direction membership.',
        limits=dict(host_bytes=8*1024**3,output_bytes=512*1024**2,wall_seconds=600))
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['audit'],authorized_gates=[3],confirmation_reference='User 继续 following explicit no-training nonexclusive observation-support minimal implementation proposal',scope='One original12 frozen-prediction nonexclusive interface diagnostic; no training or graph',scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=scope,approval=approval)
    from mtare_topo.governance_nonexclusive_support import validate_card
    assert validate_card(card).passed
    write(ROOT/CARD,card);pins={CARD:sha(ROOT/CARD)}
    for i in range(12):
        for p in [f'{FIT}/artifacts/prediction_0500_{i:02d}.npz',f'{CACHE}/artifacts/common_{i:02d}.npz']:pins[p]=sha(ROOT/p)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ['src/mtare_topo','tools/v3'] for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=approval,
        command=[sys.executable,'tools/v3/run_nonexclusive_support.py','--execute'],question='Does a nontransitive nonexclusive interface preserve anchor distinctions and unresolved evidence on the same predictions?',method=scope['method'],baseline='Archived exclusive-grouping failures retained; not a like-for-like branch metric comparison',fallback='Report unsupported/overlapping evidence; no training, labels, threshold tuning or branch claims',
        estimated_cost=dict(compute='CPU original12 cached outputs, zero forwards',host_ram_gb=8,gpu_vram_gb=0,disk_gb=.5,wall_time_hours=1/6),acceptance_criteria=['Every valid ray retained','No anchor merging or inferred transitive relation','All direct high and low threshold pairs at anchors accounted for','Unknown and shared support reported; not converted to physical edges'],expected_evidence=['Anchors, direct supports, repulsions, unresolved rays per case','Summary, command, inputs, source hashes, environment, seal'],input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np
    from dataclasses import asdict
    from mtare_topo.topology.nonexclusive_relation_support import anchored_relation_support
    from mtare_topo.representation.observed_query_sampling_v1 import sample_observed_queries
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    for p,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():assert sha(ROOT/p)==h,p
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[]
    write(out/'config/replay_environment.json',dict(python=sys.version,numpy=np.__version__))
    for i in range(12):
        with np.load(ROOT/FIT/f'artifacts/prediction_0500_{i:02d}.npz',allow_pickle=False) as d:ids=d['ray_ids'];pairs=d['pairs'];prob=d['probabilities']
        with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:xyz=d['registered_returns_xyz_m'][4*11520+ids].astype(np.float64)
        direction=xyz/np.linalg.norm(xyz,axis=1)[:,None];index=sample_observed_queries(direction,maximum=32);anchors=ids[index].tolist()
        active=np.isin(ids[pairs[:,0]],anchors)|np.isin(ids[pairs[:,1]],anchors)
        relations=[(int(ids[a]),int(ids[b]),float(p)) for (a,b),p in zip(pairs[active],prob[active])]
        r=anchored_relation_support(ids.tolist(),anchors,relations)
        assert r==anchored_relation_support(ids[::-1].tolist(),anchors[::-1],relations[::-1])
        expected_positive=sum((int(ids[a]) in anchors)+(int(ids[b]) in anchors) for (a,b),p in zip(pairs[active],prob[active]) if p>=np.float32(.9))
        actual=sum(len(x) for _,x in r.support_by_anchor);assert actual==expected_positive
        write(out/f'artifacts/support_{i:02d}.json',asdict(r))
        row=dict(observation=i,rays=len(ids),anchors=len(anchors),direct_positive_memberships=actual,shared_rays=len(r.shared_rays),unsupported_rays=len(r.unsupported_rays),repulsive_memberships=sum(len(x) for _,x in r.repulsion_by_anchor),anchor_merges=0)
        rows.append(row);print(json.dumps(row),flush=True)
        assert time.monotonic()-start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
    write(out/'logs/replay.json',dict(raw_case_records=rows))
    write(out/'metrics/summary.json',dict(status='GATE_MIXED',observations=rows,elapsed_s=time.monotonic()-start,model_forwards=0,training_steps=0,method_advantage_proven=False,branch_identity_verified=False,meaning='Nonexclusive interface diagnostic completed; zero anchor merges is by construction, not a detection achievement'))
    write(out/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    freeze() if a.freeze else execute()
