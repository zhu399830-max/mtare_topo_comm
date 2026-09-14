"""Same saved final prediction, confidence-preserving local conflict replay."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
from dataclasses import asdict
import argparse,hashlib,json,sys,time,traceback,resource,signal,zipfile
NAME='gse_local_conflict_replay_v1';CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
FIT='results/gate3_semantics/gate3_20260913_gse_branch_core_fit_v1_seed0'
REVIEW='results/gate3_semantics/gate3_20260913_gse_ai_branch_review_v1r1_seed0'

def freeze():
    previous=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_branch_core_fit_v1.json').read_text())
    scope=dict(entries=previous['scope']['entries'],observations=12,parents=3,rays=135699,pairs=1815651,training_steps=0,model_forwards=0,
        split='same fit C01/C06,24frames/3fragments, no new payload or independent test',
        sampling='All12 existing final-step outputs, no score-selected cases',
        teacher='Only sealed nonblind AI surface-core labels for post-output scoring; no physical full branch truth',
        partial_ai_reference_only=True,thresholds=[.1,.9],unknown_filtered=False,
        old='Archived binary confidence grouping: conflicting tied component rejected as whole',
        new='Keep all threshold-qualified repulsions as constraints; attraction ordered by original probabilities; exact-tie conflicting mergers deferred, earlier consistent components preserved',
        differences='Preserves raw attraction ranking rather than binary tie; preinstalls all threshold-qualified repulsion; cannot attribute benefit solely to one of these implementation changes',
        leakage='No new source frames,GT IDs,checkpoint choice or thresholds; labels read only after grouping',
        limits=dict(host_bytes=8*1024**3,output_bytes=512*1024**2,wall_seconds=600))
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['audit'],authorized_gates=[3],
        confirmation_reference='User: 开始干, after explicit same-prediction local conflict plan',scope='Same12 final outputs local conflict replay only',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=scope,approval=approval)
    from mtare_topo.governance_local_conflict import validate_card
    assert validate_card(card).passed
    pins={}
    for folder in (FIT,REVIEW):
        seal=ROOT/folder/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
        for line in seal.read_text().splitlines():
            h,p=line.split('  ',1);assert sha(ROOT/p)==h;pins[p]=h
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',sys.executable,'tools/v3/run_local_conflict_replay.py','--execute'],
        question='Can local conflict handling preserve useful groups on the identical saved predictions without increased core errors?',
        method=scope['new'],baseline=scope['old'],fallback='Keep model and old result; no retraining,threshold tuning or follow-on model search',
        estimated_cost=dict(compute='CPU same12 predictions,0GPU',host_ram_gb=8,gpu_vram_gb=0,disk_gb=.5,wall_time_hours=1/6),
        acceptance_criteria=['All135699 rays retained grouped or unresolved','Same masks/thresholds/no GT filtering','Order-invariant with tie abstention','Report all12 core restoration,merges,fragments,unknown-only groups and unknown membership','No method advantage or complete branch qualification from partial AI cores'],
        expected_evidence=['Old/new percase metrics,all groups and decisions,environment,source snapshot,logs,seal'],
        input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def score(h,ids,labels):
    lookup=dict(zip(ids,labels));per=[0,0];mixed=0;unknown_grouped=0;unknown_only=0
    for group in h['groups']:
        kinds={int(lookup[r]) for r in group if lookup[r]>=0}
        mixed+=int(len(kinds)>1);unknown_only+=int(not kinds)
        for k in kinds:per[k]+=1
        unknown_grouped+=sum(lookup[r]<0 for r in group)
    return dict(groups=len(h['groups']),groups_per_core=per,cross_core_merges=mixed,
        labeled_unresolved=sum(lookup[r]>=0 for r in h['unresolved_ray_ids']),unresolved_rays=len(h['unresolved_ray_ids']),
        unknown_grouped_rays=unknown_grouped,unknown_only_groups=unknown_only,
        group_sizes=sorted([len(g) for g in h['groups']],reverse=True))

def execute():
    import numpy as np
    from collections import Counter
    from mtare_topo.topology.local_conflict_grouping import group_local_conflicts
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());limits=card['scope']['limits']
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    for p,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():assert sha(ROOT/p)==h,p
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None
    def timeout(*_):raise TimeoutError('wall limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(limits['wall_seconds'])
    try:
        write(out/'config/replay_environment.json',dict(python=sys.version,numpy=np.__version__))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(ROOT/p,p)
        with (out/'logs/replay.jsonl').open('x') as log:
            for i in range(12):
                p=ROOT/FIT/f'artifacts/prediction_0500_{i:02d}.npz'
                with np.load(p,allow_pickle=False) as d:ids=d['ray_ids'].tolist();pairs=d['pairs'];prob=d['probabilities']
                # Exactly the old float32 threshold masks, original raw scores.
                pos=prob>=.9;neg=prob<=.1
                edges=[(ids[a],ids[b],float(prob[j]) if pos[j] else -float(1-float(prob[j]))) for j,(a,b) in enumerate(pairs) if pos[j] or neg[j]]
                result=group_local_conflicts(ids,edges);h=asdict(result.hypotheses)
                reverse=group_local_conflicts(ids[::-1],edges[::-1])
                assert result.hypotheses==reverse.hypotheses
                assigned=[x for g in h['groups'] for x in g]+h['unresolved_ray_ids'] if isinstance(h['unresolved_ray_ids'],list) else [x for g in h['groups'] for x in g]+list(h['unresolved_ray_ids'])
                assert len(assigned)==len(ids) and set(assigned)==set(ids)
                write(out/f'artifacts/groups_{i:02d}.json',h)
                with (out/f'artifacts/decisions_{i:02d}.jsonl').open('x') as f:
                    for row in result.decisions:f.write(json.dumps(row)+'\n')
                # Reference only after grouping is immutable on disk.
                with np.load(ROOT/REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:assert d['ray_ids'].tolist()==ids;labels=d['core_labels'].tolist()
                old=json.loads((ROOT/FIT/f'artifacts/groups_0500_{i:02d}.json').read_text())
                record=dict(observation=i,old=score(old,ids,labels),new=score(h,ids,labels),positive_pairs=int(pos.sum()),negative_pairs=int(neg.sum()),uncertain_pairs=int((~(pos|neg)).sum()),
                    decision_counts=dict(Counter(r[3] for r in result.decisions)),order_invariant=True,prediction_sha256=sha(p))
                # NumPy-free booleans/integers in score; standard JSON only.
                rows.append(record);log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limits['host_bytes']
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=limits['output_bytes']
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:signal.alarm(0)
    complete=lambda x:x['groups_per_core']==[1,1] and x['cross_core_merges']==0 and x['labeled_unresolved']==0
    n=sum(complete(r['new']) for r in rows)
    summary=dict(status='GATE_FAIL' if error or n!=12 else 'GATE_MIXED',error=error,observations=rows,
        completed=len(rows),old_complete_cases=sum(complete(r['old']) for r in rows),new_complete_cases=n,
        new_cross_core_merges=sum(r['new']['cross_core_merges'] for r in rows),new_unknown_only_groups=sum(r['new']['unknown_only_groups'] for r in rows),
        partial_reference_only=True,model_forwards=0,training_steps=0,method_advantage_proven=False,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'metrics/summary.json',summary);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='observations'}),flush=True);return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
