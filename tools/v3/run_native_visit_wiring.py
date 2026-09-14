"""Source-only visit/registration/task integration over existing native records."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter, defaultdict
import json
import platform
import resource
import time
import traceback
import zipfile

from run_native_token_retrieval import SOURCE,sha,write
from mtare_topo.governance_native_structure_capture import scope_digest,validate_visit_wiring_card

REGISTRATION='results/gate6_single_robot/gate6_20260914_gse_native_token_registration_v1_seed11'
NAME='gse_native_visit_wiring_v1';RUN='results/gate6_single_robot/gate6_20260914_'+NAME+'_seed11'
CARD='configs/v3/gate6/data_cards/'+NAME+'.json';SPEC='configs/v3/gate6/'+NAME+'.json'
TRACE=SOURCE+'/artifacts/model_inputs/native_traces.jsonl'
PAIRS=REGISTRATION+'/artifacts/registration.jsonl'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise ValueError('fresh paths required')
    s=dict(source_run=SOURCE,registration_run=REGISTRATION,world='tunnel',seed=11,trajectories=1,
        trace_records=2024,feedback_records=1681,visit_records=121,epochs=115,registration_pairs=560,
        raw_frames=617,selected_raw_frames=575,
        independent_units='one120s historically used development trajectory, not115independent places',
        sampling='all original receipt-ordered native messages and all560saved pair results; exact keys, not nearest time',
        spacing='no resampling; original native scan/pose stamps and message order retained',
        split='development interface integration, no train/test/threshold selection',teacher=None,protected_worlds_read=[],
        scope='native_metric_visit_provenance_not_direction_or_place_identity',
        training_steps=0,model_forwards=0,registration_calls=0,new_simulations=0,control_changes=False)
    a=dict(status='APPROVED',approved_by='user',approved_at='2026-09-14',authorized_operations=['audit'],authorized_gates=[6],
        scope_sha256=scope_digest(s),scope='Same-native recorded visits, frozen alignment outputs and task lifecycle wiring only',
        confirmation_reference='User 继续推进 and active goal explicitly connects geometry verification to task state and native route advice; standing same-record integration authority')
    card=dict(schema_version='gse_native_visit_wiring_card_v1',card_id=NAME,scope=s,approval=a)
    if not validate_visit_wiring_card(card).passed:raise ValueError('card mismatch')
    write(ROOT/CARD,card)
    sources=['tools/v3/run_native_visit_wiring.py','tools/v3/run_native_token_retrieval.py','tools/v3/_bootstrap.py',
        'src/mtare_topo/integration/native_epoch_visit_binding.py','src/mtare_topo/integration/native_region_task_registry.py',
        'src/mtare_topo/integration/native_route_advice.py','src/mtare_topo/topology/structural_task_lifecycle.py',
        'src/mtare_topo/governance.py','src/mtare_topo/governance_native_structure_capture.py',CARD]
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260914',slug=NAME,seed=11,
        operation='audit',data_card=CARD,user_authorization=a,command=['/home/zeng-workstation/anaconda3/bin/python','tools/v3/run_native_visit_wiring.py','--execute'],
        question='Which saved local matches refer to the same recorded native visit and which remain distinct or source-unbound?',
        method='Exact source/pose/depot visit binding and original task registry; preserve all560geometry verdicts',
        baseline='156unverified local fits,44epochs with multiple scan candidates; native region states retained',
        fallback='explicit unknown bindings, no nearest-time recovery, no direction/level/identity from grid ID or completion transfer',
        estimated_cost=dict(wall_time_hours=1/60,disk_gb=.02,compute='CPU standard library only,1GiB/60s'),
        acceptance_criteria=['all115epochs and560pairs preserved', 'strict causal history, same visit only if exact declared provenance',
            'prefix replay equality and no changed registration/model/old task completion evidence'],
        expected_evidence=['epoch-to-visit records, pair classifications, original registry replay, prefix check, source snapshot, environment/log/seal'],
        source_sha256={p:sha(ROOT/p) for p in sources},input_sha256={TRACE:sha(ROOT/TRACE),PAIRS:sha(ROOT/PAIRS)}))
    print(SPEC)


def execute():
    from mtare_topo.integration.native_epoch_visit_binding import NativeEpochVisitBinding,classify_local_matches
    from mtare_topo.integration.native_region_task_registry import NativeRegionTaskRegistry
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((out/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('run already used')
    if json.loads((out/'config/run_spec.json').read_text())!=spec:raise ValueError('spec drift')
    if not validate_visit_wiring_card(json.loads((ROOT/CARD).read_text())).passed:raise ValueError('scope drift')
    for path,digest in {**spec['source_sha256'],**spec['input_sha256']}.items():
        if sha(ROOT/path)!=digest:raise ValueError('source drift: '+path)
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in spec['source_sha256']:z.write(ROOT/p,p)
    write(out/'config/execution_environment.json',dict(python=platform.python_version(),gpu_used=False,training_steps=0))
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    start=time.monotonic();error=None;bindings=[];matches=[];decisions=[];prefix_equal=False;registry_state=None
    try:
        rows=[json.loads(l) for l in (ROOT/TRACE).read_text().splitlines()]
        pairs=[dict(json.loads(l),record_ref=PAIRS+':'+str(i+1)) for i,l in enumerate((ROOT/PAIRS).read_text().splitlines())]
        if len(rows)!=2024 or len(pairs)!=560:raise ValueError('population mismatch')
        by_current=defaultdict(list)
        for pair in pairs:by_current[pair['current_epoch']].append(pair)
        binder=NativeEpochVisitBinding();history={}
        registry=NativeRegionTaskRegistry(segment='tunnel_seed11_shadow',trajectory_id='tunnel_seed11_native')
        midpoint=next(i for i,row in enumerate(rows) if row['topic'].endswith('/candidates') and row['payload']['epoch'].endswith(':57'))
        prefix_expected=None
        with (out/'logs/wiring.log').open('x') as log:
            for i,row in enumerate(rows):
                if time.monotonic()-start>60:raise RuntimeError('wall budget exceeded')
                binding=binder.consume(row)
                if binding is not None:
                    bindings.append(binding)
                    matches.append(classify_local_matches(binding,by_current.get(binding['epoch'],[]),history))
                    history[binding['epoch']]=binding
                    log.write(binding['epoch']+' '+binding['binding_status']+' '+binding['reason']+'\n');log.flush()
                suffix=row['topic'].rsplit('/',1)[-1]
                if suffix in ('candidates','decision','feedback'):
                    decisions.append(registry.consume({'candidates':'snapshot','decision':'decision','feedback':'feedback'}[suffix],
                        row['payload'],order=i,record_ref=row['record_ref']))
                if i==midpoint:prefix_expected=json.dumps(bindings,sort_keys=True)
        prefix_binder=NativeEpochVisitBinding();prefix=[]
        for row in rows[:midpoint+1]:
            bound=prefix_binder.consume(row)
            if bound is not None:prefix.append(bound)
        prefix_equal=json.dumps(prefix,sort_keys=True)==prefix_expected
        if len(bindings)!=115 or sum(len(m['scan_candidates']) for m in matches)!=560 or not prefix_equal:raise ValueError('incomplete or noncausal output')
        registry_state=registry.snapshot()
    except Exception:error=traceback.format_exc()
    write(out/'artifacts/epoch_visits.json',bindings);write(out/'artifacts/registered_visit_candidates.json',matches)
    write(out/'artifacts/registry_decisions.json',decisions);write(out/'artifacts/native_region_registry.json',registry_state)
    binding_counts=Counter(b['reason'] for b in bindings)
    kinds=Counter(r['kind'] for m in matches for r in m['scan_candidates'])
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,epochs=len(bindings),
        exact_bound_epochs=sum(b['binding_status']=='EXACT_NATIVE_METRIC_VISIT' for b in bindings),
        binding_reasons=dict(binding_counts),pair_kinds=dict(kinds),
        exact_metric_visit_groups=len({b['metric_visit_id'] for b in bindings if b['metric_visit_id'] is not None}),
        epochs_with_multiple_fit_visits=sum(m['accepted_visit_group_count']>1 for m in matches),
        prefix_equal=prefix_equal,confirmed_associations=0,direction_tasks_created=0,control_changes=0,
        model_forwards=0,registration_calls=0,training_steps=0,elapsed_s=time.monotonic()-start,
        limitation='Visit grouping is recorded native provenance only, not a unique structural place or completed direction')
    write(out/'metrics/summary.json',summary);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
