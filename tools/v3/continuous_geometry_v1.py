"""Freeze and execute the exact30 existing-scan geometry front-end diagnostic."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from mtare_topo.governance_continuous_geometry import SCHEMA,SLUG,compile_scope,digest,validate_card

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed0'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value,mode='x'):
    with p.open(mode) as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def freeze():
    scope=compile_scope(ROOT)
    authority=dict(status='APPROVED',approved_by='user-standing-development-authorization',
        approved_at='2026-09-10',authorized_operations=['audit'],authorized_gates=[3],
        scope_sha256=digest(scope),scope='Exact existing30 C04 observations; geometry proposals only; no training, teacher or protected test access.',
        confirmation_reference='User: set goal and run complete system, maintain geometry main path; standing autonomous development authorization. PLAN exact existing30 front-end execution.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='audit',scope=scope,
        scope_sha256=digest(scope),approval=authority)
    if not validate_card(card).passed:raise ValueError(validate_card(card).errors)
    cost=dict(compute='CPU single-thread; GPU0; existing6.1MB inputs; RAM4GiB cap',
        disk_gb=.5,wall_time_hours=1/6)
    write(ROOT/CARD,card)
    # Sources are pinned independently from mutable progress documents.
    paths=[Path(__file__).resolve(),ROOT/CARD]
    paths+=list((ROOT/'src/mtare_topo/semantics').glob('*.py'))
    paths += [ROOT/'src/mtare_topo'/p for p in (
        'integration/continuous_geometry_frontend.py','integration/geometry_structure_trace.py',
        'data/gse_supplement_feature_input_v1.py','data/gse_surface_feature_input_v1.py',
        'evaluation/continuous_anchor_motion_v1.py','governance_continuous_geometry.py')]
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',sys.executable,
        'tools/v3/continuous_geometry_v1.py','--execute']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,
        seed=0,operation='audit',data_card=CARD,config_path=CARD,user_authorization=authority,
        question='What finite primitive compositions does the existing front end produce on the exact continuous development scans?',
        method='Full sensor sector proposals, observed10m fit, finite-axis composition; no new model or parameter search.',
        baseline='Existing geometric implementation unchanged; not comparative research.',
        command=command,estimated_cost=cost,acceptance_criteria=[
            'All30 processed or explicit failure; no teacher, no missing outputs silently removed.',
            'Source hashes preserved; proposed structures not confirmed openings/edges; per-variant local odometry.'],
        expected_evidence=['30 geometry records, summary, source hashes, log, RUN_STATE, seal'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print(SPEC)


def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':
        raise ValueError('fresh created run required')
    if json.loads((run/'config/run_spec.json').read_text())!=spec:raise ValueError('spec snapshot drift')
    start=time.monotonic();rows=[];error=None
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name),'w')
    def expire(*_):raise TimeoutError('600s cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        import numpy as np
        from mtare_topo.integration.continuous_geometry_frontend import replay_package
        card=json.loads((ROOT/CARD).read_text())
        if not validate_card(card).passed:raise ValueError('card drift')
        for p,h in spec['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        for item in card['scope']['tasks']:
            payload=(ROOT/item['path']).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('input drift')
            records=replay_package(payload,item,composition_policy=card['scope']['composition_policy'])
            for i,r in enumerate(records):
                write(run/'artifacts'/f"{item['task']}_{i:02d}.json",r)
                rows.append(dict(task=item['task'],row=i,primitives=len(r['primitives']),
                    fitted=sum(p['axis_controls_world_m'] is not None for p in r['primitives']),
                    structures=len(r['structures']),rejected_axes=r['rejected_axes']))
            print(item['task'],len(records),'completed',flush=True)
        if len(rows)!=30:raise ValueError('incomplete population')
    except BaseException:
        error=traceback.format_exc()
    finally:
        signal.alarm(0)
        summary=dict(status='FAILED' if error else 'GEOMETRY_FRONTEND_EXECUTED',
            observations=len(rows),per_observation=rows,error=error,elapsed_s=time.monotonic()-start,
            trained_model=False,graph_verified=False,closed_loop=False)
        write(run/'metrics/summary.json',summary)
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name),'w')
        write(run/'logs/raw.json',summary)
        with (run/'artifacts/evidence_sha256.txt').open('x') as f:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.execute:raise SystemExit(execute())
    else:parser.error('choose --freeze or --execute')
