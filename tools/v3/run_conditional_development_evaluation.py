"""Prepare only after all240references seal; one final-checkpoint evaluation.

Governance registration is deferred until the active geometry export ends so
that no file bound by that live run is edited. No training entry point exists.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from ai_junction_pilot import sha,write
from export_conditional_development_features import RUN as FEATURE, MANIFEST, PYTHON
from export_development_geometry import RUN as TARGET
NAME='gse_conditional_development_evaluation_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
MODELS='configs/v3/gate3/gse_conditional_frozen_models_v1.json'


def require_complete_reference(state,summary):
    if state.get('state')!='COMPLETED' or summary.get('error') is not None or summary.get('completed')!=240:
        raise ValueError('complete successful240reference export required; no partial selection')
    cases=[r['case'] for r in summary['windows']]
    if sorted(cases)!=list(range(240)):raise ValueError('duplicate/missing reference cases')


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        s,a=card['scope'],card['approval']
        assert card['schema_version']=='gse_conditional_development_evaluation_card_v1'
        assert len(s['entries'])==240 and s['training_steps']==0 and s['variants']==list('ABC')
        assert [e['identity']['case'] for e in s['entries']]==list(range(240))
        expected={f'S{x}_C07' for x in ('03_flat_unicyclic_small','04_3d_unicyclic_small','05_flat_branch_medium','06_3d_branch_medium','10_3d_complex')}
        assert {e['identity']['parent_id'] for e in s['entries']}==expected
        assert all(e['identity']['split']=='development' for e in s['entries'])
        assert s['checkpoint_step']==2000 and s['split_audit']['strict_unseen'] is False
        assert s['target_schema']=='construction_conditioned_geometry_targets_v1'
        expected_hashes=dict(A='247132276bf54ec1f2be79fea2966c9bc481ecbf2de02d6b906350e6eb37378d',B='e2b6691134793a337e571cfa6d19693425b0e71717120ce9e0bf5ec12260210b',C='1000e0ae0ae559915965e9e1cefbe8f08adf9c4aad16a89340fdee60328d995d')
        assert {v:s['models'][v]['sha256'] for v in 'ABC'}==expected_hashes
        assert s['limits']==dict(wall_seconds=1800,host_bytes=8*1024**3,gpu_bytes=28*1024**3,output_bytes=2*1024**3)
        assert a['status']=='APPROVED' and a['authorized_operations']==['data_export'] and a['authorized_gates']==[3]
        assert a['scope_sha256']==hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
    except (KeyError,TypeError,AssertionError):errors.append('fixed conditional development evaluation scope drift')
    return ValidationReport(passed=not errors,errors=errors,warnings=[])


def freeze():
    # Deliberately before manifest/model reads or any card/run creation.
    state=json.loads((ROOT/TARGET/'RUN_STATE.json').read_text())
    if state.get('state')!='COMPLETED':raise ValueError('reference export still running or failed; do not freeze')
    summary=json.loads((ROOT/TARGET/'metrics/summary.json').read_text());require_complete_reference(state,summary)
    manifest=json.loads((ROOT/MANIFEST).read_text());models=json.loads((ROOT/MODELS).read_text())
    pins={MANIFEST:sha(ROOT/MANIFEST),MODELS:sha(ROOT/MODELS)};seals={}
    for base in (FEATURE,TARGET):
        p=ROOT/base/'artifacts/evidence_sha256.txt';pins[str(p.relative_to(ROOT))]=sha(p)
        seals.update({p:h for h,p in (line.split('  ',1) for line in p.read_text().splitlines())})
    entries=[]
    for identity in manifest['entries']:
        i=identity['case'];e=dict(identity=identity,feature=FEATURE+f'/artifacts/window_{i:02d}.npz',target=TARGET+f'/artifacts/case_{i:03d}/targets.npz')
        for name in ('feature','target'):
            path=e[name]
            if sha(ROOT/path)!=seals[path]:raise ValueError('sealed input drift '+path)
            pins[path]=seals[path]
        entries.append(e)
    for v in 'ABC':
        r=models[v]
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('model drift')
        pins[r['path']]=r['sha256']
    s=dict(entries=entries,models=models,variants=list('ABC'),checkpoint_step=2000,training_steps=0,
        target_schema='construction_conditioned_geometry_targets_v1',split_audit=manifest['split_audit'],
        scoring='same compiled component-pair weights as fit; all/same/cross, constant axis1/height0, parent-paired descriptive intervals; no threshold tuning',
        limits=dict(wall_seconds=1800,host_bytes=8*1024**3,gpu_bytes=28*1024**3,output_bytes=2*1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-frozen-development-evaluation',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['data_export'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='720frozen model observation forwards,240shared inputs; zero training/selection/graph',
        confirmation_reference='Confirmed conditional supervision and active goal authorize independent-head-parent development comparison after full references')
    card=dict(schema_version='gse_conditional_development_evaluation_card_v1',scope=s,approval=a);assert validate_card(card).passed
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src',PYTHON,'tools/v3/run_conditional_development_evaluation.py','--execute'],
        question='Does the12fit CvsB conditional-geometry benefit transfer to all5head-held-out development parents?',
        method=s['scoring'],baseline='Frozen A/B/C plus axis1/height0; all variants reported, no checkpoint/threshold selection',
        fallback='Seal numerical/input/coverage failure; no retraining or favorable subset evaluation',
        estimated_cost=dict(compute='720final-model forwards on one GPU; stream one observation; no optimizer',host_ram_gb=8,gpu_vram_gb=28,disk_gb=2,wall_time_hours=.5),
        acceptance_criteria=['240complete paired observations,720forwards','All valid outputs including unknown retained and finite','Model state unchanged','Parent paired metrics and unknown coverage; no strict-test/topology claim'],
        expected_evidence=['720raw predictions, per-observation and parent metrics, fixed comparisons, runtime, environment, source snapshot, seal'],input_sha256=pins,source_sha256=sources))
    print(json.dumps(dict(spec=SPEC,observations=240,forwards=720,training_steps=0)))


def execute():
    import numpy as np,torch
    from evaluate_conditional_development_core import evaluate
    from conditional_development_statistics import summarize
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope'];limits=s['limits']
    assert validate_card(card).passed and json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    start=time.monotonic();error=None;rows=[];write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*_):raise TimeoutError('fixed evaluation wall cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(limits['wall_seconds'])
    def check_limits():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limits['host_bytes']:raise MemoryError('host cap')
        if torch.cuda.max_memory_reserved()>limits['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>limits['output_bytes']:raise RuntimeError('evidence cap')
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('bound drift '+p)
        torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(limits['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,gpu=torch.cuda.get_device_name(0)))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        rows=evaluate(ROOT,run,s['entries'],s['models'],check_limits=check_limits)
        write(run/'metrics/paired_parents.json',summarize(rows,[e['identity'] for e in s['entries']]))
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,prediction_rows=len(rows),training_steps=0,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        meaning='Frozen-head development evaluation; scientific gain determined by paired metrics, not successful execution')
    write(run/'metrics/summary.json',result);write(run/'logs/raw.json',result);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as stream:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':stream.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
