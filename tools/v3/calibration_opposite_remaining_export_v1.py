"""Freeze or execute one exact, CPU-only six-field continuous input export."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import json
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile
from surface_features_v1 import PYTHON,environment,sha
from development_grids_v1 import write
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from calibration_opposite_remaining_card_v1 import SCHEMA,SLUG,validate_card
from calibration_opposite_remaining_scope_v1 import compile_scope
SELECTION='configs/v3/gate3/calibration_opposite_remaining_scope_v1.json'
from calibration_opposite_remaining_population_v1 import export_population

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no refreeze or overwrite')
    scope=compile_scope(ROOT)
    authority=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-09',
        scope_sha256=digest(scope),authorized_operations=['data_export'],authorized_gates=[3],
        scope='Opposite146 windows,all calibration,2parents4tasks,exact six fields; sealed143 inputs reused unchanged; development excluded.',
        confirmation_reference='User continuous autonomous authorization; PLAN permits exact missing-only multiview input export under standing autonomous authorization. No absolute pose input to any model.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=digest(scope),approval=authority)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,SELECTION})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',PYTHON,'tools/v3/calibration_opposite_remaining_export_v1.py',
             '--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=authority,command=command,
        question='Export exact original scan and relative-motion inputs for146 missing multiview windows without teacher labels.',
        method='Reuse SurfaceInputReader variable population and existing encoder input binder; same6fields,50m normalization,shared frame byte equality; no feature inference.',
        baseline='Original sealed six-field input interface; not a model performance comparison.',
        fallback='Fail and seal without retry or resampling on input or numerical inconsistency.',
        wall_time_cap_s=600,estimated_cost=dict(compute='CPU only;25085952 decoded padded bytes;0model/optimizer',disk_gb=2,wall_time_hours=1/6,host_ram_gb=4,gpu_vram_gb=0),
        acceptance_criteria=['Exact146 missing windows198 unique variant frames with original source seals and no out-of-scope reads.',
            'Six fields source-bound,finite and original dtype; exact current-origin zero and source-window identities.',
            '13 traversal-specific six-field NPZ packages bind all146 encoder inputs; overlapping scan bytes agree; no invented safety/graph success.'],
        expected_evidence=['config,source snapshot,environment,source-read hashes,input packages and manifest,raw log,RUN_STATE,seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    start=time.monotonic();error=None;result=None;opened={}
    def expire(*args):raise TimeoutError('600s source audit cap')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/bound_runtime.json',spec['environment'],'x')
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        result=export_population(ROOT,run,card['scope'],opened)
        for p,h in opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during run')
        write(run/'artifacts/input_manifest.json',result,'x')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>2*1024**3:raise OSError('2GiB output cap')
    except BaseException:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'SIX_FIELD_INPUT_EXPORT_COMPLETE',error=error,
            observations=146 if result else 0,elapsed_s=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            model_calls=0,optimizer_steps=0,scientific_gate_pass=False,physical_safety_verified=False)
        write(run/'artifacts/source_reads_sha256.json',opened,'x');write(run/'metrics/summary.json',summary,'x')
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        (run/'logs/raw.log').write_text(json.dumps(summary)+'\n')
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
