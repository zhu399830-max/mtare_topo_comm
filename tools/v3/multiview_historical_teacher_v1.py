"""One immutable all-window historical partial-supervision export, no training."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import json
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from surface_features_v1 import PYTHON, environment, sha
from development_grids_v1 import write
from historical_teacher_client_v1 import HistoricalTeacherClient, PYTHON as OLD_PYTHON
from teacher_snapshot_probe_v1 import ZIP, SHA
from mtare_topo.governance import load_json, build_run_id
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_multiview_teacher_v1 import SCHEMA, SLUG, validate_card
from mtare_topo.data.multiview_teacher_scope_v1 import compile_scope
from mtare_topo.data.multiview_teacher_reader_v1 import MultiviewTeacherReader
from mtare_topo.data.multiview_teacher_population_v1 import export_population

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'
WALL=10800
OUTPUT=30*1024**3


def old_environment():
    return dict(executable_sha256=sha(Path(OLD_PYTHON).resolve()),
        pip_freeze=subprocess.check_output([OLD_PYTHON,'-m','pip','freeze','--all'],text=True),
        python=subprocess.check_output([OLD_PYTHON,'--version'],text=True).strip())


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite/refreeze')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-09',
        scope_sha256=digest(scope),authorized_operations=['data_export'],authorized_gates=[3],
        scope='Exact2259 fit2079/cal180 windows from65 C01-C07 parents; original inputs and historicalV6 partial reference supervision; no development135 or C08-C10; no training.',
        confirmation_reference='User standing authorization to autonomously verify supervision on existing development data; PLAN explicitly permits the fixed2259 historical teacher verification. No new teacher method, negative promotion or training authorization.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=digest(scope),approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    if sha(ZIP)!=SHA:raise ValueError('historical archive drift')
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD})
    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONHASHSEED=20260906',
        PYTHON,'tools/v3/multiview_historical_teacher_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Do all fixed multiview windows yield diverse historical partial anchor targets and preserve unobserved/background gaps?',
        method='Original scans and relative motion; isolated complete historicalV6 archive and old CPU environment; recompute original raw interface diagnostic and V6 for every window; retain full raw witnesses and unknowns; no cached-source metadata rewriting.',
        baseline='Archived307 target-position distribution; not a new model ranking or complete background qualification.',
        fallback='Fail and seal on source/runtime drift, teacher error or resource cap; no retry, resampling, hidden-negative promotion or teacher replacement.',
        wall_time_cap_s=WALL,estimated_cost=dict(compute='CPU only,2259 historical raw-interface and target evaluations; no LiDAR rendering/model/optimizer',host_ram_gb=4,gpu_vram_gb=0,disk_gb=30,wall_time_hours=3),
        resource_policy=dict(parent_address_space_bytes=1024**3,worker_address_space_bytes=3*1024**3,
            wall_seconds=WALL,output_bytes=OUTPUT,request_timeout_seconds=WALL),
        acceptance_criteria=['Exact2259 fit/cal windows,195 tasks,65 parents; original source bindings; no development/test reads.',
            'All windows retained including zero-positive and unknown; full original raw and target evidence; fixed historical source archive.',
            'Report position distributions and missing-scoring-region limitations, not complete labels or method success; stop at caps.'],
        expected_evidence=['source snapshot,old archive binding,both environments,command,source read hashes,full per-window gzip evidence,position summaries,raw logs,RUN_STATE,seal'],
        historical_archive=dict(path=str(ZIP.relative_to(ROOT)),sha256=SHA),
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment(),teacher_environment=old_environment())
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    started=time.monotonic();error=None;reader=None;result=None
    def expire(*args):raise TimeoutError('10800s total wall cap')
    def check():
        if time.monotonic()-started>WALL:raise TimeoutError('total wall cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>OUTPUT:raise OSError('total output cap')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(WALL)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment'] or old_environment()!=spec['teacher_environment']:raise ValueError('runtime drift')
        if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise ValueError('parent executable mismatch')
        if spec['historical_archive']!=dict(path=str(ZIP.relative_to(ROOT)),sha256=SHA) or sha(ZIP)!=SHA:raise ValueError('archive drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec['source_sha256'].items():archive.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/bound_runtime.json',dict(parent=spec['environment'],teacher=spec['teacher_environment']),'x')
        reader=MultiviewTeacherReader(ROOT,card['scope'])
        # Spawn before lowering the parent hard limit; worker installs its own3GiB limit.
        with (run/'logs/teacher.log').open('xb') as log:
            with HistoricalTeacherClient(log,timeout_s=WALL,memory_bytes=3*1024**3) as client:
                resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
                result=export_population(run,card['scope'],reader,client,check,output_cap_bytes=OUTPUT)
        for p,h in reader.opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during run')
        write(run/'artifacts/target_manifest.json',result,'x');check()
    except BaseException:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'HISTORICAL_PARTIAL_TARGET_EXPORT_COMPLETE_NOT_QUALIFIED',
            error=error,completed_observations=len(result['observations']) if result else None,
            elapsed_s=time.monotonic()-started,parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            model_calls=0,optimizer_steps=0,scientific_gate_pass=False)
        write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {},'x')
        write(run/'metrics/summary.json',summary,'x')
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        (run/'logs/raw.log').write_text(json.dumps(summary)+'\n')
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
