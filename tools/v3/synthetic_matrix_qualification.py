"""Freeze once, preflight/create externally, stream one synthetic matrix."""
import argparse
import json
from pathlib import Path
import resource
import shlex
import signal
import sys
import time
import traceback
import zipfile
from _bootstrap import PROJECT_ROOT as ROOT
from v8_original_ten_probe import PYTHON,sha,environment,write
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_synthetic_matrix import SCHEMA,SLUG,POLICY,scope,validate_card
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed20260906'
ENTRY='tools/v3/synthetic_matrix_qualification.py'


def freeze():
    if (ROOT/CARD).exists() or (ROOT/SPEC).exists():raise FileExistsError('no refreeze')
    if Path(sys.executable).resolve()!=Path(PYTHON).resolve():raise ValueError('exact environment required')
    data=scope();authority=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',
        authorized_operations=['data_export'],authorized_gates=[3],scope_sha256=digest(data),
        scope='Fixed 144 synthetic primary observations and 12 same-observation controls, no real data or training.',
        confirmation_reference='User standing autonomous execution authority; PLAN and GSE synthetic matrix documents specify these software qualification conditions.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=data,scope_sha256=digest(data),policy=POLICY,approval=authority)
    if not validate_card(card).passed:raise ValueError('invalid synthetic card')
    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONHASHSEED=20260906',PYTHON,ENTRY,'--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    files=sorted(str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py'))
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=authority,command=command,
        question='Which declared synthetic conditions recover correct partial geometry and preserve unknown/hidden-map invariance?',
        method='Shared finite CSG scene per type/section; V8 explicit source precision; fixed-observation hidden controls; independent limited prototype scoring.',
        baseline='Analytic convex straight and central-star geometry for 45 conditions; other 99 explicitly unscored, not pass.',
        fallback='Preserve all fixed-case semantic failures without adapting cases; stop implementation/resource errors; no repeat or real training.',
        estimated_cost=dict(compute='CPU only; 144 primary five-frame observations, 12 fixed-observation controls, 36 cached scenes',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=3),
        wall_time_cap_s=10800,acceptance_criteria=['Exact fixed scope/source/environment; no real data reads or training.',
            'Covered fixtures count missed/extra anchors and openings; all-unknown cannot pass; uncovered fields remain unqualified.',
            'Hidden control requires recorded-segment exclusion, unchanged student arrays and matching local targets.',
            'Complete case evidence, logs, resource counts and seal; software completion never promotes full label qualification.'],
        expected_evidence=['144 input NPZs and complete construction/raw/target records, 12 controls, per-condition scores, logs, environment, source snapshot, RUN_STATE and seal.'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    for p,value in ((CARD,card),(SPEC,spec)):
        with (ROOT/p).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(spec=SPEC,primary=144,controls=12,rendered_frames=720,covered=45,unscored=99)))


def execute(spec,run,*,validator=None,execute_fn=None):
    run=run.resolve(strict=True)
    if run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED':
        raise ValueError('fresh exact run required')
    if load_json(run/'config/run_spec.json')!=spec:raise ValueError('spec drift')
    start=time.monotonic();summary={};error=None
    def expired(*args):raise TimeoutError('matrix resource protection deadline')
    def progress(row):
        row=dict(row,elapsed_s=time.monotonic()-start);write(run/'metrics/progress.json',row)
        print(json.dumps(row),flush=True)
    def resources():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes']:raise MemoryError('4GiB RSS cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:raise RuntimeError('2GiB evidence cap')
    signal.signal(signal.SIGALRM,expired);signal.alarm(POLICY['wall_time_s'])
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not (validator or validate_card)(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        write(run/'config/environment.json',spec['environment'])
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec['source_sha256'].items():archive.writestr(p,read_pinned(ROOT,p,h))
        from mtare_topo.data.gse_synthetic_matrix_execution import execute_matrix
        summary=(execute_fn or execute_matrix)(run,progress=progress,resource_check=resources)
        for p,h in spec['source_sha256'].items():read_pinned(ROOT,p,h)
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
        summary.update(status='FAILED' if error else 'MATRIX_DIAGNOSTIC_COMPLETE_NOT_FULL_QUALIFICATION',error=error,
            elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            formal_optimizer_steps=0,full_matrix_qualification=False)
        write(run/'metrics/summary.json',summary)
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(dict(status=summary['status'],error=error)),flush=True)
    return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path)
    args=p.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
