#!/usr/bin/env python3
"""Freeze and export only2676 missing six-field causal sensor observations."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
import zipfile

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_supplement_input_v1 import SCHEMA, SLUG, POLICY, validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_supplement_input_scope_v1 import compile_missing_input_scope
from surface_coverage_v1 import sha, write, PYTHON

CARD = 'configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC = 'configs/v3/gate3/'+SLUG+'.json'


def freeze():
    root = PROJECT_ROOT
    if (root/CARD).exists() or (root/SPEC).exists():
        raise FileExistsError('no refreeze')
    scope = compile_missing_input_scope(root)
    approval = dict(status='APPROVED', approved_by='user-standing-supplementary-authorization',
        approved_at='2026-09-07', authorized_operations=['data_export'], authorized_gates=[3],
        scope_sha256=digest(scope), scope='C01-C07 missing2676 six-field sensor/history/motion export; no labels or training',
        confirmation_reference='User requested 那你想办法呀 then 继续执行 after the same-world supplementary sampling proposal; PLAN records bounded amendment and standing autonomous execution. No protected worlds or training authorized here.')
    card = dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,
                scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report = validate_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    with (root/CARD).open('x') as f:
        json.dump(card,f,ensure_ascii=False,indent=2)
    files = sorted({str(p.relative_to(root)) for folder in ('src/mtare_topo','tools/v3')
                    for p in (root/folder).rglob('*.py')} | {CARD})
    run = 'results/gate3_semantics/gate3_20260907_'+SLUG+'_seed20260906'
    command = ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','PYTHONHASHSEED=20260906',PYTHON,'tools/v3/supplement_input_v1.py',
        '--spec',str(root/SPEC),'--run-dir',str(root/run)]
    packages = subprocess.check_output([PYTHON,'-m','pip','freeze','--all'],text=True)
    spec = dict(schema_version='v3_run_spec_v1',gate=3,date='20260907',slug=SLUG,seed=20260906,
        operation='data_export',data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Can the missing2676 observations be exported losslessly without re-exporting original3360 or leaking teacher targets?',
        method='Frozen supplementary source rows; six-field lossless export with exact chunk hashes, causal history verification and original relative motion; task-wise release.',
        baseline='Original3360 six-field inputs retained unchanged and reused; source arrays are numerical reference, not model baseline scores.',
        fallback='Fail and seal on drift or resource violation; no retry, alternate direction, scope or threshold adjustment.',
        wall_time_cap_s=3600,estimated_cost=dict(compute='CPU only,2676 missing observations,14589 exact files;3.55GB cumulative chunk decode',
            host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=1),
        acceptance_criteria=['Exact2676 missing observations and13374 unique variant frames;210 tasks;original split and history.',
            'Exact source hashes and six-field numerical values; no absolute pose,models,labels,C08-C10 or training.',
            'Immutable source snapshot, selected frames, distances, counts and SHA256 seal; no research PASS.'],
        expected_evidence=['Per-task NPZ six-field arrays, selection/input manifest, source reads, logs, environment, metrics, RUN_STATE and SHA256 seal.'],
        source_sha256={p:sha(root/p) for p in files},environment=dict(python=platform.python_version(),
            executable_sha256=sha(Path(PYTHON).resolve()),pip_freeze=packages,
            pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest()))
    with (root/SPEC).open('x') as f:
        json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    from mtare_topo.data.gse_surface_input_export_v1 import SurfaceInputReader
    import numpy as np
    root=PROJECT_ROOT;run=run.resolve(strict=True)
    if (run != root/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state'] != 'CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json') != spec):
        raise ValueError('fresh exact created run required')
    started=time.monotonic(); error=None; reader=None; outputs=[]
    def expire(signum,frame):
        raise TimeoutError('3600s limit')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(POLICY['wall_time_s'])
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(root/spec['data_card'])
        if not validate_card(card).passed or card != load_json(run/'config/data_card.json'):
            raise ValueError('card drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec['source_sha256'].items():
                archive.writestr(p,read_pinned(root,p,h))
        env=spec['environment']; packages=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
        if (platform.python_version()!=env['python'] or sha(Path(sys.executable).resolve())!=env['executable_sha256']
                or hashlib.sha256(packages.encode()).hexdigest()!=env['pip_freeze_sha256']):
            raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':
            raise ValueError('command drift')
        write(run/'config/environment.json',env)
        scope=card['scope']
        if digest(compile_missing_input_scope(root))!=digest(scope):
            raise ValueError('compiled scope drift')
        reader=SurfaceInputReader(root,task_sources=scope['task_sources'],
            selection=scope['population']['new_observations'],sealed_keys=scope['file_sha256'],
            variable_population=True)
        (run/'artifacts/inputs').mkdir()
        with (run/'logs/tasks.jsonl').open('x') as log:
            for task in sorted(scope['task_sources']):
                data=reader.read_task(task)
                path=run/'artifacts/inputs'/(task+'.npz')
                with path.open('xb') as f:
                    np.savez_compressed(f,ranges_m=data.ranges_m,valid_mask=data.valid_mask,
                        relative_translation_current_sensor_m=data.relative_translation_current_sensor_m,
                        relative_yaw_current_sensor_deg=data.relative_yaw_current_sensor_deg,
                        frame_rows=data.frame_rows,source_sequence_ids=data.source_sequence_ids)
                entry=dict(task=task,path=str(path.relative_to(run)),sha256=sha(path),report=data.read_report)
                outputs.append(entry)
                log.write(json.dumps(entry)+'\n');log.flush()
                print(json.dumps(dict(completed=len(outputs),task=task,elapsed_s=time.monotonic()-started)),flush=True)
                del data
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes']:
                    raise MemoryError('4GiB host limit')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:
                    raise RuntimeError('output limit')
        if set(reader.opened)!=set(scope['file_sha256']):
            raise ValueError('incomplete exact six-field access')
        for p,h in reader.opened.items():
            if sha(root/p)!=h:
                raise ValueError('source changed during run')
        write(run/'artifacts/input_manifest.json',dict(schema_version='gse_supplement_input_manifest_v1',
            observations=scope['population']['new_observations'],task_shards=outputs,
            combined_requests=scope['population']['combined_requests'],labels=0))
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'MISSING_INPUT_EXPORT_COMPLETE',error=error,
            completed_tasks=len(outputs),observations=sum(o['report']['selected_observations'] for o in outputs),
            unique_variant_frames=sum(o['report']['selected_unique_sensor_frames'] for o in outputs),
            elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,labels=0,optimizer_steps=0,
            scientific_gate_pass=False)
        write(run/'metrics/summary.json',summary)
        write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
