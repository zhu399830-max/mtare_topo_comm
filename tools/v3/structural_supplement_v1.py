#!/usr/bin/env python3
"""Freeze and execute one bounded supplementary position selection."""
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
from mtare_topo.governance_structural_supplement_v1 import SCHEMA, SLUG, POLICY, validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_structural_supplement_scope_v1 import compile_supplement_scope
from surface_coverage_v1 import sha, write, PYTHON

CARD = 'configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC = 'configs/v3/gate3/'+SLUG+'.json'


def freeze():
    root = PROJECT_ROOT
    if (root/CARD).exists() or (root/SPEC).exists():
        raise FileExistsError('no refreeze')
    scope = compile_supplement_scope(root)
    approval = dict(status='APPROVED', approved_by='user-standing-supplementary-authorization',
        approved_at='2026-09-07', authorized_operations=['data_export'], authorized_gates=[3],
        scope_sha256=digest(scope), scope='C01-C07 original pose-only supplementary selection; no labels or scans',
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
        'MKL_NUM_THREADS=1','PYTHONHASHSEED=20260906',PYTHON,'tools/v3/structural_supplement_v1.py',
        '--spec',str(root/SPEC),'--run-dir',str(root/run)]
    packages = subprocess.check_output([PYTHON,'-m','pip','freeze','--all'],text=True)
    spec = dict(schema_version='v3_run_spec_v1',gate=3,date='20260907',slug=SLUG,seed=20260906,
        operation='data_export',data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Which paired causal observations cover all926 nominated development structure entities spatially?',
        method='One hash-fixed incident traversal per entity; minimize maximum original3D pose-to-anchor distance across three variants; retain far/missing candidates.',
        baseline='Original3360 background retained unchanged in its existing seal; this separate supplement contains no negative or positive labels.',
        fallback='Fail and seal on drift or resource violation; no retry, alternate direction, scope or threshold adjustment.',
        wall_time_cap_s=1800,estimated_cost=dict(compute='CPU only,32676 candidate pose rows,480 exact files; no scans',
            host_ram_gb=4,gpu_vram_gb=0,disk_gb=.5,wall_time_hours=.5),
        acceptance_criteria=['Exact70 parents/210 tasks/926 nominations, original split and paired source histories.',
            'All selected original pose hashes verified; no scans/models/labels/C08-C10; far and missing preserved.',
            'Immutable source snapshot, selected frames, distances, counts and SHA256 seal; no research PASS.'],
        expected_evidence=['Per-parent selections, source histories/distances, far/missing counts, raw logs, environment, command, metrics, RUN_STATE and SHA256 seal.'],
        source_sha256={p:sha(root/p) for p in files},environment=dict(python=platform.python_version(),
            executable_sha256=sha(Path(PYTHON).resolve()),pip_freeze=packages,
            pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest()))
    with (root/SPEC).open('x') as f:
        json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    from mtare_topo.data.gse_structural_supplement_reader_v1 import SupplementPoseReader
    root=PROJECT_ROOT;run=run.resolve(strict=True)
    if (run != root/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state'] != 'CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json') != spec):
        raise ValueError('fresh exact created run required')
    started=time.monotonic(); error=None; reader=None; outputs=[]
    def expire(signum,frame):
        raise TimeoutError('1800s limit')
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
        reader=SupplementPoseReader(root,card['scope'])
        with (run/'logs/parents.jsonl').open('x') as log:
            for p in card['scope']['parents']:
                out=reader.read_parent(p['parent_id'])
                write(run/'artifacts'/(p['parent_id']+'.json'),out)
                outputs.append(out)
                entry=dict(parent_id=p['parent_id'],completed=len(outputs),elapsed_s=time.monotonic()-started)
                log.write(json.dumps(entry)+'\n');log.flush(); print(json.dumps(entry),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes']:
                    raise MemoryError('4GiB host limit')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:
                    raise RuntimeError('output limit')
        if set(reader.opened)!=set(card['scope']['file_sha256']):
            raise ValueError('incomplete exact pose access')
        for p,h in reader.opened.items():
            if sha(root/p)!=h:
                raise ValueError('pose changed during run')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        selections=[s for out in outputs for s in out['selections']]
        rows=[s for selection in selections for s in selection['sources']]
        summary=dict(status='FAILED' if error else 'SUPPLEMENTARY_SELECTION_COMPLETE',error=error,
            completed_parents=len(outputs),selected_entities=len(selections),observation_requests=len(rows),
            unique_observations=len({(r['task'],r['source_sequence_id']) for r in rows}),
            unique_variant_frames=len({(r['task'],f) for r in rows for f in r['frame_rows']}),
            entities_all_variants_within_10m=sum(s['all_variants_within_10m'] for s in selections),
            missing=sum(len(o['missing']) for o in outputs),elapsed_s=time.monotonic()-started,
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
