#!/usr/bin/env python3
"""Freeze or execute the single307 existing-observation grid export."""
import argparse
import base64
import gzip
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

from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_development_grids_v1 import SCHEMA, SLUG, SCOPE_SHA, validate_card
from mtare_topo.governance_surface_material import read_pinned

PYTHON = '/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'
CARD = 'configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC = 'configs/v3/gate3/'+SLUG+'.json'
PREPARATION = 'configs/v3/gate3/development_reference_grid_preparation_v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, mode='w'):
    with Path(path).open(mode) as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def freeze():
    if (ROOT/CARD).exists() or (ROOT/SPEC).exists():
        raise FileExistsError('no refreeze')
    scope = load_json(ROOT/PREPARATION)
    # Metadata-only freeze: original data payloads are not accessed here.
    for key in ('source_card', 'selection_binding'):
        item = scope[key]; read_pinned(ROOT, item['path'], item['sha256'])
    approval = dict(status='APPROVED', approved_by='user-standing-scope-authorization', approved_at='2026-09-09',
        authorized_operations=['data_export'], authorized_gates=[3], scope_sha256=SCOPE_SHA,
        confirmation_reference='User requests autonomous in-scope C01-C07 execution; exact selected307 existing observations, no training, no new semantic labels, no protected data.',
        scope='156 tasks decoded2184 observations; only307 selected observations create grids;1877 collateral excluded;1529 unique selected variant frames; fixed CPU ray evidence algorithm.')
    card = dict(schema_version=SCHEMA, card_id=SLUG, operation='data_export', scope=scope,
                scope_sha256=SCOPE_SHA, approval=approval)
    report = validate_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    write(ROOT/CARD, card, 'x')
    files = sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3')
                    for p in (ROOT/folder).rglob('*.py')} |
                   {CARD, PREPARATION, scope['source_card']['path'], scope['selection_binding']['path']})
    run = 'results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'
    command = ['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
               'PYTHONHASHSEED=20260906','CUDA_VISIBLE_DEVICES=',PYTHON,
               'tools/v3/development_grids_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/run)]
    packages = subprocess.check_output([PYTHON,'-m','pip','freeze','--all'], text=True)
    spec = dict(schema_version='v3_run_spec_v1', gate=3, date='20260909', slug=SLUG, seed=20260906,
        operation='data_export', data_card=CARD, config_path=CARD, user_authorization=approval, command=command,
        question='Can all307 selected existing observations yield lossless source-bound observed grids for shared partial-supervision training?',
        method='Reuse original CPU projection and fixed0.25m five-ray-frame grid; exact source binding; unknown remains unknown; per-observation exclusive NPZ export.',
        baseline='Existing source-bound grid algorithm unchanged; no model comparison in this export.',
        fallback='Fail and seal on any input drift, missing/duplicate selected observation or resource breach; no retry or substitution.',
        wall_time_cap_s=7200,
        estimated_cost=dict(compute='CPU only;156 tasks,2184 decoded observations,307 grid targets',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=2),
        acceptance_criteria=['Exactly307 grids from1529 unique variant frames and156 tasks; all original hashes and source bindings valid.',
                             'Unknown preserved; no semantic label promotion, optimizer steps, or C08-C10 payload.',
                             'Exclusive artifacts, source manifest, hashes, raw progress log and terminal seal;7200s/4GiB host/2GiB output caps.'],
        expected_evidence=['307 source-bound NPZ grids and manifest, source reads, command/environment, raw logs, metrics, RUN_STATE and SHA256 seal.'],
        source_sha256={p:sha(ROOT/p) for p in files},
        environment=dict(python=platform.python_version(),executable_sha256=sha(Path(PYTHON).resolve()),
                         pip_freeze=packages,pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest()))
    write(ROOT/SPEC, spec, 'x')
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec, run):
    from mtare_topo.data.gse_supplement_teacher_reader_v1 import SupplementTeacherReader
    from mtare_topo.data.development_grid_export import export_observation
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    run = run.resolve(strict=True)
    if (run != ROOT/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state'] != 'CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json') != spec):
        raise ValueError('fresh exact created run required')
    started=time.monotonic(); entries=[]; reader=None; error=None; active=None; decoded=0
    def expire(*args):
        raise TimeoutError('7200s export limit')
    previous=signal.signal(signal.SIGALRM,expire); signal.alarm(7200)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card != load_json(run/'config/data_card.json'):
            raise ValueError('card drift')
        scope=card['scope']; caps=scope['resource_caps']
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path,h in spec['source_sha256'].items():
                archive.writestr(path,read_pinned(ROOT,path,h))
        env=spec['environment']; packages=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
        if (platform.python_version()!=env['python'] or sha(Path(sys.executable).resolve())!=env['executable_sha256']
                or hashlib.sha256(packages.encode()).hexdigest()!=env['pip_freeze_sha256']):
            raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':
            raise ValueError('command drift')
        write(run/'config/environment.json',env)
        selection=scope['selection_binding']
        wrapper=json.loads(read_pinned(ROOT,selection['path'],selection['sha256']))
        raw=gzip.decompress(base64.b64decode(wrapper['payload'],validate=True))
        if hashlib.sha256(raw).hexdigest()!=selection['decoded_sha256']:
            raise ValueError('selection drift')
        selected=json.loads(raw); rows=selected['rows']
        keys=[(r['source']['task'],r['source']['source_sequence_id']) for r in rows]
        if len(rows)!=307 or len(set(keys))!=307 or sorted({k[0] for k in keys})!=scope['selected_tasks']:
            raise ValueError('population mismatch')
        selected_by_key=dict(zip(keys,rows)); seen=set()
        item=scope['source_card']; original=json.loads(read_pinned(ROOT,item['path'],item['sha256']))
        reader=SupplementTeacherReader(ROOT,original['scope'])
        output=run/'artifacts/grids'; output.mkdir()
        with (run/'logs/observations.jsonl').open('x') as log:
            for task in scope['selected_tasks']:
                active=task; bundles=reader.read_task(task); decoded+=len(bundles)
                for bundle in bundles:
                    source=bundle['source']; key=(task,source['source_sequence_id'])
                    if key not in selected_by_key:
                        continue
                    row=selected_by_key[key]
                    actual={k:source[k] for k in ('task','source_sequence_id','frame_rows')}
                    if key in seen or actual!=row['source']:
                        raise ValueError('duplicate or changed selected observation')
                    path=row['construction_path']
                    doc=json.loads(read_pinned(ROOT,path,selected['file_sha256'][path]))
                    reader.opened[path]=selected['file_sha256'][path]
                    binding=dict(source=actual,construction_sha256=canonical_sha(doc))
                    entry=export_observation(bundle,expected_binding=binding,output_dir=output)
                    entry['split']=row['split']; entries.append(entry); seen.add(key)
                    log.write(json.dumps(entry)+'\n'); log.flush()
                    print(json.dumps(dict(completed=len(entries),total=307,task=task,elapsed_s=time.monotonic()-started)),flush=True)
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>caps['host_ram_bytes']:
                        raise MemoryError('4GiB host cap')
                    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>caps['output_bytes']:
                        raise RuntimeError('2GiB output cap')
                del bundle,bundles
        if seen!=set(keys) or decoded!=2184:
            raise ValueError('incomplete population or collateral count drift')
        for path,h in reader.opened.items():
            if sha(ROOT/path)!=h:
                raise ValueError('source changed during export')
        write(run/'artifacts/grid_manifest.json',dict(status='COMPLETE',observations=entries,whole_region_complete=False),'x')
    except Exception:
        error=traceback.format_exc(); (run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'OBSERVED_GRID_EXPORT_COMPLETE',error=error,active_task=active,
                     completed_observations=len(entries),decoded_observations=decoded,elapsed_s=time.monotonic()-started,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     optimizer_steps=0,new_semantic_labels=0,scientific_gate_pass=False)
        write(run/'metrics/summary.json',summary)
        write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze',action='store_true'); parser.add_argument('--spec',type=Path); parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze: freeze()
    elif args.spec and args.run_dir: raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else: parser.error('--freeze or --spec/--run-dir required')
