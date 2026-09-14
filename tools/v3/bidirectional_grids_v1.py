"""Exact missing833 original observed grids; no new semantic teacher."""
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
from _bootstrap import PROJECT_ROOT as ROOT
from bidirectional_grid_inventory_v1 import compile_inventory, identity
from development_grids_v1 import PYTHON, sha, write
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_bidirectional_grids_v1 import SCHEMA, SLUG, SCOPE_SHA, validate_card

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def environment():
    packages=subprocess.check_output([PYTHON,'-m','pip','freeze','--all'],text=True)
    return dict(python=platform.python_version(),executable_sha256=sha(Path(PYTHON).resolve()),
                pip_freeze=packages,pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest())


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no refreeze')
    scope=compile_inventory(ROOT)
    if digest(scope)!=SCOPE_SHA:raise ValueError('inventory drift')
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-09',
        scope_sha256=SCOPE_SHA,authorized_operations=['data_export'],authorized_gates=[3],
        scope='Existing860 observations across four sealed C01-C07 source cards; export833 missing grids and reuse27; input container scope unchanged; no model or new labels.',
        confirmation_reference='User standing autonomous development permission; current PLAN explicitly permits exact833 original observed-grid export after preflight; no protected test data, retry or rule change.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=SCOPE_SHA,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=20260906',
             'CUDA_VISIBLE_DEVICES=',PYTHON,'tools/v3/bidirectional_grids_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=20260906,
        operation='data_export',data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Can the exact833 missing observations supply unchanged source-bound free/occupied/unknown grids for paired partial supervision?',
        method='Original MultiviewTeacherReader and fixed0.25m CPU observed ray grid, no semantic label generation; reuse27 grids.',
        baseline='Original grid algorithm unchanged; no model performance comparison.',fallback='Fail and seal on drift, error or cap; no retry, changed labels or substituted observations.',
        wall_time_cap_s=7200,estimated_cost=dict(compute='CPU833 grids,860 source bundles; no GPU',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=2),
        acceptance_criteria=['Exactly833 missing source-bound grids; reused27 verified but not regenerated; all860 source bundles accounted for.',
            'Original grid algorithm and unknown states; no model, optimizer or semantic teacher invocation.',
            '4GiB host RSS,2GiB new output,7200s caps; immutable evidence and terminal seal.'],
        expected_evidence=['833 NPZ grids,source bindings,manifest,config,environment,logs,metrics,RUN_STATE,SHA256 seal.'],
        environment=environment(),source_sha256={p:sha(ROOT/p) for p in files})
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    from mtare_topo.data.multiview_teacher_reader_v1 import MultiviewTeacherReader
    from mtare_topo.data.development_grid_export import export_observation, decode_grid
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    run=run.resolve(strict=True)
    if (run!=ROOT/RUN or build_run_id(spec)!=run.name or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    started=time.monotonic();entries=[];seen=set();opened={};error=None;active=None;reader=None
    def expire(*args):raise TimeoutError('7200s cap')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(7200)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json') or digest(compile_inventory(ROOT))!=SCOPE_SHA:
            raise ValueError('scope/card drift')
        if environment()!=spec['environment'] or str(Path(sys.executable).resolve())!=str(Path(PYTHON).resolve()):raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',spec['environment'])
        scope=card['scope'];rows={identity(r['source']):r for r in scope['observations']}
        output=run/'artifacts/grids';output.mkdir()
        with (run/'logs/observations.jsonl').open('x') as log:
            for item in scope['source_cards']:
                p=item['path'];h=item['sha256'];source_card=json.loads(read_pinned(ROOT,p,h));opened[p]=h
                def compiler(root, p=p, h=h):return json.loads(read_pinned(root,p,h))['scope']
                reader=MultiviewTeacherReader(ROOT,source_card['scope'],scope_compiler=compiler)
                for task in sorted(reader.entries):
                    bundles=reader.read_task(task)
                    for bundle in bundles:
                        active=bundle['source'];key=identity(active)
                        if key in seen or key not in rows:raise ValueError('unexpected or duplicate source')
                        row=rows[key]
                        if row['source_card']!=p or row['split']!=active['split']:raise ValueError('source card/split mismatch')
                        binding=dict(source=row['source'],construction_sha256=canonical_sha(bundle['construction_teacher_only']))
                        old=row['reused_grid']
                        if old:
                            payload=read_pinned(ROOT,old['path'],old['sha256']);opened[old['path']]=old['sha256']
                            if binding!=old['binding']:raise ValueError('reused binding drift')
                            decode_grid(payload,expected_sha256=old['sha256'],expected_binding=binding)
                        else:
                            entry=export_observation(bundle,expected_binding=binding,output_dir=output)
                            entry['split']=row['split'];entries.append(entry)
                            log.write(json.dumps(entry)+'\n');log.flush()
                            print(json.dumps(dict(completed=len(entries),total=833,elapsed_s=time.monotonic()-started)),flush=True)
                        seen.add(key)
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:raise MemoryError('4GiB RSS cap')
                        if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>2*1024**3:raise RuntimeError('2GiB output cap')
                    del bundle,bundles
                opened.update(reader.opened);reader=None
        if seen!=set(rows) or len(entries)!=833:raise ValueError('incomplete population')
        for p,h in opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during export')
        write(run/'artifacts/grid_manifest.json',dict(status='COMPLETE',observations=entries,reused_observations=27,whole_region_complete=False),'x')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if reader is not None:opened.update(reader.opened)
        summary=dict(status='FAILED' if error else 'OBSERVED_GRID_EXPORT_COMPLETE',error=error,active_source=active,
            completed_observations=len(entries),seen_observations=len(seen),elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_steps=0,new_semantic_labels=0,scientific_gate_pass=False)
        write(run/'metrics/summary.json',summary);write(run/'artifacts/source_reads_sha256.json',opened)
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
