"""Single bounded860 authenticated reader probe; no model or new labels."""
import argparse
import json
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile
from collections import Counter
from _bootstrap import PROJECT_ROOT as ROOT
from bidirectional_grids_v1 import environment, PYTHON, sha, write
from bidirectional_paired_scope_v1 import compile_scope, identity
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_bidirectional_reader_probe_v1 import SCHEMA, SLUG, SCOPE_SHA, validate_card

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no refreeze')
    scope=compile_scope(ROOT)
    if digest(scope)!=SCOPE_SHA:raise ValueError('scope drift')
    authority=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-09',
        scope_sha256=SCOPE_SHA,authorized_operations=['data_export'],authorized_gates=[3],
        scope='Exact860 existing paired C01-C07 observations282fit578calibration; stream authenticated input/target/grid/partitions; compact diagnostic records only; no model or semantic teacher.',
        confirmation_reference='User standing autonomous development authorization and PLAN exact860 read-only adapter verification before paired training. No protected worlds, new targets, model inference or optimizer.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=SCOPE_SHA,approval=authority)
    if not validate_card(card).passed:raise ValueError('invalid card')
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=20260906','CUDA_VISIBLE_DEVICES=',
             PYTHON,'tools/v3/bidirectional_reader_probe_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=20260906,operation='data_export',
        data_card=CARD,config_path=CARD,user_authorization=authority,command=command,
        question='Can all860 source-bound observations use unchanged partial target adaptation and three paired representations under bounded streamed memory?',
        method='Read existing frozen V8 records, authenticate and adapt with original branch target/grid binding; drop raw diagnostics before yielding; no model execution.',
        baseline='Original authenticated loss-side adapter; software probe not a method comparison.',fallback='Fail and seal; no retry, label change, sample deletion or threshold change.',
        wall_time_cap_s=3600,estimated_cost=dict(compute='CPU860 existing observations read once; zero GPU/model',host_ram_gb=4,gpu_vram_gb=0,disk_gb=1,wall_time_hours=1),
        acceptance_criteria=['Exactly860 unique sources282fit578calibration; each has r0/r1/r2, source-bound grid and authenticated partial targets.',
            'All raw teacher records released per observation; finite detached targets, unknown not negative; no complete detection claim.',
            '4GiB peak RSS,1GiB new evidence,3600s wall cap; source hashes and immutable terminal seal.'],
        expected_evidence=['Per-observation target counts,retained grid bytes,elapsed/RSS,source reads,config,logs,summary,RUN_STATE,seal.'],
        environment=environment(),source_sha256={p:sha(ROOT/p) for p in files})
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    from bidirectional_paired_reader_v1 import BidirectionalPairedReader
    import torch
    run=run.resolve(strict=True)
    if (run!=ROOT/RUN or build_run_id(spec)!=run.name or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    start=time.monotonic();reader=None;iterator=None;seen=set();splits=Counter();counts=Counter();error=None
    def expire(*args):raise TimeoutError('3600s wall cap')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(3600)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',spec['environment'])
        reader=BidirectionalPairedReader(ROOT,card['scope']);iterator=reader.observations()
        with (run/'logs/observations.jsonl').open('x') as log:
            for obs in iterator:
                key=identity(obs.source)
                if key in seen or set(obs.student_representations)!={'r0','r1','r2'}:raise ValueError('duplicate/missing representation')
                loss=obs.loss_only;t=loss['target']
                if (set(loss['produced_targets'])!={'record','source_binding','target_record_sha256'}
                        or t.anchors_complete or loss['full_detection_eligible'] or loss['aperture_supervised']):
                    raise ValueError('partial payload contract drift')
                for tensor in (t.position_m,*t.directions):
                    if tensor.requires_grad or not torch.isfinite(tensor).all():raise ValueError('finite detached targets required')
                seen.add(key);splits[obs.split]+=1;counts['anchors']+=len(t.position_m);counts['branches']+=loss['observed_branch_count']
                counts['observations_without_anchor']+=int(len(t.position_m)==0)
                entry=dict(source=obs.source,split=obs.split,anchors=len(t.position_m),branches=loss['observed_branch_count'],
                    reference_branches=loss['reference_branch_count'],branches_complete=list(t.branches_complete),
                    target_record_sha256=loss['frozen_manifest']['target_record_sha256'],
                    grid_bytes=sum(getattr(loss['grid'],n).nbytes for n in ('state','free_frame_bits','occupied_frame_bits')),
                    elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                log.write(json.dumps(entry)+'\n');log.flush()
                if entry['peak_rss_bytes']>4*1024**3:raise MemoryError('4GiB RSS cap')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>1024**3:raise RuntimeError('1GiB output cap')
                if len(seen)%10==0 or len(seen)==1:print(json.dumps(dict(completed=len(seen),total=860,elapsed_s=entry['elapsed_s'])),flush=True)
                del obs,loss,t,tensor
        if len(seen)!=860 or splits!=dict(fit=282,calibration=578):raise ValueError('population incomplete')
        for p,h in reader.opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during probe')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        if iterator is not None:iterator.close()
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'PAIRED_READER_PROBE_COMPLETE',error=error,observations=len(seen),splits=dict(splits),counts=dict(counts),
            elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            optimizer_steps=0,model_calls=0,scientific_gate_pass=False)
        write(run/'metrics/summary.json',summary);write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
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
