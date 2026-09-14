"""Single CPU export of30 paired voxel/native-SPG partitions."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile
import numpy as np
import libcp
import libply_c
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_continuous_partitions_v1 import SCHEMA,SLUG,validate_card
from mtare_topo.data.continuous_partition_scope_v1 import compile_scope
from mtare_topo.data.development_partition_handoff import produce_partitions,decode_partitions
from mtare_topo.representation.gse_spg_runtime import sha,runtime_snapshot

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUNTIME='configs/v3/environments/'+SLUG+'.json'
PREFIX='build/gse_spg_cpu_v1'


def write(path,value,mode='w'):
    with Path(path).open(mode) as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n')


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUNTIME)):raise FileExistsError('no refreeze')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-09',
                  authorized_operations=['data_export'],authorized_gates=[3],scope_sha256=digest(scope),
                  scope='30 fixed continuous C04 compact caches,fit only;42 unique variant frames;CPU partitions only.',
                  confirmation_reference='User standing autonomous authorization; PLAN exact continuous30 frozen feature population and original native partitions; no training or new labels.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=digest(scope),approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    runtime=runtime_snapshot(ROOT)
    write(ROOT/CARD,card,'x');write(ROOT/RUNTIME,runtime,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,RUNTIME})
    run='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONHASHSEED=0',
             'PYTHONPATH='+PREFIX+':'+PREFIX+'/prefix/usr/lib/python3/dist-packages',
             'LD_LIBRARY_PATH='+PREFIX+'/prefix/usr/lib/x86_64-linux-gnu','/usr/bin/python3',
             'tools/v3/continuous_partitions_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/run)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=0,operation='data_export',
              data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
              question='Can all30 fixed observations provide identical-point voxel/SPG inputs for frozen-weight continuous prediction diagnostics?',
              method=scope['method'],baseline='Shared scanning-layout mapping from identical compact cache; no performance comparison at export.',fallback=scope['fallback'],
              wall_time_cap_s=3600,estimated_cost=dict(compute='CPU30 paired partitions; no GPU',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=1),
              acceptance_criteria=['Exactly30 original source identities and full ROI ray mappings; no point deletion or missing representation substitution.',
                                   'Finite bounded partitions,30 immutable packets,complete source hashes and seal; no training or semantic success claim.'],
              expected_evidence=['Paired NPZ packets,per-observation counts/runtime,source snapshot and reads,environment,command,raw log,summary,RUN_STATE,seal.'],
              source_sha256={p:sha(ROOT/p) for p in files},environment_manifest=RUNTIME,environment_sha256=sha(ROOT/RUNTIME))
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    start=time.monotonic();entries=[];opened={};error=None;active=None
    def expire(*args):raise TimeoutError('3600s limit')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(3600)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        runtime=load_json(ROOT/spec['environment_manifest'])
        if sha(ROOT/spec['environment_manifest'])!=spec['environment_sha256'] or runtime_snapshot(ROOT)!=runtime:raise ValueError('runtime drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/partition_runtime.json',runtime)
        output=run/'artifacts/partitions';output.mkdir()
        with (run/'logs/observations.jsonl').open('x') as log:
            for row in card['scope']['observations']:
                active=row['source'];tick=time.monotonic()
                raw=read_pinned(ROOT,row['cache_path'],row['cache_sha256']);opened[row['cache_path']]=row['cache_sha256']
                payload=produce_partitions(raw,source=active,cache_sha256=row['cache_sha256'],
                    geof_backend=libply_c.compute_geof,partition_backend=libcp.cutpursuit)
                # Independent original-cache mapping comparison, not nearest-point lookup.
                with np.load(io.BytesIO(raw),allow_pickle=False) as a:
                    xyz=a['points_xyz_m'][0];valid=a['valid'][0]
                indices=np.flatnonzero(valid & (np.linalg.norm(xyz.astype(float),axis=1)<=10.));points=xyz[indices]
                h=hashlib.sha256(payload).hexdigest()
                assignments=decode_partitions(payload,expected_sha256=h,expected_source=active,
                    expected_cache_sha256=row['cache_sha256'],points_xyz_m=points,source_flat_ray_index=indices)
                name=digest(active)+'.npz'
                with (output/name).open('xb') as f:f.write(payload)
                entry=dict(source=active,split=row['split'],file=name,sha256=h,cache_sha256=row['cache_sha256'],
                           points=len(points),groups={k:len(np.unique(v)) for k,v in assignments.items()},elapsed_s=time.monotonic()-tick)
                entries.append(entry);log.write(json.dumps(entry)+'\n');log.flush()
                print(json.dumps(dict(completed=len(entries),total=30,groups=entry['groups'],elapsed_s=time.monotonic()-start)),flush=True)
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>2*1024**3:raise RuntimeError('output cap')
        if len(entries)!=30:raise ValueError('population incomplete')
        for p,h in opened.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift during run')
        write(run/'artifacts/partition_manifest.json',dict(status='COMPLETE',observations=entries),'x')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'PARTITION_EXPORT_COMPLETE',error=error,active_source=active,
                     observations=len(entries),elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     optimizer_steps=0,scientific_gate_pass=False)
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
