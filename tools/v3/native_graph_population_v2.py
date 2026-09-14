"""Execute one preflight-created, immutable-scope native candidate graph export."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile

from mtare_topo.governance import load_json, build_run_id
from mtare_topo.governance_native_graph_v2 import validate_card
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.representation.gse_spg_runtime import sha
from mtare_topo.evaluation.native_graph_population_v2 import export_population


def write(path, value, mode='w'):
    with Path(path).open(mode) as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def execute(spec, run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):
        raise ValueError('fresh exact preflight-created run required')
    started=time.monotonic(); error=None; manifest=None
    def expire(*args): raise TimeoutError('population wall cap')
    previous=signal.signal(signal.SIGALRM,expire)
    signal.alarm(int(spec['wall_time_cap_s']))
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name,pid=os.getpid()))
    try:
        card=load_json(ROOT/spec['data_card'])
        report=validate_card(card)
        if not report.passed or card!=load_json(run/'config/data_card.json'):
            raise ValueError(f'card drift: {report.errors}')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':
            raise ValueError('command drift')
        runtime=load_json(ROOT/spec['environment_manifest'])
        if sha(ROOT/spec['environment_manifest'])!=spec['environment_sha256']:
            raise ValueError('runtime manifest drift')
        for path, expected in runtime['files'].items():
            if sha(ROOT/path)!=expected: raise ValueError(f'runtime file drift: {path}')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for path, expected in spec['source_sha256'].items():
                z.writestr(path,read_pinned(ROOT,path,expected))
        # Preserve create_run's host snapshot; bound native runtime is separate.
        write(run/'config/native_runtime.json',runtime,'x')
        resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            GLOG_logtostderr='1', LD_LIBRARY_PATH=str(ROOT/runtime['native_library_path']))
        manifest=export_population(ROOT,run,card['scope'],
            command=[str(ROOT/runtime['native_executable'])],environment=env,
            wall_cap_s=spec['wall_time_cap_s'])
        for path, expected in runtime['files'].items():
            if sha(ROOT/path)!=expected: raise ValueError(f'runtime drift during run: {path}')
    except BaseException:
        error=traceback.format_exc()
        with (run/'logs/error.log').open('x') as f: f.write(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        progress=run/'logs/native_observations.jsonl'
        completed=0
        if progress.exists():
            for line in progress.read_text().splitlines():
                if json.loads(line)['event']=='COMPLETE': completed+=1
        summary=dict(status='FAILED' if error else 'CANDIDATE_GEOMETRY_EXPORT_COMPLETE',
            observations=completed,elapsed_s=time.monotonic()-started,error=error,
            parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            optimizer_steps=0,semantic_graph_pass=False,continuous_safety_proof=False)
        write(run/'metrics/summary.json',summary,'x')
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        with seal.open('x') as f:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p!=seal: f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args()
    raise SystemExit(execute(load_json(args.spec),args.run_dir))

