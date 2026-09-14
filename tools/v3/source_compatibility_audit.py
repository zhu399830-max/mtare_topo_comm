"""Single immutable source compatibility audit; never relabel or train."""
import argparse
import json
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile
from _bootstrap import PROJECT_ROOT as ROOT
from double_frame_diagnostic_run import environment, write
from mtare_topo.data.covered_sensor_conversion import file_sha
from mtare_topo.evaluation.source_compatibility_population import audit
from mtare_topo.governance import load_json, build_run_id
from mtare_topo.governance_source_compatibility import validate_card


def execute(spec,run):
    run=run.resolve(strict=True)
    if run != ROOT/'results/gate3_semantics'/build_run_id(spec): raise ValueError('wrong run')
    if load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED': raise ValueError('no retry')
    if load_json(run/'config/run_spec.json')!=spec: raise ValueError('spec drift')
    start=time.monotonic(); metrics={}; error=None
    def verify():
        for p,h in spec['source_sha256'].items():
            if file_sha(ROOT/p)!=h: raise ValueError('source drift: '+p)
    def expired(*_): raise TimeoutError('conversion time cap')
    write(run/'RUN_STATE.json',dict(state='RUNNING'))
    signal.signal(signal.SIGALRM,expired);signal.alarm(spec['wall_time_cap_s'])
    try:
        verify(); card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'): raise ValueError('card drift')
        if environment()!=spec['environment']: raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n': raise ValueError('command drift')
        write(run/'config/environment.json',spec['environment'])
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for p in spec['source_sha256']: archive.write(ROOT/p,p)
        with (run/'logs/progress.jsonl').open('x') as log:
            def progress(row):
                row=dict(row,elapsed_s=time.monotonic()-start)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>1024**3: raise MemoryError('1GiB RAM cap')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>256*1024**2: raise RuntimeError('256MiB output cap')
            metrics=audit(ROOT,run,card['scope'],progress)
        verify()
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally: signal.alarm(0)
    summary=dict(metrics,state='AUDIT_COMPLETE_NOT_SOURCE_CERTIFICATE' if error is None else 'AUDIT_EXECUTION_FAIL',
        elapsed_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        error=error,gate_pass=False)
    write(run/'metrics/summary.json',summary);write(run/'RUN_STATE.json',summary)
    write(run/'SHA256_SEAL.json',{str(p.relative_to(run)):file_sha(p) for p in sorted(run.rglob('*')) if p.is_file()})
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(execute(load_json(args.spec),args.run_dir))
