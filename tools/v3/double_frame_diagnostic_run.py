"""One immutable synthetic software diagnostic, not a sensor dataset export."""
import argparse
from collections import Counter
import hashlib
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
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.governance import build_run_id, load_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {'python': sys.version, 'executable': sys.executable,
            'executable_sha256': sha(sys.executable),
            'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(spec, run):
    run = run.resolve(strict=True)
    if run != ROOT/'results/gate3_semantics'/build_run_id(spec):
        raise ValueError('wrong run directory')
    if load_json(run/'RUN_STATE.json')['state'] != 'CREATED_NOT_EXECUTED':
        raise ValueError('run already consumed')
    if load_json(run/'config/run_spec.json') != spec:
        raise ValueError('spec drift')
    counts = Counter(); failures = 0; checked = 0; max_error = 0.; error = None
    start = time.monotonic()
    def expired(*_):
        raise TimeoutError('fixed diagnostic wall time cap')
    def verify():
        for p, digest in spec['source_sha256'].items():
            if sha(ROOT/p) != digest:
                raise ValueError('source/input drift: '+p)
    write(run/'RUN_STATE.json', {'state': 'RUNNING', 'run_id': run.name})
    signal.signal(signal.SIGALRM, expired); signal.alarm(spec['wall_time_cap_s'])
    try:
        verify()
        if environment() != spec['environment']:
            raise ValueError('environment drift')
        if (run/'config/command.txt').read_text() != shlex.join(spec['command'])+'\n':
            raise ValueError('command drift')
        write(run/'config/environment.json', spec['environment'])
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for p in spec['source_sha256']:
                archive.write(ROOT/p, p)
        from mtare_topo.evaluation.double_frame_diagnostic import frame_inputs, score_return
        from mtare_topo.data.gse_synthetic_matrix import construction_document
        from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
        from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
        from mtare_topo.teacher.exact_reference_diagnostic import ExactReferenceDiagnostic
        case, origin, directions, expected, manifest = frame_inputs()
        if manifest != load_json(ROOT/spec['config_path']):
            raise ValueError('exact frame input drift')
        _, primitives = load_p1a_realized_construction(construction_document(case))
        meshes = [mesh_swept_superellipse(p, axial_spacing_m=.05, angular_segments=64) for p in primitives]
        diagnostic = ExactReferenceDiagnostic(meshes)
        with (run/'metrics/rays.jsonl').open('x') as stream:
            for index, direction in enumerate(directions):
                result = diagnostic.query(origin, direction, maximum_m=50.)
                score = score_return(result, expected[index])
                counts[result['status']] += 1; checked += 1
                failures += not score['passed']
                if score['error_m'] is not None:
                    max_error = max(max_error, score['error_m'])
                stream.write(json.dumps({'ray_index': index, 'direction': direction.tolist(),
                    'expected_m': float(expected[index]), 'result': result, 'score': score}, allow_nan=False)+'\n')
                stream.flush()
                if checked % 128 == 0:
                    print(json.dumps({'checked': checked, 'failures': failures, 'counts': dict(counts),
                                      'elapsed_s': time.monotonic()-start}), flush=True)
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3:
                        raise MemoryError('4GiB RSS cap')
                    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) > 1024**3:
                        raise RuntimeError('1GiB evidence cap')
                if not score['passed']:
                    raise RuntimeError('first unsafe/unknown numerical diagnostic at ray '+str(index))
        verify()
    except Exception:
        error = traceback.format_exc()
        (run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
    passed = error is None and checked == 11520 and failures == 0
    summary = {'state': 'DIAGNOSTIC_PASS' if passed else 'DIAGNOSTIC_FAIL',
               'checked_rays': checked, 'planned_rays': 11520, 'failures': failures,
               'counts': dict(counts), 'max_error_m': max_error if checked else None,
               'elapsed_s': time.monotonic()-start,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'labels_exported': 0, 'optimizer_steps': 0, 'gate_pass': False, 'error': error}
    write(run/'metrics/summary.json', summary)
    write(run/'RUN_STATE.json', summary)
    seal = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write(run/'SHA256_SEAL.json', seal)
    print(json.dumps(summary), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(execute(load_json(args.spec), args.run_dir))
