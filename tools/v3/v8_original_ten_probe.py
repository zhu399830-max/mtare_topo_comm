#!/usr/bin/env python3
"""Freeze and execute exactly one original-ten partial supervision probe."""
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
from mtare_topo.governance_v8_probe import SCHEMA, SLUG, SCOPE_SHA, POLICY, validate_card
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_v8_probe_scope import compile_scope

PYTHON = '/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python'
CARD = 'configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC = 'configs/v3/gate3/'+SLUG+'.json'
RUN = 'results/gate3_semantics/gate3_20260908_'+SLUG+'_seed20260906'
ENTRYPOINT = 'tools/v3/v8_original_ten_probe.py'
COMPLETION_STATUS = 'ORIGINAL_TEN_V8_PARTIAL_DIAGNOSTIC_COMPLETE'
ORIGINAL_RUNNER = 'tools/v3/run_primitive_relation_p1a_sensor_provenance_export_v1.py'
ORIGINAL_SHA = '33c33d2db0b7e52cf3a35dfe431754b018de30162053fdae896baaf05f0d4049'
STORAGE_RUNNER = 'tools/v3/run_primitive_relation_p1a_lossless_storage_corrective_v1.py'
STORAGE_SHA = '9e7fc00f2435456a924b9e772bf8cfd3f37eb7403202fa66a5fe68b60d96e4ac'
ORIGINAL_SPEC = 'results/gate3_semantics/gate3_20260830_primitive_relation_p1a_sensor_provenance_export_v1_seed0/config/run_spec.json'
ORIGINAL_SPEC_SHA = 'edb9c93994a18471d33a1b3e747d23a19aff194c18d9caf3f30d8b9b5bbb63d4'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')


def environment():
    packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
    return dict(python=platform.python_version(), executable_sha256=sha(Path(sys.executable).resolve()),
        pip_freeze=packages, pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest())


def freeze(*, spec_overrides=None):
    root = PROJECT_ROOT
    if (root/CARD).exists() or (root/SPEC).exists():
        raise FileExistsError('no refreeze')
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('freeze in the exact execution environment')
    read_pinned(root, ORIGINAL_RUNNER, ORIGINAL_SHA)
    read_pinned(root, STORAGE_RUNNER, STORAGE_SHA)
    read_pinned(root, ORIGINAL_SPEC, ORIGINAL_SPEC_SHA)
    scope = compile_scope(root)
    approval = dict(status='APPROVED', approved_by='user-standing-scope-authorization',
        approved_at='2026-09-08', authorized_operations=['data_export'], authorized_gates=[3],
        scope_sha256=SCOPE_SHA, scope='Original ten failed fit observations; source-bound V8 partial targets and comparison only.',
        confirmation_reference='User requests autonomous execution and permits evidence-backed efficiency improvements without repeated approval; docs/PLAN.md 2026-09-08 explicitly authorizes original-ten V8 probe. No manual annotation signature or training approval asserted.')
    card = dict(schema_version=SCHEMA, card_id=SLUG, operation='data_export', scope=scope,
        scope_sha256=SCOPE_SHA, policy=POLICY, approval=approval)
    report = validate_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    files = sorted(str(p.relative_to(root)) for folder in ('src/mtare_topo','tools/v3')
                   for p in (root/folder).rglob('*.py'))
    command = ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','PYTHONHASHSEED=20260906',PYTHON,ENTRYPOINT,
        '--spec',str(root/SPEC),'--run-dir',str(root/RUN)]
    spec = dict(schema_version='v3_run_spec_v1', gate=3, date='20260908', slug=SLUG,
        seed=20260906, operation='data_export', data_card=CARD, config_path=CARD,
        user_authorization=approval, command=command,
        question='Does source-bound V8 recover observed structural membership and nonmembership on the original ten failures without corrupting old supported targets?',
        method='Original mesh .05m/64 and field .025m; original cached first returns, causal motion and raw interfaces; complete operand surface intersections only for evidence; V8 partial anchors/openings/relations; no full annotation promotion.',
        baseline='Sealed V6 targets from the same ten observations, preserving old known memberships by exact anchor/opening geometry matching.',
        fallback='Seal failures, retain unknown and original assets; no retry, population expansion or training; separate implementation failure from research nonqualification.',
        estimated_cost=dict(compute='CPU only; ten five-frame observations /44 unique variant frames /3 parents; NPZ containers103 observations',
            host_ram_gb=4, gpu_vram_gb=0, disk_gb=2, wall_time_hours=3),
        wall_time_cap_s=POLICY['wall_time_s'],
        acceptance_criteria=['All exact source hashes, identities, original float precision and motion binding hold.',
            'No first-return rerender, C08-C10, training or full label qualification.',
            'All old supported targets and score regions preserved; differences and unknowns explicitly counted.',
            'Resource caps enforced and immutable failure evidence retained. Diagnostic completion alone is not scientific PASS.'],
        expected_evidence=['Source snapshot, original parameter pins, input reads, per-observation partial targets, prior comparison, raw logs, runtime, memory, summary, RUN_STATE and SHA256 seal.'],
        source_sha256={p:sha(root/p) for p in files}, environment=environment(),
        original_source_tools={ORIGINAL_RUNNER:ORIGINAL_SHA, STORAGE_RUNNER:STORAGE_SHA,
                               ORIGINAL_SPEC:ORIGINAL_SPEC_SHA})
    # Prepare everything before writing either immutable contract.
    if spec_overrides:
        allowed={'question','method','baseline','acceptance_criteria','fallback'}
        if not set(spec_overrides)<=allowed:raise ValueError('only research descriptions may be overridden')
        spec.update(spec_overrides)
    for p, value in ((CARD,card),(SPEC,spec)):
        with (root/p).open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(spec=SPEC, counts=scope['counts'], executed=False)))


def compare(old, new):
    a, b = old['record'], new['record']
    if old['source_binding'] != new['source_binding']:
        raise ValueError('source binding changed')
    for name in ('openings','score_region','source_frame_indices'):
        if a[name] != b[name]:
            raise ValueError('unexpected '+name+' change')
    changes = []
    for i, anchor in enumerate(a['anchors']):
        matches = [j for j, value in enumerate(b['anchors']) if value == anchor]
        if len(matches) != 1:
            raise ValueError('old anchor lost/changed/duplicated')
        j = matches[0]
        for oi, row in enumerate(a['membership']):
            before, after = row[i], b['membership'][oi][j]
            if before is not None and after is not before:
                raise ValueError('old known membership changed')
            if before is not after:
                changes.append(dict(opening=oi, old_anchor=i, new_anchor=j, before=before, after=after))
    return dict(old_anchors=len(a['anchors']), new_anchors=len(b['anchors']), membership_changes=changes,
        old_counts=counts(a), new_counts=counts(b), complete_annotation=False)


def counts(record):
    values = [v for row in record['membership'] for v in row]
    return dict(positive=sum(v is True for v in values), negative=sum(v is False for v in values),
                unknown=sum(v is None for v in values))


def make_reader(root,card):
    from mtare_topo.data.gse_v8_probe_reader import V8ProbeReader
    return V8ProbeReader(root,card['scope'])


def produce_output(bundle,raw,card):
    from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets
    return produce_joint_reference_targets(bundle,raw,**card['scope']['geometry_settings'])


def prior_output(root,card,reader,old):
    return old


def execute(spec, run):
    from mtare_topo.data.gse_v8_probe_reader import V8ProbeReader
    from mtare_topo.data.gse_lossless_evidence_v1 import write_evidence
    from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets
    root=PROJECT_ROOT; run=run.resolve(strict=True)
    if (run != root/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state'] != 'CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json') != spec):
        raise ValueError('fresh exact run required')
    started=time.monotonic(); error=None; completed=[]; reader=None; active=None
    def expire(*args):
        raise TimeoutError('diagnostic runaway protection')
    signal.signal(signal.SIGALRM, expire); signal.alarm(POLICY['wall_time_s'])
    write(run/'RUN_STATE.json', dict(state='RUNNING', run_id=run.name))
    try:
        card=load_json(root/spec['data_card'])
        if not validate_card(card).passed or card != load_json(run/'config/data_card.json'):
            raise ValueError('data card drift')
        if environment() != spec['environment']:
            raise ValueError('environment drift')
        if (run/'config/command.txt').read_text() != shlex.join(spec['command'])+'\n':
            raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec['source_sha256'].items():
                archive.writestr(p,read_pinned(root,p,h))
        for p,h in spec['original_source_tools'].items():
            read_pinned(root,p,h)
        write(run/'config/environment.json', spec['environment'])
        reader=make_reader(root,card)
        with (run/'logs/observations.jsonl').open('x') as log:
            for row in card['scope']['entries']:
                active=row['source']; write(run/'artifacts/active_observation.json',active)
                print(json.dumps(dict(starting=active['task'], sequence=active['source_sequence_id'], elapsed_s=time.monotonic()-started)),flush=True)
                bundle,raw,old=reader.read_observation(active['task'],active['source_sequence_id'])
                old=prior_output(root,card,reader,old)
                output=produce_output(bundle,raw,card)
                name=active['task']+'_'+str(active['source_sequence_id'])+'.json.gz'
                # Preserve the actual candidate even if the comparison fails.
                write_evidence(run/'artifacts'/('candidate_'+name),dict(produced_targets=output,
                    raw_interfaces_reference=row['references'][-1]))
                comparison=compare(old,output)
                storage=write_evidence(run/'artifacts'/name,dict(produced_targets=output,
                    prior_comparison=comparison,raw_interfaces_reference=row['references'][-1]))
                item=dict(source=active, comparison=comparison, storage=storage, elapsed_s=time.monotonic()-started)
                completed.append(item); log.write(json.dumps(item)+'\n'); log.flush()
                print(json.dumps(dict(completed=len(completed),counts=comparison['new_counts'],elapsed_s=item['elapsed_s'])),flush=True)
                del output,bundle,raw,old
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > POLICY['host_ram_bytes']:
                    raise MemoryError('4GiB host resource cap')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) > POLICY['output_bytes']:
                    raise RuntimeError('2GiB output cap')
        for p,h in reader.opened.items():
            if sha(root/p) != h:
                raise ValueError('input drift during run')
    except Exception:
        error=traceback.format_exc(); (run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
        summary=dict(status='FAILED' if error else COMPLETION_STATUS,
            error=error,active_source=active,completed_observations=len(completed),observations=completed,
            elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            formal_optimizer_steps=0,full_label_qualification=False,scientific_gate_pass=False)
        write(run/'metrics/summary.json',summary)
        write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(dict(status=summary['status'],completed=len(completed),error=error)),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:
        freeze()
    elif args.spec and args.run_dir:
        raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:
        parser.error('--freeze or --spec/--run-dir required')
