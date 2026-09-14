"""Bounded causal retrieval from all saved native tokens; no model or control."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import os
import platform
import resource
import time
import traceback
import zipfile

from mtare_topo.governance_native_structure_capture import scope_digest, validate_retrieval_card

SOURCE = 'results/gate6_single_robot/gate6_20260913_gse_explicit_source_capture_v1r1_seed11'
SEAL = 'ff41069885bfa0a6ab372795da5428ab4cbc709209424853b7047e85cf55c3bb'
NAME = 'gse_native_token_retrieval_v1'
RUN = 'results/gate6_single_robot/gate6_20260913_' + NAME + '_seed11'
CARD = 'configs/v3/gate6/data_cards/' + NAME + '.json'
SPEC = 'configs/v3/gate6/' + NAME + '.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, mode='x'):
    with path.open(mode) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def freeze():
    if any((ROOT / p).exists() for p in (CARD, SPEC, RUN)):
        raise ValueError('fresh paths required')
    scope = dict(source_run=SOURCE, world='tunnel', seed=11, trajectories=1, raw_frames=617,
        selected_raw_frames=575, windows=115, independent_units='one development episode, not115independent structures',
        sampling='all native epochs with exact5raw sources; actual times bound in source; no score selection',
        split='historically used development tunnel; no train/test or checkpoint selection',
        training_steps=0, model_forwards=0, new_simulations=0, teacher=None, protected_worlds_read=[],
        operation='saved_token_causal_retrieval_only', threshold_selection=False, control_changes=False)
    approval = dict(status='APPROVED', approved_by='user', approved_at='2026-09-13',
        authorized_operations=['audit'], authorized_gates=[6], scope_sha256=scope_digest(scope),
        scope='Existing authorized native-input task integration; all saved tokens to strictly past retrieval; no training/control',
        confirmation_reference='User 继续推进 and active goal connecting frozen geometry features to native task correspondence; standing same-data integration authorization')
    card = dict(schema_version='gse_native_token_retrieval_card_v1', card_id=NAME, scope=scope, approval=approval)
    if not validate_retrieval_card(card).passed:
        raise ValueError('card scope mismatch')
    write(ROOT / CARD, card)
    sources = [__file__, 'tools/v3/_bootstrap.py', 'src/mtare_topo/integration/native_token_archive.py',
        'src/mtare_topo/integration/structural_token_retrieval.py',
        'src/mtare_topo/governance_native_structure_capture.py', 'src/mtare_topo/governance.py', CARD]
    sources = [str((ROOT / p).relative_to(ROOT)) for p in sources]
    write(ROOT / SPEC, dict(schema_version='v3_run_spec_v1', gate=6, date='20260913', slug=NAME, seed=11,
        operation='audit', data_card=CARD, user_authorization=approval, command=['/home/zeng-workstation/anaconda3/bin/python',
            'tools/v3/run_native_token_retrieval.py', '--execute'],
        question='Do authenticated saved local tokens produce causal historical candidates for every native epoch?',
        method='Existing symmetric local-token retrieval, top5, fixed10m/256view, all115epochs, no global pooling',
        baseline='identity/shadow original controller retained; no claim of comparator accuracy',
        fallback='no-support explicitly unknown; candidates never become task identity or completion automatically',
        estimated_cost=dict(wall_time_hours=1/6, disk_gb=.05, compute='CPU only,8GiB limit,600s budget'),
        acceptance_criteria=['115epochs retained in causal order', 'all input/source/pose hashes exact',
            'no future candidate, zero training/control and no verified identity claim'],
        expected_evidence=['all per-epoch ranked candidates, source refs, gaps/unknowns, raw log, summary, source snapshot and seal'],
        source_sha256={p:sha(ROOT / p) for p in sources},
        input_sha256={SOURCE+'/artifacts/evidence_sha256.txt':SEAL}))
    print(SPEC)


def execute():
    from mtare_topo.integration.native_token_archive import NativeTokenArchive
    from mtare_topo.integration.structural_token_retrieval import retrieve_structural_candidates
    import numpy as np
    out = ROOT / RUN
    spec = json.loads((ROOT / SPEC).read_text())
    if json.loads((out/'RUN_STATE.json').read_text())['state'] != 'CREATED_NOT_EXECUTED':
        raise ValueError('run already used')
    if json.loads((out/'config/run_spec.json').read_text()) != spec:
        raise ValueError('spec drift')
    for path, digest in {**spec['source_sha256'], **spec['input_sha256']}.items():
        if sha(ROOT/path) != digest:
            raise ValueError('frozen source drift')
    if not validate_retrieval_card(json.loads((ROOT/CARD).read_text())).passed:
        raise ValueError('scope mismatch')
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in spec['source_sha256']:
            archive.write(ROOT/p,p)
    write(out/'config/execution_environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        omp_threads=os.environ.get('OMP_NUM_THREADS'),gpu_used=False))
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    start=time.monotonic();error=None;history=[];rows=[]
    try:
        archive=NativeTokenArchive(ROOT,SOURCE,seal_sha256=SEAL)
        with (out/'artifacts/retrieval.jsonl').open('x') as f, (out/'logs/retrieval.log').open('x') as log:
            for current, meta in archive.records():
                if time.monotonic()-start > 600:
                    raise RuntimeError('wall budget exceeded')
                result=retrieve_structural_candidates(current,history)
                result.update(meta)
                result['order']=current.order
                result['source_refs']=list(current.source_refs)
                rows.append(result)
                f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
                log.write(f"epoch={current.record_id} past={len(history)} candidates={len(result['candidates'])}\n");log.flush()
                history.append(current)
        if len(rows)!=115:
            raise ValueError('missing native epochs')
    except Exception:
        error=traceback.format_exc()
    gaps=[r['candidates'][0]['score']-r['candidates'][1]['score'] for r in rows if len(r['candidates'])>1]
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,epochs=len(rows),
        total_candidates=sum(len(r['candidates']) for r in rows),
        epochs_with_candidates=sum(bool(r['candidates']) for r in rows),
        candidate_score_gaps=dict(minimum=min(gaps) if gaps else None, median=float(np.median(gaps)) if gaps else None),
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        model_forwards=0,training_steps=0,confirmed_associations=0,control_changes=0,
        real_revisit_accuracy=None,method_advantage_proven=False,
        limitation='retrieval is not identity; no registration, directional correspondence or execution completion inferred')
    write(out/'metrics/summary.json',summary)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':
                f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true'); group.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze()
    else: raise SystemExit(execute())
