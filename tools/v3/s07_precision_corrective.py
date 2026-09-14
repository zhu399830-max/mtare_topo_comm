"""One original-case corrective diagnostic using the existing sealed executor."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import v8_original_ten_probe as runner
from mtare_topo.data.gse_s07_precision_scope import compile_scope,reader
from mtare_topo.governance_s07_precision import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.governance_surface_selection import digest

runner.SCHEMA=SCHEMA;runner.SLUG=SLUG;runner.POLICY=POLICY;runner.validate_card=validate_card
runner.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
runner.SPEC='configs/v3/gate3/'+SLUG+'.json'
runner.RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed20260906'
runner.ENTRYPOINT='tools/v3/s07_precision_corrective.py'
runner.COMPLETION_STATUS='SINGLE_S07_CORRECTIVE_DIAGNOSTIC_COMPLETE_NOT_QUALIFICATION'
runner.make_reader=lambda root,card:reader(root,card['scope'])


def prior(root,card,source_reader,old):
    pin=card['scope']['baseline']
    return json.loads(gzip.decompress(source_reader._read(pin['path'],pin['sha256'])))['produced_targets']


def produce(bundle,raw,card):
    from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets
    return produce_joint_reference_targets(bundle,raw,qualify_cap_precision=True,**card['scope']['geometry_settings'])


def compare(old,new):
    a,b=old['record'],new['record']
    if old['source_binding']!=new['source_binding']:raise ValueError('source binding changed')
    for name in ('openings','score_region','source_frame_indices','coordinate_frame'):
        if a[name]!=b[name]:raise ValueError('unexpected '+name+' change')
    changes=[];used=set()
    for i,anchor in enumerate(a['anchors']):
        matches=[j for j,x in enumerate(b['anchors']) if x['position_m']==anchor['position_m']]
        if len(matches)>1:raise ValueError('duplicate anchor match')
        j=matches[0] if matches else None
        if j is not None:used.add(j)
        changes.append(dict(old_anchor=i,new_anchor=j,position_m=anchor['position_m'],
            status='WITHDRAWN' if j is None else 'RETAINED',old_memberships=[r[i] for r in a['membership']],
            new_memberships=None if j is None else [r[j] for r in b['membership']]))
    if len(used)!=len(b['anchors']):raise ValueError('unexpected new anchor in fixed corrective case')
    return dict(old_anchors=len(a['anchors']),new_anchors=len(b['anchors']),changes=changes,
        old_counts=runner.counts(a),new_counts=runner.counts(b),complete_annotation=False,
        expected_withdrawal_observed=any(x['status']=='WITHDRAWN' for x in changes))


runner.prior_output=prior;runner.produce_output=produce;runner.compare=compare


def freeze():
    root=runner.PROJECT_ROOT
    if (root/runner.CARD).exists() or (root/runner.SPEC).exists():raise FileExistsError('no refreeze')
    if Path(sys.executable).resolve()!=Path(runner.PYTHON).resolve():raise ValueError('exact execution environment required')
    scope=compile_scope(root);scope_sha=digest(scope)
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',
        scope='Only original S07 97000 five-frame corrective diagnostic; no training or population expansion.',
        authorized_operations=['data_export'],authorized_gates=[3],scope_sha256=scope_sha,
        confirmation_reference='User continuously authorizes evidence-backed autonomous corrections; PLAN 2026-09-08 authorizes exactly original S07 97000 five-frame diagnostic. No human review, training, or population approval asserted.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=scope_sha,policy=POLICY,approval=approval)
    if not validate_card(card).passed:raise ValueError('invalid card')
    command=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONHASHSEED=20260906',runner.PYTHON,runner.ENTRYPOINT,'--spec',str(root/runner.SPEC),'--run-dir',str(root/runner.RUN)]
    files=sorted(str(p.relative_to(root)) for folder in ('src/mtare_topo','tools/v3') for p in (root/folder).rglob('*.py'))
    pins={runner.ORIGINAL_RUNNER:runner.ORIGINAL_SHA,runner.STORAGE_RUNNER:runner.STORAGE_SHA,runner.ORIGINAL_SPEC:runner.ORIGINAL_SPEC_SHA}
    for p,h in pins.items():runner.read_pinned(root,p,h)
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=SLUG,seed=20260906,
        operation='data_export',data_card=runner.CARD,config_path=runner.CARD,user_authorization=approval,command=command,
        question='Does source-precision qualification withdraw the archived unsupported S07 anchor, and why?',
        method='Original five scans and source geometry settings; V8 explicit entering/leaving source interval qualification; exact duplicate sample correction; partial references only.',
        baseline='Sealed failed V1R S07 target, not a valid truth label. Compare anchor positions and all membership changes; old known answers may be withdrawn with preserved evidence.',
        fallback='Seal even if withdrawal absent; inspect actual alternate evidence; no repeat, resampling, expansion, training or scientific PASS.',
        estimated_cost=dict(compute='CPU only; one observation/five frames/one parent; original container collateral in card',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=.5),
        wall_time_cap_s=1800,
        acceptance_criteria=['Exact original source hashes, pose, frame and scope binding.',
            'Preserve opening geometry and score-region contract; log every anchor and membership difference.',
            'No scientific qualification inferred; expected withdrawal reported as a separate boolean.',
            'One immutable run with complete output, errors, hashes and zero training.'],
        expected_evidence=['Original source pins and snapshot, old/new target comparison, precision evidence, runtime, logs, RUN_STATE and SHA256 seal.'],
        source_sha256={p:runner.sha(root/p) for p in files},environment=runner.environment(),original_source_tools=pins)
    for p,value in ((runner.CARD,card),(runner.SPEC,spec)):
        with (root/p).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(spec=runner.SPEC,counts=scope['counts'],payload_files=len(scope['file_sha256']),metadata_files=len(scope['metadata_reads_sha256']),scope_sha256=scope_sha)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(runner.execute(runner.load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
