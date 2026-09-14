"""Qualify exact opposite-route observations with unchanged frozen V8."""
import argparse
import gzip
import json
from pathlib import Path
import v8_multiview_run_v1 as original
from fit_opposite_teacher_scope_v1 import compile_scope
from request_scoped_teacher_client_v1 import RequestScopedTeacherClient

base=original.base;ROOT=base.ROOT
SLUG='gse_v8_fit_opposite141_v1';SCHEMA='v3_v8_fit_opposite141_card_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if set(card)!={'schema_version','card_id','operation','scope','scope_sha256','approval'}:raise ValueError('closed opposite card required')
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):raise ValueError('opposite operation mismatch')
        expected=base.digest(compile_scope(ROOT));a=card['approval']
        if card['scope_sha256']!=expected or base.digest(card['scope'])!=expected:raise ValueError('opposite scope drift')
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=expected or a.get('authorized_operations')!=['data_export']
                or a.get('authorized_gates')!=[3] or not a.get('confirmation_reference')):raise ValueError('opposite standing authority required')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite/refreeze')
    scope=compile_scope(ROOT)
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-09',
        scope_sha256=base.digest(scope),authorized_operations=['data_export'],authorized_gates=[3],
        scope='Exact141 opposite-direction fit observations,5 C01-C06 parents,13 variant tasks; historical-route diagnostic, not fitting or threshold selection.',
        confirmation_reference='User standing autonomous permission for evidence-backed execution improvements; current PLAN permits fit_opposite141 teacher diagnosis after scan/motion verification; no automatic training qualification; standing authorization covers scoped development execution after preflight. No old run overwrite, no training/test or threshold changes.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=base.digest(scope),approval=a)
    if not validate_card(card).passed:raise ValueError('opposite card invalid')
    spec=base.load_json(ROOT/original.SPEC)
    spec.update(slug=SLUG,data_card=CARD,config_path=CARD,user_authorization=a,
        question='Does unchanged V8 provide multi-position observed junction references on the fixed141 opposite-direction observations, including approaching junctions?',
        method='Exact frozen V8 without batching or geometry overrides. Only request-local lifetime differs; input/raw/targets/response are released before the next request. Same original mesh/field, rays, uncertainty, precision and caps. Fresh process age remains explicit.',
        baseline='Original-direction141 V8 coverage: all84 positive anchors behind sensor. Opposite observations are distinct inputs, so compare visibility and position coverage, not pointwise paired accuracy.',
        fallback='Fail and seal; no further lifetime retry, raising cap, dropped rays or training. Completion is not training qualification; label quality and calibration adequacy must be evaluated separately.',
        acceptance_criteria=['Exactly141 fit observations193 variant frames5 parents; no calibration, development-evaluation or test payload.',
            'Keep all original sources in order, same frozen archive, original geometry and unknown masks.',
            'All141 finish under the original cap, no dropped observations or retries; no training or threshold calibration.'],
        expected_evidence=['141 full outputs, source hashes, logs, runtime, memory, RUN_STATE, seal'],
        environment=base.environment(),teacher_environment=base.old_environment(),
        estimated_cost=dict(compute='CPU only141 fit diagnostic observations; no sensor rerendering/model/training',host_ram_gb=4,gpu_vram_gb=0,disk_gb=30,wall_time_hours=3),
        execution_correction=dict(index=1,only_override='request_object_lifetime',batching=False,explicit_gc=False,
            original_kernel_sha256='a1e768bffee8d5b0e06dff9dca1b4c3ec0fa78799c28151a26b4196a7072f11a',
            fresh_worker_confound=True,reused_input_observations=141))
    spec['command']=['env','MALLOC_ARENA_MAX=2','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'PYTHONHASHSEED=20260906',base.PYTHON,'tools/v3/v8_fit_opposite141_run_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    base.write(ROOT/CARD,card,'x')
    files={str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    files.add(CARD);spec['source_sha256']={p:base.sha(ROOT/p) for p in sorted(files)}
    base.write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    original.compile_scope=compile_scope;original.validate_card=validate_card
    original.V8HistoricalTeacherClient=RequestScopedTeacherClient
    return original.execute(spec,run)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);args=p.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(base.load_json(args.spec),args.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')

