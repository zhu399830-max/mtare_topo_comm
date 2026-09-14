"""Immutable141 mechanism diagnostic, reusing sealed V8 and existing executor."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import multiview_historical_teacher_v1 as base
from v8_multiview_reader_scope_v1 import compile_scope
from v8_multiview_manifest_v1 import compile_manifest
from v8_historical_teacher_client_v1 import V8HistoricalTeacherClient
from v8_multiview_population_v1 import export_population

ROOT=base.ROOT
SLUG='gse_v8_multiview_mechanism_v1'
SCHEMA='v3_v8_multiview_mechanism_card_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if set(card)!={'schema_version','card_id','operation','scope','scope_sha256','approval'}:
            raise ValueError('closed141 card required')
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):
            raise ValueError('exact V8 diagnostic card required')
        expected=base.digest(compile_scope(ROOT))
        if card['scope_sha256']!=expected or base.digest(card['scope'])!=expected:
            raise ValueError('141 scope drift')
        a=card['approval']
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=expected
                or a.get('authorized_operations')!=['data_export'] or a.get('authorized_gates')!=[3]
                or not a.get('confirmation_reference')):raise ValueError('exact standing authority required')
    except Exception as error:errors.append(str(error))
    return ValidationReport(not errors,tuple(errors))


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite/refreeze')
    scope=compile_scope(ROOT);manifest=compile_manifest()
    authority=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-09',
        scope_sha256=base.digest(scope),authorized_operations=['data_export'],authorized_gates=[3],
        scope='Exact141 fit-only observations/5 C01-C06 parents; sealedV8 mechanism diagnostic, no training/calibration/test.',
        confirmation_reference='User standing authorization: autonomously verify supervision, choose evidence-backed alternatives and continue without routine approval. Current PLAN limits this to141 complete-window observations, preserves old evidence and forbids training. This is diagnostic authorization, not training approval.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=base.digest(scope),approval=authority)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    # Reuse complete resource/environment/evidence contract, not its population or method.
    spec=base.load_json(ROOT/base.SPEC)
    spec.update(slug=SLUG,data_card=CARD,config_path=CARD,user_authorization=authority,
        question='Does sealed source-precision V8 retain observed junction evidence across all141 fixed windows where historicalV6 attrition was diagnosed?',
        method='Original scans, frozen V8 complete archive, original .05/64 mesh and .025 field, qualify_cap_precision=True. No rerender; full raw diagnostics and partial targets retained. Selected mechanism cohort, not unbiased evaluation.',
        baseline='SealedV6 results for the same141 identities; counts/unknown and position transitions, not complete F1.',
        fallback='Fail and seal on drift, error or resource cap; no retries, dropped windows, teacher mutation or training.',
        historical_archive=dict(path=manifest['archive_path'],sha256=manifest['archive_sha256']),
        environment=base.environment(),teacher_environment=base.old_environment(),
        acceptance_criteria=['Exactly141 complete windows/13 tasks/5 fit parents; no calibration/test payload.',
            'Original geometry settings, V8 archive and source binding; all unknown and zero-target windows retained.',
            'Report per-window evidence/anchor attrition, no complete-label or model qualification claim.'],
        estimated_cost=dict(compute='CPU only141 V8 evaluations; no model or rendering',host_ram_gb=4,gpu_vram_gb=0,disk_gb=30,wall_time_hours=3))
    spec['command']=['env','MALLOC_ARENA_MAX=2','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'PYTHONHASHSEED=20260906',base.PYTHON,'tools/v3/v8_multiview_run_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    base.write(ROOT/CARD,card,'x')
    files={str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    files.update([CARD,'configs/v3/gate3/v8_multiview_mechanism_manifest_v1.json'])
    spec['source_sha256']={p:base.sha(ROOT/p) for p in sorted(files)}
    base.write(ROOT/SPEC,spec,'x')
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    if os.environ.get('MALLOC_ARENA_MAX')!='2':raise ValueError('bound allocator required')
    manifest=compile_manifest()
    base.ZIP=ROOT/manifest['archive_path'];base.SHA=manifest['archive_sha256']
    base.validate_card=validate_card
    base.MultiviewTeacherReader=partial(base.MultiviewTeacherReader,scope_compiler=compile_scope)
    base.HistoricalTeacherClient=V8HistoricalTeacherClient
    base.export_population=export_population
    return base.execute(spec,run)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);args=p.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(base.load_json(args.spec),args.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
