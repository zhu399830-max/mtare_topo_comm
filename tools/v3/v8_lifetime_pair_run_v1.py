"""Single exact-pair request-lifetime diagnostic; no geometry or batching override."""
import argparse
import gzip
import json
from pathlib import Path
import v8_multiview_run_v1 as original
from v8_batch_pair_scope_v1 import compile_scope
from request_scoped_teacher_client_v1 import RequestScopedTeacherClient

base=original.base;ROOT=base.ROOT
SLUG='gse_v8_lifetime_pair_v1';SCHEMA='v3_v8_lifetime_pair_card_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if set(card)!={'schema_version','card_id','operation','scope','scope_sha256','approval'}:raise ValueError('closed pair card required')
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):raise ValueError('pair operation mismatch')
        expected=base.digest(compile_scope(ROOT));a=card['approval']
        if card['scope_sha256']!=expected or base.digest(card['scope'])!=expected:raise ValueError('pair scope drift')
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=expected or a.get('authorized_operations')!=['data_export']
                or a.get('authorized_gates')!=[3] or not a.get('confirmation_reference')):raise ValueError('pair standing authority required')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite/refreeze')
    scope=compile_scope(ROOT)
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-09',
        scope_sha256=base.digest(scope),authorized_operations=['data_export'],authorized_gates=[3],
        scope='Two exact fit observations192653/192654, one S10_C04 mixed traversal; request-lifetime diagnostic only.',
        confirmation_reference='User standing autonomous permission for evidence-backed execution improvements; current PLAN permits preparing the exact pair request-lifetime diagnostic after16 software checks; standing authorization covers scoped development execution after preflight. No retry of old141 run, no training/test or threshold changes.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,scope_sha256=base.digest(scope),approval=a)
    if not validate_card(card).passed:raise ValueError('pair card invalid')
    spec=base.load_json(ROOT/original.SPEC)
    spec.update(slug=SLUG,data_card=CARD,config_path=CARD,user_authorization=a,
        question='Does releasing each completed request before the next calculation resolve the failed location under the unchanged cap?',
        method='Exact frozen V8 without batching or geometry overrides. Only request-local lifetime differs; input/raw/targets/response are released before the next request. Same original mesh/field, rays, uncertainty, precision and caps. Fresh process age remains explicit.',
        baseline='Full raw and produced_targets of immediately preceding sealed observation192653; failure192654 has no complete output.',
        fallback='Fail and seal; no further lifetime retry, raising cap, dropped rays or training. Passing pair does not complete141.',
        acceptance_criteria=['Exactly source192653 then192654;6 original frames, one fit parent; no calibration/test.',
            'First full raw and targets equal sealed predecessor; any difference fails before accepting evidence.',
            'Second finishes under same cap; preserve all witnesses. No claim of full141 completion or teacher qualification; process age is a confound.'],
        expected_evidence=['both full outputs, exact preceding comparison, source hashes, logs, runtime, memory, RUN_STATE, seal'],
        environment=base.environment(),teacher_environment=base.old_environment(),
        estimated_cost=dict(compute='CPU only two diagnostic observations; no rendering/model/training',host_ram_gb=4,gpu_vram_gb=0,disk_gb=30,wall_time_hours=3),
        execution_correction=dict(index=1,only_override='request_object_lifetime',batching=False,explicit_gc=False,
            original_kernel_sha256='a1e768bffee8d5b0e06dff9dca1b4c3ec0fa78799c28151a26b4196a7072f11a',
            fresh_worker_confound=True,completed_reference=scope['completed_reference']))
    spec['command']=['env','MALLOC_ARENA_MAX=2','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
        'PYTHONHASHSEED=20260906',base.PYTHON,'tools/v3/v8_lifetime_pair_run_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    base.write(ROOT/CARD,card,'x')
    files={str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    files.add(CARD);spec['source_sha256']={p:base.sha(ROOT/p) for p in sorted(files)}
    base.write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],executed=False)))


def execute(spec,run):
    pin=spec['execution_correction']['completed_reference']
    prior=json.loads(gzip.decompress(base.read_pinned(ROOT,pin['path'],pin['sha256'])))
    class ComparingClient(RequestScopedTeacherClient):
        completed=0
        def request(self,bundle):
            response=super().request(bundle)
            if self.completed==0:
                for k in ('raw_interfaces','produced_targets'):
                    if response[k]!=prior[k]:raise ValueError('lifetime adapter changed preceding complete output: '+k)
                base.write(run/'artifacts/preceding_comparison.json',dict(exact_equal=True,reference=pin),'x')
            self.completed+=1
            return response
    original.compile_scope=compile_scope;original.validate_card=validate_card
    original.V8HistoricalTeacherClient=ComparingClient
    return original.execute(spec,run)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);args=p.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(base.load_json(args.spec),args.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')

