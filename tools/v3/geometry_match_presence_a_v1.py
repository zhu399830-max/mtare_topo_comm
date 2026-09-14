"""Restore repaired presence loss only; reuse original B, no decoder or position rerun."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import grouping_center_fit_v1 as executor
from mtare_topo.governance_geometry_presence_v1 import SCHEMA,SLUG,POLICY,PARENT,PARENT_SEAL,AUTH,compile_scope,validate_card,index
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.geometry_match_presence_v1 import center_objective
from mtare_topo.representation.development_paired_training import batch_schedule,state_sha256
from mtare_topo.evaluation.grouping_localization_ceiling_v1 import localization_funnel

ROOT=executor.ROOT;CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed0'
original_evaluate=executor.evaluate


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-latest-explicit-scope',approved_at='2026-09-10',
        authorized_operations=['training'],authorized_gates=[3],scope_sha256=executor.digest(scope),confirmation_reference=AUTH,
        scope='Only original16 fit observations,1000same scheduled batches, original step0, restored presence loss but no confidence matching cost; B reused after equivalence; no decoder/position rerun, no protected data/labels.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=executor.digest(scope),policy=POLICY,approval=approval)
    if not validate_card(card).passed:raise ValueError('invalid card')
    executor.write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,*POLICY['extra_sources']})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8',executor.PYTHON,'tools/v3/geometry_match_presence_a_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=0,operation='training',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question=POLICY['question'],method=POLICY['loss'],baseline=POLICY['baseline'],
        fallback='Report actual A result; if full fit passes retain protocol for later fair validation, no more root-cause blocking. No new models/maps/branches.',
        wall_time_cap_s=43200,estimated_cost=dict(compute='one same16 A1000 fit on5090; B reused',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['Same actual16/input/query/initial/schedule and repaired loss weights; confidence matching cost off in A only.',
            'Final1000 known-domain1m precision AND recall>=.90 at unchanged.5 threshold; unknown separate; localization coverage reported separately.',
            'Keep1/2/4m and.5m scores, all predictions/checkpoints; no test/branch/graph expansion; B reuse verified.'],
        expected_evidence=['A full detection and localization every100; B equivalence; gradients; same16 manifest; original inputs;plots;checkpoints;logs;SHA seal'],
        source_sha256={p:executor.sha(ROOT/p) for p in files},environment=executor.environment())
    executor.write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],B_reused=True)),flush=True)


def verify_b(model,examples,run):
    idx=index(ROOT,PARENT,PARENT_SEAL)
    def read(rel):return executor.read_pinned(ROOT,PARENT+'/'+rel,idx[PARENT+'/'+rel])
    import io
    checkpoint=torch.load(io.BytesIO(read('checkpoints/step_0000.pt')),map_location='cpu',weights_only=False)
    if any(not torch.equal(v.cpu(),checkpoint['model'][k]) for k,v in model.state_dict().items()):raise ValueError('B initial mismatch')
    if json.loads(read('config/schedule.json'))!=[list(x) for x in batch_schedule(16,updates=1000)]:raise ValueError('B schedule mismatch')
    for o in examples:
        key=executor.digest(o.source);b=o.student_representations['PRIMITIVE']['blocks']
        with np.load(io.BytesIO(read('artifacts/input_'+key+'.npz'))) as z:
            for name,value in [('xyz_m',b.xyz_m),('frame_index',b.frame_index),('primitive_assignment',b.point_to_block),('target_positions_m',o.loss_only['target'].position_m.numpy())]:
                if not np.array_equal(value,z[name]):raise ValueError('B actual input mismatch '+name)
        with torch.no_grad():out=executor.forward(model,o,'PRIMITIVE')
        with np.load(io.BytesIO(read('artifacts/prediction_0000_'+key+'.npz'))) as z:
            if not np.array_equal(out.query_source_indices.cpu().numpy(),z['query_indices']) or not np.array_equal(out.query_positions_m.cpu().numpy(),z['query_positions_m']):raise ValueError('B query mismatch')
    executor.write(run/'metrics/b_equivalence.json',dict(B_run=PARENT,B_reused=True,B_new_updates=0,
        same_actual16=True,same_queries=True,same_initial=True,same_schedule=True,
        initial_sha256=state_sha256(model),same_loss_weights_and_masks=True,
        only_training_change='geometry-only assignment instead of confidence-augmented assignment'), 'x')


def evaluate(model,examples,run,step):
    if step==0:verify_b(model,examples,run)
    result=original_evaluate(model,examples,run,step)
    scores=json.loads((run/'metrics'/f'evaluation_{step:04d}.json').read_text())
    rows=[]
    for o,row in zip(examples,scores['observations']):
        if o.source!=row['source']:raise ValueError('evaluation order changed')
        with np.load(run/'artifacts'/f'prediction_{step:04d}_{executor.digest(o.source)}.npz') as p:
            funnel=localization_funnel(p['position_m'],p['presence_logits'],o.loss_only['target'].position_m.numpy(),row['scores']['1.0'])
        rows.append(dict(source=o.source,**funnel))
    totals={k:sum(r[k] for r in rows) for k in ('reference_count','raw_one_to_one_tp','selected_one_to_one_tp','actual_tp')}
    executor.write(run/'metrics'/f'localization_{step:04d}.json',dict(summary=totals,observations=rows),'x')
    print(json.dumps(dict(step=step,localization=totals)),flush=True)
    return result


def configure():
    executor.SCHEMA=SCHEMA;executor.SLUG=SLUG;executor.POLICY=POLICY;executor.GROUPING='PRIMITIVE'
    executor.compile_scope=compile_scope;executor.validate_card=validate_card
    executor.build_model=build_model;executor.center_objective=center_objective;executor.evaluate=evaluate


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(executor.execute(executor.load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
