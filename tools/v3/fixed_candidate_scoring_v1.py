"""One scoring-only stage on passed frozen dynamic-position checkpoint."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import torch
import grouping_center_fit_v1 as executor
from mtare_topo.governance_frozen_scoring_v1 import SCHEMA,SLUG,POLICY,PARENT,SEAL,AUTH,index,compile_scope,validate_card
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model as original_model
from mtare_topo.representation.frozen_candidate_scoring_v1 import (
    freeze_except_existence,forward,center_objective as loss,assert_frozen_parameters,verify_full_path)
from mtare_topo.representation.development_paired_training import batch_schedule,state_sha256
from mtare_topo.evaluation.grouping_localization_ceiling_v1 import localization_funnel

ROOT=executor.ROOT;CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed0'
original_evaluate=executor.evaluate
active_model=None


def parent_read(rel):
    path=PARENT+'/'+rel
    return executor.read_pinned(ROOT,path,index(ROOT)[path])


def build_model(**kwargs):
    global active_model
    model=original_model(**kwargs)
    ckpt=torch.load(io.BytesIO(parent_read('checkpoints/step_1000.pt')),map_location=kwargs.get('device','cpu'),weights_only=False)
    model.load_state_dict(ckpt['model'],strict=True)
    active_model=freeze_except_existence(model)
    return active_model


def center_objective(output,observation):return loss(output,observation,model=active_model)


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-latest-exact-scope',approved_at='2026-09-10',
        authorized_operations=['training'],authorized_gates=[3],scope_sha256=executor.digest(scope),confirmation_reference=AUTH,
        scope='Same16; frozen dynamic-position final checkpoint; all512queries; only original existence129 effective parameters; one1000batch schedule; fixed repaired supervision, no protected worlds or new labels.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=executor.digest(scope),policy=POLICY,approval=approval)
    if not validate_card(card).passed:raise ValueError('invalid card')
    executor.write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,'tests/v3/unit/test_frozen_candidate_scoring_v1.py'})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8',executor.PYTHON,'tools/v3/fixed_candidate_scoring_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=0,operation='training',data_card=CARD,config_path=CARD,
        user_authorization=approval,command=command,question=POLICY['question'],method=POLICY['loss']+';'+POLICY['trainable'],baseline=POLICY['baseline'],
        fallback='If final detection fails stop at fixed candidate scoring; no automatic joint training or new model search.',
        wall_time_cap_s=43200,estimated_cost=dict(compute='one frozen feature extraction and129parameter scoring1000batches; no encoder backprop',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['All512valid queries and their positions remain bitwise equal to parent through all evaluations; nonexistence parameters unchanged.',
            'Original1000batches, fixed geometry matching and repaired masks; original head only, unknown not background.',
            'Final known1m precision and recall>=.90 at unchanged.5 threshold; all original radii; positives,confirmed negatives and unknown score distributions reported.',
            'No A/B,decoder or position retraining. If pass retain two-stage protocol for same-protocol comparison; otherwise stop fixed scoring.'],
        expected_evidence=['fixed512candidates/features/masks,all checkpoints and predictions,full-path and frozen-parameter checks,per-class scores,full detection,logs,seal'],
        source_sha256={p:executor.sha(ROOT/p) for p in files},environment=executor.environment())
    executor.write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],effective_trainable_parameters=129)),flush=True)


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),selected=int((a>=.5).sum()),min=float(a.min()) if len(a) else None,
        mean=float(a.mean()) if len(a) else None,max=float(a.max()) if len(a) else None,
        quantiles=np.quantile(a,[.1,.5,.9]).tolist() if len(a) else [])


def evaluate(model,examples,run,step):
    model.eval();assert_frozen_parameters(model);rows=[];groups={k:[] for k in ('positive','negative','unknown')};queries=0
    with torch.no_grad():
        for o in examples:
            out=verify_full_path(model,o);key=executor.digest(o.source);queries+=len(out.prediction.position_m)
            with np.load(io.BytesIO(parent_read('artifacts/prediction_1000_'+key+'.npz'))) as z:
                for name,value in [('position_m',out.prediction.position_m.cpu().numpy()),('query_indices',out.query_source_indices.cpu().numpy()),('query_positions_m',out.query_positions_m.cpu().numpy())]:
                    if not np.array_equal(value,z[name]):raise ValueError('parent candidate changed '+name)
            if step==0:
                b=o.student_representations['PRIMITIVE']['blocks']
                with np.load(io.BytesIO(parent_read('artifacts/input_'+key+'.npz'))) as z:
                    for name,value in [('xyz_m',b.xyz_m),('primitive_assignment',b.point_to_block),('frame_index',b.frame_index),('target_positions_m',o.loss_only['target'].position_m.numpy())]:
                        if not np.array_equal(value,z[name]):raise ValueError('parent input changed')
            r=center_objective(out,o);positive=torch.zeros(len(out.prediction.position_m),dtype=torch.bool,device=out.prediction.position_m.device)
            positive[r['assignment']]=True;negative=r['negative_mask'];unknown=~(positive|negative)
            probability=out.prediction.presence_logits.sigmoid()
            for name,mask in [('positive',positive),('negative',negative),('unknown',unknown)]:groups[name].extend(probability[mask].cpu().tolist())
            rows.append(dict(source=o.source,assignment=r['assignment'].tolist(),positive=positive.cpu().tolist(),negative=negative.cpu().tolist(),unknown=unknown.cpu().tolist(),
                logits=out.prediction.presence_logits.cpu().tolist(),probabilities=probability.cpu().tolist()))
            if step==0:
                c=model._candidate_cache[key]
                np.savez_compressed(run/'artifacts'/f'fixed_candidates_{key}.npz',feature=c['feature'].cpu().numpy(),position_m=out.prediction.position_m.cpu().numpy(),
                    assignment=r['assignment'].cpu().numpy(),positive=positive.cpu().numpy(),negative=negative.cpu().numpy(),unknown=unknown.cpu().numpy())
                executor.write(run/'artifacts'/f'fixed_supervision_evidence_{key}.json',r['evidence'],'x')
    if queries!=512:raise ValueError('all512queries required, no filtering')
    if json.loads(parent_read('config/schedule.json'))!=[list(b) for b in batch_schedule(16,updates=1000)]:raise ValueError('schedule mismatch')
    executor.write(run/'metrics'/f'frozen_check_{step:04d}.json',dict(all_queries=queries,all_positions_bitwise_parent_equal=True,all_nonexistence_parameters_equal=True,
        cache_full_forward_bitwise_equal=True,effective_trainable_parameters=129,initialization='parent dynamic final',full_state_sha256=state_sha256(model)),'x')
    executor.write(run/'metrics'/f'candidate_scores_{step:04d}.json',dict(groups={k:stats(v) for k,v in groups.items()},observations=rows),'x')
    result=original_evaluate(model,examples,run,step)
    scored=json.loads((run/'metrics'/f'evaluation_{step:04d}.json').read_text());funnels=[]
    for o,row in zip(examples,scored['observations']):
        with np.load(run/'artifacts'/f'prediction_{step:04d}_{executor.digest(o.source)}.npz') as p:
            f=localization_funnel(p['position_m'],p['presence_logits'],o.loss_only['target'].position_m.numpy(),row['scores']['1.0'])
        funnels.append(dict(source=o.source,**f))
    totals={k:sum(r[k] for r in funnels) for k in ('reference_count','raw_one_to_one_tp','selected_one_to_one_tp','actual_tp')}
    executor.write(run/'metrics'/f'localization_{step:04d}.json',dict(summary=totals,observations=funnels),'x')
    print(json.dumps(dict(step=step,groups={k:stats(v) for k,v in groups.items()},localization=totals)),flush=True)
    return result


def configure():
    executor.SCHEMA=SCHEMA;executor.SLUG=SLUG;executor.POLICY=POLICY;executor.GROUPING='PRIMITIVE'
    executor.compile_scope=compile_scope;executor.validate_card=validate_card;executor.build_model=build_model
    executor.forward=forward;executor.center_objective=center_objective;executor.evaluate=evaluate


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(executor.execute(executor.load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
