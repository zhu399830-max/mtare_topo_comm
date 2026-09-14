"""Same corrected16 fit, only the selected grouping becomes SPATIAL.

Reuse the completed PRIMITIVE result. No old gradient/mask audit is rerun.
"""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import grouping_center_fit_v1 as executor
from grouping_localization_ceiling_v1 import reduce_step
from mtare_topo.governance_spatial_fit_v1 import SCHEMA,SLUG,POLICY,PARENT,PARENT_SEAL,PRELIM,compile_scope,validate_card
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.grouping_supervision_v2 import center_objective

original_evaluate=executor.evaluate
original_preview=executor.preview


def spatial_preview(observation,prediction,path):
    proxy=SimpleNamespace(source=dict(observation.source,diagnostic_grouping='SPATIAL'),
        student_representations={'r2':observation.student_representations['SPATIAL']},loss_only=observation.loss_only)
    original_preview(proxy,prediction,path)


def verify_reuse(model,examples,run):
    root=executor.ROOT;scope=executor.load_json(run/'config/data_card.json')['scope']
    if executor.environment()!=scope['parent_environment']:raise ValueError('paired environment differs')
    seal=root/PARENT/'artifacts/evidence_sha256.txt'
    if executor.sha(seal)!=PARENT_SEAL:raise ValueError('parent seal drift')
    index={p:h for h,p in (line.split('  ',1) for line in seal.read_text().splitlines())}
    reads={}
    def read(rel):
        path=PARENT+'/'+rel;reads[path]=index[path]
        return executor.read_pinned(root,path,index[path])
    cp=torch.load(io.BytesIO(read('checkpoints/step_0000.pt')),map_location='cpu',weights_only=False)
    if set(cp['model'])!=set(model.state_dict()) or any(not torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items()):
        raise ValueError('initial state not exactly the reused PRIMITIVE initialization')
    del cp
    if json.loads(read('config/schedule.json'))!=json.loads((run/'config/schedule.json').read_text()):
        raise ValueError('sample order differs')
    rows=[];model.eval()
    with torch.no_grad():
        for o in examples:
            key=executor.digest(o.source)
            with np.load(io.BytesIO(read(f'artifacts/input_{key}.npz'))) as old:
                primitive=o.student_representations['PRIMITIVE']['blocks'];spatial=o.student_representations['SPATIAL']['blocks']
                for name,value in [('xyz_m',primitive.xyz_m),('frame_index',primitive.frame_index),
                    ('primitive_assignment',primitive.point_to_block),('spatial_assignment',spatial.point_to_block),
                    ('target_positions_m',o.loss_only['target'].position_m.numpy())]:
                    if not np.array_equal(old[name],value):raise ValueError('paired input/target/group drift: '+name)
            if not np.array_equal(primitive.xyz_m,spatial.xyz_m) or not np.array_equal(primitive.frame_index,spatial.frame_index):
                raise ValueError('groupings do not preserve identical observations')
            if len(primitive.block_ids)!=len(spatial.block_ids):raise ValueError('M differs')
            a=executor.forward(model,o,'PRIMITIVE');b=executor.forward(model,o,'SPATIAL')
            if not torch.equal(a.query_positions_m,b.query_positions_m) or not torch.equal(a.query_source_indices,b.query_source_indices):
                raise ValueError('observed query populations differ')
            rows.append(dict(source=o.source,points=len(spatial.xyz_m),blocks=len(spatial.block_ids),
                identical_points=True,matched_M=True,identical_query_coordinates=True,identical_query_indices=True,
                original_group_arrays_preserved=True,
                primitive_initial_logits=a.prediction.presence_logits.cpu().tolist(),
                spatial_initial_logits=b.prediction.presence_logits.cpu().tolist()))
    # This is configuration/input comparability, not a repeated teacher audit.
    executor.write(run/'metrics/reused_primitive_comparability.json',dict(status='COMPARABILITY_PASS',
        initial_parameters_equal=True,schedule_equal=True,environment_equal=True,
        common_implementation=scope['common_implementation'],observations=rows,
        source_reads_sha256=reads,primitive_new_updates=0,old_loss_gradient_reaudits=0),'x')
    executor.write(run/'metrics/primitive_reused_localization_ceiling.json',
        json.loads(executor.read_pinned(root,PRELIM,scope['prerequisite']['sha256'])),'x')
    print('COMPARABILITY_PASS: same16/points/M/query indices/initial parameters/environment/schedule; only SPATIAL grouping is trained.',flush=True)


def evaluate(model,examples,run,step):
    if step==0:verify_reuse(model,examples,run)
    result=original_evaluate(model,examples,run,step)
    ceiling=reduce_step(run,step)
    executor.write(run/'metrics'/f'localization_ceiling_{step:04d}.json',ceiling,'x')
    print(json.dumps(dict(step=step,grouping='SPATIAL',funnel=ceiling['summary'])),flush=True)
    if step==1000:
        parent=executor.load_json(run/'metrics/primitive_reused_localization_ceiling.json')
        spatial=[executor.load_json(run/'metrics'/f'localization_ceiling_{s:04d}.json') for s in range(0,1001,100)]
        executor.write(run/'metrics/paired_fit_diagnostic.json',dict(primitive_reused=PARENT,
            primitive_new_updates=0,spatial_updates=1000,formal_comparison=False,method_winner_claim=False,
            primitive=parent['steps'],spatial=spatial,
            next='compare shared localization vs selection failure; no automatic training,branches or graph expansion'),'x')
    return result


def configure():
    executor.SCHEMA=SCHEMA;executor.SLUG=SLUG;executor.POLICY=POLICY;executor.GROUPING='SPATIAL'
    executor.compile_scope=compile_scope;executor.validate_card=validate_card
    executor.build_model=build_model;executor.center_objective=center_objective
    executor.evaluate=evaluate;executor.preview=spatial_preview
    executor.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
    executor.SPEC='configs/v3/gate3/'+SLUG+'.json'
    executor.RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
    executor.SCRIPT='tools/v3/spatial_center_fit_v1.py'


if __name__=='__main__':
    configure()
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:executor.freeze()
    elif a.spec and a.run_dir:raise SystemExit(executor.execute(executor.load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
