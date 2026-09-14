"""Finite fitting core, supplied authorized inputs only; no filesystem access.

Every snapshot preserves all predictions. Evaluation does not use loss matches
or filter predictions with a GT region. Unmatched output counts are unconfirmed,
not automatically false positives on partial references.
"""
from dataclasses import replace
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .gse_dual_path_encoder_adapter_v1 import expand_features
from .gse_surface_patches_v1 import extract_surface_patches
from .gse_surface_relation_model_v1 import collate_surface_patches
from .gse_observed_membership_loss import observed_membership_loss


@torch.no_grad()
def prepare_membership_example(adapter, student, record, *, device):
    rv=np.stack((student.ranges_m/np.float32(50),student.valid_mask.astype(np.float32)),axis=1)
    args=[torch.from_numpy(np.array(a,copy=True))[None].to(device) for a in (
        rv,student.relative_translation_current_sensor_m,student.relative_yaw_current_sensor_deg)]
    compact=adapter.extract_compact_features(*args)
    local=compact.valid & (torch.linalg.vector_norm(compact.points_xyz_m.double(),dim=-1)<=10.)
    compact=replace(compact,valid=local)
    xyz=compact.points_xyz_m[0].cpu().numpy()
    frames=np.repeat(np.arange(5),16*720)
    origins=np.repeat(student.relative_translation_current_sensor_m,16*720,axis=0)
    patches=extract_surface_patches(xyz,local[0].cpu().numpy(),frames,ray_origins_m=origins)
    patch=collate_surface_patches([patches],device=device)
    return compact,patch,record


def predict_membership(model, example):
    compact,patch,_=example
    f=expand_features(compact)
    return model(f.points_xyz_m,f.frozen_point_context,f.valid,patch,
                 sensor_token_index=f.sensor_token_index)


def evaluate_membership(prediction, record):
    """Independent1m,0.5 partial-reference diagnostic; never full detection AP."""
    if prediction is None:raise ValueError('empty prediction for supported fitting observation')
    report={};maps={}
    for role in ('anchor','opening'):
        positions=getattr(prediction,role+'_position_m').detach().cpu().numpy()
        scores=getattr(prediction,role+'_presence_logits').detach().sigmoid().cpu().numpy()
        selected=np.flatnonzero(scores>=.5)
        truth=np.asarray([x['position_m'] for x in record[role+'s']],float).reshape(-1,3)
        recovered={};errors=[]
        if len(selected) and len(truth):
            distances=np.linalg.norm(positions[selected,None]-truth[None],axis=-1)
            # Maximum number inside1m, then shortest distance; no class/score matching.
            costs=(distances>1.)*(1.+distances.max())*(1+min(distances.shape))+distances
            q,t=linear_sum_assignment(costs)
            for qi,ti in zip(q,t):
                errors.append(float(distances[qi,ti]))
                if distances[qi,ti]<=1.:recovered[int(ti)]=int(selected[qi])
        maps[role]=recovered
        report[role]=dict(references=len(truth),correct=len(recovered),selected=len(selected),
            unmatched_unconfirmed=len(selected)-len(recovered),matched_distances_m=errors,
            reference_to_prediction=recovered)
    scores=prediction.membership_logits.detach().sigmoid().cpu().numpy()
    counts={'positive_total':0,'negative_total':0,'positive_correct':0,'negative_correct':0,
            'unlocalized_known':0,'unknown_reference_pairs':0}
    for o,row in enumerate(record['membership']):
        for a,label in enumerate(row):
            if label is None:counts['unknown_reference_pairs']+=1;continue
            side='positive' if label else 'negative';counts[side+'_total']+=1
            if o not in maps['opening'] or a not in maps['anchor']:
                counts['unlocalized_known']+=1;continue
            decision=bool(scores[maps['opening'][o],maps['anchor'][a]]>=.5)
            counts[side+'_correct']+=int(decision==label)
    report['membership']=counts
    report['all_predictions']={name:value.detach().cpu().tolist() for name,value in vars(prediction).items()}
    report['full_detection_precision_available']=False
    return report


def fit_membership(model, examples, *, on_update, updates=500, seed=0):
    """One fixed run: AdamW.001/wd.0001, microbatch1x4, initial/final only.

    Callback must persist progress and enforce wall/RAM/GPU/output caps. Errors
    propagate; no retry, replacement examples, intermediate selection or tuning.
    """
    if updates!=500 or seed!=0 or len(examples)!=16 or not callable(on_update):
        raise ValueError('fixed16,500 updates,seed0 and bounded progress callback required')
    generator=torch.Generator().manual_seed(seed)
    schedule=[]
    while len(schedule)<updates:
        order=torch.randperm(16,generator=generator).tolist()
        schedule.extend([order[i:i+4] for i in range(0,16,4)])
    def snapshot():
        model.eval()
        with torch.no_grad():return [evaluate_membership(predict_membership(model,e),e[2]) for e in examples]
    initial=snapshot()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for step,batch in enumerate(schedule,1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for i in batch:
            result=observed_membership_loss(predict_membership(model,examples[i]),examples[i][2])
            (result['total']/4.).backward()
            losses.append({k:float(v.detach()) for k,v in result['terms'].items()})
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite gradient; no update')
        optimizer.step()
        on_update(dict(step=step,sample_indices=batch,loss_terms=losses))
    return dict(initial=initial,final=snapshot(),schedule=schedule,actual_updates=updates)
