"""Microbatch-one, accumulation-four mechanics for the bounded detector pilot.

Does not read data, select a checkpoint, authorize a run, or launch pair training.
"""
from dataclasses import replace
import torch
from .gse_block_point_encoder import BlockPointEncoder
from .observed_anchor_detector_v1 import ObservedAnchorDetectorV1
from .conditional_observed_anchor_loss_v1 import conditional_observed_anchor_loss


def build_model(*,relation_attributes=False,seed=0,device='cpu'):
    torch.manual_seed(seed)
    return torch.nn.ModuleDict(dict(point=BlockPointEncoder(),
        head=ObservedAnchorDetectorV1(relation_attributes=relation_attributes))).to(device)


def forward_observation(model,observation):
    student=observation.student_representations['r2']
    parameter=next(model.parameters())
    context=torch.tensor(student['context'],dtype=parameter.dtype,device=parameter.device)
    # No source ID, split or loss_only read on the forward path.
    return model['head'](student['blocks'],model['point'](student['blocks']),context)


def objective(output,observation):
    if output is None:
        raise ValueError('empty observation requires explicit run accounting, never guessed queries')
    p=output.prediction; loss=observation.loss_only; target=loss['target']
    target=replace(target,position_m=target.position_m.to(device=p.position_m.device,dtype=p.position_m.dtype),
                   directions=tuple(d.to(device=p.position_m.device,dtype=p.position_m.dtype) for d in target.directions))
    return conditional_observed_anchor_loss(p,target,bundle=loss['bundle'],grid=loss['grid'],
        produced_targets=loss['produced_targets'],frozen_manifest=loss['frozen_manifest'])


def train_update(model,optimizer,batch):
    if len(batch)!=4 or any(o.split!='fit' for o in batch):
        raise ValueError('exactly four fit observations required before any update')
    model.train();optimizer.zero_grad(set_to_none=True);records=[];active=0
    try:
        for observation in batch:
            output=forward_observation(model,observation);result=objective(output,observation)
            if not torch.isfinite(result['total']):raise FloatingPointError('nonfinite loss')
            if result['has_supervision']:
                (result['total']/4.).backward();active+=1
            records.append(dict(source=observation.source,loss=float(result['total'].detach()),
                terms={k:float(v.detach()) for k,v in result['terms'].items()},counts=result['counts'],
                anchor_assignment=result['anchor_assignment'].tolist(),
                branch_assignments=[v.tolist() for v in result['branch_assignments']],
                anchor_group_counts=result['anchor_group_counts'],branch_group_counts=result['branch_group_counts'],
                query_source_indices=output.query_source_indices.detach().cpu().tolist()))
            del output,result
        if not active:return dict(optimizer_step=False,observations=records)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite gradient; no optimizer step')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite updated parameters; stop run')
        return dict(optimizer_step=True,observations=records)
    finally:
        optimizer.zero_grad(set_to_none=True)
