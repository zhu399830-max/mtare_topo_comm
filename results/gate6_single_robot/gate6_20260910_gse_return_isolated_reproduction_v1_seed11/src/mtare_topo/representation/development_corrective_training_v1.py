"""Separate corrective entry: same forward and schedule, new branch loss only."""
from dataclasses import replace
import torch
from .development_paired_training import forward_observation
from .conditional_branch_selection_v1 import conditional_branch_selection


def objective(prediction, observation):
    loss=observation.loss_only;target=loss['target'];device=prediction.position_m.device
    target=replace(target,position_m=target.position_m.to(device=device,dtype=prediction.position_m.dtype),
                   directions=tuple(d.to(device=device,dtype=prediction.position_m.dtype) for d in target.directions))
    return conditional_branch_selection(prediction,target,bundle=loss['bundle'],grid=loss['grid'],
        produced_targets=loss['produced_targets'],frozen_manifest=loss['frozen_manifest'])


def train_update(model, optimizer, batch, representation):
    if len(batch)!=4 or any(o.split!='fit' for o in batch):
        raise ValueError('four fit-only observations required before any update')
    model.train();optimizer.zero_grad(set_to_none=True);records=[];active=0
    try:
        for observation in batch:
            prediction=forward_observation(model,observation,representation)
            result=objective(prediction,observation)
            if not torch.isfinite(result['total']):raise FloatingPointError('nonfinite loss')
            if result['has_supervision']:
                (result['total']/4.).backward();active+=1
            records.append(dict(source=observation.source,loss=float(result['total'].detach()),
                terms={k:float(v.detach()) for k,v in result['terms'].items()},counts=result['counts'],
                anchor_assignment=result['anchor_assignment'].tolist(),
                branch_assignments=[v.tolist() for v in result['branch_assignments']],
                branch_group_counts=result['branch_group_counts']))
            del prediction,result
        if not active:return dict(optimizer_step=False,observations=records)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite gradient; no optimizer step')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite updated parameters; stop run')
        return dict(optimizer_step=True,observations=records)
    finally:
        optimizer.zero_grad(set_to_none=True)
