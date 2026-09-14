"""A: unchanged repaired presence supervision; geometry-only assignment.

Use neutral matching logits solely in the LOSS adapter to reuse all existing
binding/mask validation. -sigmoid(0) is a row-constant and cannot favor a query.
Actual logits are restored for the unchanged positive/negative-group loss.
No output or input is changed in forward, prediction saving or evaluation.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from .grouping_supervision_v2 import center_objective as repaired_objective
from .center_position_diagnostic_v1 import geometry_assignment


def restore_presence(result,logits):
    groups=[];matched=result['assignment'];negative=result['negative_mask']
    if len(matched):groups.append(F.softplus(-logits[matched]).mean())
    if negative.any():groups.append(F.softplus(logits[negative]).mean())
    presence=sum(groups)/len(groups) if groups else logits.new_zeros((),requires_grad=True)
    return dict(result,presence=presence,total=result['position']+presence,has_supervision=bool(groups))


def center_objective(output,observation):
    p=output.prediction
    neutral=replace(output,prediction=replace(p,presence_logits=torch.zeros_like(p.presence_logits)))
    result=repaired_objective(neutral,observation)
    geometry=geometry_assignment(p.position_m,observation.loss_only['target'].position_m.to(p.position_m))
    if not torch.equal(geometry,result['assignment']):raise ValueError('neutral cost and geometry assignment disagree')
    return restore_presence(result,p.presence_logits)
