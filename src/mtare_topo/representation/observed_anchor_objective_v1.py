"""Training assignment: distance/10m - sigmoid(logit); scoring stays separate.

Confirmed negatives must come from the authenticated evidence binder. Positive
and negative group means have equal weight; unknown has no gradient.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def observed_anchor_objective(positions, logits, targets, *, confirmed_negative):
    if (positions.ndim!=2 or positions.shape[1]!=3 or not 1<=len(positions)<=32
            or logits.shape!=(len(positions),) or targets.ndim!=2 or targets.shape[1]!=3
            or confirmed_negative.shape!=logits.shape or confirmed_negative.dtype!=torch.bool
            or confirmed_negative.device!=positions.device):
        raise ValueError('bounded aligned centers, logits, targets and evidence mask required')
    if len(targets)>len(positions): raise ValueError('capacity exceeded; never truncate targets')
    if positions.dtype not in (torch.float32,torch.float64) or targets.requires_grad:
        raise ValueError('floating prediction and detached target required')
    for value in (positions,logits,targets):
        if value.dtype!=positions.dtype or value.device!=positions.device or not torch.isfinite(value).all():
            raise ValueError('finite same-device inputs required')
    for value in (positions,targets):
        if torch.any(torch.linalg.vector_norm(value,dim=-1)>10.+640*torch.finfo(positions.dtype).eps):
            raise ValueError('outside frozen 10m domain')
    cost=(torch.cdist(targets.detach().double(),positions.detach().double())/10.
          -logits.detach().double().sigmoid()[None]).cpu().numpy()
    assignment=np.empty(len(targets),np.int64)
    if len(targets):
        rows,columns=linear_sum_assignment(cost)
        optimum=float(cost[rows,columns].sum())
        tolerance=64*np.finfo(float).eps*max(1.,float(np.abs(cost).max())*len(targets))
        for row,column in zip(rows,columns):
            alternative=cost.copy(); alternative[row,column]=np.inf
            try:
                rr,cc=linear_sum_assignment(alternative)
                second=float(alternative[rr,cc].sum())
            except ValueError:
                second=np.inf
            if second<=optimum+tolerance: raise ValueError('ambiguous center training assignment')
        assignment[rows]=columns
    matched=torch.as_tensor(assignment,device=positions.device)
    negative=confirmed_negative.clone(); negative[matched]=False
    zero=logits.new_zeros((),requires_grad=True)
    groups=[]
    if len(matched): groups.append(F.softplus(-logits[matched]).mean())
    if negative.any(): groups.append(F.softplus(logits[negative]).mean())
    presence=sum(groups)/len(groups) if groups else zero
    position=torch.linalg.vector_norm(positions[matched]-targets,dim=-1).mean()/10. if len(matched) else zero
    unknown=~negative; unknown[matched]=False
    return dict(total=presence+position,presence=presence,position=position,
                assignment=matched,negative_mask=negative,positive_count=len(matched),
                negative_count=int(negative.sum()),unknown_count=int(unknown.sum()),
                has_supervision=bool(groups),scoring_policy_changed=False)
