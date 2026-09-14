"""Training-only branch-set correction; never used by independent scoring."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def branch_selection_objective(directions, logits, targets, *, complete):
    if type(complete) is not bool:
        raise ValueError('explicit reference completeness required')
    if (directions.ndim != 2 or directions.shape[1] != 3 or not 1 <= len(directions) <= 64
            or logits.shape != (len(directions),) or targets.ndim != 2 or targets.shape[1] != 3):
        raise ValueError('bounded Kx3, K and Tx3 tensors required')
    if len(targets) > len(directions):
        raise ValueError('capacity exceeded; no truncation')
    if directions.dtype not in (torch.float32, torch.float64) or targets.requires_grad:
        raise ValueError('floating predictions and detached targets required')
    for value in (directions, logits, targets):
        if value.dtype != directions.dtype or value.device != directions.device or not torch.isfinite(value).all():
            raise ValueError('finite shared dtype/device required')
    for value in (directions, targets):
        if not torch.allclose(torch.linalg.vector_norm(value, dim=-1), torch.ones(len(value), device=value.device, dtype=value.dtype), atol=64*torch.finfo(value.dtype).eps, rtol=0):
            raise ValueError('unit directions required')
    cost = (torch.cdist(targets.detach().double(), directions.detach().double()) - logits.detach().double().sigmoid()[None]).cpu().numpy()
    assignment = np.empty(len(targets), dtype=np.int64)
    if len(targets):
        rows, columns = linear_sum_assignment(cost)
        optimum = float(cost[rows, columns].sum())
        tolerance = 64*np.finfo(float).eps*max(1., float(np.abs(cost).max())*len(targets))
        for row, column in zip(rows, columns):
            alternative = cost.copy(); alternative[row, column] = np.inf
            try:
                rr, cc = linear_sum_assignment(alternative)
                second = float(alternative[rr, cc].sum())
            except ValueError:
                second = np.inf
            if second <= optimum+tolerance:
                raise ValueError('ambiguous training assignment')
        assignment[rows] = columns
    matched = torch.as_tensor(assignment, device=logits.device)
    negative = torch.full_like(logits, complete, dtype=torch.bool)
    negative[matched] = False
    zero = logits.new_zeros((), requires_grad=True)
    groups = []
    if len(matched): groups.append(F.softplus(-logits[matched]).mean())
    if negative.any(): groups.append(F.softplus(logits[negative]).mean())
    presence = sum(groups)/len(groups) if groups else zero
    direction = (1-(directions[matched]*targets).sum(-1).clamp(-1,1)).mean()/2 if len(matched) else zero
    return dict(total=presence+direction, presence=presence, direction=direction,
                assignment=matched, positive_count=len(matched), negative_count=int(negative.sum()),
                has_supervision=bool(groups), scoring_policy_changed=False)
