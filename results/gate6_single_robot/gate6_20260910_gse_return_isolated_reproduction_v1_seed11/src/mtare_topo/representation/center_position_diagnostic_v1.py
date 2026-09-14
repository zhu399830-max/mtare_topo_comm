"""Loss-only correspondences. Never changes the observation-only forward."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .observed_anchor_detector_v1 import residual_positions


def geometry_assignment(positions, targets):
    if targets.requires_grad or positions.ndim != 2 or targets.ndim != 2:
        raise ValueError('detached 2D targets required')
    if positions.shape[1] != 3 or targets.shape[1] != 3 or len(targets)>len(positions):
        raise ValueError('aligned capacity; do not truncate')
    if not torch.isfinite(positions).all() or not torch.isfinite(targets).all():
        raise ValueError('finite coordinates required')
    cost=torch.cdist(targets.detach().double(),positions.detach().double()).cpu().numpy()
    rows,cols=linear_sum_assignment(cost)
    assignment=np.empty(len(targets),dtype=np.int64);assignment[rows]=cols
    # Stable scipy row/column order resolves equal costs; no confidence tie-break.
    return torch.as_tensor(assignment,device=positions.device)


def position_objective(positions,targets,assignment):
    if (targets.requires_grad or assignment.dtype!=torch.long or assignment.shape!=(len(targets),)
        or assignment.unique().numel()!=len(assignment) or torch.any(assignment<0)
        or torch.any(assignment>=len(positions))):
        raise ValueError('detached targets and valid fixed one-to-one assignment required')
    if not torch.isfinite(positions).all() or not torch.isfinite(targets).all():
        raise ValueError('nonfinite position')
    return (torch.linalg.vector_norm(positions[assignment]-targets,dim=-1).mean()/10.
            if len(targets) else positions.sum()*0.)


def independent_tensor_check(queries,targets,updates=1000):
    """Oracle residual tensors test only decoder/loss; never network inputs/weights."""
    if queries.shape!=targets.shape or not len(targets):raise ValueError('paired tensors required')
    q=queries.detach().cpu().float();t=targets.detach().cpu().float()
    delta=(t-q)/20.
    if torch.any(delta.abs()>=1) or torch.any(t.norm(dim=-1)>10.+1e-6):
        raise ValueError('actual selected target outside finite inverse range')
    inverse=torch.atanh(delta);decoded,_=residual_positions(q,inverse)
    exact_error=(decoded-t).norm(dim=-1)
    raw=torch.zeros_like(q,requires_grad=True)
    optimizer=torch.optim.AdamW([raw],lr=.001,weight_decay=0.)
    assignment=torch.arange(len(t));history=[]
    for step in range(updates+1):
        p,_=residual_positions(q,raw)
        if step%100==0:history.append(dict(step=step,error_m=(p-t).norm(dim=-1).detach().tolist()))
        if step==updates:break
        optimizer.zero_grad();loss=position_objective(p,t,assignment);loss.backward()
        if raw.grad is None or not torch.isfinite(raw.grad).all():raise ValueError('invalid tensor gradient')
        optimizer.step()
    final=torch.tensor(history[-1]['error_m'])
    return dict(passed=bool(exact_error.max()<=1e-5 and final.max()<=.1),
        independent_network=False,network_updates=0,tensor_updates=updates,
        max_abs_delta_m=float((t-q).abs().max()),max_target_norm_m=float(t.norm(dim=-1).max()),
        analytic_inverse_error_m=exact_error.tolist(),history=history,
        finite_inverse_raw=inverse.tolist(),final_max_m=float(final.max()))
