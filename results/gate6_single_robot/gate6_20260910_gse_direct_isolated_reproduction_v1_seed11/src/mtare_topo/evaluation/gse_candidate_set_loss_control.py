"""One-to-one hard-negative objective control; not a frozen training method.

No spatial NMS in this loss. Each target gets at most one candidate; surplus
known candidates are negatives, including duplicate high positives. Matching
prioritizes feasible cardinality before normalized distance and confidence.
Identical candidate features can still make unique output unlearnable; this
function establishes an objective, not representational sufficiency.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def one_to_one_hard_negative_control(logits,positions,expected,*,complete_region):
    if (logits.ndim!=1 or logits.dtype not in (torch.float32,torch.float64)
            or not torch.isfinite(logits).all() or type(complete_region)!=bool):
        raise ValueError('finite logits and explicit completeness required')
    for xyz in (positions,expected):
        if (not isinstance(xyz,np.ndarray) or xyz.ndim!=2 or xyz.shape[1]!=3
                or not np.isfinite(xyz).all()):raise ValueError('finite N,3 geometry required')
    n=len(logits);m=len(expected)
    if len(positions)!=n:raise ValueError('position/logit mismatch')
    matched=[]
    if n and m:
        distance=np.linalg.norm(expected[:,None]-positions[None],axis=-1)
        score=logits.detach().sigmoid().cpu().numpy()
        cost=distance-score[None]
        penalty=4*(max(n,m)+1)
        a,b=linear_sum_assignment(np.where(distance<=1.,cost,penalty))
        matched=[int(j) for i,j in zip(a,b) if distance[i,j]<=1.]
    if len(matched)!=m:raise ValueError('target lacks distinct in-radius candidates')
    positive=torch.zeros(n,dtype=torch.bool,device=logits.device)
    positive[matched]=True
    negative=~positive if complete_region else torch.zeros_like(positive)
    terms=[];hard_indices=[]
    if positive.any():terms.append(F.softplus(-logits[positive]).mean())
    if negative.any():
        # At most one hard negative per target, at least one for empty scenes.
        # Extra easy candidates cannot dilute an existing hardest error.
        ids=torch.nonzero(negative,as_tuple=True)[0]
        k=min(max(m,1),len(ids))
        values,order=torch.topk(F.softplus(logits[ids]),k)
        hard_indices=ids[order].detach().cpu().tolist();terms.append(values.mean())
    loss=torch.stack(terms).mean() if terms else logits.sum()*0
    return loss,dict(matched_candidate_indices=matched,hard_negative_indices=hard_indices,
                     unmatched_unknown=n-len(matched) if not complete_region else 0,
                     spatial_suppression_used=False,training_qualified=False)
