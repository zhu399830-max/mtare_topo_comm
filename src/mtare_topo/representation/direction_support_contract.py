"""Loss-side nonexclusive directional support; not a new prediction network.

Rows address observed rays, columns address deployed observation-derived queries.
Source references and targets must never be passed to the model forward path.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class DirectionSupportTargets:
    query_ray_ids: tuple
    values: torch.Tensor
    known: torch.Tensor
    evidence_refs: tuple
    qualified: bool = False

def validate_targets(targets, prediction_query_ids, logits):
    if not isinstance(targets,DirectionSupportTargets):raise ValueError('explicit loss-side target type required')
    if tuple(prediction_query_ids)!=targets.query_ray_ids:raise ValueError('supervision columns do not match deployed queries')
    if len(set(targets.query_ray_ids))!=len(targets.query_ray_ids):raise ValueError('duplicate query')
    if any(type(i)!=int or i<0 for i in targets.query_ray_ids):raise ValueError('observed query ray IDs required, not teacher direction names')
    if logits.ndim!=2 or logits.shape!=targets.values.shape or logits.shape!=targets.known.shape or logits.shape[1]!=len(targets.query_ray_ids):raise ValueError('ray-query layout mismatch')
    if targets.known.dtype!=torch.bool or targets.known.device!=logits.device or targets.values.device!=logits.device:raise ValueError('mask/device mismatch')
    if len(targets.evidence_refs)!=logits.numel():raise ValueError('one source slot per ray-query entry')
    selected=targets.values[targets.known]
    if not bool(torch.isfinite(selected).all()) or bool(((selected!=0)&(selected!=1)).any()):raise ValueError('known binary support required')
    if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite predicted score')
    known=targets.known.flatten().detach().cpu().tolist()
    if any(k and not ref for k,ref in zip(known,targets.evidence_refs)):raise ValueError('known target missing evidence')

def direction_support_loss(logits, prediction_query_ids, targets):
    validate_targets(targets,prediction_query_ids,logits)
    values=targets.values[targets.known];selected=logits[targets.known]
    # Independent Bernoulli outputs. No per-row softmax or sum-to-one loss.
    terms=[F.binary_cross_entropy_with_logits(selected[values==v],values[values==v]) for v in (0,1) if bool((values==v).any())]
    loss=torch.stack(terms).mean() if terms else selected.sum()
    return loss,dict(positive=int((values==1).sum()),negative=int((values==0).sum()),unknown=int((~targets.known).sum()),shared_positive_rows=int(((targets.values==1)&targets.known).sum(1).gt(1).sum()))

def require_direction_training_contract(logits, prediction_query_ids, targets):
    validate_targets(targets,prediction_query_ids,logits)
    values=targets.values[targets.known]
    if not targets.qualified:raise ValueError('direction support reference is not qualified')
    if not bool((values==0).any()) or not bool((values==1).any()):raise ValueError('selective-support training requires evidenced positive and negative entries; unknown is not negative')
