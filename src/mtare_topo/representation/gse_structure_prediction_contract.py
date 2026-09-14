"""Partial structural output and score-free geometric correspondence.

Predictions are supplied before reference matching. No reference locations are
queries. This contract deliberately makes no calibrated presence/full detector
claim on the incomplete background population.
"""
from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PartialStructuralPrediction:
    structure_positions_m: torch.Tensor
    window_section_positions_m: torch.Tensor
    section_structure_logits: torch.Tensor
    full_detection_qualified: bool = False


def validate_prediction(prediction):
    a=prediction.structure_positions_m;o=prediction.window_section_positions_m;m=prediction.section_structure_logits
    if a.ndim!=2 or a.shape[1]!=3 or o.ndim!=2 or o.shape[1]!=3 or m.shape!=(len(o),len(a)):
        raise ValueError('structure/section positions and matching relation matrix required')
    if len(a)>32 or len(o)>64:raise ValueError('structure32/section64 capacity exceeded')
    if a.dtype not in (torch.float32,torch.float64) or any(t.dtype!=a.dtype or t.device!=a.device for t in (o,m)):
        raise ValueError('same floating prediction device/dtype required')
    if not all(torch.isfinite(t).all() for t in (a,o,m)):raise ValueError('nonfinite prediction')


def match_for_partial_loss(prediction,targets):
    """Minimum total Euclidean distance, no confidence or membership costs."""
    validate_prediction(prediction);result={}
    for name,p,t in (('structure',prediction.structure_positions_m,targets.anchor_positions_m),
                     ('section',prediction.window_section_positions_m,targets.window_section_positions_m)):
        if len(t)>len(p):raise ValueError('insufficient output capacity; never drop references')
        distance=np.linalg.norm(p.detach().cpu().numpy()[:,None]-t[None],axis=-1)
        q,r=linear_sum_assignment(distance)
        order=np.full(len(t),-1,dtype=np.int64);order[r]=q
        order.setflags(write=False);result[name]=order
    return result


def evaluate_partial_localization(prediction,targets):
    """Independent maximum1m reference recovery, then minimum distance.

    All predictions participate. Extra outputs remain unconfirmed, not known
    false positives. This evaluator never consumes training assignments.
    """
    validate_prediction(prediction);out={}
    for name,p,t in (('structure',prediction.structure_positions_m,targets.anchor_positions_m),
                     ('section',prediction.window_section_positions_m,targets.window_section_positions_m)):
        d=np.linalg.norm(p.detach().cpu().numpy()[:,None]-t[None],axis=-1)
        recovered={}
        if d.size:
            penalty=(1+d.max())*(1+min(d.shape))
            q,r=linear_sum_assignment(d+(d>1.)*penalty)
            recovered={int(ti):int(qi) for qi,ti in zip(q,r) if d[qi,ti]<=1.}
        out[name]=dict(reference_count=len(t),correct=len(recovered),prediction_count=len(p),
            unconfirmed_predictions=len(p)-len(recovered),reference_to_prediction=recovered)
    out['full_detection_precision_supported']=False
    return out


def partial_structure_loss(prediction,targets):
    """Original normalized geometry and known-pair mean; no presence loss.

    This task has no known background certificate. It cannot learn calibrated
    existence by declaring every unmatched query negative.
    """
    from torch.nn import functional as F
    from .gse_partial_structure_contract import known_relation_loss
    matches=match_for_partial_loss(prediction,targets);terms={};counts={}
    for name,p,t in (('structure',prediction.structure_positions_m,targets.anchor_positions_m),
                     ('section',prediction.window_section_positions_m,targets.window_section_positions_m)):
        if len(t):
            ix=torch.tensor(matches[name].copy(),device=p.device)
            truth=torch.tensor(t.copy(),device=p.device,dtype=p.dtype)
            terms[name+'_position']=F.smooth_l1_loss(p[ix]/10.,truth/10.)
            counts[name+'_position']=len(t)
    oi=torch.tensor(matches['section'].copy(),device=prediction.section_structure_logits.device)
    ai=torch.tensor(matches['structure'].copy(),device=prediction.section_structure_logits.device)
    aligned=prediction.section_structure_logits[oi[:,None],ai[None,:]]
    if targets.relation_known.any():
        terms['known_relation']=known_relation_loss(aligned,targets)
        counts['known_relation']=int(targets.relation_known.sum())
    if not terms:raise ValueError('no supervised target; not a fitting sample')
    total=sum(terms.values())
    if not torch.isfinite(total):raise ValueError('nonfinite partial objective')
    return dict(total=total,terms=terms,counts=counts,matches=matches)


def evaluate_partial_structure(prediction,targets):
    result=evaluate_partial_localization(prediction,targets)
    a=result['structure']['reference_to_prediction'];o=result['section']['reference_to_prediction']
    stats=dict(positive_total=0,negative_total=0,positive_correct=0,negative_correct=0,unlocalized_known=0,
               unknown_reference_pairs=int((~targets.relation_known).sum()))
    for oi,ai in np.argwhere(targets.relation_known):
        positive=bool(targets.relation_values[oi,ai]);side='positive' if positive else 'negative'
        stats[side+'_total']+=1
        if int(oi) not in o or int(ai) not in a:
            stats['unlocalized_known']+=1;continue
        selected=bool(prediction.section_structure_logits[o[int(oi)],a[int(ai)]].detach()>=0)
        stats[side+'_correct']+=int(selected==positive)
    result['relations']=stats
    return result
