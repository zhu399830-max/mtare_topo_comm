"""Separate primary relation losses; this module never qualifies teacher evidence.

Input reconstruction or construction-source affinity cannot be substituted for
these channel-region targets. A source-bound exporter must supply the masks.
No weights/total are selected here before the real target population is frozen.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .gse_structural_representation import ObservableRelationPrediction


@dataclass(frozen=True)
class ObservableRelationTargets:
    axis_abs_dot: torch.Tensor
    axis_known: torch.Tensor
    height_difference_m: torch.Tensor
    height_known: torch.Tensor
    section_log_ratio: torch.Tensor
    section_known: torch.Tensor
    correspondence: torch.Tensor
    correspondence_known: torch.Tensor
    evidence_refs: tuple[str,...]
    schema_version: str = 'observable_channel_relation_targets_v1'


def observable_relation_losses(prediction, targets):
    if type(prediction) is not ObservableRelationPrediction or type(targets) is not ObservableRelationTargets:
        raise ValueError('channel prediction and explicit observable channel targets required')
    if targets.schema_version!='observable_channel_relation_targets_v1':
        raise ValueError('source-affinity/input-reconstruction targets are not channel supervision')
    if not isinstance(targets.evidence_refs,tuple) or any(type(r) is not str or not r.strip() for r in targets.evidence_refs):
        raise ValueError('explicit immutable evidence references required')
    results={};counts={}
    fields=(('axis',prediction.axis_abs_dot,targets.axis_abs_dot,targets.axis_known),
            ('height',prediction.height_difference_m,targets.height_difference_m,targets.height_known),
            ('section',prediction.section_log_ratio,targets.section_log_ratio,targets.section_known),
            ('correspondence',prediction.correspondence_logits,targets.correspondence,targets.correspondence_known))
    for name,p,y,known in fields:
        if (y.shape!=p.shape or known.shape!=p.shape or known.dtype!=torch.bool
                or y.device!=p.device or known.device!=p.device):raise ValueError('target shape/device/mask mismatch')
        valid=prediction.computation_valid
        if p.ndim==4:valid=valid[...,None].expand_as(p)
        if bool((known&~valid).any()):raise ValueError('known relation on absent computation pair')
        a,b=p[known],y[known]
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise ValueError('nonfinite known relation')
        counts[name]=int(known.sum())
        if counts[name] and not targets.evidence_refs:raise ValueError('known target without provenance')
        if name=='axis' and bool(((b<0)|(b>1)).any()):raise ValueError('unsigned axis dot outside [0,1]')
        if name=='correspondence':
            if bool(((b!=0)&(b!=1)).any()):raise ValueError('known correspondence must be binary')
            terms=[F.binary_cross_entropy_with_logits(a[b==v],b[b==v].to(a.dtype)) for v in (0,1) if bool((b==v).any())]
            results[name]=torch.stack(terms).mean() if terms else a.sum()
            counts['correspondence_positive']=int((b==1).sum());counts['correspondence_negative']=int((b==0).sum())
        else:
            # Native units; a later frozen training spec must choose weights.
            results[name]=F.l1_loss(a,b.to(a.dtype)) if counts[name] else a.sum()
    return results,counts
