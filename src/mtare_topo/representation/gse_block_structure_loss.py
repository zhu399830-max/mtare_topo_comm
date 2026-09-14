"""Geometry-only set matching; unknown tasks/regions have no gradient path.

This software objective must be frozen with any future experiment. It does not
authorize rerunning failed historical candidate experiments.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .gse_block_structure_readout import BlockStructurePrediction


@dataclass(frozen=True)
class LocatedStructureTargets:
    anchors_m: torch.Tensor
    openings_m: torch.Tensor
    opening_direction: torch.Tensor
    direction_known: torch.Tensor
    anchors_complete: bool
    openings_complete: bool


def located_structure_loss(prediction,target):
    if type(prediction)!=BlockStructurePrediction or type(target)!=LocatedStructureTargets:raise ValueError('typed geometry-only contract required')
    if type(prediction.observation_supported)!=bool:raise ValueError('explicit observation support required')
    zero=prediction.anchor_position_m.new_zeros((),requires_grad=True)
    terms={};counts={};assignments={}
    for kind,capacity in (('anchor',32),('opening',64)):
        xyz=getattr(prediction,kind+'_position_m');logits=getattr(prediction,kind+'_presence_logits')
        truth=target.anchors_m if kind=='anchor' else target.openings_m
        complete=target.anchors_complete if kind=='anchor' else target.openings_complete
        if type(complete)!=bool:raise ValueError('explicit region completeness required')
        if xyz.shape!=(capacity,3) or logits.shape!=(capacity,) or truth.ndim!=2 or truth.shape[1]!=3:
            raise ValueError('set shape mismatch')
        if len(truth)>capacity:raise OverflowError('target capacity, never truncate')
        if truth.requires_grad or truth.dtype!=xyz.dtype or truth.device!=xyz.device:raise ValueError('detached same-device targets required')
        if any(not torch.isfinite(t).all() for t in (xyz,logits,truth)):raise ValueError('nonfinite known geometry')
        if len(torch.unique(truth,dim=0))!=len(truth):raise ValueError('ambiguous duplicate target positions')
        if not prediction.observation_supported and len(truth):raise ValueError('positive targets without observations')
        tolerance=64*torch.finfo(xyz.dtype).eps*10
        if len(truth) and torch.any(torch.linalg.vector_norm(truth,dim=1)>10.+tolerance):raise ValueError('target outside fixed ROI')
        if kind=='opening' and prediction.observation_supported:
            for value in (xyz,truth):
                if not torch.allclose(torch.linalg.vector_norm(value,dim=1),value.new_full((len(value),),10.),atol=tolerance,rtol=0):
                    raise ValueError('openings must use declared window surface domain')
        ti=torch.empty(0,dtype=torch.long,device=xyz.device);qi=ti
        if len(truth):
            cost=torch.cdist(truth.detach().double(),xyz.detach().double())
            t,q=linear_sum_assignment(cost.cpu().numpy())
            ti=torch.as_tensor(t,device=xyz.device);qi=torch.as_tensor(q,device=xyz.device)
        assignments[kind]=(ti,qi)
        positive=F.binary_cross_entropy_with_logits(logits[qi],torch.ones_like(logits[qi]),reduction='sum') if len(qi) else zero
        negative=torch.ones(capacity,dtype=torch.bool,device=xyz.device);negative[qi]=False
        if not (complete and prediction.observation_supported):negative[:]=False
        negative_count=int(negative.sum())
        background=F.binary_cross_entropy_with_logits(logits[negative],torch.zeros_like(logits[negative]),reduction='sum') if negative_count else zero
        terms[kind+'_presence']=(positive+.1*background)/max(1.,len(qi)+.1*negative_count)
        terms[kind+'_position']=torch.linalg.vector_norm(xyz[qi]-truth[ti],dim=1).mean()/10. if len(qi) else zero
        counts[kind+'_positive']=len(qi);counts[kind+'_negative']=negative_count
    dirs=target.opening_direction;mask=target.direction_known
    if dirs.shape!=target.openings_m.shape or mask.shape!=(len(dirs),) or mask.dtype!=torch.bool or mask.device!=dirs.device:
        raise ValueError('direction validity required')
    if dirs.requires_grad or dirs.device!=target.openings_m.device or dirs.dtype!=target.openings_m.dtype:
        raise ValueError('detached direction target required')
    known=dirs[mask]
    predicted_direction=prediction.opening_direction
    if predicted_direction.shape!=(64,3) or not torch.isfinite(predicted_direction).all():raise ValueError('invalid predicted direction')
    if prediction.observation_supported and not torch.allclose(torch.linalg.vector_norm(predicted_direction,dim=1),predicted_direction.new_ones(64),atol=64*torch.finfo(predicted_direction.dtype).eps,rtol=0):
        raise ValueError('predicted directions must be unit vectors')
    if not torch.isfinite(known).all() or not torch.allclose(torch.linalg.vector_norm(known,dim=1),torch.ones(len(known),device=dirs.device,dtype=dirs.dtype),atol=1e-5,rtol=0):
        raise ValueError('known directions must be unit vectors')
    ti,qi=assignments['opening'];use=mask[ti]
    terms['opening_direction']=(1-(prediction.opening_direction[qi[use]]*dirs[ti[use]]).sum(1).clamp(-1,1)).mean()/2 if use.any() else zero
    counts['opening_direction']=int(use.sum())
    return dict(total=sum(terms.values()),terms=terms,counts=counts,assignments=assignments,
                has_supervision=any(counts.values()))
