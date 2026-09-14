"""Versioned window detection training only; independent scoring is unchanged.

Fixed normalized distance - presence probability assignment; no-object weight
0.1. Not a drop-in real-world loss: known size/reach/membership targets reject.
"""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .gse_surface_losses_v1 import _validate,SurfaceLossResult,LOSS_NAMES
from .gse_window_surface_domain import complete_window_query_mask

NO_OBJECT_WEIGHT=.1


def training_assignment(positions,logits,target_positions,target_valid):
    mapping=torch.full(target_valid.shape,-1,dtype=torch.long,device=positions.device)
    for b in range(len(positions)):
        ids=torch.nonzero(target_valid[b]).flatten()
        if len(ids)>positions.shape[1]:raise OverflowError('target capacity exceeded')
        if not len(ids):continue
        truth=target_positions[b,ids]
        if len(torch.unique(truth,dim=0))!=len(truth):raise ValueError('duplicate target centers')
        cost=torch.cdist(truth.detach().double(),positions[b].detach().double())/10.-logits[b].detach().double().sigmoid()[None]
        rows,cols=linear_sum_assignment(cost.cpu().numpy())
        mapping[b,ids[torch.as_tensor(rows,device=ids.device)]]=torch.as_tensor(cols,device=ids.device)
    return mapping


def window_set_loss(prediction,target):
    _validate(prediction,target)
    if target.dimension_valid.any() or target.reachability_valid.any() or target.membership_valid.any():
        raise ValueError('this version supports located anchors/openings/directions only; never silently drop known attributes')
    if (target.score_region_center_m!=0).any() or (target.score_region_radius_m!=10).any():
        raise ValueError('fixed current-sensor 10m window required')
    for xyz,mask in ((prediction.opening_position_m,prediction.observation_supported[:,None].expand(-1,64)),
                     (target.opening_position_m,target.opening_valid)):
        radius=torch.linalg.vector_norm(xyz[mask],dim=-1)
        if not torch.allclose(radius,torch.full_like(radius,10.),atol=64*torch.finfo(xyz.dtype).eps*10,rtol=0):
            raise ValueError('explicit sphere-surface opening positions required')
    zero=torch.zeros((),dtype=prediction.anchor_position_m.dtype,device=prediction.anchor_position_m.device,requires_grad=True)
    numerator={name:zero for name in LOSS_NAMES};counts={name:0 for name in LOSS_NAMES};weight_sum={}
    maps={}
    for kind in ('anchor','opening'):
        xyz=getattr(prediction,kind+'_position_m');logits=getattr(prediction,kind+'_presence_logits')
        truth=getattr(target,kind+'_position_m');valid=getattr(target,kind+'_valid')
        mapping=training_assignment(xyz,logits,truth,valid);maps[kind]=mapping
        complete=getattr(target,kind+'_region_complete')
        for b in range(len(xyz)):
            ids=torch.nonzero(mapping[b]>=0).flatten();matched=mapping[b,ids]
            negative=complete_window_query_mask(prediction.observation_supported[b:b+1],complete[b:b+1],xyz.shape[1])[0].clone()
            if kind=='anchor':negative &= torch.linalg.vector_norm(xyz[b].detach(),dim=-1)<=10
            negative[matched]=False
            name=kind+'_presence';weights=weight_sum.get(name,0.)
            if len(matched):
                numerator[name]=numerator[name]+F.binary_cross_entropy_with_logits(logits[b,matched],torch.ones_like(logits[b,matched]),reduction='sum')
                counts[name]+=len(matched);weights+=len(matched)
                pos=kind+'_position';numerator[pos]=numerator[pos]+torch.linalg.vector_norm(xyz[b,matched]-truth[b,ids],dim=-1).sum()/10.
                counts[pos]+=len(matched)
            n=int(negative.sum())
            if n:
                numerator[name]=numerator[name]+NO_OBJECT_WEIGHT*F.binary_cross_entropy_with_logits(logits[b,negative],torch.zeros_like(logits[b,negative]),reduction='sum')
                counts[name]+=n;weights+=NO_OBJECT_WEIGHT*n
            weight_sum[name]=weights
            if kind=='opening':
                for i,j in zip(ids.tolist(),matched.tolist()):
                    if bool(target.direction_valid[b,i]):
                        cosine=(prediction.opening_direction[b,j]*target.opening_direction[b,i]).sum().clamp(-1.,1.)
                        numerator['opening_direction']=numerator['opening_direction']+(1-cosine)/2
                        counts['opening_direction']+=1
    denominators={name:weight_sum.get(name,counts[name]) for name in LOSS_NAMES}
    terms={name:numerator[name]/(denominators[name] if denominators[name]>0 else 1.) for name in LOSS_NAMES}
    return SurfaceLossResult(sum(terms.values()),terms,counts,maps,any(counts.values()))
