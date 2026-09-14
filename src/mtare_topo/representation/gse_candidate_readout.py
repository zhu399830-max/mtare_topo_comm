"""Shared fixed-candidate readout reusing existing sparse message layers.

No candidate sampling, target input, position refinement, NMS or graph update.
A also shares observation-derived geometric candidate positions; it is not a
geometry-free system. Identical parameter layouts do not imply equal FLOPs.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .gse_surface_relation_model_v1 import DIM, SurfacePatchBatch, _MessageLayer


@dataclass(frozen=True)
class CandidateGeometryBatch(SurfacePatchBatch):
    unary_known: torch.Tensor


@dataclass(frozen=True)
class CandidatePrediction:
    position_m: torch.Tensor
    presence_logits: torch.Tensor
    valid: torch.Tensor


class SharedCandidateReadout(nn.Module):
    def __init__(self, path):
        super().__init__()
        if path not in ('A','B','C'): raise ValueError('A/B/C required')
        self.path=path
        self.observation=nn.Sequential(nn.Linear(DIM+3,DIM),nn.GELU(),nn.Linear(DIM,DIM))
        self.geometry=nn.Sequential(nn.Linear(36,DIM),nn.GELU(),nn.Linear(DIM,DIM))
        self.layers=nn.ModuleList(_MessageLayer() for _ in range(3))
        self.presence=nn.Linear(DIM,1)

    def forward(self, positions_m, observation_features, geometry):
        if not isinstance(geometry,CandidateGeometryBatch): raise ValueError('typed geometry with known masks required')
        if positions_m.ndim!=3 or positions_m.shape[-1]!=3 or not 1<=positions_m.shape[1]<=4592:
            raise ValueError('B,N,3 positions,1..4592 candidates required; never truncate')
        b,n,_=positions_m.shape
        if b<1 or observation_features.shape!=(b,n,DIM): raise ValueError('observation feature shape')
        if positions_m.dtype not in (torch.float32,torch.float64): raise ValueError('floating positions required')
        for value,shape in ((observation_features,(b,n,DIM)),(geometry.unary,(b,n,18)),(geometry.relation,(b,n,8,9))):
            if value.shape!=shape or value.dtype!=positions_m.dtype or value.device!=positions_m.device:
                raise ValueError('floating shape/dtype/device mismatch')
            if not torch.isfinite(value).all(): raise ValueError('nonfinite feature')
        if not torch.isfinite(positions_m).all(): raise ValueError('nonfinite positions')
        for value,shape in ((geometry.valid,(b,n)),(geometry.neighbor_valid,(b,n,8)),(geometry.unary_known,(b,n,18))):
            if value.shape!=shape or value.dtype!=torch.bool or value.device!=positions_m.device:
                raise ValueError('explicit boolean mask required')
        index=geometry.neighbor_index
        if index.shape!=(b,n,8) or index.dtype!=torch.long or index.device!=positions_m.device:
            raise ValueError('neighbor indices required')
        if ((index < -1)|(index >= n)).any(): raise ValueError('neighbor out of bounds')
        mask=geometry.neighbor_valid
        if (mask & (index<0)).any(): raise ValueError('valid edge needs index')
        destination=geometry.valid[torch.arange(b,device=index.device)[:,None,None],index.clamp_min(0)]
        if (mask & (~geometry.valid[...,None]|~destination)).any():
            raise ValueError('neighbor may not reference invalid candidate')
        x=self.observation(torch.cat((positions_m.detach()/10.,observation_features.detach()),-1))
        # Zero inputs rather than bypassed modules preserve paired layouts and
        # defined zero gradients. A cannot read explicit unary attributes.
        unary=torch.cat((torch.where(geometry.unary_known,geometry.unary.detach(),0.),geometry.unary_known.to(positions_m.dtype)),dim=-1)
        if self.path=='A':unary=torch.zeros_like(unary)
        x=x+self.geometry(unary)
        x=torch.where(geometry.valid[...,None],x,0.)
        for layer in self.layers:x=layer(x,geometry,self.path=='C')
        logits=self.presence(x).squeeze(-1)
        return CandidatePrediction(positions_m.detach().clone(),torch.where(geometry.valid,logits,0.),geometry.valid.clone())
