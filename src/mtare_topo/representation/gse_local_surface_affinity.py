"""Center-free local surface-source affinity, NOT junction membership.

A consumes observed features and query coordinates; B adds unary geometry;
C additionally uses the existing sparse geometry message layers. All use the
same observation-derived patches/adjacency and parameter layout. No source ID,
teacher-selected candidate or structure reference is a forward argument.
This is the low-level organization task, not a topological-edge predictor.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .gse_surface_relation_model_v1 import _MessageLayer, SurfacePatchBatch


@dataclass(frozen=True)
class SurfaceAffinityPrediction:
    logits: torch.Tensor
    pair_valid: torch.Tensor
    neighbor_index: torch.Tensor
    is_structural_membership: bool = False


class LocalSurfaceAffinity(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in ('A','B','C'):raise ValueError('A/B/C required')
        self.variant=variant
        self.observed=nn.Sequential(nn.Linear(131,128),nn.GELU(),nn.LayerNorm(128))
        self.geometry=nn.Sequential(nn.Linear(18,128),nn.GELU(),nn.Linear(128,128))
        self.layers=nn.ModuleList([_MessageLayer() for _ in range(3)])
        self.score=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))

    def encode_tokens(self, observed_features, patch):
        """Shared teacher-free token path; does not run source-affinity scoring."""
        if type(patch) is not SurfacePatchBatch:raise ValueError('observation-derived patches required')
        x=observed_features
        if x.ndim!=3 or x.shape[-1]!=128:raise ValueError('B,M,128 observed features required')
        b,m,_=x.shape
        if not 1<=m<=4096 or patch.unary.shape!=(b,m,18):raise ValueError('patch capacity/shape')
        if (patch.valid.shape!=(b,m) or patch.valid.dtype!=torch.bool
                or patch.neighbor_index.shape!=(b,m,8) or patch.neighbor_index.dtype!=torch.long
                or patch.neighbor_valid.shape!=(b,m,8) or patch.neighbor_valid.dtype!=torch.bool
                or patch.relation.shape!=(b,m,8,9)):raise ValueError('patch layout')
        if any(t.device!=x.device for t in vars(patch).values()):raise ValueError('device mismatch')
        if patch.unary.dtype!=x.dtype or patch.relation.dtype!=x.dtype:raise ValueError('dtype mismatch')
        if (bool(((patch.neighbor_index < -1)|(patch.neighbor_index>=m)).any())
                or not torch.equal(patch.neighbor_valid,patch.neighbor_index>=0)):raise ValueError('adjacency mismatch')
        safe=patch.neighbor_index.clamp_min(0);batch=torch.arange(b,device=x.device)[:,None,None]
        valid=patch.neighbor_valid&patch.valid[...,None]&patch.valid[batch,safe]
        if not torch.equal(valid,patch.neighbor_valid):raise ValueError('padded endpoint')
        if any(not bool(torch.isfinite(t).all()) for t in (x,patch.unary,patch.relation)):
            raise ValueError('nonfinite observation')
        features=self.observed(torch.cat((x.detach(),patch.unary[...,:3].detach()),-1))
        if self.variant!='A':features=features+self.geometry(patch.unary.detach())
        features=torch.where(patch.valid[...,None],features,0.)
        # Same layers/shape in every branch; A/B self updates, C neighbor messages.
        for layer in self.layers:features=layer(features,patch,self.variant=='C')
        return features,valid

    def forward(self, observed_features, patch):
        features,valid=self.encode_tokens(observed_features,patch)
        b=features.shape[0];safe=patch.neighbor_index.clamp_min(0)
        batch=torch.arange(b,device=features.device)[:,None,None]
        neighbor=features[batch,safe];sender=features[:,:,None,:].expand_as(neighbor)
        pair=torch.cat((sender+neighbor,torch.abs(sender-neighbor)),-1)
        logits=self.score(pair).squeeze(-1)
        return SurfaceAffinityPrediction(torch.where(valid,logits,0.),valid,patch.neighbor_index)


def affinity_loss(prediction, targets, known):
    """Loss-side labels only. Unknown is masked before checking values.

    Do not derive labels from same component, adjacency or patch indices.
    Only independently bound pure-source patches may receive known labels.
    Counts refer to directed adjacency entries, not independent scenes/pairs.
    """
    logits=prediction.logits
    if targets.shape!=logits.shape or known.shape!=logits.shape or known.dtype!=torch.bool:
        raise ValueError('same-shape targets and explicit bool known mask required')
    if known.device!=logits.device or targets.device!=logits.device:raise ValueError('device mismatch')
    if bool((known&~prediction.pair_valid).any()):raise ValueError('known label on invalid pair')
    values=targets[known];selected=logits[known]
    if not bool(torch.isfinite(values).all()) or bool(((values!=0)&(values!=1)).any()):
        raise ValueError('known labels must be binary')
    if not bool(torch.isfinite(selected).all()):raise ValueError('nonfinite known prediction')
    terms=[]
    for value in (0,1):
        mask=values==value
        if bool(mask.any()):terms.append(F.binary_cross_entropy_with_logits(selected[mask],values[mask].to(logits.dtype)))
    loss=torch.stack(terms).mean() if terms else selected.sum()
    return loss,{'positive_entries':int((values==1).sum()),'negative_entries':int((values==0).sum()),
                 'unknown_entries':int((prediction.pair_valid&~known).sum())}
