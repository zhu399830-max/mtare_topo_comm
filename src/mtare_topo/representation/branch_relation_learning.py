"""Direct ray relations with teacher-free local geometry attention.

No coarse-sector pooling, section identity, center query, or physical edge
prediction. Source-frame context is metadata, never a learned feature.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .gse_local_surface_affinity import LocalSurfaceAffinity
from .gse_structural_representation import StructuralPatchInput
from .gse_structure_context import CausalFrameOrderContext


@dataclass(frozen=True, slots=True)
class LocalBranchEvidence:
    ray_ids: torch.Tensor
    endpoints_m: torch.Tensor
    observed_features: torch.Tensor
    patch_observed_features: torch.Tensor
    patches: StructuralPatchInput
    context: CausalFrameOrderContext

    def validate(self):
        n=len(self.ray_ids)
        if self.ray_ids.shape!=(n,) or self.ray_ids.dtype!=torch.long or self.endpoints_m.shape!=(n,3) or self.observed_features.shape!=(n,128):raise ValueError('ray layout')
        if type(self.context) is not CausalFrameOrderContext or type(self.patches) is not StructuralPatchInput:raise ValueError('bound causal observation required')
        if self.ray_ids.unique().numel()!=n or bool(((self.ray_ids<0)|(self.ray_ids>=11520)).any()):raise ValueError('unique original current-frame ray IDs')
        if self.patch_observed_features.ndim!=3 or self.patch_observed_features.shape[0]!=1 or self.patch_observed_features.shape[-1]!=128:raise ValueError('single patch population')
        tensors=[self.endpoints_m,self.observed_features,self.patch_observed_features]
        if any(v.device!=self.ray_ids.device or not bool(torch.isfinite(v).all()) for v in tensors):raise ValueError('device or finite observation')
        if any(v.dtype!=self.endpoints_m.dtype for v in tensors):raise ValueError('observation dtype')
        if bool((self.endpoints_m.norm(dim=-1)<=0).any()):raise ValueError('positive valid return range')
        if not torch.equal(self.patches.coordinate_scales_m,self.endpoints_m.new_tensor([[.5,10.]])):raise ValueError('fixed geometric domain')


@dataclass(frozen=True, slots=True)
class BranchRelationTargets:
    values: torch.Tensor
    known: torch.Tensor
    evidence_refs: tuple[str,...]
    training_qualified: bool = False


@dataclass(frozen=True, slots=True)
class RayRelationPrediction:
    ray_tokens: torch.Tensor
    pair_indices: torch.Tensor
    logits: torch.Tensor
    nearest_patch_indices: torch.Tensor
    geometric_support: torch.Tensor


@torch.no_grad()
def nearest_segment_patches(endpoints,centers,valid,k=8):
    """Exact point-to-clipped-current-ray-segment distance, bounded chunks."""
    if endpoints.ndim!=2 or endpoints.shape[1]!=3 or centers.ndim!=2 or centers.shape[1]!=3 or valid.shape!=(len(centers),):raise ValueError('geometry layout')
    ids=torch.nonzero(valid,as_tuple=False).flatten();count=min(k,len(ids));rows=[]
    if count==0:return torch.empty((len(endpoints),0),device=endpoints.device,dtype=torch.long)
    c=centers[ids]
    for chunk in endpoints.detach().split(128):
        length=chunk.norm(dim=-1);direction=chunk/length[:,None]
        projection=(direction@c.T).clamp_min(0)
        projection=torch.minimum(projection,length.clamp_max(10)[:,None])
        dist=((c[None]-projection[:,:,None]*direction[:,None])**2).sum(-1)
        rows.append(ids[torch.argsort(dist,dim=1,stable=True)[:,:count]])
    return torch.cat(rows) if rows else torch.empty((0,count),device=endpoints.device,dtype=torch.long)


def observed_ray_pairs(ray_ids, *, policy='local_v1'):
    """Fixed local/multiscale layout; no teacher pair selection in forward."""
    ids=ray_ids.detach().cpu().tolist();lookup={v:i for i,v in enumerate(ids)};pairs=set()
    if len(lookup)!=len(ids) or any(not isinstance(v,int) or not 0<=v<16*720 for v in ids):raise ValueError('unique current-frame ray IDs required')
    if policy=='local_v1':offsets=((0,1),(1,0),(0,8),(0,32),(0,128))
    elif policy=='dyadic_v2':
        # Uniform powers of two up to each image dimension, plus the azimuth
        # half-turn. Fixes the old64deg ceiling without looking at any labels.
        offsets=tuple((0,2**p) for p in range(10))+((0,360),)+tuple((2**p,0) for p in range(4))
    else:raise ValueError('unknown pair policy')
    for v,i in lookup.items():
        r,c=divmod(v,720)
        for dr,dc in offsets:
            rr=r+dr
            if 0<=rr<16:
                j=lookup.get(rr*720+(c+dc)%720)
                if j is not None and i!=j:pairs.add(tuple(sorted((i,j))))
    return torch.tensor(sorted(pairs),device=ray_ids.device,dtype=torch.long).reshape(-1,2)


class RayBranchRelationModel(nn.Module):
    def __init__(self,variant):
        super().__init__()
        self.variant=variant;self.patch_encoder=LocalSurfaceAffinity(variant);del self.patch_encoder.score
        self.ray_adapter=nn.Sequential(nn.Linear(135,128),nn.GELU(),nn.LayerNorm(128))
        self.query=nn.Linear(128,128,bias=False);self.key=nn.Linear(128,128,bias=False);self.value=nn.Linear(128,128,bias=False)
        self.fusion=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.LayerNorm(128))
        self.relation=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,evidence,pair_indices):
        if type(evidence) is not LocalBranchEvidence:raise ValueError('student observation type required')
        evidence.validate();e=evidence;p=e.patches.batch
        if pair_indices.ndim!=2 or pair_indices.shape[1]!=2 or pair_indices.dtype!=torch.long or pair_indices.device!=e.ray_ids.device:raise ValueError('pair layout')
        if bool(((pair_indices<0)|(pair_indices>=len(e.ray_ids))).any()):raise ValueError('pair index')
        patches,_=self.patch_encoder.encode_tokens(e.patch_observed_features,p)
        ranges=e.endpoints_m.norm(dim=-1,keepdim=True)
        raw=torch.cat((e.observed_features.detach(),e.endpoints_m.detach()/ranges.detach(),ranges.detach()/50,e.endpoints_m.detach()/50),-1)
        ray=self.ray_adapter(raw)
        centers=p.unary[0,:,:3].detach()*10
        nearest=nearest_segment_patches(e.endpoints_m,centers,p.valid[0])
        support=torch.full((len(ray),),nearest.shape[1]>0,device=ray.device,dtype=torch.bool)
        context=torch.zeros_like(ray)
        if nearest.shape[1]:
            gathered=patches[0][nearest]
            weights=(self.query(ray)[:,None]*self.key(gathered)).sum(-1)/(128**.5)
            context=(weights.softmax(-1)[...,None]*self.value(gathered)).sum(1)
        tokens=self.fusion(torch.cat((ray,context),-1))
        a,b=tokens[pair_indices[:,0]],tokens[pair_indices[:,1]]
        logits=self.relation(torch.cat((a+b,(a-b).abs()),-1)).squeeze(-1)
        return RayRelationPrediction(tokens,pair_indices,logits,nearest,support)


def branch_relation_loss(prediction,targets):
    if type(targets) is not BranchRelationTargets:raise ValueError('loss-only targets required')
    logits=prediction.logits
    if targets.known.shape!=logits.shape or targets.values.shape!=logits.shape or targets.known.dtype!=torch.bool or len(targets.evidence_refs)!=len(logits):raise ValueError('target layout')
    if targets.known.device!=logits.device or targets.values.device!=logits.device:raise ValueError('target device')
    if any(not targets.evidence_refs[i] for i in torch.nonzero(targets.known).flatten().tolist()):raise ValueError('known relation needs evidence')
    values=targets.values[targets.known];selected=logits[targets.known]
    if not bool(torch.isfinite(selected).all()) or bool(((values!=0)&(values!=1)).any()):raise ValueError('known finite binary targets')
    terms=[F.binary_cross_entropy_with_logits(selected[values==v],values[values==v]) for v in (0,1) if bool((values==v).any())]
    loss=torch.stack(terms).mean() if terms else selected.sum()
    return loss,dict(positive=int((values==1).sum()),negative=int((values==0).sum()),unknown=int((~targets.known).sum()))


def require_trainable_targets(targets):
    values=targets.values[targets.known]
    if not targets.training_qualified or not bool((values==0).any()) or not bool((values==1).any()):
        raise ValueError('qualified positive AND negative evidence required before fitting')
