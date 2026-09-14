"""Implicit local geometry representation; no center/identity/global-edge head.

Reuses the existing 128D, three-layer observation/geometry path. Predictions
are hypotheses, not certified relations. Source-affinity labels are NOT valid
substitutes for the independent channel-relation targets these heads require.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .gse_local_surface_affinity import LocalSurfaceAffinity
from .gse_structure_context import CausalStructureContext, CausalFrameOrderContext
from .gse_surface_relation_model_v1 import SurfacePatchBatch,collate_surface_patches


@dataclass(frozen=True)
class StructuralPatchInput:
    batch: SurfacePatchBatch
    coordinate_scales_m: torch.Tensor


def bind_structural_patches(patches, *, device='cpu', dtype=torch.float32):
    """Keep metric provenance without changing the legacy patch/cache schema."""
    batch=collate_surface_patches(patches,device=device,dtype=dtype)
    scales=torch.tensor([[p.voxel_size_m,p.roi_radius_m] for p in patches],device=device,dtype=dtype)
    return StructuralPatchInput(batch,scales)


@dataclass(frozen=True)
class ObservableRelationPrediction:
    neighbor_index: torch.Tensor
    computation_valid: torch.Tensor
    axis_abs_dot: torch.Tensor
    height_difference_m: torch.Tensor
    section_log_ratio: torch.Tensor
    correspondence_logits: torch.Tensor
    regression_scale: torch.Tensor
    uncertainty_calibrated: bool = False
    connectivity_certified: bool = False
    axis_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class StructuralRepresentation:
    contexts: tuple[CausalStructureContext | CausalFrameOrderContext, ...]
    local_tokens: torch.Tensor
    region_embedding: torch.Tensor
    token_valid: torch.Tensor
    region_valid: torch.Tensor
    support_centers_m: torch.Tensor
    support_extents_m: torch.Tensor
    frame_support: torch.Tensor
    relations: ObservableRelationPrediction
    schema_version: str = 'structural_representation_v1'
    is_place_identity: bool = False


class GeometryStructureEncoder(nn.Module):
    """A observed+XYZ, B adds unary geometry, C adds sparse messages/relations.

    The source score head is removed. Same parameter layouts across A/B/C do
    not imply equal FLOPs. No normalization or rotation invariance is claimed
    for this coordinate-based backbone. All patch tokens remain present.
    """
    def __init__(self, variant):
        super().__init__()
        self.backbone = LocalSurfaceAffinity(variant)
        del self.backbone.score
        self.relation_head = nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,9))

    def forward(self, observed_features, patch_input, contexts):
        if (not isinstance(contexts,tuple) or len(contexts)!=observed_features.shape[0]
                or any(type(c) not in (CausalStructureContext, CausalFrameOrderContext) for c in contexts)):
            raise ValueError('one causal context per observation required')
        if type(patch_input) is not StructuralPatchInput:
            raise ValueError('bound patch input with scales required')
        patches=patch_input.batch
        scales=patch_input.coordinate_scales_m
        if (not isinstance(scales,torch.Tensor) or scales.shape!=(len(contexts),2)
                or scales.device!=observed_features.device or scales.dtype!=observed_features.dtype
                or not torch.equal(scales,scales.new_tensor([[.5,10.]]).expand_as(scales))):
            raise ValueError('bound .5m voxel / 10m ROI scales required; legacy unbound geometry rejected')
        tokens, pair_valid = self.backbone.encode_tokens(observed_features, patches)
        b,m,_=tokens.shape
        support=patches.valid
        count=support.sum(1,keepdim=True)
        pooled=tokens.sum(1)/count.clamp_min(1)
        region_valid=count[:,0]>0
        pooled=torch.where(region_valid[:,None],F.normalize(pooled,dim=-1),0.)
        safe=patches.neighbor_index.clamp_min(0)
        neighbor=tokens[torch.arange(b,device=tokens.device)[:,None,None],safe]
        sender=tokens[:,:,None,:].expand_as(neighbor)
        raw=self.relation_head(torch.cat((sender,neighbor,neighbor-sender),dim=-1))
        reverse=self.relation_head(torch.cat((neighbor,sender,sender-neighbor),dim=-1))
        symmetric=(raw+reverse)/2
        antisymmetric=(raw-reverse)/2
        def masked(value):
            mask=pair_valid if value.ndim==3 else pair_valid[...,None]
            return torch.where(mask,value,0.)
        relations=ObservableRelationPrediction(
            patches.neighbor_index.clone(),pair_valid,
            masked(symmetric[...,0].sigmoid()),masked(antisymmetric[...,1]),masked(antisymmetric[...,2:4]),
            masked(symmetric[...,4]),masked(F.softplus(symmetric[...,5:9])),axis_logits=masked(symmetric[...,0]))
        # Units come from the unchanged .5m / 10m observation extraction.
        return StructuralRepresentation(contexts,tokens,pooled,support.clone(),region_valid,
            torch.where(support[...,None],patches.unary[...,:3].detach()*scales[:,None,1:2],0.),
            torch.where(support[...,None],patches.unary[...,7:10].detach()*scales[:,None,0:1],0.),
            (patches.unary[...,12:17].detach()>0)&support[...,None],relations)
