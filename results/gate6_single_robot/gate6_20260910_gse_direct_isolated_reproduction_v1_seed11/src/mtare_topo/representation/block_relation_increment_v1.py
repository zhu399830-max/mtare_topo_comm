"""Same-block, matched-neighbor explicit geometry increment.

Reuses the existing three message layers and trained-output head. C0 retains
coordinate-bearing observations and the same k8 neighbors; it is NOT geometry
free. Nine attributes are deltaXYZ/10, distance/10, deltaExtent/10, and two
degeneracy flags. They are not normals, ray evidence, openings or graph edges.
Neither this module nor its coordinate tie-break is rotation equivariant.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .block_anchor_branch import BlockAnchorBranch
from .gse_surface_relation_model_v1 import _MessageLayer


@dataclass(frozen=True)
class BlockRelationBatch:
    valid: torch.Tensor
    neighbor_index: torch.Tensor
    neighbor_valid: torch.Tensor
    relation: torch.Tensor


def block_relations(centers, extents, degenerate):
    m=len(centers)
    if (centers.shape!=(m,3) or extents.shape!=(m,3) or degenerate.shape!=(m,)
            or m>4096 or centers.dtype not in (torch.float32,torch.float64)
            or extents.dtype!=centers.dtype or degenerate.dtype!=torch.bool
            or extents.device!=centers.device or degenerate.device!=centers.device
            or not torch.isfinite(centers).all() or not torch.isfinite(extents).all()
            or torch.any(extents<0)):
        raise ValueError('bounded finite block geometry required')
    c=centers.detach().cpu().numpy()
    # Geometrically indistinguishable centers cannot receive an ID-based tie-break.
    if len(np.unique(c,axis=0))!=m:raise ValueError('coincident centers: ambiguous fixed-k neighborhood')
    ids=np.full((m,8),-1,np.int64)
    for i in range(m):
        distance=np.linalg.norm(c-c[i],axis=1)
        order=np.lexsort((c[:,2],c[:,1],c[:,0],distance))
        chosen=order[order!=i][:8];ids[i,:len(chosen)]=chosen
    index=torch.as_tensor(ids,device=centers.device);mask=index>=0;safe=index.clamp_min(0)
    if m:
        delta=centers[safe]-centers[:,None]
        ds=extents[safe]-extents[:,None]
        flags=torch.stack((degenerate[:,None].expand(-1,8),degenerate[safe]),-1).to(centers.dtype)
        rel=torch.cat((delta/10,torch.linalg.vector_norm(delta,dim=-1,keepdim=True)/10,ds/10,flags),-1)
        rel=torch.where(mask[...,None],rel,0.)
    else:rel=centers.new_zeros(0,8,9)
    return BlockRelationBatch(torch.ones(1,m,dtype=torch.bool,device=centers.device),
        index[None],mask[None],rel[None].detach())


class BlockRelationIncrementV1(nn.Module):
    def __init__(self, existing_head, *, relation_attributes):
        super().__init__()
        if type(existing_head) is not BlockAnchorBranch or type(relation_attributes) is not bool:
            raise ValueError('existing anchor/branch head and explicit paired control required')
        self.head=deepcopy(existing_head)
        self.relation_attributes=relation_attributes
        self.layers=nn.ModuleList(_MessageLayer() for _ in range(3)).to(
            device=self.head.queries.device,dtype=self.head.queries.dtype)

    def forward(self, point_features, frozen_context):
        if set(point_features)!={'embedding','centers_m','extent_m','degenerate'}:
            raise ValueError('only existing observation fields allowed')
        e=point_features['embedding']
        if e.ndim!=2 or e.shape[1]!=128 or not torch.isfinite(e).all():
            raise ValueError('finite Mx128 block embeddings required')
        patch=block_relations(point_features['centers_m'],point_features['extent_m'],point_features['degenerate'])
        if e.shape[0]!=patch.valid.shape[1]:raise ValueError('block/embedding mismatch')
        if not len(e):return self.head(point_features,frozen_context)
        if not self.relation_attributes:
            patch=BlockRelationBatch(patch.valid,patch.neighbor_index,patch.neighbor_valid,torch.zeros_like(patch.relation))
        feature=e[None]
        for layer in self.layers:feature=layer(feature,patch,True)
        return self.head(dict(point_features,embedding=feature[0]),frozen_context)
