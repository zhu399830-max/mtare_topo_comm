"""Observation-anchored detection, independently adapted from 3DETR mechanisms.

See docs/GSE_OBSERVATION_ANCHORED_EXECUTION_V1.md for upstream provenance.
No boxes, teacher positions, IDs, or learned free position queries enter forward.
Source indices address the ordered, bounded BlockPoints.xyz_m input, not GT IDs.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .gse_block_points import BlockPoints
from .observed_query_sampling_v1 import sample_observed_queries
from .block_relation_increment_v1 import BlockRelationBatch, block_relations
from .gse_surface_relation_model_v1 import _MessageLayer
from .anchor_branch_readout import AnchorBranchReadout
from .anchor_branch_loss import AnchorBranchPrediction


@dataclass(frozen=True)
class ObservedAnchorPrediction:
    prediction: AnchorBranchPrediction
    query_source_indices: torch.Tensor
    query_positions_m: torch.Tensor
    residual_m: torch.Tensor


def residual_positions(query_positions, raw_residual):
    """Domain diameter bounds each residual; Euclidean projection fixes support."""
    residual = 20. * torch.tanh(raw_residual)
    position = query_positions + residual
    # Compute the Euclidean projection in float64 before the final output cast:
    # float32 normalization can exceed the unchanged scorer's 1e-6 tolerance.
    precise=position.double()
    scale=(torch.linalg.vector_norm(precise,dim=-1,keepdim=True)/10.).clamp_min(1.)
    return (precise/scale).to(position.dtype), residual


class ObservedAnchorDetectorV1(nn.Module):
    def __init__(self, *, relation_attributes, branch_slots=64):
        super().__init__()
        if type(relation_attributes) is not bool:
            raise ValueError('explicit paired relation control required')
        self.relation_attributes = relation_attributes
        self.layers = nn.ModuleList(_MessageLayer() for _ in range(3))
        self.adapter = nn.Sequential(nn.Linear(263,128),nn.GELU(),nn.LayerNorm(128))
        self.position_encoding = nn.Sequential(nn.Linear(3,128),nn.GELU(),nn.Linear(128,128))
        layer = nn.TransformerDecoderLayer(128,4,256,dropout=0.,activation='gelu',
                                           batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,2,norm=nn.LayerNorm(128))
        self.head = AnchorBranchReadout(branch_slots=branch_slots)

    def forward(self, blocks, point_features, frozen_context):
        if type(blocks) is not BlockPoints or set(point_features) != {
                'embedding','centers_m','extent_m','degenerate'}:
            raise ValueError('only authenticated observation block fields allowed')
        e=point_features['embedding']; c=point_features['centers_m']
        s=point_features['extent_m']; d=point_features['degenerate']; m=len(e)
        weight=self.head.anchor.weight
        if (e.shape!=(m,128) or c.shape!=(m,3) or s.shape!=(m,3) or d.shape!=(m,)
                or frozen_context.shape!=(m,128) or m>4096 or d.dtype!=torch.bool
                or len(blocks.centers_m)!=m or d.device!=weight.device):
            raise ValueError('bounded aligned block features required')
        for value in (e,c,s,frozen_context):
            if value.dtype!=weight.dtype or value.device!=weight.device or not torch.isfinite(value).all():
                raise ValueError('feature dtype/device/numeric drift')
        if not np.array_equal(c.detach().cpu().numpy(),blocks.centers_m):
            raise ValueError('point/block coordinates do not correspond')
        indices=sample_observed_queries(blocks.xyz_m)
        if not len(indices):
            if m: raise ValueError('blocks without observations')
            return None
        point_to_block=np.asarray(blocks.point_to_block)
        if (point_to_block.shape!=(len(blocks.xyz_m),) or point_to_block.dtype.kind not in 'iu'
                or np.any(point_to_block<0) or np.any(point_to_block>=m)):
            raise ValueError('every observed point must map to a real block')
        relation=block_relations(c,s,d)
        if not self.relation_attributes:
            relation=BlockRelationBatch(relation.valid,relation.neighbor_index,
                                       relation.neighbor_valid,torch.zeros_like(relation.relation))
        feature=e[None]
        for layer in self.layers: feature=layer(feature,relation,True)
        memory=self.adapter(torch.cat((feature[0],frozen_context.detach(),c/10.,s/10.,
                                       d[:,None].to(e.dtype)),dim=-1))
        source=torch.as_tensor(indices,device=e.device,dtype=torch.long)
        xyz=torch.as_tensor(blocks.xyz_m[indices].copy(),device=e.device,dtype=e.dtype)
        parent=torch.as_tensor(point_to_block[indices].copy(),device=e.device,dtype=torch.long)
        query=memory[parent]+self.position_encoding(xyz/10.)
        decoded=self.decoder(query[None],(memory+self.position_encoding(c/10.))[None])[0]
        anchor=self.head.anchor(decoded)
        branch=self.head.branch(self.head.branch_adapter(
            decoded[:,None]+self.head.branch_slots_embedding[None]))
        norms=torch.linalg.vector_norm(branch[...,:3],dim=-1,keepdim=True)
        if (not torch.isfinite(anchor).all() or not torch.isfinite(branch).all()
                or torch.any(norms<=32*torch.finfo(e.dtype).eps)):
            raise ValueError('nonfinite output or degenerate branch direction')
        position,residual=residual_positions(xyz,anchor[:,:3])
        prediction=AnchorBranchPrediction(position,anchor[:,3],branch[...,:3]/norms,branch[...,3])
        return ObservedAnchorPrediction(prediction,source,xyz,residual)
