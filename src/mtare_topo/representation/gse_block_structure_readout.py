"""Shared block-shape/context readout; software interface, not trained result.

Outputs only the supervised geometry fields. Width, reachability, membership
and calibrated uncertainty are explicitly unavailable, not invented numbers.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .gse_window_surface_domain import window_surface_positions


@dataclass(frozen=True)
class BlockStructurePrediction:
    anchor_position_m: torch.Tensor
    anchor_presence_logits: torch.Tensor
    opening_position_m: torch.Tensor
    opening_presence_logits: torch.Tensor
    opening_direction: torch.Tensor
    observation_supported: bool
    unknown_fields: tuple=('event_type','dimensions','reachability','membership','calibrated_uncertainty')


class BlockStructureReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter=nn.Sequential(nn.Linear(263,128),nn.GELU(),nn.LayerNorm(128))
        self.queries=nn.Parameter(torch.randn(96,128)/128**.5)
        layer=nn.TransformerDecoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,2,norm=nn.LayerNorm(128))
        self.anchor=nn.Linear(128,4);self.opening=nn.Linear(128,7)

    def anchor_coordinates(self,raw):
        radius=torch.linalg.vector_norm(raw,dim=1,keepdim=True)
        return 10*raw/radius.clamp_min(1.)

    def forward(self,point_features,frozen_context):
        e=point_features['embedding'];c=point_features['centers_m'];s=point_features['extent_m'];d=point_features['degenerate']
        m=len(e)
        if (e.shape!=(m,128) or c.shape!=(m,3) or s.shape!=(m,3) or d.shape!=(m,)
                or frozen_context.shape!=(m,128) or m>4096 or d.dtype!=torch.bool):
            raise ValueError('shared bounded block features required')
        for t in (e,c,s,frozen_context):
            if t.device!=e.device or t.dtype!=e.dtype or not torch.isfinite(t).all():raise ValueError('feature dtype/device/numeric drift')
        if d.device!=e.device or torch.any(s<0):raise ValueError('invalid extent/degeneracy')
        if not m:
            return BlockStructurePrediction(e.new_zeros((32,3)),e.new_zeros(32),e.new_zeros((64,3)),
                e.new_zeros(64),e.new_zeros((64,3)),False)
        memory=self.adapter(torch.cat((e,frozen_context.detach(),c/10.,s/10.,d[:,None].to(e.dtype)),dim=1))
        q=self.decoder(self.queries[None],memory[None])[0]
        a=self.anchor(q[:32]);o=self.opening(q[32:])
        # Anchors remain independent metric locations inside the shared ROI.
        xyz=self.anchor_coordinates(a[:,:3])
        supported=torch.ones(1,dtype=torch.bool,device=e.device)
        opening=window_surface_positions(o[None,:,:3],supported)[0]
        direction=window_surface_positions(o[None,:,4:7],supported)[0]/10.
        return BlockStructurePrediction(xyz,a[:,3],opening,o[:,3],direction,True)
