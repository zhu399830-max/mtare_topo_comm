"""Small head for existing 128D anchor queries, not a new encoder.

Branch slots are explicit experiment configuration. No teacher candidates,
IDs, aperture centers, dimensions or traversability enter this forward API.
"""
import torch
from torch import nn
from .anchor_branch_loss import AnchorBranchPrediction
from .gse_block_structure_readout_v2 import smooth_anchor_coordinates


class AnchorBranchReadout(nn.Module):
    def __init__(self, *, branch_slots):
        super().__init__()
        if type(branch_slots) is not int or not 1<=branch_slots<=64:
            raise ValueError('explicit bounded branch slot count required')
        self.branch_slots=branch_slots
        self.anchor=nn.Linear(128,4)
        self.branch_slots_embedding=nn.Parameter(torch.randn(branch_slots,128)/128**.5)
        self.branch_adapter=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,128),nn.GELU())
        self.branch=nn.Linear(128,4)

    def forward(self, anchor_queries):
        q=anchor_queries
        if (q.ndim!=2 or q.shape[1]!=128 or not 1<=len(q)<=32
                or q.dtype not in (torch.float32,torch.float64) or not torch.isfinite(q).all()
                or q.dtype!=self.anchor.weight.dtype or q.device!=self.anchor.weight.device):
            raise ValueError('finite same-device Qx128 observation queries required')
        anchor=self.anchor(q)
        branch=self.branch(self.branch_adapter(q[:,None,:]+self.branch_slots_embedding[None,:,:]))
        raw=branch[...,:3];norm=torch.linalg.vector_norm(raw,dim=-1,keepdim=True)
        if not torch.isfinite(branch).all() or torch.any(norm<=32*torch.finfo(q.dtype).eps):
            raise ValueError('degenerate branch direction; do not invent a fallback vector')
        return AnchorBranchPrediction(smooth_anchor_coordinates(anchor[:,:3]),anchor[:,3],
            raw/norm,branch[...,3])
