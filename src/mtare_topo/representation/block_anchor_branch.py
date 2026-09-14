"""Independent adapter reusing a copied legacy block decoder.

All96 original decoder queries are retained for identical self-attention;
the first32 feed the new head. Unused legacy output heads are not copied.
The supplied legacy model is never mutated or shared with this module.
"""
from copy import deepcopy
import torch
from torch import nn
from .gse_block_structure_readout import BlockStructureReadout
from .anchor_branch_readout import AnchorBranchReadout


class BlockAnchorBranch(nn.Module):
    def __init__(self, legacy, *, branch_slots):
        super().__init__()
        if not isinstance(legacy,BlockStructureReadout) or legacy.queries.shape!=(96,128):
            raise ValueError('existing96x128 block decoder required')
        self.adapter=deepcopy(legacy.adapter)
        self.decoder=deepcopy(legacy.decoder)
        self.queries=nn.Parameter(legacy.queries.detach().clone())
        self.head=AnchorBranchReadout(branch_slots=branch_slots).to(
            device=legacy.queries.device,dtype=legacy.queries.dtype)

    def encode_queries(self, point_features, frozen_context):
        if set(point_features)!={'embedding','centers_m','extent_m','degenerate'}:
            raise ValueError('only observation block fields allowed')
        e=point_features['embedding'];c=point_features['centers_m']
        s=point_features['extent_m'];d=point_features['degenerate'];m=len(e)
        if (e.shape!=(m,128) or c.shape!=(m,3) or s.shape!=(m,3) or d.shape!=(m,)
                or frozen_context.shape!=(m,128) or m>4096 or d.dtype!=torch.bool):
            raise ValueError('bounded aligned observation block features required')
        for value in (e,c,s,frozen_context):
            if (value.dtype!=self.queries.dtype or value.device!=self.queries.device
                    or not torch.isfinite(value).all()):
                raise ValueError('feature dtype/device/numeric drift')
        if d.device!=e.device or torch.any(s<0):raise ValueError('invalid extent/degeneracy')
        if not m:return None  # Explicit no observation, no guessed nodes/directions.
        memory=self.adapter(torch.cat((e,frozen_context.detach(),c/10.,s/10.,d[:,None].to(e.dtype)),dim=1))
        return self.decoder(self.queries[None],memory[None])[0,:32]

    def forward(self, point_features, frozen_context):
        queries=self.encode_queries(point_features,frozen_context)
        return None if queries is None else self.head(queries)
