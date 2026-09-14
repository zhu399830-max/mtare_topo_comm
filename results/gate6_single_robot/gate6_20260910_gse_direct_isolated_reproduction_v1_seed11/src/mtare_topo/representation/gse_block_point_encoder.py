"""PointNet-style shared block encoder, not an SPG reproduction or detector.

Chunked pointwise evaluation retains one bounded N x 128 embedding tensor for
one observation. Training activations are NOT claimed constant in chunk size.
No population-wide feature expansion and no N x N attention are used.
"""
import numpy as np
import torch
from torch import nn
from .gse_block_points import BlockPoints


class BlockPointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Independent points: no BatchNorm/dropout/chunk-dependent statistics.
        self.pointwise=nn.Sequential(nn.Linear(14,128),nn.GELU(),nn.Linear(128,128))

    def forward(self,blocks:BlockPoints,*,chunk_size=2048):
        if not isinstance(blocks,BlockPoints) or not 1<=chunk_size<=57600:
            raise ValueError('bound block input and bounded positive chunk required')
        if len(blocks.xyz_m)>57600 or len(blocks.block_ids)>4096:
            raise ValueError('single-observation capacity exceeded')
        parameter=next(self.parameters());device=parameter.device;dtype=parameter.dtype
        def tensor(x,kind=dtype):return torch.as_tensor(np.array(x,copy=True),device=device,dtype=kind)
        x=tensor(blocks.xyz_m);g=tensor(blocks.point_to_block,torch.long)
        c=tensor(blocks.centers_m);extent=tensor(blocks.extent_m)
        frame=tensor(blocks.frame_index,torch.long)
        count=len(c)
        if not len(x):
            return dict(embedding=x.new_empty((0,128)),centers_m=c,extent_m=extent,
                        degenerate=tensor(blocks.degenerate,torch.bool))
        outputs=[]
        for start in range(0,len(x),chunk_size):
            end=min(len(x),start+chunk_size);group=g[start:end]
            local=x[start:end]-c[group]
            scale=extent[group].amax(dim=1,keepdim=True)
            # A zero-size block stays zero locally; the actual extent and
            # degeneracy are exposed. The denominator is not a guessed size.
            normalized=local/torch.where(scale>0,scale,torch.ones_like(scale))
            history=nn.functional.one_hot(frame[start:end],num_classes=5).to(dtype)
            features=torch.cat((x[start:end]/10.,local/10.,normalized,history),dim=1)
            outputs.append(self.pointwise(features))
        features=torch.cat(outputs,dim=0)
        pooled=features.new_full((count,128),-torch.inf)
        # One reduction across all points also preserves the same tie-gradient
        # convention across chunks, unlike successive pairwise torch.maximum.
        pooled=pooled.scatter_reduce(0,g[:,None].expand(-1,128),features,reduce='amax',include_self=False)
        if not bool(torch.isfinite(pooled).all()):raise ValueError('nonfinite block embedding')
        return dict(embedding=pooled,centers_m=c,extent_m=extent,
                    degenerate=tensor(blocks.degenerate,torch.bool))
