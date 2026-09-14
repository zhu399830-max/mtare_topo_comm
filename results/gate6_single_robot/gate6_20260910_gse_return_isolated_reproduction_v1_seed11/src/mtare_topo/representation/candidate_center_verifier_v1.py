"""Candidate-conditioned block readout. No IDs, teacher, or old query features."""
import torch
from torch import nn
from torch.nn import functional as F


class CandidateCenterVerifierV1(nn.Module):
    def __init__(self,dimension=128):
        super().__init__();self.dimension=dimension
        self.pair=nn.Sequential(nn.Linear(dimension+7,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.score=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,candidate_xyz,block_xyz,block_extent,block_features,block_valid,block_degenerate):
        q=len(candidate_xyz);m=len(block_xyz)
        if candidate_xyz.shape!=(q,3) or block_xyz.shape!=(m,3) or block_extent.shape!=(m,3) or block_features.shape!=(m,self.dimension) or block_valid.shape!=(m,) or block_valid.dtype!=torch.bool or block_degenerate.shape!=(m,1):
            raise ValueError('aligned observation-only tensors required')
        # Padding/invalid values must never enter normalization, extrema, or denominator.
        c=block_xyz[block_valid];e=block_extent[block_valid];f=block_features[block_valid];d=block_degenerate[block_valid]
        if not all(torch.isfinite(v).all() for v in (candidate_xyz,c,e,f,d)):raise ValueError('nonfinite valid input')
        if not len(c):
            # Neutral finite logit is explicitly NO_OBSERVATION, never a negative label.
            return candidate_xyz.new_zeros(q)+sum(p.sum()*0 for p in self.parameters())
        relative=(c[None]-candidate_xyz[:,None])/10.
        features=F.layer_norm(f,(self.dimension,),weight=None,bias=None,eps=1e-5)
        geometry=torch.cat((relative,e[None].expand(q,-1,-1)/10.,d[None].expand(q,-1,-1).to(f.dtype)),dim=-1)
        token=torch.cat((features[None].expand(q,-1,-1),geometry),dim=-1)
        pair=self.pair(token);pooled=torch.cat((pair.mean(1),pair.amax(1)),dim=-1)
        return self.score(pooled)[:,0]


def balanced_validity_loss(logits,positive,negative):
    if positive.dtype!=torch.bool or negative.dtype!=torch.bool or positive.shape!=logits.shape or negative.shape!=logits.shape or (positive&negative).any():
        raise ValueError('disjoint aligned masks required')
    groups=[]
    if positive.any():groups.append(F.softplus(-logits[positive]).mean())
    if negative.any():groups.append(F.softplus(logits[negative]).mean())
    return sum(groups)/len(groups) if groups else logits.sum()*0.


@torch.no_grad()
def capture_observation_memory(model,student):
    """Hook existing adapter output memory; parent source/parameters untouched."""
    from .grouping_center_training_v1 import forward
    captured={}
    def capture(_,args,out):captured['block_features']=out.detach().clone()
    handle=model['head'].adapter.register_forward_hook(capture)
    try:output=forward(model,student,'PRIMITIVE')
    finally:handle.remove()
    blocks=student.student_representations['PRIMITIVE']['blocks'];features=captured['block_features']
    def tensor(x):return torch.as_tensor(x.copy(),device=features.device,dtype=features.dtype)
    packet=dict(candidate_xyz=output.prediction.position_m.detach().clone(),block_xyz=tensor(blocks.centers_m),block_extent=tensor(blocks.extent_m),
        block_features=features,block_valid=torch.ones(len(features),dtype=torch.bool,device=features.device),block_degenerate=tensor(blocks.degenerate)[:,None])
    return output,packet
