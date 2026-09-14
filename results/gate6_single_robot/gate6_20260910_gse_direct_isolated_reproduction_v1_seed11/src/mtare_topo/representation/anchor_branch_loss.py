"""Partial anchor/branch objective, not aperture or connectivity supervision.

Reuses geometry-only assignment and rejects ambiguous matches. Unknown
background is disconnected, not a zero class. No training is authorized here.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .gse_surface_losses_v1 import _assign


@dataclass(frozen=True)
class AnchorBranchPrediction:
    position_m: torch.Tensor              # Q,3
    presence_logits: torch.Tensor         # Q
    directions: torch.Tensor              # Q,K,3 directed unit vectors
    branch_logits: torch.Tensor           # Q,K


@dataclass(frozen=True)
class PartialAnchorBranches:
    position_m: torch.Tensor              # T,3, all listed anchors are observed
    directions: tuple[torch.Tensor, ...]  # each U,3; only observed directions
    anchors_complete: bool
    branches_complete: tuple[bool, ...]


def anchor_branch_loss(prediction, target):
    if type(prediction) is not AnchorBranchPrediction or type(target) is not PartialAnchorBranches:
        raise ValueError('typed anchor/branch contract required')
    p=prediction;t=target;xyz=p.position_m;dirs=p.directions
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or not 1<=len(xyz)<=32
            or dirs.ndim!=3 or dirs.shape[0]!=len(xyz) or dirs.shape[2]!=3 or not 1<=dirs.shape[1]<=64
            or p.presence_logits.shape!=(len(xyz),) or p.branch_logits.shape!=dirs.shape[:2]):
        raise ValueError('bounded prediction shapes required')
    if xyz.dtype not in (torch.float32,torch.float64):raise ValueError('floating predictions required')
    for value in (xyz,dirs,p.presence_logits,p.branch_logits):
        if value.device!=xyz.device or value.dtype!=xyz.dtype or not torch.isfinite(value).all():
            raise ValueError('finite shared prediction dtype/device required')
    if (t.position_m.ndim!=2 or t.position_m.shape[1]!=3 or type(t.anchors_complete) is not bool
            or len(t.directions)!=len(t.position_m) or len(t.branches_complete)!=len(t.position_m)
            or any(type(v) is not bool for v in t.branches_complete)):
        raise ValueError('explicit per-anchor completeness required')
    for value in (t.position_m,*t.directions):
        if (value.ndim!=2 or value.shape[1]!=3 or value.dtype!=xyz.dtype or value.device!=xyz.device
                or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError('detached finite same-device known targets required')
    for value in (dirs.reshape(-1,3),*t.directions):
        norms=torch.linalg.vector_norm(value,dim=-1)
        if not torch.allclose(norms,torch.ones_like(norms),atol=64*torch.finfo(xyz.dtype).eps,rtol=0):
            raise ValueError('directed unit vectors required')
    for value in (xyz,t.position_m):
        if torch.any(torch.linalg.vector_norm(value,dim=-1)>10.+64*torch.finfo(xyz.dtype).eps*10):
            raise ValueError('anchor outside shared10m domain')
    # No graph from unknown tasks to model parameters (including AdamW).
    zero=xyz.new_zeros((),requires_grad=True)
    numerators={name:zero for name in ('anchor_presence','anchor_position','branch_presence','branch_direction')}
    counts={name:0 for name in numerators}
    def add(name,value,count):
        numerators[name]=numerators[name]+value;counts[name]+=count
    def match(pred,true):
        valid=torch.ones((1,len(true)),dtype=torch.bool,device=xyz.device)
        return _assign(pred[None],true[None],valid)[0]
    assignments=match(xyz,t.position_m);branch_assignments=[]
    if len(assignments):
        add('anchor_position',torch.linalg.vector_norm(xyz[assignments]-t.position_m,dim=-1).sum()/10.,len(assignments))
        add('anchor_presence',F.softplus(-p.presence_logits[assignments]).sum(),len(assignments))
    if t.anchors_complete:
        mask=torch.ones(len(xyz),dtype=torch.bool,device=xyz.device);mask[assignments]=False
        if mask.any():add('anchor_presence',F.softplus(p.presence_logits[mask]).sum(),int(mask.sum()))
    for i,q in enumerate(assignments.tolist()):
        target_dirs=t.directions[i];matched=match(dirs[q],target_dirs);branch_assignments.append(matched)
        if len(matched):
            add('branch_direction',(1-(dirs[q,matched]*target_dirs).sum(-1).clamp(-1,1)).sum()/2.,len(matched))
            add('branch_presence',F.softplus(-p.branch_logits[q,matched]).sum(),len(matched))
        if t.branches_complete[i]:
            mask=torch.ones(dirs.shape[1],dtype=torch.bool,device=xyz.device);mask[matched]=False
            if mask.any():add('branch_presence',F.softplus(p.branch_logits[q,mask]).sum(),int(mask.sum()))
    terms={k:numerators[k]/max(1,counts[k]) for k in counts}
    return dict(total=sum(terms.values()),terms=terms,counts=counts,
        has_supervision=any(counts.values()),anchor_assignment=assignments,
        branch_assignments=tuple(branch_assignments),aperture_supervised=False)
