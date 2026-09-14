"""Compact sensor-context lookup with explicit cross-artifact source agreement.

Caller must authenticate the artifacts whose stamps are supplied. Equal stamps
alone do not prove bytes authentic. Nearest-return distance is exposed: context
lookup does NOT certify that a candidate is on a surface or visible structure.
"""
from dataclasses import dataclass
import re
import numpy as np
from scipy.spatial import cKDTree
import torch
from .gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures


@dataclass(frozen=True)
class ObservationBinding:
    observation_id: str
    frame_indices: tuple[int, ...]
    coordinate_frame: str
    input_sha256: str
    encoder_sha256: str

    def validate(self):
        if not isinstance(self.observation_id,str) or not self.observation_id.strip(): raise ValueError('observation ID required')
        if (not isinstance(self.frame_indices,tuple) or len(self.frame_indices)!=5
                or any(type(x)!=int or x<0 for x in self.frame_indices)
                or tuple(sorted(set(self.frame_indices)))!=self.frame_indices): raise ValueError('five ordered frames required')
        if self.coordinate_frame!='current_sensor': raise ValueError('canonical current_sensor frame required')
        if any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in (self.input_sha256,self.encoder_sha256)):
            raise ValueError('SHA256 bindings required')


@dataclass(frozen=True)
class CandidateContext:
    features: torch.Tensor
    supported: torch.Tensor
    source_point_index: torch.Tensor
    sensor_token_index: torch.Tensor
    nearest_return_distance_m: torch.Tensor


def bind_candidate_context(compact, positions_m, candidate_binding, cache_binding):
    for stamp in (candidate_binding,cache_binding):
        if not isinstance(stamp,ObservationBinding): raise ValueError('typed binding required')
        stamp.validate()
    if candidate_binding!=cache_binding: raise ValueError('candidate/cache source mismatch')
    if not isinstance(compact,CompactFrozenDualPathFeatures): raise ValueError('typed compact cache required')
    p,f,v,idx=compact.points_xyz_m,compact.frozen_sensor_context,compact.valid,compact.sensor_token_index
    if (p.shape!=(1,57600,3) or f.shape!=(1,900,128) or v.shape!=(1,57600) or idx.shape!=(1,57600)
            or p.dtype not in (torch.float32,torch.float64) or f.dtype!=p.dtype or v.dtype!=torch.bool or idx.dtype!=torch.long):
        raise ValueError('single compact observation layout required')
    if any(t.device.type!='cpu' or t.requires_grad for t in (p,f,v,idx)):
        raise ValueError('detached CPU cache required; no population-wide point feature expansion')
    expected=(torch.arange(5)[:,None,None]*180+torch.arange(720)[None,None]//4).expand(5,16,720).reshape(1,-1)
    if not torch.equal(idx,expected): raise ValueError('sensor layout identity mismatch')
    if not torch.isfinite(p).all() or not torch.isfinite(f).all(): raise ValueError('nonfinite compact cache')
    if (not isinstance(positions_m,torch.Tensor) or positions_m.ndim!=2 or positions_m.shape[1]!=3
            or len(positions_m)>4592 or positions_m.device.type!='cpu' or positions_m.dtype!=p.dtype
            or not torch.isfinite(positions_m).all()): raise ValueError('finite CPU candidate positions required')
    n=len(positions_m);features=torch.zeros(n,128,dtype=p.dtype);support=torch.zeros(n,dtype=torch.bool)
    source=torch.full((n,),-1,dtype=torch.long);tokens=source.clone();dist=torch.full((n,),float('nan'),dtype=p.dtype)
    valid_indices=torch.nonzero(v[0],as_tuple=True)[0]
    if len(valid_indices) and n:
        xyz=p[0,valid_indices].numpy();tree=cKDTree(xyz);q=positions_m.detach().numpy()
        nearest,indices=tree.query(q,k=1)
        # Stable ties use the smallest authenticated sensor-layout index, never
        # a label or nearest target. Include roundoff-equivalent nearest returns.
        eps=64*np.finfo(xyz.dtype).eps
        for j,d in enumerate(nearest):
            ties=tree.query_ball_point(q[j],float(d)+eps*max(1.,float(d)))
            chosen=min(ties,key=lambda k:int(valid_indices[k]))
            source[j]=valid_indices[chosen]
        tokens=idx[0,source];features=f[0,tokens].clone();support[:]=True
        dist=torch.linalg.vector_norm(p[0,source]-positions_m.detach(),dim=-1)
    return CandidateContext(features,support,source,tokens,dist)
