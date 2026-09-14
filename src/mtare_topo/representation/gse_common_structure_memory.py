"""Three-path observation consumer, without detector heads or GT queries.

A: full context and observed rays. B adds unary patch updates. C adds sparse
patch relation messages. B/C parameters have the same layout; the message
mechanism itself is part of the C-versus-B comparison, not only edge attributes.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .gse_common_observation import CommonObservation
from .gse_surface_relation_model_v1 import _MessageLayer,collate_surface_patches


@dataclass(frozen=True)
class StructuralMemory:
    features: torch.Tensor
    valid: torch.Tensor
    context_slice: tuple
    ray_slice: tuple
    patch_slice: tuple
    is_structure_prediction: bool = False


class CommonStructureMemory(nn.Module):
    def __init__(self,variant):
        super().__init__()
        if variant not in ('A','B','C'):raise ValueError('A/B/C variant required')
        self.variant=variant
        self.context_adapter=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128))
        self.ray_adapter=nn.Sequential(nn.Linear(13,128),nn.GELU(),nn.Linear(128,128))
        self.patch_adapter=nn.Sequential(nn.Linear(18,128),nn.GELU(),nn.Linear(128,128))
        self.patch_layers=nn.ModuleList(_MessageLayer() for _ in range(3))
        self.kind_embedding=nn.Parameter(torch.zeros(3,128))

    def forward(self,observation,*,patch_batch=None):
        if not isinstance(observation,CommonObservation):raise ValueError('teacher-free common observation required')
        device=self.kind_embedding.device;dtype=self.kind_embedding.dtype
        ctx=observation.full_sensor_context.detach().to(device=device,dtype=dtype)
        cv=observation.full_sensor_valid.to(device=device)
        if ctx.shape!=(900,128) or cv.shape!=(900,) or cv.dtype!=torch.bool or not torch.isfinite(ctx).all():
            raise ValueError('finite full context and independent900 validity required')
        context=self.context_adapter(ctx)+self.kind_embedding[0]
        context=torch.where(cv[:,None],context,0.)
        rays=observation.local_rays;n=len(rays.ray_indices)
        if n>57600:raise ValueError('five-frame ray capacity exceeded')
        ids=np.asarray(observation.ray_sensor_token_indices)
        frames=np.asarray(observation.ray_history_indices)
        if ids.shape!=(n,) or frames.shape!=(n,) or np.any((ids<0)|(ids>=900)) or np.any((frames<0)|(frames>4)):
            raise ValueError('ray layout indices required')
        if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(frames.dtype,np.integer):raise ValueError('integer layout indices required')
        span=rays.end_xyz_m-rays.start_xyz_m
        length=np.linalg.norm(span,axis=1)
        if np.any(length<=0):raise ValueError('positive observed segment required')
        geometry=np.concatenate((rays.start_xyz_m/10.,rays.end_xyz_m/10.,span/length[:,None],
            rays.start_distance_from_origin_m[:,None]/50.,rays.end_distance_from_origin_m[:,None]/50.,
            rays.end_is_observed_return[:,None].astype(float),frames[:,None]/4.),axis=1)
        if geometry.shape!=(n,13) or not np.isfinite(geometry).all():raise ValueError('finite13D ray geometry required')
        pooled=context.new_zeros(900,128);counts=context.new_zeros(900)
        # Nonlinear encoding precedes pooling; never expand per-point128D
        # frozen context or construct all-rays pairwise attention.
        for start in range(0,n,1024):
            sl=slice(start,start+1024)
            x=torch.tensor(geometry[sl],device=device,dtype=dtype)
            index=torch.tensor(ids[sl],device=device,dtype=torch.long)
            pooled=pooled.index_add(0,index,self.ray_adapter(x))
            counts.index_add_(0,index,torch.ones(len(index),device=device,dtype=dtype))
        rv=counts>0
        ray_memory=pooled/counts.clamp_min(1)[:,None]+self.kind_embedding[1]
        ray_memory=torch.where(rv[:,None],ray_memory,0.)
        memories=[context,ray_memory];masks=[cv,rv];end=1800
        if self.variant!='A':
            p=patch_batch if patch_batch is not None else collate_surface_patches([observation.surface_patches],device=device,dtype=dtype)
            if p.unary.ndim!=3 or p.unary.shape[0]!=1 or p.unary.shape[-1]!=18 or p.unary.shape[1]>4096:
                raise ValueError('bounded single patch batch required')
            m=p.unary.shape[1]
            if m<1 or p.valid.shape!=(1,m) or p.neighbor_index.shape!=(1,m,8) or p.neighbor_valid.shape!=(1,m,8) or p.relation.shape!=(1,m,8,9):
                raise ValueError('patch layout drift')
            if p.valid.dtype!=torch.bool or p.neighbor_valid.dtype!=torch.bool or p.neighbor_index.dtype!=torch.long:
                raise ValueError('patch index and mask dtypes required')
            if p.unary.dtype!=dtype or p.relation.dtype!=dtype or any(t.device!=device for t in (p.unary,p.valid,p.neighbor_index,p.neighbor_valid,p.relation)):
                raise ValueError('patch device or floating dtype drift')
            if bool(((p.neighbor_index < -1)|(p.neighbor_index>=m)).any()) or not torch.equal(p.neighbor_valid,p.neighbor_index>=0):
                raise ValueError('patch neighbor identity/mask drift')
            target=p.valid[0,p.neighbor_index.clamp_min(0)]
            if bool((p.neighbor_valid&(~target|~p.valid[...,None])).any()):
                raise ValueError('patch edge uses absent endpoint')
            if not torch.isfinite(p.unary).all() or not torch.isfinite(p.relation).all():raise ValueError('nonfinite patch input')
            feature=self.patch_adapter(p.unary.detach())
            for layer in self.patch_layers:feature=layer(feature,p,self.variant=='C')
            feature=feature[0]+self.kind_embedding[2]
            feature=torch.where(p.valid[0,:,None],feature,0.)
            memories.append(feature);masks.append(p.valid[0]);end+=len(feature)
        return StructuralMemory(torch.cat(memories)[None],torch.cat(masks)[None],(0,900),(900,1800),(1800,end))
