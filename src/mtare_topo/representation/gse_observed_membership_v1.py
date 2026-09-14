"""Observation-seeded local structure/opening correspondence adapter.

Reuses existing raw/surface encoders and decoder, not the retired score head.
No free positional queries, teacher proposals or point-to-GT assignment enter
forward. Window openings are separate from persistent structure anchors.
This is an untrained interface, not an effective detector or a safety model.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .gse_surface_relation_model_v1 import SurfaceRelationModelV1
from .observed_anchor_detector_v1 import residual_positions


@dataclass(frozen=True)
class ObservedMembershipPrediction:
    anchor_position_m: torch.Tensor
    anchor_presence_logits: torch.Tensor
    opening_position_m: torch.Tensor
    opening_presence_logits: torch.Tensor
    opening_direction: torch.Tensor
    membership_logits: torch.Tensor
    anchor_source_indices: torch.Tensor
    opening_source_indices: torch.Tensor


def _seeds(points):
    """Up to64 unique observed coordinates; stable lexicographic FPS ties."""
    unique, source = np.unique(points, axis=0, return_index=True)
    count = min(64, len(unique))
    distance = np.full(len(unique), np.inf)
    available = np.ones(len(unique), bool)
    selected = []; index = 0
    for _ in range(count):
        selected.append(index); available[index] = False
        distance = np.minimum(distance, np.sum((unique-unique[index])**2, axis=1))
        index = int(np.argmax(np.where(available, distance, -np.inf)))
    return source[np.asarray(selected, dtype=np.int64)]


class ObservedMembershipV1(SurfaceRelationModelV1):
    def __init__(self, variant='C', *, membership_gradient_to_shared=True):
        if type(membership_gradient_to_shared) is not bool:
            raise ValueError('explicit gradient-routing boolean required')
        if variant not in ('A','B','C'):
            raise ValueError('explicit A observation / B unary / C relation variant required')
        # B/C retain the same adjacency and trainable shape; B zeros edge attributes.
        super().__init__('A' if variant == 'A' else 'C')
        self.variant = variant
        self.membership_gradient_to_shared = membership_gradient_to_shared
        for name in ('anchor_queries','opening_queries','anchor_uncertainty',
                     'opening_dimensions','dimension_evidence','opening_support',
                     'reachability','validity_anchor','validity_opening'):
            delattr(self, name)
        self.query_position = nn.Sequential(nn.Linear(3,128),nn.GELU(),nn.Linear(128,128))
        self.query_role = nn.Parameter(torch.zeros(2,128))

    def forward(self, points_xyz_m, frozen_point_context, valid, patches=None, *, sensor_token_index=None):
        if points_xyz_m.ndim != 3 or points_xyz_m.shape[0] != 1 or points_xyz_m.shape[1] > 57600:
            raise ValueError('one bounded five-frame observation per microbatch required')
        memory, memory_valid = self._raw(points_xyz_m, frozen_point_context, valid, sensor_token_index)
        observed = points_xyz_m[0,valid[0]]
        if not len(observed):
            if patches is not None and bool(patches.valid.any()):
                raise ValueError('patches cannot replace absent observations')
            return None
        if bool((torch.linalg.vector_norm(observed.double(),dim=-1)>10.).any()):
            raise ValueError('input must already use the fixed10m observation domain')
        if self.variant != 'A':
            if self.variant == 'B':
                if not bool(torch.isfinite(patches.relation).all()):
                    raise ValueError('nonfinite edge attributes')
                patches = replace(patches, relation=torch.zeros_like(patches.relation))
            pm, pv = self._patch(patches,1,memory.dtype,memory.device)
            memory = torch.cat((memory,pm),1); memory_valid = torch.cat((memory_valid,pv),1)
        selected = _seeds(observed.detach().double().cpu().numpy())
        source = torch.nonzero(valid[0],as_tuple=False).flatten()[
            torch.as_tensor(selected,device=valid.device)]
        # Sorting/deduplication choose coordinates, never GT membership or scores.
        opening_xyz = points_xyz_m[0,source]; anchor_xyz = opening_xyz[:32]
        point_features = self.raw_adapter(torch.cat((opening_xyz/10.,
            frozen_point_context[0,source].detach()),-1))
        opening_query = point_features+self.query_position(opening_xyz/10.)+self.query_role[1]
        anchor_query = point_features[:32]+self.query_position(anchor_xyz/10.)+self.query_role[0]
        decoded = self.decoder(torch.cat((anchor_query,opening_query))[None],memory,
            memory_key_padding_mask=~memory_valid)[0]
        na = len(anchor_xyz); anchor, opening = decoded[:na], decoded[na:]
        anchor_position,_ = residual_positions(anchor_xyz,self.anchor_position(anchor))
        # The reference opening is the10m window section, not a physical node.
        opening_vector = opening_xyz/10.+self.opening_position(opening)
        if bool((torch.linalg.vector_norm(opening_vector,dim=-1)<=torch.finfo(opening.dtype).eps).any()):
            raise ValueError('undefined opening position direction')
        opening_position = 10.*F.normalize(opening_vector,dim=-1)
        direction = self.opening_orientation(opening)
        if bool((torch.linalg.vector_norm(direction,dim=-1)<=torch.finfo(opening.dtype).eps).any()):
            raise ValueError('undefined opening orientation')
        member_opening = opening if self.membership_gradient_to_shared else opening.detach()
        member_anchor = anchor if self.membership_gradient_to_shared else anchor.detach()
        result = ObservedMembershipPrediction(anchor_position,self.anchor_presence(anchor).squeeze(-1),
            opening_position,self.opening_presence(opening).squeeze(-1),F.normalize(direction,dim=-1),
            self.member_opening(member_opening)@self.member_anchor(member_anchor).T/math.sqrt(128),source[:32],source)
        if any(not bool(torch.isfinite(v).all()) for v in vars(result).values()):
            raise ValueError('nonfinite structural prediction')
        return result
